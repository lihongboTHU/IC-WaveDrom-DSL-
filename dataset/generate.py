import wavedrom
import json
import random
import os
import cairosvg
import cv2
import numpy as np
import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 📂 1. 初始化存储目录
# =========================
os.makedirs("train_images", exist_ok=True)
os.makedirs("train_labels", exist_ok=True)

augmentation_records = []

# =========================
# 💦 2. 工业级防伪水印系统
# =========================
watermarks = [
    "CONFIDENTIAL", "INTERNAL USE ONLY", "DO NOT COPY",
    "FOR REVIEW ONLY", "COPYRIGHT", "CLASSIFIED",
    "TOP SECRET"
]

SIGNAL_NAME_POOLS = {
    "spi_cs": ["CSn", "CS#", "nCS", "CS_N", "SS", "NSS", "STE"],
    "spi_clk": ["SCK", "SCLK", "CLK", "SPI_CLK"],
    "spi_mosi": ["MOSI", "SDI", "DIN", "SI"],
    "spi_miso": ["MISO", "SDO", "DOUT", "SO"],
    "i2c_scl": ["SCL", "SCLK"],
    "i2c_sda": ["SDA", "SDIO"],
    "mem_clk": ["SYS_CLK", "CLK", "MCLK", "BCLK"],
    "mem_ctrl": ["WE", "WE#", "WR#", "OE", "OE#", "RD#", "CS", "CS#"],
    "mem_addr": ["ADDR", "Address", "A[15:0]", "A[7:0]"],
    "mem_data": ["DATA", "Data", "D[7:0]", "DQ[15:0]", "Address/Data"],
    "misc": [
        "RSTn", "RESET#", "INT", "IRQ", "READY", "VALID", "WDT", "GPIO",
        "TX", "RX", "BUSY", "SYNC", "ACK", "WP", "WP#", "HOLD", "HOLD#",
        "LRCLK", "FSYNC", "SDATA", "MCLK", "BCLK"
    ]
}

DATASHEET_DISTRACTORS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8",
    "MSB", "LSB", "Left", "Right", "16-bit", "20-bit", "24-bit",
    "setup", "hold", "tSU", "tH", "sample", "valid", "don't care"
]

STATIC_WAVE_CHARS = set("01hlxz")

def canonicalize_wave(wave_str):
    """
    Fold repeated static states into WaveDrom dots.
    Keep '=', p/n/P/N uncollapsed because they denote data cells or clock edges.
    """
    if not wave_str:
        return wave_str

    result = [wave_str[0]]
    prev = wave_str[0]
    for ch in wave_str[1:]:
        if ch == prev and ch in STATIC_WAVE_CHARS:
            result.append(".")
        else:
            result.append(ch)
            prev = ch
    return "".join(result)

def canonicalize_signals(signals):
    for sig in signals:
        sig["wave"] = canonicalize_wave(sig.get("wave", ""))
        if "data" in sig:
            expected = sig["wave"].count("=")
            sig["data"] = sig["data"][:expected]
            while len(sig["data"]) < expected:
                sig["data"].append(" ")
    return signals

def collect_used_names(signals):
    return {sig.get("name") for sig in signals if sig.get("name")}

def summarize_data_box_styles(signals):
    summary = {
        "DataBox_PerCycle": 0,
        "DataBox_SingleLarge": 0,
        "DataBox_Chunked": 0
    }
    for sig in signals:
        wave = sig.get("wave", "")
        if "=" not in wave:
            continue
        has_spanning_box = "=." in wave
        eq_count = wave.count("=")
        if not has_spanning_box:
            summary["DataBox_PerCycle"] += 1
        elif eq_count == 1:
            summary["DataBox_SingleLarge"] += 1
        else:
            summary["DataBox_Chunked"] += 1
    return summary

def add_single_watermark(img, text, angle, alpha, color, font_scale, thickness, center_pos=None):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = int(max(tw, th) * 0.2)
    canvas_w, canvas_h = tw + 2 * margin, th + 2 * margin
    text_x, text_y = margin, margin + th

    text_img = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    cv2.putText(text_img, text, (text_x, text_y), font, font_scale,
                (color[0], color[1], color[2], 255), thickness, cv2.LINE_AA)

    center = (canvas_w // 2, canvas_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(text_img, rot_mat, (canvas_w, canvas_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if center_pos is None:
        center_x, center_y = random.randint(0, w), random.randint(0, h)
    else:
        center_x, center_y = center_pos

    top_left_x, top_left_y = center_x - canvas_w // 2, center_y - canvas_h // 2
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    x1, y1 = max(0, top_left_x), max(0, top_left_y)
    x2, y2 = min(w, top_left_x + canvas_w), min(h, top_left_y + canvas_h)
    rx1, ry1 = max(0, -top_left_x), max(0, -top_left_y)
    rx2, ry2 = rx1 + (x2 - x1), ry1 + (y2 - y1)

    overlay[y1:y2, x1:x2, :] = rotated[ry1:ry2, rx1:rx2, :]
    img_f = img.astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    overlay_mask = (overlay[:, :, 3] > 0).astype(np.float32)

    for c in range(3):
        img_f[:, :, c] = img_f[:, :, c] * (1 - overlay_mask * alpha) + overlay_rgb[:, :, c] * overlay_mask * alpha

    return np.clip(img_f, 0, 255).astype(np.uint8)

def add_watermark(img):
    h, w = img.shape[:2]
    text = random.choice(watermarks)
    angle, alpha = random.uniform(-45, 45), random.uniform(0.15, 0.25)
    color, font_scale, thickness = (180, 180, 180), random.uniform(1.0, 2.0), random.randint(3, 5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    step = np.sqrt(tw**2 + th**2) * 0.8
    rad = np.deg2rad(angle)
    perp_x, perp_y = -np.sin(rad), np.cos(rad)

    centers = []
    for off in [-step, 0, step]:
        cx, cy = w // 2 + perp_x * off, h // 2 + perp_y * off
        centers.append((int(np.clip(cx, 0, w)), int(np.clip(cy, 0, h))))

    img_result = img.copy()
    for center in centers:
        img_result = add_single_watermark(img_result, text, angle, alpha, color, font_scale, thickness, center)
    return img_result

# =========================
# 🌪️ 3. 实拍物理退化增强核心库
# =========================
def add_perspective(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    shift_x, shift_y = w * random.uniform(0.02, 0.05), h * random.uniform(0.02, 0.05)
    pts2 = np.float32([
        [random.uniform(0, shift_x), random.uniform(0, shift_y)],
        [w - random.uniform(0, shift_x), random.uniform(0, shift_y)],
        [random.uniform(0, shift_x), h - random.uniform(0, shift_y)],
        [w - random.uniform(0, shift_x), h - random.uniform(0, shift_y)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

def add_lines(img):
    h, w = img.shape[:2]
    for _ in range(random.randint(1, 3)):
        cv2.line(img, (random.randint(0, w), random.randint(0, h)), (random.randint(0, w), random.randint(0, h)), (0, 0, 0), 1)
    return img

def add_noise(img):
    noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def add_jpeg_compression(img):
    quality = random.randint(45, 85)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def add_downsample(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.55, 0.85)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def add_occlusion(img):
    h, w = img.shape[:2]
    for _ in range(random.randint(1, 2)):
        max_box_w, max_box_h = min(int(w * 0.05), 20), min(int(h * 0.10), 20)
        if max_box_w < 5 or max_box_h < 5: continue 
        x1, y1 = random.randint(0, w - max_box_w), random.randint(0, h - max_box_h)
        x2, y2 = x1 + random.randint(5, max_box_w), y1 + random.randint(5, max_box_h)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    return img

def add_lighting(img):
    return cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))

def add_datasheet_distractors(img):
    """
    Add small annotation-like text that should be ignored by the label.
    This teaches the model that bit indices and MSB/LSB notes are not signals.
    """
    h, w = img.shape[:2]
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = random.choice([(70, 70, 70), (100, 100, 100), (130, 130, 130)])
    font_scale = random.uniform(0.35, 0.65)
    thickness = 1

    def safe_randint(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi < lo:
            return lo
        return random.randint(lo, hi)

    if random.random() < 0.65:
        count = random.randint(4, 10)
        y = min(h - 2, random.choice([max(14, int(h * 0.05)), max(14, int(h * 0.12)), h - 10]))
        x0 = safe_randint(20, max(21, int(w * 0.15)))
        step = max(22, int((w - x0 - 20) / max(1, count - 1)))
        for idx in range(count):
            cv2.putText(out, str(idx), (x0 + idx * step, y), font, font_scale, color, thickness, cv2.LINE_AA)

    for _ in range(random.randint(1, 3)):
        text = random.choice(DATASHEET_DISTRACTORS)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = safe_randint(4, max(5, w - tw - 4))
        top_y = safe_randint(max(12, th), max(max(12, th), int(h * 0.2)))
        bottom_y = safe_randint(max(12, int(h * 0.75)), max(max(12, int(h * 0.75)), h - 6))
        y = min(h - 2, random.choice([top_y, bottom_y]))
        cv2.putText(out, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return out

def apply_background(img):
    bg_type = random.choice(["white", "yellow", "gray"])
    colors = {"white": (255,255,255,255), "yellow": (245,245,220,255), "gray": (230,230,230,255)}
    bg = Image.new("RGBA", img.size, colors[bg_type])
    img = Image.alpha_composite(bg, img)
    return img.convert("RGB")

# =========================
# 📈 正态分布采样函数
# =========================
def sample_normal_int(mean, std, min_val, max_val):
    value = int(round(np.random.normal(mean, std)))
    return int(np.clip(value, min_val, max_val))

def weighted_sample_without_replacement(items, weights, k):
    pool = list(zip(items, weights))
    result = []
    for _ in range(min(k, len(pool))):
        total = sum(weight for _, weight in pool)
        pick = random.uniform(0, total)
        upto = 0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= pick:
                result.append(item)
                pool.pop(idx)
                break
    return result

# =========================
# 🧠 4. 全局对齐式总线协议引擎 
# =========================
def populate_data(wave_str):
    data_tokens = []
    for i in range(wave_str.count("=")):
        r = random.random()
        if r < 0.12:
            data_tokens.append(random.choice(["MSB", "LSB", "ADDR", "DATA", f"bit{i}"]))
        elif r < 0.25:
            data_tokens.append(f"{random.randint(0, 255):02X}h")
        else:
            data_tokens.append(f"0x{random.randint(0, 255):02X}")
    return data_tokens

def make_data_body(length):
    """
    Generate bus-valid regions with mixed WaveDrom data-box styles.
    - "====" means one data box per cycle.
    - "=..." means one large data box spanning the region.
    - "=..=.." means a few coarse data boxes.
    """
    if length <= 0:
        return ""

    style = random.choices(
        ["per_cycle", "single_box", "chunked_box"],
        weights=[0.45, 0.35, 0.20],
        k=1
    )[0]

    if style == "per_cycle" or length == 1:
        return "=" * length

    if style == "single_box":
        return "=" + "." * (length - 1)

    body = []
    pos = 0
    while pos < length:
        remaining = length - pos
        if remaining == 1:
            body.append("=")
            break
        chunk = random.randint(2, min(6, remaining))
        body.append("=" + "." * (chunk - 1))
        pos += chunk
    return "".join(body)[:length]

def protocol_spi(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)
    
    cpol, clk_char, data_phase = random.choice(['0', '1']), random.choice(['p', 'n', 'P', 'N']), random.choice([0, 0.5])
    
    cs_wave = ("1" * start + "0" * active + "1" * end)[:total_len]
    clk_wave = (cpol * start + clk_char * active + cpol * end)[:total_len]
    m_wave = ("z" * start + make_data_body(active) + "z" * end)[:total_len]

    signals = [{"name": random.choice(SIGNAL_NAME_POOLS["spi_cs"]), "wave": cs_wave},
               {"name": random.choice(SIGNAL_NAME_POOLS["spi_clk"]), "wave": clk_wave},
               {"name": random.choice(SIGNAL_NAME_POOLS["spi_mosi"]), "wave": m_wave, "data": populate_data(m_wave)}]
    if data_phase > 0: signals[-1]["phase"] = data_phase
    if random.random() < 0.6:
        sig_miso = {"name": random.choice(SIGNAL_NAME_POOLS["spi_miso"]), "wave": m_wave, "data": populate_data(m_wave)}
        if data_phase > 0: sig_miso["phase"] = data_phase
        signals.append(sig_miso)
    return signals

def protocol_i2c(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)

    scl_wave = ("1" * start + "p" * active + "1" * end)[:total_len]
    sda_wave = ("1" * max(0, start - 1) + "0" + make_data_body(max(0, active - 1)) + "0" + "1" * end)
    sda_wave = (sda_wave + "1" * total_len)[:total_len] 

    return [{"name": random.choice(SIGNAL_NAME_POOLS["i2c_scl"]), "wave": scl_wave},
            {"name": random.choice(SIGNAL_NAME_POOLS["i2c_sda"]), "wave": sda_wave, "data": populate_data(sda_wave), "phase": random.choice([0, 0.2])}]

def protocol_memory(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)
    
    clk_wave = (random.choice(['p', 'n']) * total_len)[:total_len]
    ctrl_wave = ("1" * start + "0" * active + "1" * end)[:total_len]
    addr_wave = ("x" * start + make_data_body(active) + "x" * end)[:total_len]
    data_wave = ("z" * start + make_data_body(active) + "z" * end)[:total_len]
    data_phase = random.choice([0, 0.5])

    return [{"name": random.choice(SIGNAL_NAME_POOLS["mem_clk"]), "wave": clk_wave},
            {"name": random.choice(SIGNAL_NAME_POOLS["mem_ctrl"]), "wave": ctrl_wave},
            {"name": random.choice(SIGNAL_NAME_POOLS["mem_addr"]), "wave": addr_wave, "data": populate_data(addr_wave), "phase": data_phase},
            {"name": random.choice(SIGNAL_NAME_POOLS["mem_data"]), "wave": data_wave, "data": populate_data(data_wave), "phase": data_phase}]

def generate_logical_timing():
    # Keep most samples short enough for stable autoregressive JSON learning.
    # A small tail of longer samples is still useful for hard cases.
    if random.random() < 0.8:
        total_cycles = sample_normal_int(mean=14, std=4, min_val=5, max_val=24)
    else:
        total_cycles = sample_normal_int(mean=26, std=6, min_val=18, max_val=40)
    used_names = set()

    protocol_signals = random.choice([
        protocol_spi,
        protocol_i2c,
        protocol_memory
    ])(total_cycles, used_names)
    used_names.update(collect_used_names(protocol_signals))

    if random.random() < 0.8:
        target_signal_count = sample_normal_int(mean=5, std=1.8, min_val=1, max_val=8)
    else:
        target_signal_count = sample_normal_int(mean=9, std=2.0, min_val=7, max_val=12)
    target_signal_count = max(target_signal_count, len(protocol_signals))

    extra_needed = target_signal_count - len(protocol_signals)
    bg_signals = []
    available = list(set(SIGNAL_NAME_POOLS["misc"]) - used_names)

    for _ in range(extra_needed):
        if not available: break
        name = random.choice(available)
        available.remove(name); used_names.add(name)
        
        raw_wave = ""
        while len(raw_wave) < total_cycles: 
            raw_wave += random.choice(['0', '1', 'z', 'x']) * random.randint(2, 6)
        raw_wave = raw_wave[:total_cycles]
        
        folded_wave = raw_wave[0]
        for i in range(1, len(raw_wave)):
            folded_wave += '.' if raw_wave[i] == raw_wave[i - 1] else raw_wave[i]

        bg_signals.append({"name": name, "wave": folded_wave})

    all_signals = protocol_signals + bg_signals
    for sig in all_signals:
        if random.random() < 0.5:
            sig["wave"] = sig["wave"].replace('0', 'l').replace('1', 'h')

    random.shuffle(all_signals)
    return canonicalize_signals(all_signals)

# =========================
# 🏃‍♂️ 5. 主程序启动 (带打点记录)
# =========================
num_samples = 200
print(f"Start rendering {num_samples} timing diagram samples...")

augment_pool = {
    "Blur": lambda x: cv2.GaussianBlur(x, (3, 3), 0),
    "Noise": add_noise,
    "Lighting": add_lighting,
    "Occlusion": add_occlusion,
    "Perspective": add_perspective,
    "Interference Lines": add_lines,
    "Watermark": add_watermark,
    "JPEG Compression": add_jpeg_compression,
    "Downsample": add_downsample
}
aug_names = list(augment_pool.keys())
aug_weights = {
    "Blur": 1.0,
    "Noise": 1.0,
    "Lighting": 1.4,
    "Occlusion": 0.35,
    "Perspective": 0.75,
    "Interference Lines": 0.35,
    "Watermark": 0.25,
    "JPEG Compression": 1.2,
    "Downsample": 1.0
}

for i in range(num_samples):
    signals = generate_logical_timing()
    wd_dict = {"signal": signals}
    wd_json = json.dumps(wd_dict, ensure_ascii=False, separators=(",", ":"))

    with open(f"train_labels/{i:04d}.txt", "w", encoding='utf-8') as f:
        f.write(wd_json)

    svg = wavedrom.render(wd_json)
    
    if hasattr(svg, 'tostring'):
        svg_bytes = svg.tostring().encode('utf-8')
    else:
        svg_bytes = svg.encode('utf-8')

    png_data = cairosvg.svg2png(bytestring=svg_bytes)
    
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    img = apply_background(img)

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    has_datasheet_distractors = int(random.random() < 0.45)
    if has_datasheet_distractors:
        cv_img = add_datasheet_distractors(cv_img)

    # Curriculum-like mix: keep many clean or lightly degraded samples.
    # Heavy samples are useful, but too many make small VLM decoding unstable.
    r = random.random()
    if r < 0.40:
        num_aug = 0
    elif r < 0.78:
        num_aug = 1
    elif r < 0.95:
        num_aug = 2
    else:
        num_aug = 3
    selected_aug_names = weighted_sample_without_replacement(
        aug_names,
        [aug_weights[name] for name in aug_names],
        num_aug
    )

    for name in selected_aug_names:
        cv_img = augment_pool[name](cv_img)

    cv2.imwrite(f"train_images/{i:04d}.png", cv_img)

    record = {
        "Image_ID": f"{i:04d}.png",
        "Total_Augmentations": num_aug,
        "Datasheet Distractors": has_datasheet_distractors
    }
    for name in aug_names:
        record[name] = 1 if name in selected_aug_names else 0
    record.update(summarize_data_box_styles(signals))
    augmentation_records.append(record)

    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1} / {num_samples}")

print("Image generation completed. Drawing augmentation statistics...")

# =========================
# 📊 6. 自动化统计绘图模块
# =========================
df = pd.DataFrame(augmentation_records)
df.to_csv("augmentation_records.csv", index=False)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def draw_dashboard(language='en', suffix='en'):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if language == 'cn':
        title = f'物理退化追踪仪表板 (样本数 N={num_samples})'
        ax0_title = '每张图片应用的增强数量'
        ax0_xlabel = '应用增强方式数量 (0 = 原始图像)'
        ax0_ylabel = '图像数量'
        ax1_title = '各增强策略使用频次'
        ax1_xlabel = '总使用次数'
    else:
        title = f'Physical Degradation Tracking Dashboard (N={num_samples})'
        ax0_title = 'Amount of Augmentations Applied per Image'
        ax0_xlabel = 'Number of Applied Augmentation Methods (0 = Clean/Original)'
        ax0_ylabel = 'Image Count'
        ax1_title = 'Trigger Frequency of Each Augmentation Strategy'
        ax1_xlabel = 'Total Times Triggered'

    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    aug_counts = df['Total_Augmentations'].value_counts().sort_index()
    
    # 强制确保 0, 1, 2, 3 都在图中显示
    for i in range(4):
        if i not in aug_counts: 
            aug_counts[i] = 0
    aug_counts = aug_counts.sort_index()
        
    palette = sns.color_palette("coolwarm", n_colors=len(aug_counts))
    ax00 = axes[0, 0]
    sns.barplot(x=aug_counts.index, y=aug_counts.values, ax=ax00, hue=aug_counts.index, palette=palette, legend=False)
    ax00.set_title(ax0_title, fontweight='bold')
    ax00.set_xlabel(ax0_xlabel)
    ax00.set_ylabel(ax0_ylabel)
    
    max_height = max(aug_counts.values) if len(aug_counts) > 0 else 1
    for i, v in enumerate(aug_counts.values):
        ax00.text(i, v + max_height * 0.01, str(v), ha='center', fontweight='bold')
    
    hit_counts = df[aug_names].sum().sort_values(ascending=False)
    palette2 = sns.color_palette("magma", n_colors=len(hit_counts))
    ax01 = axes[0, 1]
    sns.barplot(x=hit_counts.values, y=hit_counts.index, ax=ax01, hue=hit_counts.index, palette=palette2, legend=False)
    ax01.set_title(ax1_title, fontweight='bold')
    ax01.set_xlabel(ax1_xlabel)
    
    max_hit = max(hit_counts.values) if len(hit_counts) > 0 else 1
    for i, v in enumerate(hit_counts.values):
        ax01.text(v + max_hit * 0.01, i, str(v), va='center', fontweight='bold')

    style_cols = ["DataBox_PerCycle", "DataBox_SingleLarge", "DataBox_Chunked"]
    style_counts = df[style_cols].sum()
    style_labels = ["Per-cycle boxes", "Single large box", "Chunked boxes"]
    ax10 = axes[1, 0]
    sns.barplot(x=style_labels, y=style_counts.values, ax=ax10, hue=style_labels, palette="Set2", legend=False)
    ax10.set_title("Data Box Style Coverage", fontweight='bold')
    ax10.set_ylabel("Signal Count")
    ax10.tick_params(axis='x', rotation=12)
    max_style = max(style_counts.values) if len(style_counts) > 0 else 1
    for i, v in enumerate(style_counts.values):
        ax10.text(i, v + max_style * 0.01, str(int(v)), ha='center', fontweight='bold')

    distractor_counts = pd.Series({
        "With distractors": int(df["Datasheet Distractors"].sum()),
        "Without distractors": int(len(df) - df["Datasheet Distractors"].sum())
    })
    ax11 = axes[1, 1]
    ax11.pie(
        distractor_counts.values,
        labels=distractor_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette("pastel"),
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    ax11.set_title("Datasheet Distractor Coverage", fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'augmentation_statistics_{suffix}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"{language.upper()} chart saved to: {filename}")

draw_dashboard(language='en', suffix='en')
draw_dashboard(language='cn', suffix='cn')

print("Chinese and English charts generated.")
print("Augmentation details saved to: augmentation_records.csv")

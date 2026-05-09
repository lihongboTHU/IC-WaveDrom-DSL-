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
    "TOP SECRET", "YX always in my heart", "Missing you forever LYX"
]

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

# =========================
# 🧠 4. 全局对齐式总线协议引擎 
# =========================
def populate_data(wave_str):
    return [" " if random.random() < 0.1 else f"0x{random.randint(0, 255):02X}" for _ in range(wave_str.count("="))]

def protocol_spi(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)
    
    cpol, clk_char, data_phase = random.choice(['0', '1']), random.choice(['p', 'n', 'P', 'N']), random.choice([0, 0.5])
    
    cs_wave = ("1" * start + "0" * active + "1" * end)[:total_len]
    clk_wave = (cpol * start + clk_char * active + cpol * end)[:total_len]
    m_wave = ("z" * start + "=" * active + "z" * end)[:total_len]

    signals = [{"name": random.choice(["CSn", "SS", "NSS"]), "wave": cs_wave},
               {"name": random.choice(["SCK", "SCLK", "CLK"]), "wave": clk_wave},
               {"name": "MOSI", "wave": m_wave, "data": populate_data(m_wave)}]
    if data_phase > 0: signals[-1]["phase"] = data_phase
    if random.random() < 0.6:
        sig_miso = {"name": "MISO", "wave": m_wave, "data": populate_data(m_wave)}
        if data_phase > 0: sig_miso["phase"] = data_phase
        signals.append(sig_miso)
    return signals

def protocol_i2c(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)

    scl_wave = ("1" * start + "p" * active + "1" * end)[:total_len]
    sda_wave = ("1" * max(0, start - 1) + "0" + "=" * max(0, active - 1) + "0" + "1" * end)
    sda_wave = (sda_wave + "1" * total_len)[:total_len] 

    return [{"name": "SCL", "wave": scl_wave},
            {"name": "SDA", "wave": sda_wave, "data": populate_data(sda_wave), "phase": random.choice([0, 0.2])}]

def protocol_memory(total_len, used_names):
    start = random.randint(1, max(1, total_len // 4))
    end = random.randint(1, max(1, total_len // 4))
    active = max(1, total_len - start - end)
    
    clk_wave = (random.choice(['p', 'n']) * total_len)[:total_len]
    ctrl_wave = ("1" * start + "0" * active + "1" * end)[:total_len]
    addr_wave = ("x" * start + "=" * active + "x" * end)[:total_len]
    data_wave = ("z" * start + "=" * active + "z" * end)[:total_len]
    data_phase = random.choice([0, 0.5])

    return [{"name": random.choice(["SYS_CLK", "CLK"]), "wave": clk_wave},
            {"name": random.choice(["WE", "OE", "CS"]), "wave": ctrl_wave},
            {"name": "ADDR", "wave": addr_wave, "data": populate_data(addr_wave), "phase": data_phase},
            {"name": "DATA", "wave": data_wave, "data": populate_data(data_wave), "phase": data_phase}]

def generate_logical_timing():
    total_cycles = sample_normal_int(mean=20, std=7, min_val=5, max_val=40)
    used_names = set()

    protocol_signals = random.choice([
        protocol_spi,
        protocol_i2c,
        protocol_memory
    ])(total_cycles, used_names)

    target_signal_count = sample_normal_int(mean=6, std=2.5, min_val=1, max_val=12)
    target_signal_count = max(target_signal_count, len(protocol_signals))

    extra_needed = target_signal_count - len(protocol_signals)
    bg_signals = []
    available = list(set(["RSTn", "INT", "READY", "VALID", "WDT", "GPIO", "TX", "RX", "IRQ", "BUSY", "SYNC", "ACK"]) - used_names)

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
    return all_signals

# =========================
# 🏃‍♂️ 5. 主程序启动 (带打点记录)
# =========================
num_samples = 20000
print(f"🚀 开始渲染 {num_samples} 张物理退化逻辑时序图...")

augment_pool = {
    "Blur": lambda x: cv2.GaussianBlur(x, (3, 3), 0),
    "Noise": add_noise,
    "Lighting": add_lighting,
    "Occlusion": add_occlusion,
    "Perspective": add_perspective,
    "Interference Lines": add_lines,
    "Watermark": add_watermark
}
aug_names = list(augment_pool.keys())

for i in range(num_samples):
    signals = generate_logical_timing()
    wd_dict = {"signal": signals}
    wd_json = json.dumps(wd_dict)

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

    # 抽取 0~2 个随机增强
    num_aug = random.randint(0, 2)
    selected_aug_names = random.sample(aug_names, num_aug)

    for name in selected_aug_names:
        cv_img = augment_pool[name](cv_img)

    cv2.imwrite(f"train_images/{i:04d}.png", cv_img)

    record = {"Image_ID": f"{i:04d}.png", "Total_Augmentations": num_aug}
    for name in aug_names:
        record[name] = 1 if name in selected_aug_names else 0
    augmentation_records.append(record)

    if (i + 1) % 10 == 0:
        print(f"✅ 进度: {i + 1} / {num_samples} 张")

print("🎉 图片生成完毕！开始进行增强方式数据统计绘图...")

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
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
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
    
    # 强制确保 0, 1, 2 都在图中显示，剔除多余的 3
    for i in range(3):
        if i not in aug_counts: 
            aug_counts[i] = 0
    aug_counts = aug_counts.sort_index()
        
    palette = sns.color_palette("coolwarm", n_colors=len(aug_counts))
    sns.barplot(x=aug_counts.index, y=aug_counts.values, ax=axes[0], hue=aug_counts.index, palette=palette, legend=False)
    axes[0].set_title(ax0_title, fontweight='bold')
    axes[0].set_xlabel(ax0_xlabel)
    axes[0].set_ylabel(ax0_ylabel)
    
    max_height = max(aug_counts.values) if len(aug_counts) > 0 else 1
    for i, v in enumerate(aug_counts.values):
        axes[0].text(i, v + max_height * 0.01, str(v), ha='center', fontweight='bold')
    
    hit_counts = df[aug_names].sum().sort_values(ascending=False)
    palette2 = sns.color_palette("magma", n_colors=len(hit_counts))
    sns.barplot(x=hit_counts.values, y=hit_counts.index, ax=axes[1], hue=hit_counts.index, palette=palette2, legend=False)
    axes[1].set_title(ax1_title, fontweight='bold')
    axes[1].set_xlabel(ax1_xlabel)
    
    max_hit = max(hit_counts.values) if len(hit_counts) > 0 else 1
    for i, v in enumerate(hit_counts.values):
        axes[1].text(v + max_hit * 0.01, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f'augmentation_statistics_{suffix}.png'
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"📊 {language.upper()} 图表已保存至：{filename}")

draw_dashboard(language='en', suffix='en')
draw_dashboard(language='cn', suffix='cn')

print("✅ 中文和英文图表均已生成！")
print("📁 追踪详情仍保存至：augmentation_records.csv")
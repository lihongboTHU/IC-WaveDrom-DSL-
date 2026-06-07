import json
import re
import difflib
import paddle
from PIL import Image
from tqdm import tqdm
from paddleformers.transformers import (
    AutoModelForConditionalGeneration,
    AutoProcessor
)
from paddleformers.generation import GenerationConfig

# =========================================================
# 1. 模型加载
# =========================================================

# 基于paddleOCR-VL的微调模型
# model_path = "./PaddleOCR-VL-SFT-WaveDrom-LoRA/export"

# 基于paddleOCR-VL1.5的微调模型
# model_path = "./output_PaddleOCR-VL-1.5-SFT-ICOCR-lora_withTextIntensify_model/export"

# 原始模型
# model_path = "./PaddlePaddle/PaddleOCR-VL" 

# model_path = "./PaddleOCR-VL-SFT-IC-lora_model2/export" # 多轮训练+高秩lora微调结果损失函数

# model_path = "./output_PaddleOCR-VL-1.5-SFT-ICOCR-lora_model2/export" # 多轮训练+高秩lora微调结果损失函数 + 大规模数据增强（文本增强） + 权重合并（merge）

model_path = "./output_lora_stage4_full/export"

model = AutoModelForConditionalGeneration.from_pretrained(
    model_path,
    convert_from_hf=True
).eval()

model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(model_path)

generation_config = GenerationConfig(
    do_sample=True,         # 开启采样
    temperature=0.1,        # 极低的温度，几乎等同于贪婪解码，但保留了一丝随机性
    top_p=0.9,
    max_new_tokens=2048,
    pad_token_id=0,
    eos_token_id=2,
    repetition_penalty=1.1
)

# =========================================================
# 2. WaveDrom 处理函数
# =========================================================

def expand_wave(wave_str):
    """
    展开 WaveDrom wave 字符串中的 '.'
    """
    if not wave_str:
        return ""

    result = []
    prev = ""
    for ch in wave_str:
        if ch == ".":
            result.append(prev)
        else:
            result.append(ch)
            prev = ch

    return "".join(result)

def normalize_json_text(text):
    """
    清理 Markdown 符号，尝试提取干净的 JSON
    """
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text

def extract_signal_list(obj):
    """
    将 signal 提取为列表，保留从上到下的顺序，方便按行对比
    """
    signals = []
    if "signal" not in obj:
        return signals

    for item in obj["signal"]:
        # 防御性编程：跳过非字典类型的异常数据（如嵌套组）
        if isinstance(item, list) or not isinstance(item, dict):
            continue 
            
        signals.append({
            "name": str(item.get("name", "")),
            "wave": expand_wave(str(item.get("wave", ""))),
            "data": [str(x) for x in item.get("data", [])]
        })
    return signals


# =========================================================
# 3. 全新打分函数 (基于相似度和按行匹配)
# =========================================================

def calc_text_similarity(str1, str2):
    """
    基于编辑距离(SequenceMatcher)计算两个字符串的相似度 (0.0 到 1.0)
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def calc_data_similarity(data1, data2):
    """
    比较 Data 列表的相似度。
    拼接成字符串并忽略空格，完美解决模型多生成或少生成空格的问题。
    """
    str1 = "|".join(data1).replace(" ", "") 
    str2 = "|".join(data2).replace(" ", "")
    return calc_text_similarity(str1, str2)

def evaluate_single_row(gt_sig, pred_sig):
    """
    综合评估单行信号的准确率：Name, Wave, Data 综合加权
    """
    name_sim = calc_text_similarity(gt_sig["name"], pred_sig["name"])
    wave_sim = calc_text_similarity(gt_sig["wave"], pred_sig["wave"])
    
    # 动态权重分配：
    if len(gt_sig["data"]) == 0:
        # 如果 GT 这行本来就没有数据，权重分配给 Name 和 Wave
        row_score = (name_sim * 0.3) + (wave_sim * 0.7)
        data_sim = 1.0 # 占位
    else:
        # 如果有数据，三者共同评分
        data_sim = calc_data_similarity(gt_sig["data"], pred_sig["data"])
        row_score = (name_sim * 0.2) + (wave_sim * 0.4) + (data_sim * 0.4)
        
    return {
        "signal": gt_sig["name"] if gt_sig["name"] else "Unnamed",
        "pred_name": pred_sig["name"],
        "name_acc": name_sim,
        "wave_acc": wave_sim,
        "data_acc": data_sim,
        "row_accuracy": row_score
    }

def evaluate(gt_obj, pred_obj):
    """
    综合评估整个 JSON
    """
    gt_signals = extract_signal_list(gt_obj)
    pred_signals = extract_signal_list(pred_obj)

    signal_scores = []
    total_score = 0.0
    
    # 取最大行数。如果模型漏行或者产生幻觉多出几行，都会作为分母惩罚总分
    max_len = max(len(gt_signals), len(pred_signals))

    for i in range(max_len):
        if i < len(gt_signals) and i < len(pred_signals):
            # GT 和 预测 都有这一行，进行综合比对
            row_eval = evaluate_single_row(gt_signals[i], pred_signals[i])
            signal_scores.append(row_eval)
            total_score += row_eval["row_accuracy"]
            
        elif i < len(gt_signals):
            # 预测漏了这行
            signal_scores.append({
                "signal": gt_signals[i]["name"],
                "error": "Missing in prediction",
                "row_accuracy": 0.0
            })
            
        else:
            # 预测多出了幻觉行
            signal_scores.append({
                "signal": f"Hallucination (Extra): {pred_signals[i].get('name', 'Unknown')}",
                "error": "Extra output",
                "row_accuracy": 0.0
            })

    overall_acc = (total_score / max_len) if max_len > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "signal_scores": signal_scores
    }


# =========================================================
# 4. 推理函数
# =========================================================

def inference_one(image_path, prompt="Return only one valid WaveDrom JSON object. Do not output extra signals. Do not include time-axis labels."):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pd"
    )

    with paddle.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    output_text = processor.decode(
        outputs[0].tolist()[0],
        skip_special_tokens=True
    )
    return output_text


# =========================================================
# 5. 批量评估
# =========================================================

jsonl_path = "./dataset/eval_messages_labeled.jsonl"

all_results = []
overall_scores = []
difficulty_scores = {
    "易": [],
    "中": [],
    "难": [],
    "未知": []
}

with open(jsonl_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

    for idx, line in enumerate(tqdm(lines)):
        sample = json.loads(line)
        difficulty = sample.get("difficulty", "未知")
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = []

        # GT
        gt_text = normalize_json_text(sample["messages"][1]["content"])
        try:
            gt_obj = json.loads(gt_text)
        except Exception as e:
            print(f"\nGT JSON 解析失败: {idx}")
            continue

        image_path = sample["images"][0]

        # 推理
        pred_text = normalize_json_text(inference_one(image_path))

        # 解析与评估
        try:
            pred_obj = json.loads(pred_text)
            eval_result = evaluate(gt_obj, pred_obj)
            overall_acc = eval_result["overall_accuracy"]
            
            # 记录结果
            overall_scores.append(overall_acc)
            difficulty_scores[difficulty].append(overall_acc)

            result = {
                "index": idx,
                "image": image_path,
                "difficulty": difficulty,
                "accuracy": overall_acc,
                "details": eval_result["signal_scores"],
                "prediction": pred_obj
            }
            all_results.append(result)
            print(f"\n[{idx}] 难度: {difficulty} | ACC = {overall_acc:.4f}")

        except Exception as e:
            print(f"\n预测 JSON 解析失败: {idx}")
            
            overall_scores.append(0.0)
            difficulty_scores[difficulty].append(0.0)

            result = {
                "index": idx,
                "image": image_path,
                "difficulty": difficulty,
                "accuracy": 0.0,
                "error": "json_parse_failed",
                "raw_output": pred_text
            }
            all_results.append(result)
            print(f"\n[{idx}] 难度: {difficulty} | ACC = 0.0000 (解析失败)")


# =========================================================
# 6. 输出最终结果
# =========================================================

final_acc = (sum(overall_scores) / len(overall_scores)) if len(overall_scores) > 0 else 0

print("\n=================================================")
print(f"总体评估结果 (共 {len(overall_scores)} 个样本):")
print(f"最终平均准确率: {final_acc:.4f}")
print("-------------------------------------------------")
print("各难度细分准确率:")

for diff in ["易", "中", "难", "未知"]:
    scores = difficulty_scores.get(diff, [])
    if len(scores) > 0:
        diff_acc = sum(scores) / len(scores)
        print(f" - [{diff}] 难度 (样本数: {len(scores)}): {diff_acc:.4f}")

print("=================================================")

# =========================================================
# 7. 保存结果
# =========================================================

with open("eval_results2.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n详细评测结果已保存到 eval_results.json")
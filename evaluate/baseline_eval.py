import json
import paddle
from PIL import Image
from tqdm import tqdm
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from evaluation import evaluate, normalize_json_text

"""
基线模型评估脚本
该脚本用于评估未微调的基础模型 (PaddleOCR-VL)，获取其在 WaveDrom 评估集上的真实指标，
用于对比证明微调(LoRA)所带来的巨大性能提升。
"""

# =========================================================
# 1. 模型加载 (使用原始基础模型，非微调模型)
# =========================================================

# 请确保您的网络能够访问 HuggingFace，或者提前下载好基础模型放在本地并修改此路径
baseline_model_path = "PaddlePaddle/PaddleOCR-VL"

print(f"正在加载基础模型: {baseline_model_path}")
model = AutoModelForConditionalGeneration.from_pretrained(
    baseline_model_path,
    convert_from_hf=True
).eval()

model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained(baseline_model_path)

generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
    pad_token_id=0,
    eos_token_id=2
)

# =========================================================
# 2. 推理函数
# =========================================================

def inference_one(image_path, prompt="WaveDrom Recognition:"):
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
# 3. 批量评估
# =========================================================

jsonl_path = "./dataset/annotations.jsonl"

all_results = []
overall_scores = []
difficulty_scores = {
    "易": [],
    "中": [],
    "难": [],
    "未知": []
}

print("开始基线模型评估...")

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
                "prediction": pred_obj
            }
            all_results.append(result)

        except Exception as e:
            # 基础模型极大概率无法生成合法的 JSON，这是正常的
            overall_scores.append(0.0)
            difficulty_scores[difficulty].append(0.0)

            result = {
                "index": idx,
                "image": image_path,
                "difficulty": difficulty,
                "accuracy": 0.0,
                "error": "json_parse_failed_baseline_expected",
                "raw_output": pred_text
            }
            all_results.append(result)

# =========================================================
# 4. 输出最终结果
# =========================================================

final_acc = (sum(overall_scores) / len(overall_scores)) if len(overall_scores) > 0 else 0

print("\n=================================================")
print(f"基础模型 (Baseline) 总体评估结果 (共 {len(overall_scores)} 个样本):")
print(f"最终平均准确率: {final_acc:.4f}  (预期会非常低甚至为0)")
print("-------------------------------------------------")
print("各难度细分准确率:")

for diff in ["易", "中", "难", "未知"]:
    scores = difficulty_scores.get(diff, [])
    if len(scores) > 0:
        diff_acc = sum(scores) / len(scores)
        print(f" - [{diff}] 难度 (样本数: {len(scores)}): {diff_acc:.4f}")

print("=================================================")

# =========================================================
# 5. 保存结果
# =========================================================

with open("baseline_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n详细基线评测结果已保存到 baseline_eval_results.json")
print("提示：您可以将此脚本的运行结果与微调后的 eval_results.json 进行对比，放入报告中以凸显微调的收益。")

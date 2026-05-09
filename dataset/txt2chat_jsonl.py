import os
import json
import glob

# ==================== 配置区域 ====================
IMG_DIR = "train_images"
LABEL_DIR = "train_labels"
# 输出文件路径
OUTPUT_JSONL = "train_chat.jsonl"

# 目标图片路径前缀
IMAGE_BASE_PATH = "./dataset/train_images"
USER_CONTENT = "<image>\nWaveDrom Recognition:"

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
# =================================================

def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_JSONL)), exist_ok=True)

    # 收集所有图片文件
    image_files = []
    for ext in IMG_EXTENSIONS:
        pattern = os.path.join(IMG_DIR, f"*{ext}")
        image_files.extend(glob.glob(pattern))
    image_files.sort()

    total = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

            if not os.path.exists(txt_path):
                print(f"⚠️ 跳过 {img_path}：找不到对应的标签文件 {txt_path}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                label_text = f.read().strip()
                if not label_text:
                    print(f"⚠️ 跳过 {txt_path}：内容为空")
                    continue

            if os.path.isdir(IMAGE_BASE_PATH):
                image_full_path = os.path.join(IMAGE_BASE_PATH, os.path.basename(img_path))
            else:
                image_full_path = os.path.join(IMAGE_BASE_PATH, os.path.basename(img_path))
            image_full_path = image_full_path.replace("\\", "/")  # 统一为 Unix 风格

            sample = {
                "messages": [
                    {"role": "user", "content": USER_CONTENT},
                    {"role": "assistant", "content": label_text}
                ],
                "images": [image_full_path]
            }

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ 转换完成！共生成 {total} 条数据，输出文件：{OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
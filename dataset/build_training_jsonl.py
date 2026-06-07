import argparse
import csv
import glob
import json
import os


IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]
DEFAULT_USER_CONTENT = (
    "<image>\n"
    "Return only one valid WaveDrom JSON object. "
    "Do not output extra signals. Do not include time-axis labels."
)


def load_augmentation_counts(csv_path):
    counts = {}
    if not os.path.exists(csv_path):
        return counts

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("Image_ID", "")
            try:
                counts[image_id] = int(float(row.get("Total_Augmentations", 0)))
            except ValueError:
                counts[image_id] = 0
    return counts


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def image_files(img_dir):
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(files)


def max_wave_len(signals):
    lengths = [len(str(sig.get("wave", ""))) for sig in signals if isinstance(sig, dict)]
    return max(lengths) if lengths else 0


def difficulty_score(label_obj, augmentation_count):
    signals = label_obj.get("signal", [])
    row_count = len(signals)
    return row_count * max_wave_len(signals) + augmentation_count * 10


def stage1_label(label_obj):
    signals = []
    for sig in label_obj.get("signal", []):
        if not isinstance(sig, dict):
            continue
        signals.append({"name": str(sig.get("name", ""))})
    return {"signal": signals}


def make_sample(image_path, assistant_obj, user_content):
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": compact_json(assistant_obj)}
        ],
        "images": [image_path.replace("\\", "/")]
    }


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def percentile(sorted_values, ratio):
    if not sorted_values:
        return 0
    idx = int(round((len(sorted_values) - 1) * ratio))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def main():
    parser = argparse.ArgumentParser(
        description="Build full, stage-1, and curriculum JSONL files for WaveDrom training."
    )
    parser.add_argument("--img_dir", default="train_images")
    parser.add_argument("--label_dir", default="train_labels")
    parser.add_argument("--augmentation_csv", default="augmentation_records.csv")
    parser.add_argument("--image_base_path", default="./dataset/train_images")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--easy_ratio", type=float, default=0.35)
    parser.add_argument("--medium_ratio", type=float, default=0.75)
    parser.add_argument("--user_content", default=DEFAULT_USER_CONTENT)
    args = parser.parse_args()

    aug_counts = load_augmentation_counts(args.augmentation_csv)
    full_samples = []
    stage1_samples = []
    scored_full_samples = []

    for img_path in image_files(args.img_dir):
        image_id = os.path.basename(img_path)
        base_name = os.path.splitext(image_id)[0]
        label_path = os.path.join(args.label_dir, f"{base_name}.txt")
        if not os.path.exists(label_path):
            print(f"Skip {image_id}: missing label {label_path}")
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            label_obj = json.load(f)

        image_ref = os.path.join(args.image_base_path, image_id)
        augmentation_count = aug_counts.get(image_id, 0)
        score = difficulty_score(label_obj, augmentation_count)

        full_sample = make_sample(image_ref, label_obj, args.user_content)
        stage1_sample = make_sample(image_ref, stage1_label(label_obj), args.user_content)

        full_samples.append(full_sample)
        stage1_samples.append(stage1_sample)
        scored_full_samples.append((score, full_sample))

    scores = sorted(score for score, _ in scored_full_samples)
    easy_threshold = percentile(scores, args.easy_ratio)
    medium_threshold = percentile(scores, args.medium_ratio)

    easy_samples = [sample for score, sample in scored_full_samples if score <= easy_threshold]
    easy_medium_samples = [sample for score, sample in scored_full_samples if score <= medium_threshold]

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {
        "train_stage1_names.jsonl": stage1_samples,
        "train_easy.jsonl": easy_samples,
        "train_easy_medium.jsonl": easy_medium_samples,
        "train_chat.jsonl": full_samples,
    }

    for filename, samples in outputs.items():
        path = os.path.join(args.output_dir, filename)
        write_jsonl(path, samples)
        print(f"Wrote {len(samples)} samples -> {path}")

    print(f"Difficulty formula: rows * max_wave_len + augmentations * 10")
    print(f"Easy threshold: {easy_threshold}")
    print(f"Easy+Medium threshold: {medium_threshold}")
    if scores:
        print(f"Score range: min={scores[0]}, max={scores[-1]}")


if __name__ == "__main__":
    main()

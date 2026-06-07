# Curriculum LoRA Training Guide

This project uses one rank-16 LoRA adapter and keeps training it through four curriculum stages.

The key idea is:

```text
Base PaddleOCR-VL
  + Stage 1 LoRA -> output_lora_stage1_names
  + continue same LoRA on Stage 2 -> output_lora_stage2_easy
  + continue same LoRA on Stage 3 -> output_lora_stage3_easy_medium
  + continue same LoRA on Stage 4 -> output_lora_stage4_full
```

Do not merge after every stage. Merge only once after Stage 4.

## Config Files

Run these configs in order:

```text
stage1_names_lora16.yaml
stage2_easy_lora16_continue.yaml
stage3_easy_medium_lora16_continue.yaml
stage4_full_lora16_continue.yaml
```

## Important Fields

Stage 1 starts from the base model:

```yaml
model_name_or_path: PaddlePaddle/PaddleOCR-VL
lora_rank: 16
output_dir: ./output_lora_stage1_names
```

Stage 2 starts from the base model and loads the Stage 1 LoRA:

```yaml
model_name_or_path: PaddlePaddle/PaddleOCR-VL
lora_rank: 16
lora_model_path: ./output_lora_stage1_names
output_dir: ./output_lora_stage2_easy
```

Stage 3 loads Stage 2 LoRA:

```yaml
lora_model_path: ./output_lora_stage2_easy
output_dir: ./output_lora_stage3_easy_medium
```

Stage 4 loads Stage 3 LoRA:

```yaml
lora_model_path: ./output_lora_stage3_easy_medium
output_dir: ./output_lora_stage4_full
```

## If Checkpoints Are Saved as checkpoint-xxx

Some trainers save LoRA weights inside checkpoint folders, for example:

```text
output_lora_stage1_names/checkpoint-1200
```

If that happens, use the latest checkpoint folder as `lora_model_path`:

```yaml
lora_model_path: ./output_lora_stage1_names/checkpoint-1200
```

The rule is simple:

```text
Stage N lora_model_path = Stage N-1 final LoRA directory or latest checkpoint directory
```

## Final Merge

After Stage 4, merge only the final LoRA:

```yaml
model_name_or_path: PaddlePaddle/PaddleOCR-VL
lora_model_path: ./output_lora_stage4_full
output_dir: ./output_lora_stage4_full
export_dir: ./IC-WaveDrom-DSL-final
```

If Stage 4 also saves checkpoint folders, use the latest checkpoint folder as `lora_model_path`.

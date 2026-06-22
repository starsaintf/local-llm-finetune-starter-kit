# Fine-tuning Commands

Run all commands from the repo root.

## 1. Set up Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## 2. Check GPU

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If CUDA is not available, fix your PyTorch/CUDA install before training.

## 3. Clean your SFT data

Put raw examples here:

```text
finetuning/data/raw_sft.jsonl
```

Then run:

```bash
python finetuning/scripts/preprocess_jsonl.py \
  --input finetuning/data/raw_sft.jsonl \
  --output finetuning/data/clean_sft.jsonl
```

## 4. Split train and eval

Create two files:

```text
finetuning/data/train_sft.jsonl
finetuning/data/eval_sft.jsonl
```

Use about 90% for train and 10% for eval.

Do not train on the eval examples.

## 5. Train with QLoRA

Use this first if you have a normal consumer GPU.

```bash
python finetuning/scripts/train_sft_hf.py \
  --model_path /models/base-model \
  --train_file finetuning/data/train_sft.jsonl \
  --eval_file finetuning/data/eval_sft.jsonl \
  --output_dir runs/finetuning/sft-qlora \
  --use_qlora \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_use_double_quant \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --bf16
```

Replace `/models/base-model` with your actual base model path or Hugging Face model id.

## 6. Train with LoRA

Use this if you have more VRAM.

```bash
python finetuning/scripts/train_sft_hf.py \
  --model_path /models/base-model \
  --train_file finetuning/data/train_sft.jsonl \
  --eval_file finetuning/data/eval_sft.jsonl \
  --output_dir runs/finetuning/sft-lora \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --bf16
```

## 7. Watch training

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

Loss should trend down, but loss is not quality. Always test the model on real prompts.

## 8. Merge adapter

```bash
python finetuning/scripts/merge_lora.py \
  --base_model /models/base-model \
  --adapter runs/finetuning/sft-lora \
  --output_dir artifacts/finetuning/merged
```

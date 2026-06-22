# Starter Commands and Guidelines

This file keeps the heavy command/manual layer out of the X article.

## 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## 2. Verify GPU

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If CUDA is not visible, fix PyTorch/CUDA before training.

## 3. Prepare data

Raw SFT data can be chat-style:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Or prompt-completion style:

```json
{"prompt":"...","completion":"..."}
```

Clean it:

```bash
python scripts/preprocess_jsonl.py \
  --input data/raw_sft.jsonl \
  --output data/clean_sft.jsonl
```

Then split it into:

```text
data/train_sft.jsonl
data/eval_sft.jsonl
```

Keep eval examples separate.

## 4. Tiny pilot run

Before the full run, use a small subset of the data and train for one epoch.

The goal is not quality yet. The goal is to confirm that:

- the script runs,
- the model loads,
- the data tokenizes,
- loss moves,
- generation still works.

## 5. QLoRA run

Best default for consumer GPUs.

```bash
python scripts/train_sft_hf.py \
  --model_path /models/base-model \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir runs/sft-qlora \
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

## 6. LoRA run

Use when you have more VRAM.

```bash
python scripts/train_sft_hf.py \
  --model_path /models/base-model \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir runs/sft-lora \
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

## 7. Full fine-tuning

Do not start here unless you have enough VRAM and a good reason.

```bash
python scripts/train_sft_hf.py \
  --model_path /models/base-model \
  --train_file data/train_sft.jsonl \
  --eval_file data/eval_sft.jsonl \
  --output_dir runs/sft-full \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16
```

## 8. TensorBoard

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

Watch training loss and eval loss.

Loss should trend down. Eval loss should not sharply rise while train loss falls.

## 9. Merge adapter

```bash
python scripts/merge_lora.py \
  --base_model /models/base-model \
  --adapter runs/sft-lora \
  --output_dir artifacts/merged
```

## 10. DPO run

Use DPO after SFT, not before.

```bash
python scripts/train_dpo.py \
  --model_path /models/base-or-sft-model \
  --train_file data/train_pref.jsonl \
  --eval_file data/eval_pref.jsonl \
  --output_dir runs/dpo \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 5e-6 \
  --beta 0.1 \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16
```

## Practical rules

Bad data beats good settings.

Loss is not quality.

Do not train on secrets.

Do not start with DPO.

Do not start with full fine-tuning.

Run the base model first.

Save the failures.

Turn those failures into examples.

Evaluate on prompts the model never trained on.

Keep the previous model around in case the new one gets weird.

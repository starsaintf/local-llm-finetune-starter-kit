# Local LLM Finetune Starter Kit

A practical starter kit for running, fine-tuning, evaluating, and serving a local LLM.

The idea is simple:

> Running a local model is easy. Making it useful after download is the real work.

This repo is not about training a language model from scratch. It is about taking an open-weight model that already knows language and adapting it to your work, your format, your tone, your workflow, and your failure cases.

## What is included

- `scripts/preprocess_jsonl.py` — clean and deduplicate SFT JSONL datasets.
- `scripts/train_sft_hf.py` — supervised fine-tuning with LoRA, QLoRA, or full fine-tuning.
- `scripts/train_dpo.py` — preference training with DPO.
- `scripts/merge_lora.py` — merge a LoRA adapter into the base model for serving.
- `data/sample_sft.jsonl` — tiny supervised fine-tuning sample dataset.
- `data/sample_pref.jsonl` — tiny preference-training sample dataset.
- `configs/deepspeed_zero2.json` — basic DeepSpeed ZeRO-2 config.
- `requirements.txt` — Python dependencies.
- `commands.md` — copy/paste commands and workflow notes.

## The workflow

1. Run the base model first.
2. Test it on real prompts from your work.
3. Save the failures.
4. Turn failures into clean training examples.
5. Split data into train and eval files.
6. Run a tiny pilot.
7. Watch the loss curve.
8. Generate real answers.
9. Fix the data.
10. Run the full QLoRA or LoRA job.
11. Evaluate on held-out prompts.
12. Merge the adapter if you need a clean serving artifact.
13. Serve locally.
14. Keep the old model around in case the new one gets weird.

That is the loop.

Not: download model, become genius.

## Hardware guidance

### 12–16GB VRAM

Start with QLoRA on a 7B or 8B model.

Good starting point:

- `--max_length 1024`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 16`
- `--lora_r 16`
- `--learning_rate 2e-4`

### 24GB VRAM

Use QLoRA comfortably or LoRA without quantization on smaller models.

Good starting point:

- `--max_length 2048`
- `--per_device_train_batch_size 2`
- `--gradient_accumulation_steps 8`
- `--lora_r 16` or `32`

### 48GB+ VRAM

You can try longer context, larger models, bigger adapters, or selective/full fine-tuning.

### CPU only

You can experiment with tiny models and short sequences, but it will be slow. Treat it as learning-only, not a serious training setup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If this prints `False`, fix CUDA/PyTorch before debugging training.

## Dataset formats

Chat format, recommended for chat models:

```json
{"messages":[{"role":"user","content":"What does this clause mean?"},{"role":"assistant","content":"It means the supplier has 30 days to deliver."}]}
```

Prompt-completion format:

```json
{"prompt":"Summarize this clause in plain English:","completion":"The supplier must deliver goods within 30 days."}
```

## Clean your data

```bash
python scripts/preprocess_jsonl.py --input data/raw_sft.jsonl --output data/clean_sft.jsonl
```

If many examples are skipped, inspect your raw data before training.

## Train with QLoRA

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

## Watch training

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

Loss should trend down, but loss is not quality. Always evaluate with real prompts the model never trained on.

## Merge a LoRA adapter

```bash
python scripts/merge_lora.py \
  --base_model /models/base-model \
  --adapter runs/sft-lora \
  --output_dir artifacts/merged
```

## DPO preference training

Do not start with DPO. First get an SFT model that is already useful.

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

## Common failures

`nan` loss usually means learning rate, precision, or corrupted data.

CUDA out of memory usually means the model is too large, the sequence is too long, the batch is too big, or you should be using QLoRA.

Wrong answer format usually means your chat template or dataset structure is wrong.

Generic answers usually mean your examples are weak, not that the model is bad.

Eval loss rising while train loss falls usually means overfitting.

The model getting worse at normal tasks after training usually means catastrophic forgetting.

## Safety and privacy

Do not train on private data you do not have permission to use.

Do not train on secrets, API keys, credentials, customer records, or anything you would not want the model to repeat later.

Do not publish trained model weights if the training-data license does not allow it.

Check the license of your base model before commercial use.

## The real rule

If the quality is not there, the answer is usually not a different optimizer.

It is better data.

More specific examples.

Cleaner formatting.

Sharper eval prompts.

Less wishful thinking.

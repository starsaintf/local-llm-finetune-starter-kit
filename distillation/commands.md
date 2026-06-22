# Distillation Commands

Run all commands from the repo root.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## 2. Create prompts

Create:

```text
distillation/data/prompts.jsonl
```

Each line:

```json
{"prompt":"Explain LoRA to a non-technical founder in two short paragraphs."}
```

## 3. Generate teacher answers

```bash
python distillation/scripts/generate_teacher.py \
  --teacher_model /models/teacher-model \
  --input_file distillation/data/prompts.jsonl \
  --output_file distillation/data/teacher_raw.jsonl \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --do_sample
```

Output:

```text
distillation/data/teacher_raw.jsonl
```

## 4. Filter teacher answers

```bash
python distillation/scripts/filter_teacher_outputs.py \
  --input distillation/data/teacher_raw.jsonl \
  --output distillation/data/teacher_filtered.jsonl \
  --min_words 5 \
  --max_words 1000 \
  --min_unique_ratio 0.45 \
  --sample_preview 20
```

Output:

```text
distillation/data/teacher_filtered.jsonl
```

Read the preview before training.

## 5. Split train/eval

```bash
python distillation/scripts/split_jsonl.py \
  --input distillation/data/teacher_filtered.jsonl \
  --train_output distillation/data/teacher_train.jsonl \
  --eval_output distillation/data/teacher_eval.jsonl \
  --eval_ratio 0.1
```

Outputs:

```text
distillation/data/teacher_train.jsonl
distillation/data/teacher_eval.jsonl
```

## 6. Train the student

```bash
python finetuning/scripts/train_sft_hf.py \
  --model_path /models/student-model \
  --train_file distillation/data/teacher_train.jsonl \
  --eval_file distillation/data/teacher_eval.jsonl \
  --output_dir runs/distillation/student-qlora \
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

Output:

```text
runs/distillation/student-qlora
```

## 7. Watch training

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

## 8. Merge adapter if needed

```bash
python finetuning/scripts/merge_lora.py \
  --base_model /models/student-model \
  --adapter runs/distillation/student-qlora \
  --output_dir artifacts/distillation/student-merged
```

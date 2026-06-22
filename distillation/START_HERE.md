# Start Here: Distillation for Beginners

This is the simplest path.

You are not training a model from zero.

You are doing this:

```text
strong teacher model -> writes answers -> small student model learns those answers
```

## What you need

You need:

1. A teacher model
2. A student model
3. A file of prompts
4. This repo
5. A GPU if you want this to run at a reasonable speed

## Teacher vs student

The teacher is the stronger model.

Example:

```text
/models/teacher-model
```

The student is the smaller model you want to run locally.

Example:

```text
/models/student-model
```

Replace those paths with your actual models.

## The folder you are using

You are working inside:

```text
distillation/
```

The fine-tuning trainer lives in:

```text
finetuning/scripts/train_sft_hf.py
```

That is intentional. Distillation creates the dataset. Fine-tuning trains the student on that dataset.

## Step 1: create prompts

Create this file:

```text
distillation/data/prompts.jsonl
```

Each line should look like this:

```json
{"prompt":"Explain LoRA to a non-technical founder in two short paragraphs."}
```

Start small.

Use 20 to 50 prompts first.

Do not start with 10,000 prompts. Prove the workflow works first.

## Step 2: generate teacher answers

Run:

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

This creates:

```text
distillation/data/teacher_raw.jsonl
```

That file contains prompt + teacher answer pairs.

## Step 3: filter teacher answers

Run:

```bash
python distillation/scripts/filter_teacher_outputs.py \
  --input distillation/data/teacher_raw.jsonl \
  --output distillation/data/teacher_filtered.jsonl \
  --sample_preview 20
```

This creates:

```text
distillation/data/teacher_filtered.jsonl
```

Now read the preview.

If the answers look bad, stop. Fix the prompts or use a better teacher.

## Step 4: split train and eval

Run:

```bash
python distillation/scripts/split_jsonl.py \
  --input distillation/data/teacher_filtered.jsonl \
  --train_output distillation/data/teacher_train.jsonl \
  --eval_output distillation/data/teacher_eval.jsonl \
  --eval_ratio 0.1
```

This creates:

```text
distillation/data/teacher_train.jsonl
distillation/data/teacher_eval.jsonl
```

The student trains on the train file.

The eval file checks whether it is learning.

## Step 5: train the student

Run:

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

Replace `/models/student-model` with your real student model path.

## Step 6: check training

Run:

```bash
tensorboard --logdir runs/
```

Open:

```text
http://localhost:6006
```

Loss should generally go down.

But loss is not enough.

You still need to ask the model real prompts and read the answers.

## Step 7: test the student

Ask the student prompts it did not see during training.

Good test:

```text
Explain LoRA to a startup founder who has never trained a model before.
```

Bad test:

```text
The exact same prompt you gave the teacher.
```

You are checking whether the student learned the task, not whether it memorized the dataset.

## Step 8: only then scale up

If the small test works, then create more prompts.

Go from 50 to 500.

Then 500 to 5,000.

Do not scale bad data.

## Common mistakes

Do not train on raw teacher outputs without reading them.

Do not use the same clean prompt format every time.

Do not test only on prompts that look like the training prompts.

Do not assume the teacher is always correct.

Do not start with logit distillation. Start with text answers first.

## The real rule

Distillation is dataset manufacturing.

The teacher makes the examples.

The student learns from them.

The quality of the student depends on the quality of the examples.

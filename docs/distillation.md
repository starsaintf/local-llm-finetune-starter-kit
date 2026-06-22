# Distillation Guide

Distillation is how you use a stronger model to create training signal for a smaller model.

The teacher does not run forever. It creates the dataset. The student learns from that dataset. Then the student runs locally at your cost.

## The practical local path

For most local projects, start with sequence-level distillation.

That means:

1. Write prompts that look like your real use case.
2. Generate teacher answers.
3. Filter and inspect the teacher outputs.
4. Split the filtered data into train and eval.
5. Train the student with the normal SFT script.
6. Evaluate the student on organic prompts it has never seen.

Do not start with online logit distillation unless sequence-level distillation has clearly plateaued and you know why you need the extra complexity.

## Step 1 — generate teacher outputs

```bash
python scripts/generate_teacher.py \
  --teacher_model /models/teacher-model \
  --input_file data/sample_distill_prompts.jsonl \
  --output_file data/teacher_raw.jsonl \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --do_sample
```

Use sampling when generating teacher data. Greedy decoding can produce narrow, repetitive examples.

## Step 2 — filter teacher outputs

```bash
python scripts/filter_teacher_outputs.py \
  --input data/teacher_raw.jsonl \
  --output data/teacher_filtered.jsonl \
  --min_words 5 \
  --max_words 1000 \
  --min_unique_ratio 0.45 \
  --sample_preview 20
```

Always read a sample of the filtered outputs before training. The teacher is not ground truth.

## Step 3 — split train/eval

Create:

```text
data/teacher_filtered_train.jsonl
data/teacher_filtered_eval.jsonl
```

A simple split is 90% train and 10% eval.

## Step 4 — train the student

```bash
python scripts/train_sft_hf.py \
  --model_path /models/student-model \
  --train_file data/teacher_filtered_train.jsonl \
  --eval_file data/teacher_filtered_eval.jsonl \
  --output_dir runs/student-distilled \
  --use_qlora \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_use_double_quant \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --bf16
```

## What to watch

If the student sounds fluent but is factually worse than expected, spot-check the teacher outputs. You may have distilled repeated teacher mistakes.

If the student performs well only on prompts shaped like your generation prompts, vary your prompt phrasing and mix in real user examples.

If the student repeats the same phrases, increase diversity in teacher generation and filter duplicates harder.

If quality is flat despite good loss, your dataset is probably too repetitive or too shallow.

If refusal behavior gets worse, run safety checks before and after every distillation pass.

## The rule

Distillation is not magic compression.

It is dataset manufacturing.

The teacher's job is to help you create examples you could not have written by hand at the same quality and volume.

Once that dataset exists, the teacher has done its job.

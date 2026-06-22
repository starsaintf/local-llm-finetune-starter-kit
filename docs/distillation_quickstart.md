# Distillation Quickstart for Beginners

This is the noob-friendly version.

Use this when you want to make a smaller local model copy the behavior of a stronger model.

You do **not** need to understand research papers before using this.

## What you are doing

You have two models:

- **Teacher model**: the stronger model that writes good answers.
- **Student model**: the smaller model you want to run locally.

The teacher writes example answers.

You clean those answers.

Then you train the student on them.

That is sequence-level distillation.

Think of it like this:

```text
prompts -> teacher model -> teacher answers -> clean dataset -> student fine-tune -> better small model
```

## Before you start

You need:

1. A working Python environment.
2. A GPU if you want this to be practical.
3. A teacher model path or Hugging Face model id.
4. A student model path or Hugging Face model id.
5. A list of prompts that look like your real use case.

Example teacher:

```text
/models/teacher-model
```

Example student:

```text
/models/student-model
```

You can replace those with real model paths or Hugging Face ids.

## Step 0 — install the repo

Clone the repo:

```bash
git clone https://github.com/starsaintf/local-llm-finetune-starter-kit.git
cd local-llm-finetune-starter-kit
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install packages:

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Check that your GPU is visible:

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If this says `cuda available: False`, stop. Fix your CUDA/PyTorch setup before training.

## Step 1 — create your prompt file

Create a file like this:

```text
data/my_prompts.jsonl
```

Each line should be one prompt.

Example:

```json
{"prompt":"Summarize this contract clause in plain English: The supplier shall deliver the goods within thirty calendar days of receipt of purchase order."}
{"prompt":"Explain LoRA to a non-technical founder in two short paragraphs."}
{"prompt":"Rewrite this support reply to sound firm but respectful: We can maybe extend your deadline again if you need it."}
```

Do not use random toy prompts.

Use prompts that look like the work you actually want the small model to do.

Bad prompt:

```text
write a poem about bananas
```

Good prompt:

```text
Rewrite this customer support message so it sounds firm, respectful, and clear: ...
```

## Step 2 — generate teacher answers

Run this command:

```bash
python scripts/generate_teacher.py \
  --teacher_model /models/teacher-model \
  --input_file data/my_prompts.jsonl \
  --output_file data/teacher_raw.jsonl \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --do_sample
```

Replace this:

```text
/models/teacher-model
```

with your actual teacher model path or model id.

After the command runs, you should see this file:

```text
data/teacher_raw.jsonl
```

That file contains your original prompts plus the teacher's answers.

## Step 3 — clean the teacher answers

The teacher can still make mistakes.

Do not train the student on raw teacher output without checking it.

Run:

```bash
python scripts/filter_teacher_outputs.py \
  --input data/teacher_raw.jsonl \
  --output data/teacher_filtered.jsonl \
  --min_words 5 \
  --max_words 1000 \
  --min_unique_ratio 0.45 \
  --sample_preview 20
```

After this, you should see:

```text
data/teacher_filtered.jsonl
```

The script will also print a preview of random examples.

Read them.

If the answers look bad, do not continue.

Fix the prompts, use a better teacher, or change the generation settings, then generate again.

## Step 4 — split into train and eval files

You need two files:

```text
data/teacher_filtered_train.jsonl
data/teacher_filtered_eval.jsonl
```

The train file teaches the student.

The eval file checks whether training is going well.

Simple rule:

- 90% of examples go into train.
- 10% go into eval.

For a quick manual split, copy most lines into `teacher_filtered_train.jsonl` and the last few lines into `teacher_filtered_eval.jsonl`.

For a tiny test, this is okay.

For a serious run, use a real split script or dataset tool later.

## Step 5 — train the student model

Run this:

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

Replace this:

```text
/models/student-model
```

with your actual student model path or model id.

When training finishes, your trained adapter will be inside:

```text
runs/student-distilled
```

## Step 6 — watch the loss curve

Open another terminal and run:

```bash
tensorboard --logdir runs/
```

Open this in your browser:

```text
http://localhost:6006
```

What you want:

- Training loss should generally go down.
- Eval loss should not shoot upward while training loss goes down.

If training loss goes down but eval loss rises hard, the student may be overfitting.

If loss becomes `nan`, your learning rate may be too high, precision may be unstable, or your data may be broken.

## Step 7 — test the student with real prompts

Do not only trust the loss number.

Test the model manually.

Use prompts the student never saw during training.

Bad test:

```text
A prompt almost identical to your teacher-generation prompt.
```

Good test:

```text
A messy real prompt written the way an actual user would write it.
```

You are checking whether the student learned the task, not just the teacher's formatting.

## Step 8 — merge the adapter if needed

If you trained with LoRA or QLoRA, you may want to merge the adapter into the base model for cleaner serving.

Run:

```bash
python scripts/merge_lora.py \
  --base_model /models/student-model \
  --adapter runs/student-distilled \
  --output_dir artifacts/student-distilled-merged
```

After this, the merged model will be here:

```text
artifacts/student-distilled-merged
```

## What each file does

```text
scripts/generate_teacher.py
```

Uses the teacher model to answer your prompts.

```text
scripts/filter_teacher_outputs.py
```

Removes very short, repetitive, duplicate, or broken teacher answers.

```text
scripts/train_sft_hf.py
```

Trains the student model on the filtered teacher data.

```text
scripts/merge_lora.py
```

Merges a trained LoRA adapter back into the base student model.

```text
docs/distillation.md
```

More detailed distillation guide.

## Common beginner mistakes

### Mistake 1 — using random prompts

Do not generate random examples.

Use prompts that match the work you want the student model to do.

### Mistake 2 — trusting raw teacher output

The teacher can be wrong.

Always filter and read samples before training.

### Mistake 3 — using too few examples

A tiny dataset is fine for testing the pipeline.

It is not enough for serious quality.

### Mistake 4 — only testing on clean prompts

Real users write messy prompts.

Test with messy prompts.

### Mistake 5 — thinking distillation is magic

Distillation is not magic.

It is dataset manufacturing.

The teacher makes examples.

The student learns from those examples.

If the examples are bad, the student will be bad.

## Simple mental model

Fine-tuning means:

```text
human examples -> student model
```

Distillation means:

```text
teacher model -> teacher examples -> student model
```

Same training pipeline.

Different source of data.

## The beginner rule

Start small.

Run the pipeline once with 20–50 examples.

Make sure it works.

Then scale up.

Do not generate 10,000 teacher examples before you know the workflow works.

Get one tiny run working first.

Then improve the data.

Then train again.

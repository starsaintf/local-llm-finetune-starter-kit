# Fine-tuning Workflow

Use this folder when you already have examples and want a local model to behave better on your task.

Fine-tuning means:

> I already have a model. I want it to follow my format, tone, domain, or workflow better.

This is different from distillation.

Distillation uses a stronger teacher model to create training examples first. Fine-tuning uses examples you already have.

## What is in this folder

```text
finetuning/
├── README.md
├── commands.md
├── scripts/
│   ├── preprocess_jsonl.py
│   ├── train_sft_hf.py
│   ├── train_dpo.py
│   └── merge_lora.py
└── data/
    ├── sample_sft.jsonl
    └── sample_pref.jsonl
```

## Start here

1. Install dependencies from the repo root.
2. Put your training examples in `finetuning/data/raw_sft.jsonl`.
3. Clean the file with `preprocess_jsonl.py`.
4. Split it into train and eval files.
5. Train with QLoRA first.
6. Test the model on prompts it never saw.
7. Merge the LoRA adapter only when you are happy with the output.

## When to use SFT

Use supervised fine-tuning when you know the answer you want the model to produce.

Example:

```json
{"messages":[{"role":"user","content":"What does this clause mean?"},{"role":"assistant","content":"It means the supplier has 30 days to deliver."}]}
```

Or:

```json
{"prompt":"Summarize this clause:","completion":"The supplier must deliver within 30 days."}
```

## When to use DPO

Use DPO after SFT, not before.

DPO is for preference shaping. It teaches the model which answer is better when two possible answers exist.

Use it when you have examples like:

```json
{"prompt":"Rewrite this reply to sound firm but respectful.","chosen":"I understand the concern, but we cannot extend the deadline again.","rejected":"Sorry! We can probably extend it again."}
```

The important thing is that `chosen` must clearly be better than `rejected`.

Do not use DPO to teach the model a whole domain from zero. Use SFT first.

## The simple rule

Bad data beats good settings.

If the model is still bad after fine-tuning, do not immediately change the optimizer.

First check the examples.

# Distillation Workflow

Use this folder when you want a stronger model to create examples for a smaller local model.

Distillation means:

> Teacher model writes examples. Student model learns from those examples.

Fine-tuning starts with examples you already have.

Distillation starts by creating examples with a teacher model.

## Folder structure

```text
distillation/
├── README.md
├── commands.md
├── START_HERE.md
├── scripts/
│   ├── generate_teacher.py
│   ├── clean_teacher_dataset.py
│   └── split_jsonl.py
└── data/
    └── sample_prompts.jsonl
```

## Simple path

1. Write prompts in `distillation/data/prompts.jsonl`.
2. Generate teacher answers with `generate_teacher.py`.
3. Save raw answers to `distillation/data/teacher_raw.jsonl`.
4. Clean the raw file with `clean_teacher_dataset.py`.
5. Split the clean file with `split_jsonl.py`.
6. Train the student with `finetuning/scripts/train_sft_hf.py`.
7. Test the student on new prompts.

## Start here

Read `distillation/START_HERE.md` first.

Then use `distillation/commands.md` for copy/paste commands.

## Reminder

Read a sample of the teacher answers before training.

The student learns from whatever examples you give it.

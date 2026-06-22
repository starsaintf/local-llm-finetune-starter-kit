# Distillation Workflow

Use this folder when you do not have enough good examples, but you do have access to a stronger model that can write good answers for you.

Distillation means:

> Use a stronger teacher model to create training examples, then train a smaller student model on those examples.

This is different from normal fine-tuning.

Fine-tuning starts with examples you already have.

Distillation starts by making examples with a teacher model.

## What is in this folder

```text
distillation/
├── README.md
├── commands.md
├── START_HERE.md
├── scripts/
│   ├── generate_teacher.py
│   ├── filter_teacher_outputs.py
│   └── split_jsonl.py
└── data/
    └── sample_prompts.jsonl
```

## The noob version

You need two models:

- Teacher model: smarter, bigger, or more expensive. It writes the answers.
- Student model: smaller and cheaper. It learns from the teacher answers.

You will do this:

1. Write prompts in `distillation/data/prompts.jsonl`.
2. Run the teacher model on those prompts.
3. Save teacher answers to `distillation/data/teacher_raw.jsonl`.
4. Filter weak teacher answers.
5. Split the filtered file into train/eval.
6. Train the student using the fine-tuning trainer.
7. Test the student on new prompts.

## Start with this

Read `distillation/START_HERE.md` first.

Then use `distillation/commands.md` for copy/paste commands.

## Important warning

The teacher is not always right.

Read some of the teacher outputs before training the student.

Bad teacher data becomes bad student behavior.

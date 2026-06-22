# Local LLM Starter Kit

A practical starter kit for running, fine-tuning, distilling, evaluating, and serving local LLMs.

This repo is split into separate workflows so beginners do not mix up commands.

## Choose your path

### 1. Fine-tuning

Use this when you already have training examples.

Example:

> I have prompts and ideal answers. I want the model to follow my format, tone, or domain better.

Start here:

- `finetuning/README.md`
- `finetuning/commands.md`

Main scripts:

- `finetuning/scripts/preprocess_jsonl.py`
- `finetuning/scripts/train_sft_hf.py`
- `finetuning/scripts/merge_lora.py`

Sample data:

- `finetuning/data/sample_sft.jsonl`

### 2. Distillation

Use this when you do not have enough examples yet, but you have access to a stronger model that can write good answers for you.

Example:

> I want a big teacher model to create examples, then I want a smaller student model to learn from them.

Start here:

- `distillation/START_HERE.md`
- `distillation/README.md`
- `distillation/commands.md`

Main scripts:

- `distillation/scripts/generate_teacher.py`
- `distillation/scripts/clean_teacher_dataset.py`
- `distillation/scripts/split_jsonl.py`

Sample data:

- `distillation/data/sample_prompts.jsonl`

The distillation workflow uses the fine-tuning trainer at the final step:

```text
finetuning/scripts/train_sft_hf.py
```

That is intentional. Distillation creates the dataset. Fine-tuning trains the student.

## Folder map

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── deepspeed_zero2.json
├── finetuning/
│   ├── README.md
│   ├── commands.md
│   ├── scripts/
│   │   ├── preprocess_jsonl.py
│   │   ├── train_sft_hf.py
│   │   └── merge_lora.py
│   └── data/
│       └── sample_sft.jsonl
└── distillation/
    ├── START_HERE.md
    ├── README.md
    ├── commands.md
    ├── scripts/
    │   ├── generate_teacher.py
    │   ├── clean_teacher_dataset.py
    │   └── split_jsonl.py
    └── data/
        └── sample_prompts.jsonl
```

## Install once

Run this from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
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

If CUDA is not available, fix your PyTorch/CUDA setup before training.

## Which one should I use?

Use fine-tuning if:

- you already have examples,
- you know the answer style you want,
- you want the model to follow your format or domain better.

Use distillation if:

- you do not have enough examples,
- a bigger model can produce good answers,
- you want a smaller model to imitate that bigger model.

## The real rule

Bad data beats good settings.

Loss is not quality.

Always test with prompts the model never saw during training.

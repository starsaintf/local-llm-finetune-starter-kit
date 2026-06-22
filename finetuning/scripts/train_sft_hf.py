import argparse
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning with LoRA, QLoRA, or full fine-tuning.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file")
    parser.add_argument("--eval_file")
    parser.add_argument("--dataset_disk_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--deepspeed", default=None)
    return parser.parse_args()


def load_ds(args, tokenizer):
    if args.dataset_disk_path:
        return load_from_disk(args.dataset_disk_path)

    if not args.train_file or not args.eval_file:
        raise ValueError("Provide --train_file and --eval_file, or --dataset_disk_path.")

    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

    def preprocess(batch):
        texts = []

        if "messages" in batch:
            for messages in batch["messages"]:
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
        else:
            for prompt, completion in zip(batch["prompt"], batch["completion"]):
                texts.append(f"{prompt}\n{completion}")

        return tokenizer(texts, truncation=True, max_length=args.max_length)

    return dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}

    if args.use_qlora:
        compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", **model_kwargs)

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    if args.use_lora or args.use_qlora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    dataset = load_ds(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

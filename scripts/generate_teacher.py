import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teacher-model completions for sequence-level distillation.")
    parser.add_argument("--teacher_model", required=True, help="Path or HF id of the teacher model.")
    parser.add_argument("--input_file", required=True, help="JSONL file with one {'prompt': '...'} record per line.")
    parser.add_argument("--output_file", required=True, help="Where to write prompt/completion JSONL records.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt for chat-template models.")
    return parser.parse_args()


def build_text(tokenizer, prompt: str, system_prompt: str | None):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def main():
    args = parse_args()
    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            prompt = row["prompt"]
            input_text = build_text(tokenizer, prompt, args.system_prompt)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            f.write(
                json.dumps(
                    {"prompt": prompt, "completion": completion},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote teacher completions to {output_path}")


if __name__ == "__main__":
    main()

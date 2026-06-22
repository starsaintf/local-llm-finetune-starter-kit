import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()

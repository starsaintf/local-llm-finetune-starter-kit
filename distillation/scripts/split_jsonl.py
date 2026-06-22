import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into train and eval files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--eval_output", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.input)
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.seed(args.seed)
    random.shuffle(lines)

    eval_count = max(1, int(len(lines) * args.eval_ratio)) if len(lines) > 1 else 0
    eval_lines = lines[:eval_count]
    train_lines = lines[eval_count:]

    train_path.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    eval_path.write_text("\n".join(eval_lines) + ("\n" if eval_lines else ""), encoding="utf-8")

    print(f"train={len(train_lines)} eval={len(eval_lines)}")


if __name__ == "__main__":
    main()

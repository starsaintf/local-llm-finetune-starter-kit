import argparse
import json
import random
from hashlib import sha1
from pathlib import Path


def is_repetitive(text: str, min_unique_ratio: float) -> bool:
    words = text.split()
    if not words:
        return True
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < min_unique_ratio


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def main():
    parser = argparse.ArgumentParser(description="Filter teacher-generated distillation outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=1000)
    parser.add_argument("--min_unique_ratio", type=float, default=0.45)
    parser.add_argument("--sample_preview", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                skipped += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt = normalize(obj.get("prompt", ""))
            completion = normalize(obj.get("completion", ""))
            words = completion.split()

            if not prompt or len(words) < args.min_words or len(words) > args.max_words:
                skipped += 1
                continue

            if is_repetitive(completion, args.min_unique_ratio):
                skipped += 1
                continue

            key = sha1(json.dumps({"prompt": prompt, "completion": completion}, sort_keys=True).encode("utf-8")).hexdigest()
            if key in seen:
                skipped += 1
                continue

            seen.add(key)
            kept.append({"prompt": prompt, "completion": completion})

    with output_path.open("w", encoding="utf-8") as fout:
        for obj in kept:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"kept={len(kept)} skipped={skipped}")

    if kept and args.sample_preview > 0:
        print("\nPreview random kept examples:")
        for obj in random.sample(kept, min(args.sample_preview, len(kept))):
            print("- prompt:", obj["prompt"][:160])
            print("  completion:", obj["completion"][:240])


if __name__ == "__main__":
    main()

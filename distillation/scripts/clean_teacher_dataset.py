import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_words", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    skipped = 0

    with source.open("r", encoding="utf-8") as fin, target.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                item = json.loads(line)
            except Exception:
                skipped += 1
                continue

            prompt = " ".join(item.get("prompt", "").split())
            answer = " ".join(item.get("completion", "").split())

            if not prompt or len(answer.split()) < args.min_words:
                skipped += 1
                continue

            key = prompt + "\n" + answer
            if key in seen:
                skipped += 1
                continue
            seen.add(key)

            fout.write(json.dumps({"prompt": prompt, "completion": answer}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()

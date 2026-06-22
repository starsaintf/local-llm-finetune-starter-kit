import argparse
import json
import re
import sys
from hashlib import sha1
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_messages(messages):
    if not isinstance(messages, list):
        return None

    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None

        role = msg.get("role")
        content = msg.get("content")

        if role not in VALID_ROLES:
            return None

        if not isinstance(content, str) or not content.strip():
            return None

        cleaned.append({"role": role, "content": normalize_text(content)})

    if len(cleaned) < 2:
        return None

    return cleaned


def to_sft_record(obj):
    if "messages" in obj:
        messages = normalize_messages(obj["messages"])
        if not messages:
            raise ValueError("Malformed messages")
        return {"messages": messages}

    if "prompt" in obj and "completion" in obj:
        prompt = normalize_text(obj.get("prompt", ""))
        completion = normalize_text(obj.get("completion", ""))

        if not prompt or not completion:
            raise ValueError("Empty prompt or completion")

        return {"prompt": prompt, "completion": completion}

    raise ValueError("Unsupported schema")


def main(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    skipped = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                skipped += 1
                continue

            try:
                obj = json.loads(line)
                rec = to_sft_record(obj)
            except Exception:
                skipped += 1
                continue

            key = sha1(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

            if key in seen:
                skipped += 1
                continue

            seen.add(key)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} skipped={skipped}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate JSONL SFT data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)

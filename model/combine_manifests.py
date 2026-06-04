from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine shard manifest JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        samples.extend(json.loads(path.read_text(encoding="utf-8")))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} samples -> {output}")


if __name__ == "__main__":
    main()

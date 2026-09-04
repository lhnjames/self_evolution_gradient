#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in args.shards:
        with Path(shard, "trace.jsonl").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    for index, row in enumerate(rows):
        row["decision_index"] = index
    with (output / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "metadata.json").write_text(
        json.dumps({"shards": args.shards, "decisions": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "decisions": len(rows)}))


if __name__ == "__main__":
    main()

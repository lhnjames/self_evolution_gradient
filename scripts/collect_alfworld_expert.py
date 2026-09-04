#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_evolve.alfworld_data import collect_expert_decisions, save_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("train", "valid_seen", "valid_unseen"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decisions, metadata = collect_expert_decisions(
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    save_decisions(args.output, decisions)
    metadata_path = Path(args.output).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

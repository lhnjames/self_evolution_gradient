#!/usr/bin/env python3
"""Data preparation CLI for CoEvoKG.

This adapter wraps already filtered question/answer parquet files as CoEvoKG question pools.
It does not split datasets or remove held-out SSP examples; do that upstream.
KG chain pools are produced directly by data_preprocess/chain_pool_pipeline/clean_chains.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("prepare_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOLVER_SYSTEM = "You are a helpful and harmless assistant."
SOLVER_USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as you want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def solver_record(
    *,
    question: str,
    ground_truth: Any,
    data_source: str,
    split: str,
    index: int,
    reward_model: dict[str, Any] | None = None,
    ability: str = "fact-reasoning",
    metadata: Any = None,
) -> dict[str, Any]:
    reward = reward_model or {"style": "rule", "ground_truth": ground_truth}
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": SOLVER_USER_PREFIX + question},
        ],
        "ability": ability,
        "reward_model": reward,
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                        "question": question,
                        "data_source": data_source,
                    }
                }
            },
        },
        "metadata": metadata,
    }


def write_parquet(records: list[dict[str, Any]], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    LOGGER.info("Wrote %d records -> %s", len(records), output)


def cmd_questions_parquet(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    records = []
    for idx, row in df.iterrows():
        question = str(row.get(args.question_column, "")).strip()
        answer = str(row.get(args.answer_column, "")).strip()
        source = str(row.get(args.source_column, args.default_source)).strip() or args.default_source
        if not question or not answer:
            continue
        records.append(
            solver_record(
                question=question,
                ground_truth={"target": answer},
                data_source=args.data_source_prefix + source,
                split=args.split,
                index=int(idx),
            )
        )
    write_parquet(records, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CoEvoKG question-pool preparation CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "questions-parquet",
        help="Wrap an already filtered question/answer parquet file as a CoEvoKG question pool",
    )
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--split", default="seed_fallback", help="Metadata tag written to extra_info.split; this command does not filter rows")
    p.add_argument("--question-column", default="question")
    p.add_argument("--answer-column", default="answer")
    p.add_argument("--source-column", default="source_dataset")
    p.add_argument("--default-source", default="unknown")
    p.add_argument("--data-source-prefix", default="coevokg_selfplay_en_")
    p.set_defaults(func=cmd_questions_parquet)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

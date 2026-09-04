#!/usr/bin/env python3
"""Clean KILT random-walk chain pools for CoEvoKG proposer training.

Default policy is train-only: records from dev/test splits are filtered out to
avoid leakage into held-out evaluation. The script keeps only successful chains
with enough nodes and entity entries, removes upstream QA fields that could leak
answers, and writes the project chain-pool format consumed directly by the
proposer. With --label-relations, it also adds one short edge label per
consecutive entity pair.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """Given a knowledge chain of Wikipedia entities, output a short relation label for each consecutive hop.

Rules:
- Each label must be 3-8 words, describing the relationship from entity A to entity B.
- Use active voice and lowercase when possible, e.g., "directed by" or "member of".
- Output only a JSON array of strings, one per hop, in order.
- Do not include explanations or markdown."""


def project_chain_record(record: dict[str, Any]) -> dict[str, Any]:
    relations = [str(entity) for entity in (record.get("relations") or [])]
    nodes_raw = record.get("nodes") or {}
    nodes = {entity: str(nodes_raw.get(entity, "")).strip() for entity in relations}
    chain = {
        "id": str(record.get("id") or record.get("qid") or ""),
        "relations": relations,
        "nodes": nodes,
        "chain_length": len(relations),
    }
    relation_labels = record.get("relation_labels")
    if isinstance(relation_labels, list) and len(relation_labels) == max(len(relations) - 1, 0):
        chain["relation_labels"] = [str(label).strip() for label in relation_labels]
    return chain


def valid_record(record: dict[str, Any], allowed_splits: set[str], min_chain_length: int) -> tuple[bool, str]:
    split = str(record.get("split", "")).lower()
    if allowed_splits and "all" not in allowed_splits and split not in allowed_splits:
        return False, "split"
    if record.get("fill_status") != "ok":
        return False, "fill_status"
    relations = record.get("relations") or []
    nodes = record.get("nodes") or {}
    chain_length = int(record.get("chain_length") or len(relations))
    if chain_length < min_chain_length or len(relations) < min_chain_length:
        return False, "short_chain"
    if not isinstance(nodes, dict) or any(not str(nodes.get(entity, "")).strip() for entity in relations):
        return False, "missing_node_text"
    return True, "ok"


def build_user_prompt(relations: list[str], nodes: dict[str, str], max_snippet: int) -> str:
    lines = ["Chain: " + " -> ".join(relations), "", "Entity descriptions:"]
    for entity in relations:
        snippet = str(nodes.get(entity, "")).strip().replace("\n", " ")[:max_snippet]
        lines.append(f"[{entity}]: {snippet}")
    lines.extend(["", f"Output a JSON array of {max(len(relations) - 1, 0)} relation labels:"])
    return "\n".join(lines)


def parse_label_json(text: str, expected: int) -> list[str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        labels = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start:
            return None
        try:
            labels = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(labels, list) or len(labels) != expected:
        return None
    cleaned = [str(label).strip() for label in labels]
    if any(not label for label in cleaned):
        return None
    return cleaned


def label_record(record: dict[str, Any], client: Any, model: str, max_snippet: int, max_retries: int) -> dict[str, Any]:
    relations = [str(x) for x in (record.get("relations") or [])]
    nodes = record.get("nodes") or {}
    expected = max(len(relations) - 1, 0)
    if expected <= 0:
        record["relation_labels"] = []
        return record
    if record.get("relation_labels") and len(record.get("relation_labels", [])) == expected:
        return record

    user_prompt = build_user_prompt(relations, nodes, max_snippet)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            content = response.choices[0].message.content or ""
            labels = parse_label_json(content, expected)
            if labels is not None:
                record["relation_labels"] = labels
                return record
            last_error = ValueError(f"could not parse labels: {content[:200]}")
        except Exception as exc:  # noqa: BLE001 - retry transient API failures in CLI mode.
            last_error = exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"labeling failed for id={record.get('id')}: {last_error}")


def label_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.label_base_url or not args.label_api_key or not args.label_model:
        raise SystemExit(
            "Set --label-base-url/--label-api-key/--label-model or "
            "COEVOKG_LABEL_* environment variables when using --label-relations."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise SystemExit("Install the openai package to use --label-relations.") from exc

    client = OpenAI(api_key=args.label_api_key, base_url=args.label_base_url)
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.label_workers) as executor:
        futures = {
            executor.submit(
                label_record,
                record,
                client,
                args.label_model,
                args.label_max_snippet,
                args.label_max_retries,
            ): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            completed[idx] = future.result()
            if len(completed) % 100 == 0:
                print(f"labeled {len(completed)}/{len(records)}")
    return [completed[idx] for idx in range(len(records))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean train-only CoEvoKG chain pools")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--allowed-splits", default="train")
    parser.add_argument("--min-chain-length", type=int, default=2)
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate exact relation sequences")
    parser.add_argument(
        "--label-relations",
        action="store_true",
        help="Add short relation_labels for each consecutive entity pair after cleaning",
    )
    parser.add_argument("--label-base-url", default=os.getenv("COEVOKG_LABEL_BASE_URL", ""))
    parser.add_argument("--label-api-key", default=os.getenv("COEVOKG_LABEL_API_KEY", ""))
    parser.add_argument("--label-model", default=os.getenv("COEVOKG_LABEL_MODEL", ""))
    parser.add_argument("--label-workers", type=int, default=16)
    parser.add_argument("--label-max-snippet", type=int, default=200)
    parser.add_argument("--label-max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_splits = {s.strip().lower() for s in args.allowed_splits.split(",") if s.strip()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    kept_records: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue
            keep, reason = valid_record(record, allowed_splits, args.min_chain_length)
            if not keep:
                stats[reason] += 1
                continue
            key = tuple(record.get("relations") or [])
            if args.deduplicate and key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            kept_records.append(project_chain_record(record))
            stats["kept"] += 1

    if args.label_relations and kept_records:
        kept_records = label_records(kept_records, args)
        stats["labeled"] = len(kept_records)

    with output.open("w", encoding="utf-8") as fout:
        for record in kept_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"output -> {output}")


if __name__ == "__main__":
    main()

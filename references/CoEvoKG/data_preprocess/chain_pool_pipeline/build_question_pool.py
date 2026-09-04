"""
build_question_pool.py
======================
Sample source benchmark train splits proportionally and build a unified question pool.

Outputs:
  all_questions.parquet   -- train-only seed questions used for chain construction
  all_chains.jsonl        -- chain skeletons consumed by fill_chains_fast.py

Data source: source benchmark train splits.

Dev/test splits must not enter the chain pool, to avoid leakage into held-out
evaluation or final test data. Benchmarks with only held-out splits should not be
used for training-time chain-pool construction.

Usage:
  # Sample 50,000 examples proportionally across strata.
  python build_question_pool.py --manifest /path/to/source_train_manifest.json --total 50000 --output-dir ./pool

  # Set a random seed for reproducibility.
  python build_question_pool.py --manifest /path/to/source_train_manifest.json --total 10000 --seed 42 --output-dir ./pool

  # Show allocation without loading data or writing files.
  python build_question_pool.py --manifest /path/to/source_train_manifest.json --total 50000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ============================================================
# Data source definition
# ============================================================

# The source benchmark paths are intentionally not hard-coded. Provide a JSON
# manifest with records of the following form:
#   {"dataset": "nq", "split": "train", "path": "/path/to/file",
#    "format": "qa_jsonl", "size": 10000}
Stratum = Tuple[str, str, str, str, int]


def load_strata_manifest(path: str | Path) -> List[Stratum]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a non-empty JSON list of source datasets.")

    strata: List[Stratum] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {idx} is not an object: {item!r}")
        dataset = str(item.get("dataset") or item.get("source_dataset") or "").strip()
        split = str(item.get("split") or "train").strip().lower()
        file_path = str(item.get("path") or "").strip()
        fmt = str(item.get("format") or "").strip()
        size = int(item.get("size") or 0)
        if not dataset or not file_path or not fmt or size <= 0:
            raise ValueError(
                f"Manifest item {idx} must define dataset/path/format/positive size: {item!r}"
            )
        if split != "train":
            raise ValueError(
                f"Manifest item {idx} has split={split!r}; chain-pool sources must be train-only."
            )
        if fmt not in LOADERS:
            raise ValueError(f"Manifest item {idx} uses unsupported format {fmt!r}.")
        strata.append((dataset, split, file_path, fmt, size))
    return strata


def total_pool(strata: List[Stratum]) -> int:
    return sum(s[4] for s in strata)


# ============================================================
# Format loaders returning normalized dict rows
# ============================================================

def _search_turns_from_supporting(item: Dict[str, Any]) -> int:
    """Infer hop count from supporting_facts in HotpotQA or 2Wiki format."""
    facts = item.get("supporting_facts", [])
    titles = {f[0] for f in facts if isinstance(f, (list, tuple)) and f}
    return max(len(titles), 2)


def load_hotpot_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        answer = item.get("answer", "").strip()
        if not answer:
            continue
        rows.append({
            "source_id":    str(item.get("_id", "")),
            "question":     item.get("question", ""),
            "answer":       answer,
            "search_turns": _search_turns_from_supporting(item),
            "type":         item.get("type", ""),
        })
    return rows


def load_wiki_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        answer = item.get("answer", "").strip()
        if not answer:
            continue
        rows.append({
            "source_id":    str(item.get("_id", "")),
            "question":     item.get("question", ""),
            "answer":       answer,
            "search_turns": _search_turns_from_supporting(item),
            "type":         item.get("type", ""),
        })
    return rows


def load_musique_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            answer = item.get("answer", "").strip()
            if not answer:
                continue
            supporting = [
                p["title"] for p in item.get("paragraphs", [])
                if p.get("is_supporting") and p.get("title")
            ]
            rows.append({
                "source_id":    str(item.get("id", "")),
                "question":     item.get("question", ""),
                "answer":       answer,
                "search_turns": max(len(supporting), 2),
                "type":         "",
            })
    return rows


def load_nq_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            answers = item.get("golden_answers", [])
            answer = answers[0] if answers else ""
            if not answer:
                continue
            rows.append({
                "source_id":    str(item.get("id", "")),
                "question":     item.get("question", ""),
                "answer":       answer,
                "search_turns": 1,
                "type":         "",
            })
    return rows


def _first_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value.replace("'", '"'))
                return _first_answer(parsed)
            except Exception:
                return value
        return value
    return str(value).strip()


def _qa_row(item: Dict[str, Any], default_turns: int = 1) -> Dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    answer = _first_answer(item.get("answer") or item.get("golden_answers") or item.get("answers"))
    if not question or not answer:
        return None
    return {
        "source_id": str(item.get("id") or item.get("_id") or item.get("source_id") or ""),
        "question": question,
        "answer": answer,
        "search_turns": int(item.get("search_turns") or default_turns),
        "type": str(item.get("type", "")),
    }


def load_qa_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = _qa_row(json.loads(line), default_turns=1)
            if row:
                rows.append(row)
    return rows


def load_qa_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or []
    rows = []
    for item in data:
        row = _qa_row(item, default_turns=1)
        if row:
            rows.append(row)
    return rows


def load_qa_parquet(path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = []
    for item in df.to_dict(orient="records"):
        row = _qa_row(item, default_turns=1)
        if row:
            rows.append(row)
    return rows


def load_bamboogle_parquet(path: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    rows = []
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.schema.names}
        answers_raw = row.get("golden_answers", "[]")
        if isinstance(answers_raw, str):
            try:
                answers = json.loads(answers_raw.replace("'", '"'))
            except Exception:
                answers = [answers_raw]
        else:
            answers = answers_raw if isinstance(answers_raw, list) else [str(answers_raw)]
        answer = answers[0] if answers else ""
        if not answer:
            continue
        rows.append({
            "source_id":    str(row.get("id", "")),
            "question":     row.get("question", ""),
            "answer":       answer,
            "search_turns": 2,
            "type":         "",
        })
    return rows


LOADERS = {
    "hotpot_json":       load_hotpot_json,
    "wiki_json":         load_wiki_json,
    "musique_jsonl":     load_musique_jsonl,
    "nq_jsonl":          load_nq_jsonl,
    "bamboogle_parquet": load_bamboogle_parquet,
    "qa_jsonl":          load_qa_jsonl,
    "qa_json":           load_qa_json,
    "qa_parquet":        load_qa_parquet,
}


# ============================================================
# Stratified sampling allocation
# ============================================================

def compute_allocation(total: int, strata: List[Stratum]) -> List[Tuple[str, str, str, str, int, int]]:
    """
    Allocate samples to each stratum in proportion to its pool size.
    Returns: [..., (dataset, split, path, fmt, pool_size, allocated), ...].
    Remainders are assigned by descending fractional part.
    """
    pool_size_total = total_pool(strata)
    if total > pool_size_total:
        print(f"[WARN] --total {total} exceeds pool size {pool_size_total}; adjusted to {pool_size_total}")
        total = pool_size_total

    exact = [(s, total * s[4] / pool_size_total) for s in strata]
    floors = [math.floor(e) for _, e in exact]
    remainder = total - sum(floors)

    # Assign remaining samples by descending fractional part.
    fracs = [(i, e - floors[i]) for i, (_, e) in enumerate(exact)]
    fracs.sort(key=lambda x: -x[1])
    for i in range(remainder):
        floors[fracs[i][0]] += 1

    result = []
    for (dataset, split, path, fmt, pool_size), alloc in zip(strata, floors):
        result.append((dataset, split, path, fmt, pool_size, alloc))
    return result


def print_allocation(allocation: List[Tuple], total: int, pool_size_total: int) -> None:
    print(f"\n{'='*70}")
    print(f"  Stratified allocation  total={total}  pool={pool_size_total}")
    print(f"{'='*70}")
    print(f"  {'dataset':<16} {'split':<8} {'pool':>8} {'alloc':>7} {'ratio':>7}")
    print(f"  {'-'*55}")
    for dataset, split, path, fmt, pool_size, alloc in allocation:
        ratio = alloc / total * 100 if total else 0
        print(f"  {dataset:<16} {split:<8} {pool_size:>8,} {alloc:>7,} {ratio:>6.1f}%")
    print(f"  {'-'*55}")
    print(f"  {'total':<25} {pool_size_total:>8,} {total:>7,} {'100.0%':>7}")
    print(f"{'='*70}\n")


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="build_question_pool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sample proportionally from a train-only source manifest and build a unified question pool",
    )
    parser.add_argument("--manifest", required=True,
                        help="Train-only source manifest JSON. Must exclude SSP held-out evaluation subsets.")
    parser.add_argument("--total", type=int, required=True,
                        help="Total sample count, capped by the sum of sizes in the manifest")
    parser.add_argument("--output-dir", "-o", default="./pool",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the allocation only; do not load data or write files")
    args = parser.parse_args()

    random.seed(args.seed)
    strata = load_strata_manifest(args.manifest)
    pool_size_total = total_pool(strata)

    allocation = compute_allocation(args.total, strata)
    print_allocation(allocation, min(args.total, pool_size_total), pool_size_total)

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for dataset, split, path, fmt, pool_size, alloc in allocation:
        if alloc == 0:
            print(f"[SKIP] {dataset}/{split}: allocation is 0; skipping")
            continue

        print(f"[LOAD] {dataset}/{split} loading...")
        loader = LOADERS[fmt]
        rows = loader(path)
        print(f"       loaded {len(rows)} rows (expected {pool_size})")

        # Sample rows.
        k = min(alloc, len(rows))
        sampled = random.sample(rows, k)

        for row in sampled:
            row["id"]             = uuid.uuid4().hex
            row["source_dataset"] = dataset
            row["split"]          = split

        all_rows.extend(sampled)
        print(f"       sampled {k} rows")

    # Shuffle rows.
    random.shuffle(all_rows)

    # Write Parquet.
    parquet_path = out_dir / "all_questions.parquet"
    df = pd.DataFrame(all_rows, columns=[
        "id", "source_dataset", "source_id", "split",
        "question", "answer", "search_turns", "type",
    ])
    df.to_parquet(str(parquet_path), index=False)
    print(f"\n[OUTPUT] all_questions.parquet  {len(df):,} rows  ->  {parquet_path}")

    # Write chain-skeleton JSONL.
    chains_path = out_dir / "all_chains.jsonl"
    written = 0
    with open(chains_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            placeholder = {
                "id":             row["id"],
                "source_dataset": row["source_dataset"],
                "source_id":      row["source_id"],
                "split":          row["split"],
                # Chain fields filled by fill_chains_fast.py.
                "relations":      [],
                "nodes":          {},
                "chain_length":   0,
                "answer_in_kilt": False,
                "answer_wiki_title": None,
            }
            f.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
            written += 1
    print(f"[OUTPUT] all_chains.jsonl       {written:,} rows  ->  {chains_path}")

    # Summary statistics.
    print(f"\n{'='*50}")
    print("  Dataset distribution (sampled rows)")
    print(f"{'='*50}")
    from collections import Counter
    dist = Counter((r["source_dataset"], r["split"]) for r in all_rows)
    for (ds, sp), cnt in sorted(dist.items()):
        print(f"  {ds:<16} {sp:<8}  {cnt:>6,}")
    print(f"  {'total':<25}  {len(all_rows):>6,}")
    print(f"{'='*50}")

    turns_dist = Counter(r["search_turns"] for r in all_rows)
    print(f"\n  search_turns distribution")
    for t in sorted(turns_dist):
        bar = "#" * min(40, turns_dist[t] // max(1, len(all_rows) // 400))
        print(f"  {t:2d} hops: {turns_dist[t]:>6,}  {bar}")


if __name__ == "__main__":
    main()

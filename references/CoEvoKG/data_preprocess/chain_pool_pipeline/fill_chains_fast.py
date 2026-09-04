"""
fill_chains_fast.py
===================
High-throughput chain filler based on answer-anchored random walks.

Main optimizations:
  1. Thread-local KiltClient instances avoid connection-pool contention.
  2. Batched MongoDB $in queries use one batch find per hop.
  3. Out-of-order writes emit completed samples immediately.
  4. Answer search fetches only the best matching document text.
  5. Local anchor prefiltering runs before MongoDB queries.

Usage:
  python fill_chains_fast.py \\
    --questions-parquet /path/to/pool/all_questions.parquet \\
    --chains-jsonl      /path/to/pool/all_chains.jsonl \\
    --output            /path/to/pool/all_chains_filled.jsonl \\
    --min-hops 1 --max-hops 3 --max-workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# KiltClient lives in the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kilt_query import KiltClient


# ============================================================
# Text utilities
# ============================================================

def safe_search(pattern: str, text: Any, flags: int = 0):
    if not isinstance(text, str):
        return None
    try:
        return re.search(pattern, text, flags=flags)
    except re.error:
        return None


_SEE_ALSO_PATS = [
    r"\bSee also\b\.?", r"\bReferences\b", r"\bExternal links\b",
    r"\bFurther reading\b", r"\bNotes\b", r"\bBibliography\b",
    r"\bSee also\s*:",
]


def cut_see_also(text: Any) -> str:
    text = text if isinstance(text, str) else ""
    for pat in _SEE_ALSO_PATS:
        m = safe_search(pat, text, flags=re.IGNORECASE)
        if m:
            return text[: m.start()]
    return text


def normalize_title(s: str) -> str:
    return (s or "").replace("_", " ").replace("%20", " ").replace("%23", "#").strip()


def unique_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _extract_text_from_doc(
    doc: Dict[str, Any],
    min_chars: int = 500,
    max_paragraphs: int = 6,
    hard_max_chars: int = 5000,
) -> str:
    tf = doc.get("text")
    if isinstance(tf, list) and tf:
        buf: List[str] = []
        total = 0
        for i, para in enumerate(tf):
            if not isinstance(para, str) or not para.strip():
                continue
            buf.append(para.strip())
            total += len(para)
            if total >= min_chars or i + 1 >= max_paragraphs:
                break
        return cut_see_also("\n\n".join(buf)[:hard_max_chars])
    if isinstance(tf, str) and tf.strip():
        return cut_see_also(tf[:hard_max_chars])
    return ""


# ============================================================
# Batched MongoDB query helper
# ============================================================

def batch_query_text(
    col,
    titles: List[str],
    min_chars: int = 500,
    max_paragraphs: int = 6,
    hard_max_chars: int = 5000,
    min_text_len: int = 80,
) -> Dict[str, str]:
    """Fetch text for multiple titles with one $in query; return {title: text}."""
    if not titles:
        return {}
    try:
        docs = list(col.find(
            {"wikipedia_title": {"$in": titles}},
            {"wikipedia_title": 1, "text": 1},
        ))
    except Exception:
        return {}

    result: Dict[str, str] = {}
    for doc in docs:
        title = doc.get("wikipedia_title", "")
        if not title:
            continue
        text = _extract_text_from_doc(doc, min_chars, max_paragraphs, hard_max_chars)
        if text and len(text) >= min_text_len:
            result[title] = text
    return result


# ============================================================
# Thread-local KiltClient
# ============================================================

_tl = threading.local()


def get_client(mongo_uri: str, db_name: str, coll_name: str) -> KiltClient:
    if not hasattr(_tl, "client"):
        _tl.client = KiltClient(
            mongo_uri=mongo_uri,
            db_name=db_name,
            coll_name=coll_name,
            use_text_index=True,
            text_language=None,
            ensure_basic_indexes=False,
        )
    return _tl.client


# ============================================================
# Answer -> KILT title; fetch only the best matching full text
# ============================================================

def find_answer_in_kilt(
    client: KiltClient,
    answer: str,
    search_topk: int,
    min_text_len: int,
) -> Tuple[Optional[str], str]:
    answer = (answer or "").strip()
    if not answer:
        return None, ""

    try:
        result = client.kilt_search(answer, size=search_topk)
        hits = result.get("hits", {}).get("hits", [])
    except Exception:
        return None, ""

    if not hits:
        return None, ""

    answer_lower = answer.lower()

    # Rank candidate titles by match quality without fetching full text.
    best_title: Optional[str] = None
    fallback_title: Optional[str] = None

    for h in hits:
        title = normalize_title(h.get("_source", {}).get("wikipedia_title", ""))
        if not title:
            continue
        if fallback_title is None:
            fallback_title = title
        tl = title.lower()
        if answer_lower in tl or tl in answer_lower:
            best_title = title
            break

    target = best_title or fallback_title
    if not target:
        return None, ""

    # Fetch full text for only one title.
    texts = batch_query_text(
        client.col, [target],
        min_text_len=min_text_len,
    )
    text = texts.get(target, "")
    if text:
        return target, text
    return None, ""


# ============================================================
# Batched random walk with one batched query per hop
# ============================================================

def random_walk_batch(
    client: KiltClient,
    start_title: str,
    start_text: str,
    min_hops: int,
    max_hops: int,
    min_text_len: int,
    max_candidates: int,
    rng: random.Random,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Run a random walk from start_title using batched $in queries per hop.
    Return (relations, nodes), with the answer at the chain tail after reversal.
    """
    path: List[str] = [start_title]
    nodes: Dict[str, str] = {start_title: start_text}
    visited: Set[str] = {start_title.lower()}
    target_hops = rng.randint(min_hops, max_hops)

    for _ in range(target_hops):
        current = path[-1]

        # Fetch anchors.
        try:
            anchors = client.query_relations(current) or []
        except Exception:
            break

        if not anchors:
            break

        rng.shuffle(anchors)

        # Local prefilter: remove visited, disambiguation, and empty titles.
        candidates: List[str] = []
        for a in anchors:
            nxt = normalize_title(a.get("href", "") or a.get("title", ""))
            if not nxt:
                continue
            nxt_lower = nxt.lower()
            if nxt_lower in visited:
                continue
            if "(disambiguation)" in nxt_lower:
                continue
            candidates.append(nxt)
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            break

        # Batch-fetch text with one MongoDB query.
        texts = batch_query_text(
            client.col, candidates,
            min_text_len=min_text_len,
        )

        # Choose the first valid node in candidate order.
        chosen: Optional[str] = None
        for cand in candidates:
            text = texts.get(cand)
            if text:
                chosen = cand
                nodes[cand] = text
                visited.add(cand.lower())
                path.append(cand)
                break

        if chosen is None:
            break

    if len(path) < 2:
        return [], {}

    # Reverse so the answer sits at the chain tail.
    path_rev = list(reversed(path))
    relations = unique_keep_order(path_rev)
    return relations, {t: nodes[t] for t in relations if t in nodes}


# ============================================================
# Single-sample processing
# ============================================================

def _empty(obj: Dict[str, Any], status: str, **kw) -> Dict[str, Any]:
    obj.update({
        "relations": [], "nodes": {}, "chain_length": 0,
        "answer_in_kilt": False, "answer_wiki_title": None,
        "fill_status": status,
        **kw,
    })
    return obj


def process_one(
    idx: int,
    line: str,
    answers_map: Dict[str, str],
    mongo_uri: str,
    db_name: str,
    coll_name: str,
    min_hops: int,
    max_hops: int,
    search_topk: int,
    min_text_len: int,
    max_candidates: int,
    seed: int,
) -> Tuple[int, Dict[str, Any], str]:
    rng = random.Random(seed + idx)
    obj = json.loads(line)
    qid = str(obj.get("id", ""))
    answer = answers_map.get(qid, "")

    if not answer:
        return idx, _empty(obj, "missing_answer"), "missing_answer"

    client = get_client(mongo_uri, db_name, coll_name)

    ans_title, ans_text = find_answer_in_kilt(
        client, answer, search_topk, min_text_len,
    )
    if not ans_title:
        return idx, _empty(obj, "answer_not_found"), "answer_not_found"

    relations, nodes = random_walk_batch(
        client=client,
        start_title=ans_title,
        start_text=ans_text,
        min_hops=min_hops,
        max_hops=max_hops,
        min_text_len=min_text_len,
        max_candidates=max_candidates,
        rng=rng,
    )

    if len(relations) < 2:
        return idx, _empty(obj, "walk_too_short", answer_in_kilt=True,
                           answer_wiki_title=ans_title), "walk_too_short"

    obj.update({
        "relations": relations,
        "nodes": {k: nodes[k] for k in relations if k in nodes},
        "chain_length": len(relations),
        "answer_in_kilt": True,
        "answer_wiki_title": ans_title,
        "fill_status": "ok",
    })
    return idx, obj, "ok"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="fill_chains_fast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="High-throughput chain filler using batched MongoDB queries and thread-local clients",
    )
    p.add_argument("--questions-parquet", required=True)
    p.add_argument("--chains-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mongo-uri",  default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--mongo-db",   default=os.getenv("MONGO_DB", "kilt"))
    p.add_argument("--mongo-coll", default=os.getenv("MONGO_COLL", "knowledgesource"))
    p.add_argument(
        "--allowed-splits",
        default="train",
        help="Comma-separated splits allowed in the chain skeleton; use 'all' to disable filtering.",
    )

    walk = p.add_argument_group("Walk parameters")
    walk.add_argument("--min-hops",       type=int, default=1,  help="Minimum random-walk hops")
    walk.add_argument("--max-hops",       type=int, default=3,  help="Maximum random-walk hops")
    walk.add_argument("--search-topk",    type=int, default=5,  help="Number of answer-search candidates")
    walk.add_argument("--min-text-len",   type=int, default=80, help="Minimum node text length in characters")
    walk.add_argument("--max-candidates", type=int, default=15,
                      help="Maximum anchor candidates per hop; controls batched $in size")

    perf = p.add_argument_group("Performance parameters")
    perf.add_argument("--max-workers", type=int, default=32, help="Number of worker threads")
    perf.add_argument("--flush-every", type=int, default=100, help="Flush output every N rows")
    perf.add_argument("--log-every",   type=int, default=500, help="Print statistics every N rows")
    perf.add_argument("--limit",       type=int, default=0,   help="Process only the first N rows; 0 means all rows")
    perf.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load answer map.
    print(f"[LOAD] {args.questions_parquet}")
    df = pd.read_parquet(args.questions_parquet, columns=["id", "answer"])
    answers_map: Dict[str, str] = {
        str(r["id"]): ("" if pd.isna(r["answer"]) else str(r["answer"]))
        for _, r in df.iterrows()
    }
    print(f"       {len(answers_map):,} questions")

    # Read chain skeletons.
    print(f"[LOAD] {args.chains_jsonl}")
    with open(args.chains_jsonl, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    allowed_splits = {
        s.strip().lower()
        for s in str(args.allowed_splits).split(",")
        if s.strip()
    }
    if allowed_splits and "all" not in allowed_splits:
        kept = []
        skipped_split = 0
        for line in lines:
            try:
                split = str(json.loads(line).get("split", "")).lower()
            except Exception:
                kept.append(line)
                continue
            if split in allowed_splits:
                kept.append(line)
            else:
                skipped_split += 1
        print(
            f"       split filter allowed={sorted(allowed_splits)} "
            f"kept={len(kept):,} skipped={skipped_split:,}"
        )
        lines = kept

    if args.limit > 0:
        lines = lines[: args.limit]
    total = len(lines)
    print(f"       {total:,} rows to process")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Statistics.
    stats = {"ok": 0, "answer_not_found": 0, "walk_too_short": 0, "missing_answer": 0}
    write_count = 0
    write_lock = threading.Lock()

    pbar = tqdm(total=total, desc="fill_chains", ncols=100, unit="sample")

    with open(args.output, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.max_workers) as ex:

        submit_kwargs = dict(
            answers_map=answers_map,
            mongo_uri=args.mongo_uri,
            db_name=args.mongo_db,
            coll_name=args.mongo_coll,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            search_topk=args.search_topk,
            min_text_len=args.min_text_len,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )

        futures = {
            ex.submit(process_one, idx, line, **submit_kwargs): idx
            for idx, line in enumerate(lines)
        }

        for fut in as_completed(futures):
            try:
                _, obj, status = fut.result()
            except Exception as e:
                pbar.update(1)
                tqdm.write(f"[ERR] {e}")
                continue

            with write_lock:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                write_count += 1
                stats[status] = stats.get(status, 0) + 1

                if write_count % args.flush_every == 0:
                    fout.flush()
                if write_count % args.log_every == 0:
                    tqdm.write(
                        f"[{write_count:>6}/{total}] "
                        f"ok={stats['ok']} "
                        f"not_found={stats['answer_not_found']} "
                        f"short={stats['walk_too_short']} "
                        f"missing={stats['missing_answer']}"
                    )

            pbar.update(1)

        fout.flush()

    pbar.close()

    ok_rate = stats["ok"] / total * 100 if total else 0
    print(f"\n[DONE] total={total:,}")
    print(f"       ok={stats['ok']:,} ({ok_rate:.1f}%)")
    print(f"       answer_not_found={stats['answer_not_found']:,}")
    print(f"       walk_too_short={stats['walk_too_short']:,}")
    print(f"       missing_answer={stats['missing_answer']:,}")
    print(f"       output -> {args.output}")


if __name__ == "__main__":
    main()

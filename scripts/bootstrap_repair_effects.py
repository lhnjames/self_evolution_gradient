#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from self_evolve.logic_repair_head import (
    SignedLogicRepairHead,
    StageCalibrationHead,
    VerbVocabulary,
    _inputs,
    load_trace,
)


def load_head(path: Path, kind: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    verbs = checkpoint["verbs"]
    if kind == "skill":
        model = SignedLogicRepairHead(len(verbs))
    else:
        model = StageCalibrationHead(len(verbs))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    vocabulary = object.__new__(VerbVocabulary)
    vocabulary.names = verbs
    vocabulary.index = {name: idx for idx, name in enumerate(verbs)}
    return model, vocabulary


@torch.no_grad()
def correctness_rows(trace_path: str, checkpoint_dir: str):
    states = load_trace(trace_path)
    root = Path(checkpoint_dir)
    skill, vocabulary = load_head(root / "constrained_logic_repair_head.pt", "skill")
    stage, _ = load_head(root / "constrained_stage_calibration_head.pt", "stage")
    rows = []
    for state in states:
        plain_pred = int(state.plain_scores.argmax())
        evolved_pred = int(state.skill_scores.argmax())
        stage_pred = int(stage(*_inputs(state, vocabulary))[0].argmax())
        skill_pred = int(skill(*_inputs(state, vocabulary))[0].argmax())
        rows.append(
            {
                "episode": state.episode_key or state.episode_id,
                "plain": int(plain_pred == state.gold_index),
                "evolved_skill": int(evolved_pred == state.gold_index),
                "constrained_stage": int(stage_pred == state.gold_index),
                "constrained_skill": int(skill_pred == state.gold_index),
            }
        )
    return rows


def cluster_bootstrap(rows, comparisons, samples: int, seed: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(row)
    episodes = sorted(grouped)
    rng = np.random.default_rng(seed)
    result = {}
    for left, right in comparisons:
        observed = np.mean([row[left] - row[right] for row in rows])
        draws = []
        for _ in range(samples):
            chosen = rng.choice(episodes, size=len(episodes), replace=True)
            sampled_rows = [row for episode in chosen for row in grouped[episode]]
            draws.append(np.mean([row[left] - row[right] for row in sampled_rows]))
        result[f"{left}_minus_{right}"] = {
            "observed": float(observed),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "probability_positive": float(np.mean(np.asarray(draws) > 0)),
            "episodes": len(episodes),
            "decisions": len(rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seen-trace", required=True)
    parser.add_argument("--unseen-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    comparisons = [
        ("evolved_skill", "plain"),
        ("constrained_stage", "plain"),
        ("constrained_skill", "plain"),
        ("constrained_skill", "constrained_stage"),
    ]
    results = {}
    for name, trace in (("valid_seen", args.seen_trace), ("valid_unseen", args.unseen_trace)):
        results[name] = cluster_bootstrap(
            correctness_rows(trace, args.checkpoint_dir),
            comparisons,
            samples=args.samples,
            seed=args.seed + len(results),
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    runs = []
    for result_path in sorted(args.run_root.glob("seed_*/results.json")):
        runs.append(json.loads(result_path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit(f"No results found under {args.run_root}")
    metric_names = sorted(runs[0]["metrics"])
    summary = {"run_count": len(runs), "metrics": {}}
    for method in metric_names:
        summary["metrics"][method] = {}
        preferred_keys = (
            "strict_accuracy", "tool_accuracy", "choice_accuracy",
            "mean_correct_action_probability", "mean_brier", "ece",
            "mean_kl_to_plain", "mean_final_answer_nll",
        )
        keys = tuple(key for key in preferred_keys if key in runs[0]["metrics"][method])
        for key in keys:
            values = [float(run["metrics"][method][key]) for run in runs]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(values))
            summary["metrics"][method][key] = {
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "values": values,
            }
    output_path = args.run_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for method, metrics in summary["metrics"].items():
        strict = metrics["strict_accuracy"]
        print(
            f"{method:34s} strict={strict['mean']:.3f} ± {strict['std']:.3f} "
            f"(95% CI ±{strict['ci95']:.3f})"
        )
    print(output_path)


if __name__ == "__main__":
    main()

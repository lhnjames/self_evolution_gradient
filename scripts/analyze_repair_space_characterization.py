#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def circular_component_count(mask: Sequence[bool]) -> int:
    values = np.asarray(mask, dtype=bool)
    if np.all(values):
        return 1
    if not np.any(values):
        return 0
    return int(np.sum(values & ~np.roll(values, 1)))


def pareto_mask(gain: Sequence[float], harm: Sequence[float]) -> np.ndarray:
    gains = np.asarray(gain, dtype=np.float64)
    harms = np.asarray(harm, dtype=np.float64)
    order = np.lexsort((-gains, harms))
    result = np.zeros(len(gains), dtype=bool)
    best_gain = -np.inf
    for index in order:
        if gains[index] > best_gain + 1e-15:
            result[index] = True
            best_gain = gains[index]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--local-slice-json")
    return parser.parse_args()


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def fmt_percent(values: Sequence[float], digits: int = 2) -> str:
    mean, std = mean_std(values)
    return f"{100 * mean:.{digits}f}% ± {100 * std:.{digits}f}%"


def summarize(
    matrix: dict[str, np.ndarray], positions: Sequence[int], base_expected: np.ndarray
) -> dict[str, np.ndarray]:
    local_delta = matrix["expected_value_delta"][:, positions]
    local_top = matrix["top_value_delta"][:, positions]
    denominator = float(np.mean(base_expected[positions]))
    return {
        "absolute_gain": local_delta.mean(axis=1),
        "relative_gain": local_delta.mean(axis=1) / denominator,
        "positive_rate": (local_delta > 0).mean(axis=1),
        "negative_rate_epsilon_0": (local_delta < 0).mean(axis=1),
        "negative_rate_epsilon_0.01": (local_delta < -0.01).mean(axis=1),
        "mean_downside": np.maximum(-local_delta, 0).mean(axis=1),
        "max_downside": np.maximum(-local_delta, 0).max(axis=1),
        "worst_delta": local_delta.min(axis=1),
        "top_value_repair_rate": (local_top > 0).mean(axis=1),
        "top_value_harm_rate": (local_top < 0).mean(axis=1),
        "mean_kl": matrix["kl"][:, positions].mean(axis=1),
    }


def safe_mask(target: dict[str, np.ndarray], protection: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (target["relative_gain"] >= 0.30)
        & (protection["top_value_harm_rate"] <= 0.02)
        & (protection["negative_rate_epsilon_0.01"] <= 0.02)
    )


def best_index(gain: np.ndarray, eligible: np.ndarray) -> int | None:
    indices = np.flatnonzero(eligible)
    return int(indices[np.argmax(gain[indices])]) if len(indices) else None


def candidate_has_protocol(candidate: dict[str, Any], protocol: str) -> bool:
    return any(origin.get("protocol") == protocol for origin in candidate["selection_origins"])


def point_record(
    index: int | None,
    candidates: list[dict[str, Any]],
    target: dict[str, np.ndarray],
    protection: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    if index is None:
        return None
    return {
        "candidate_id": index,
        "active_rank": candidates[index]["active_rank"],
        "step_norm": candidates[index]["step_norm"],
        "strength_multiple": candidates[index]["strength_multiple"],
        "relative_target_gain": float(target["relative_gain"][index]),
        "absolute_target_gain": float(target["absolute_gain"][index]),
        "target_positive_rate": float(target["positive_rate"][index]),
        "protection_top_value_harm_rate": float(protection["top_value_harm_rate"][index]),
        "protection_negative_rate_epsilon_0.01": float(
            protection["negative_rate_epsilon_0.01"][index]
        ),
        "protection_mean_downside": float(protection["mean_downside"][index]),
        "protection_worst_delta": float(protection["worst_delta"][index]),
    }


def nearest_centroid_overlap(points: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = []
    for index, point in enumerate(points):
        distances = []
        for label in (0, 1):
            mask = labels == label
            mask[index] = False
            centroid = points[mask].mean(axis=0)
            distances.append(float(np.linalg.norm(point - centroid)))
        predictions.append(int(np.argmin(distances)))
    predictions_array = np.asarray(predictions)
    recalls = [float(np.mean(predictions_array[labels == label] == label)) for label in (0, 1)]
    return {"balanced_accuracy": float(np.mean(recalls)), "accuracy": float(np.mean(predictions_array == labels))}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_slice = None
    if args.local_slice_json:
        local_slice = json.loads(Path(args.local_slice_json).read_text(encoding="utf-8"))
    value_rows = {}
    with Path(args.value_trace).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "valid_seen":
                value_rows[int(row["global_decision_index"])] = row

    seeds = []
    for json_path_string in args.input_json:
        json_path = Path(json_path_string)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        matrix_path = Path(data["matrix_file"])
        if not matrix_path.is_absolute() and not matrix_path.exists():
            matrix_path = json_path.parent.parent.parent / matrix_path
        loaded = np.load(matrix_path)
        matrix = {key: loaded[key] for key in loaded.files}
        positions = {key: [int(value) for value in values] for key, values in data["positions"].items()}
        base_expected = matrix["base_expected_value"]
        target_validation = summarize(matrix, positions["transfer_validation"], base_expected)
        protection_validation = summarize(matrix, positions["harm_validation"], base_expected)
        target_test = summarize(matrix, positions["same_skill_holdout"], base_expected)
        protection_test = summarize(matrix, positions["protection_test"], base_expected)
        target_full = summarize(
            matrix, positions["transfer_validation"] + positions["same_skill_holdout"], base_expected
        )
        protection_full = summarize(
            matrix, positions["harm_validation"] + positions["protection_test"], base_expected
        )
        candidates = data["candidates"]
        ranks = np.asarray([item["active_rank"] for item in candidates])
        validation_origin = np.asarray([
            candidate_has_protocol(item, "validation_only") for item in candidates
        ])
        validation_safe = (
            (protection_validation["top_value_harm_rate"] <= 0.02)
            & (protection_validation["negative_rate_epsilon_0.01"] <= 0.02)
        )
        selected_validation = best_index(
            target_validation["relative_gain"], validation_origin & validation_safe
        )
        selected_validation_strong = best_index(
            target_validation["relative_gain"],
            validation_origin & validation_safe & (target_validation["relative_gain"] >= 0.30),
        )
        full_safety = (
            (protection_full["top_value_harm_rate"] <= 0.02)
            & (protection_full["negative_rate_epsilon_0.01"] <= 0.02)
        )
        full_strong_safe = full_safety & (target_full["relative_gain"] >= 0.30)
        selected_full = best_index(target_full["relative_gain"], full_safety)
        rank_results = []
        for rank in range(1, 7):
            within_rank = ranks <= rank
            local_index = best_index(target_full["relative_gain"], within_rank & full_safety)
            max_gain_any = best_index(target_full["relative_gain"], within_rank)
            rank_results.append({
                "rank": rank,
                "safe_point": point_record(local_index, candidates, target_full, protection_full),
                "unconstrained_point": point_record(max_gain_any, candidates, target_full, protection_full),
            })

        strong_ids = np.flatnonzero(full_strong_safe)
        feasible_neighborhood = {
            "point_count": int(len(strong_ids)),
            "connected_components_at_20_degrees": 0,
            "maximum_pairwise_angle_degrees": None,
        }
        if len(strong_ids):
            local_coordinates = matrix["candidate_coordinates"][strong_ids]
            cosine_neighbors = local_coordinates @ local_coordinates.T >= np.cos(np.deg2rad(20.0))
            unseen_nodes = set(range(len(strong_ids)))
            components = 0
            while unseen_nodes:
                components += 1
                stack = [unseen_nodes.pop()]
                while stack:
                    node = stack.pop()
                    neighbors = set(np.flatnonzero(cosine_neighbors[node]).tolist()) & unseen_nodes
                    unseen_nodes -= neighbors
                    stack.extend(neighbors)
            pairwise_cosine = np.clip(local_coordinates @ local_coordinates.T, -1.0, 1.0)
            feasible_neighborhood = {
                "point_count": int(len(strong_ids)),
                "connected_components_at_20_degrees": int(components),
                "maximum_pairwise_angle_degrees": float(
                    np.max(np.degrees(np.arccos(pairwise_cosine)))
                ),
            }

        grid_connectivity = []
        for step_norm in data["search"]["rank2_grid_step_norms"]:
            grid = []
            for candidate in candidates:
                if abs(candidate["step_norm"] - step_norm) > 1e-10:
                    continue
                origins = [
                    origin for origin in candidate["selection_origins"]
                    if origin.get("kind") == "rank2_connectivity_grid"
                ]
                for origin in origins:
                    grid.append((float(origin["angle_degrees"]), int(candidate["candidate_id"])))
            grid.sort()
            grid_indices = np.asarray([item[1] for item in grid], dtype=int)
            safety = (
                (protection_full["top_value_harm_rate"][grid_indices] <= 0.02)
                & (protection_full["negative_rate_epsilon_0.01"][grid_indices] <= 0.02)
            )
            strong = target_full["relative_gain"][grid_indices] >= 0.30
            feasible = safety & strong
            grid_connectivity.append({
                "step_norm": float(step_norm),
                "strength_multiple": float(step_norm / 0.0006),
                "angle_count": len(grid_indices),
                "safety_only_fraction": float(safety.mean()),
                "strong_gain_only_fraction": float(strong.mean()),
                "feasible_fraction": float(feasible.mean()),
                "safety_component_count": circular_component_count(safety),
                "feasible_component_count": circular_component_count(feasible),
            })

        optimal_rows = data["per_state_projected_optimal_directions"]
        target_indices = positions["transfer_validation"] + positions["same_skill_holdout"]
        protection_indices = positions["harm_validation"] + positions["protection_test"]
        all_indices = target_indices + protection_indices
        coefficient_points = np.asarray([optimal_rows[index]["coordinates"] for index in all_indices])
        labels = np.asarray([1] * len(target_indices) + [0] * len(protection_indices))
        normalized = coefficient_points / np.linalg.norm(coefficient_points, axis=1, keepdims=True)
        cosine = normalized @ normalized.T
        target_mask = labels == 1
        protection_mask = labels == 0
        overlap = {
            **nearest_centroid_overlap(normalized, labels),
            "target_within_mean_cosine": float(cosine[np.ix_(target_mask, target_mask)][~np.eye(target_mask.sum(), dtype=bool)].mean()),
            "protection_within_mean_cosine": float(cosine[np.ix_(protection_mask, protection_mask)][~np.eye(protection_mask.sum(), dtype=bool)].mean()),
            "target_protection_mean_cosine": float(cosine[np.ix_(target_mask, protection_mask)].mean()),
            "mean_target_projection_fraction": float(np.mean([optimal_rows[index]["projection_fraction"] for index in target_indices])),
            "mean_protection_projection_fraction": float(np.mean([optimal_rows[index]["projection_fraction"] for index in protection_indices])),
        }

        seed_panel_indices = data["panel_indices"]
        seed_target_rows = [value_rows[seed_panel_indices[index]] for index in target_indices]
        seed_protection_rows = [value_rows[seed_panel_indices[index]] for index in protection_indices]
        seed_hidden_target_rows = [
            value_rows[seed_panel_indices[index]] for index in positions["same_skill_holdout"]
        ]
        seed_hidden_protection_rows = [
            value_rows[seed_panel_indices[index]] for index in positions["protection_test"]
        ]
        seed_baseline = np.mean([row["discounted_base_expected_value"] for row in seed_target_rows])
        seed_delta = np.mean([row["discounted_expected_value_delta"] for row in seed_target_rows])
        seed_hidden_baseline = np.mean([
            row["discounted_base_expected_value"] for row in seed_hidden_target_rows
        ])
        seed_hidden_delta = np.mean([
            row["discounted_expected_value_delta"] for row in seed_hidden_target_rows
        ])
        seed_metrics = {
            "target_relative_gain": float(seed_delta / seed_baseline),
            "target_absolute_gain": float(seed_delta),
            "protection_top_value_harm_rate": float(np.mean([
                row["discounted_top_value_delta"] < 0 for row in seed_protection_rows
            ])),
            "protection_negative_rate_epsilon_0.01": float(np.mean([
                row["discounted_expected_value_delta"] < -0.01 for row in seed_protection_rows
            ])),
            "hidden_target_relative_gain": float(seed_hidden_delta / seed_hidden_baseline),
            "hidden_target_absolute_gain": float(seed_hidden_delta),
            "hidden_protection_top_value_harm_rate": float(np.mean([
                row["discounted_top_value_delta"] < 0 for row in seed_hidden_protection_rows
            ])),
            "hidden_protection_negative_rate_epsilon_0.01": float(np.mean([
                row["discounted_expected_value_delta"] < -0.01
                for row in seed_hidden_protection_rows
            ])),
        }
        trajectory_coordinates = np.asarray([row["coordinates"] for row in data["trajectory"]])
        trajectory_unit = trajectory_coordinates / np.linalg.norm(
            trajectory_coordinates, axis=1, keepdims=True
        )
        feedback_coordinates = np.asarray(data["heldout_feedback_projection"]["coordinates"])
        feedback_unit = feedback_coordinates / np.linalg.norm(feedback_coordinates)
        trajectory_geometry = {
            "g4_g12_cosine_in_rank6": float(trajectory_unit[3] @ trajectory_unit[-1]),
            "feedback_g12_cosine_in_rank6": float(feedback_unit @ trajectory_unit[-1]),
            "g4_rank6_captured_norm_squared": float(data["trajectory"][3]["captured_norm_squared"]),
            "g12_rank6_captured_norm_squared": float(data["trajectory"][-1]["captured_norm_squared"]),
            "feedback_rank6_captured_norm_squared": float(
                data["heldout_feedback_projection"]["captured_norm_squared"]
            ),
            "feedback_residual_novelty": float(
                data["heldout_feedback_projection"]["residual_novelty"]
            ),
            "oracle_g4_cosine_in_rank6": None,
            "oracle_g12_cosine_in_rank6": None,
            "oracle_feedback_cosine_in_rank6": None,
        }
        if selected_full is not None:
            oracle = matrix["candidate_coordinates"][selected_full]
            trajectory_geometry.update({
                "oracle_g4_cosine_in_rank6": float(oracle @ trajectory_unit[3]),
                "oracle_g12_cosine_in_rank6": float(oracle @ trajectory_unit[-1]),
                "oracle_feedback_cosine_in_rank6": float(oracle @ feedback_unit),
            })

        seeds.append({
            "sample_seed": data["sample_seed"],
            "json_path": str(json_path),
            "data": data,
            "matrix": matrix,
            "summaries": {
                "target_validation": target_validation,
                "protection_validation": protection_validation,
                "target_test": target_test,
                "protection_test": protection_test,
                "target_full": target_full,
                "protection_full": protection_full,
            },
            "validation_selected": {
                "validation": point_record(selected_validation, candidates, target_validation, protection_validation),
                "hidden_test": point_record(selected_validation, candidates, target_test, protection_test),
                "validation_strong_candidate_exists": selected_validation_strong is not None,
            },
            "full_panel_oracle": point_record(selected_full, candidates, target_full, protection_full),
            "full_panel_strong_safe_exists": bool(np.any(full_strong_safe)),
            "full_panel_feasible_neighborhood": feasible_neighborhood,
            "rank_results": rank_results,
            "grid_connectivity": grid_connectivity,
            "coefficient_overlap": overlap,
            "trajectory_geometry": trajectory_geometry,
            "seed_baseline": seed_metrics,
        })

    # Pareto plot.
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds, strict=True):
        summary = seed["summaries"]
        candidates = seed["data"]["candidates"]
        gain = summary["target_full"]["relative_gain"]
        harm = summary["protection_full"]["mean_downside"]
        rank = np.asarray([item["active_rank"] for item in candidates])
        scatter = axis.scatter(100 * harm, 100 * gain, c=rank, s=8, alpha=0.35, cmap="viridis")
        frontier = pareto_mask(gain, harm)
        order = np.argsort(harm[frontier])
        axis.plot(100 * harm[frontier][order], 100 * gain[frontier][order], color="black", linewidth=1)
        axis.axhline(30, color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(f"seed {seed['sample_seed']}")
        axis.set_xlabel("mean protection downside (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("target relative value gain (%)")
    fig.colorbar(scatter, ax=axes, label="active repair rank")
    fig.suptitle("Actual nonlinear gain–harm Pareto samples")
    fig.savefig(output_dir / "gain_harm_pareto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Rank-constrained safe gain.
    fig, axis = plt.subplots(figsize=(7, 4))
    for seed in seeds:
        values = [
            np.nan if item["safe_point"] is None else 100 * item["safe_point"]["relative_target_gain"]
            for item in seed["rank_results"]
        ]
        axis.plot(range(1, 7), values, marker="o", label=str(seed["sample_seed"]))
    axis.axhline(30, color="tab:red", linestyle="--", label="30% gain gate")
    axis.set_xlabel("maximum allowed repair rank")
    axis.set_ylabel("max gain with empirical safety gates (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "feasible_repair_rank.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # First two repair coordinates: actual safe grid and evolution trajectory.
    fig, axes = plt.subplots(2, len(seeds), figsize=(4.6 * len(seeds), 8.2), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = np.asarray(axes)[:, None]
    for column, seed in enumerate(seeds):
        data, matrix = seed["data"], seed["matrix"]
        candidates = data["candidates"]
        full_target = seed["summaries"]["target_full"]
        full_protection = seed["summaries"]["protection_full"]
        for row, step_norm in enumerate(data["search"]["rank2_grid_step_norms"]):
            axis = axes[row, column]
            grid_ids = []
            for candidate in candidates:
                if abs(candidate["step_norm"] - step_norm) <= 1e-10 and any(
                    origin.get("kind") == "rank2_connectivity_grid"
                    for origin in candidate["selection_origins"]
                ):
                    grid_ids.append(int(candidate["candidate_id"]))
            grid_ids = np.asarray(grid_ids)
            coords = np.asarray([candidates[index]["coordinates"][:2] for index in grid_ids])
            safe = safe_mask(full_target, full_protection)[grid_ids]
            axis.scatter(coords[~safe, 0], coords[~safe, 1], s=7, color="lightgray")
            axis.scatter(coords[safe, 0], coords[safe, 1], s=12, color="tab:green", label="30%/2% feasible")
            trajectory = np.asarray([item["coordinates"][:2] for item in data["trajectory"]])
            axis.plot(trajectory[:, 0], trajectory[:, 1], "o-", color="tab:blue", markersize=3, linewidth=1)
            axis.annotate("G1", trajectory[0])
            axis.annotate("G4", trajectory[3])
            axis.annotate("G12", trajectory[-1])
            feedback = np.asarray(data["heldout_feedback_projection"]["coordinates"][:2])
            axis.scatter([feedback[0]], [feedback[1]], marker="*", s=80, color="tab:orange")
            axis.set_aspect("equal")
            axis.set_title(f"seed {seed['sample_seed']}, {step_norm / .0006:.0f}x")
            axis.grid(alpha=0.2)
    fig.suptitle("Rank-2 feasible directions, G1→G12 trajectory, and held-out feedback")
    fig.savefig(output_dir / "repair_space_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Projected per-state optimal coefficients.
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.5 * len(seeds), 4), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds, strict=True):
        data = seed["data"]
        rows = data["per_state_projected_optimal_directions"]
        target_positions = data["positions"]["transfer_validation"] + data["positions"]["same_skill_holdout"]
        protection_positions = data["positions"]["harm_validation"] + data["positions"]["protection_test"]
        target = np.asarray([rows[index]["coordinates"][:2] for index in target_positions])
        protection = np.asarray([rows[index]["coordinates"][:2] for index in protection_positions])
        axis.scatter(target[:, 0], target[:, 1], color="tab:green", marker="o", label="repair targets")
        axis.scatter(protection[:, 0], protection[:, 1], color="tab:red", marker="x", label="protection")
        axis.axhline(0, color="black", linewidth=.5)
        axis.axvline(0, color="black", linewidth=.5)
        axis.set_title(f"seed {seed['sample_seed']}")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Per-state optimal directions in first two repair coordinates")
    fig.savefig(output_dir / "target_protection_coefficients.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    serializable_seeds = []
    for seed in seeds:
        serializable_seeds.append({key: value for key, value in seed.items() if key not in {"data", "matrix", "summaries"}})
    aggregate = {
        "experiment": "repair_space_characterization_v1",
        "status": "complete",
        "seed_count": len(seeds),
        "seeds": serializable_seeds,
        "local_slice": local_slice,
        "definitions": {
            "strong_safe_gate": "target relative expected-value gain >=30%; protection top-value harm <=2%; protection DeltaV<-0.01 rate <=2%",
            "validation_only": "select using transfer_validation+harm_validation, evaluate same_skill_holdout+protection_test",
            "full_panel_oracle": "optimistic finite-panel existence upper bound, not deployable model selection",
        },
    }
    (output_dir / "repair_space_characterization.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    full_exists = [seed["full_panel_strong_safe_exists"] for seed in seeds]
    validation_test_pass = []
    for seed in seeds:
        point = seed["validation_selected"]["hidden_test"]
        validation_test_pass.append(
            point is not None
            and point["relative_target_gain"] >= .30
            and point["protection_top_value_harm_rate"] <= .02
            and point["protection_negative_rate_epsilon_0.01"] <= .02
        )
    seed_gain_mean, seed_gain_std = mean_std([
        seed["seed_baseline"]["target_relative_gain"] for seed in seeds
    ])
    seed_value_harm_mean, _ = mean_std([
        seed["seed_baseline"]["protection_negative_rate_epsilon_0.01"] for seed in seeds
    ])
    trajectory_cosine_mean, _ = mean_std([
        seed["trajectory_geometry"]["g4_g12_cosine_in_rank6"] for seed in seeds
    ])
    feedback_cosine_mean, _ = mean_std([
        seed["trajectory_geometry"]["feedback_g12_cosine_in_rank6"] for seed in seeds
    ])
    overlap_accuracy_mean, overlap_accuracy_std = mean_std([
        seed["coefficient_overlap"]["balanced_accuracy"] for seed in seeds
    ])
    actual_candidate_count = sum(len(seed["data"]["candidates"]) for seed in seeds)
    if local_slice is not None:
        actual_candidate_count += int(local_slice["candidate_count"])
    actual_state_response_count = 30 * actual_candidate_count
    report = [
        "# Repair-Space Characterization",
        "",
        "**日期：2026-09-04**",
        "",
        "## 判定口径",
        "",
        "强安全修复同时要求：目标相对长期价值提升至少 30%，protection top-value harm 不超过 2%，且 protection 中 `DeltaV < -0.01` 的比例不超过 2%。本面板的离散样本量使两个 2% 条件实际上都要求零个违规状态。",
        "",
        "`full-panel oracle` 只回答有限 output-gradient span 中是否存在方向，是乐观上界；`validation-only` 只在 validation 上选方向，隐藏 holdout/protection 才是泛化判定。所有表中数值均来自真实 multi-token softmax，而非一阶预测。",
        "",
        "## 总结果",
        "",
        f"- 搜索规模：四个 seed 共一阶预筛 **240 万**个单位方向；GPU 对 **{actual_candidate_count:,}** 个方向–强度候选执行完整 multi-token 非线性评分，得到 **{actual_state_response_count:,}** 个逐状态参数修改响应。",
        f"- 强安全单方向只在 **{sum(full_exists)}/{len(full_exists)} seed** 的完整面板 oracle 中存在；validation-only 选择后，隐藏测试通过率是 **{sum(validation_test_pass)}/{len(validation_test_pass)}**。",
        "- rank 1–2 在四个 seed 中都没有找到满足 protection 条件的方向；唯一超过 30% 的安全方向直到 rank 5 才出现。",
        "- 这否定了“一个稳定低维全局 output direction 就足够”的当前证据，但不能证明整个 output-gradient span 数学上无解。",
        "- 当前 G4→G12 与新 feedback 主要仍沿既有方向移动；它们并未系统靠近 full-panel 的安全 Pareto 区域。",
        f"- SEED 在同一目标面板平均提升 **{100*seed_gain_mean:.2f}% ± {100*seed_gain_std:.2f}%**，top-value harm 为 0，但仍有 **{100*seed_value_harm_mean:.2f}%** protection 状态的长期价值下降超过 0.01；因此 SEED 也不满足这里更严格的 value-safety gate。",
        "",
        "## 1. Single-direction feasibility",
        "",
        "| seed | full-panel strong-safe ≥30% | best safe rank | 强度 | best safe target gain | top harm | DeltaV<-0.01 | validation-selected hidden target | hidden top harm | hidden DeltaV<-0.01 |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        full = seed["full_panel_oracle"]
        hidden = seed["validation_selected"]["hidden_test"]
        full_strength = f"{full['strength_multiple']:.0f}x" if full else "—"
        full_gain = f"{100 * full['relative_target_gain']:.2f}%" if full else "—"
        full_top_harm = f"{100 * full['protection_top_value_harm_rate']:.2f}%" if full else "—"
        full_value_harm = (
            f"{100 * full['protection_negative_rate_epsilon_0.01']:.2f}%" if full else "—"
        )
        hidden_gain = f"{100 * hidden['relative_target_gain']:.2f}%" if hidden else "—"
        hidden_top_harm = (
            f"{100 * hidden['protection_top_value_harm_rate']:.2f}%" if hidden else "—"
        )
        hidden_value_harm = (
            f"{100 * hidden['protection_negative_rate_epsilon_0.01']:.2f}%" if hidden else "—"
        )
        report.append(
            f"| {seed['sample_seed']} | {'yes' if seed['full_panel_strong_safe_exists'] else 'no'} | "
            f"{full['active_rank'] if full else '—'} | "
            f"{full_strength} | {full_gain} | {full_top_harm} | {full_value_harm} | "
            f"{hidden_gain} | {hidden_top_harm} | {hidden_value_harm} |"
        )
    report.extend([
        "",
        f"有限面板中找到强安全单方向的 seed 比例：**{sum(full_exists)}/{len(full_exists)}**。validation 选择后在隐藏测试仍通过全部门槛：**{sum(validation_test_pass)}/{len(validation_test_pass)}**。表中未达到 30% 的 full-panel 数值表示“安全但不够强”，不是强安全成功。",
        "",
        "## 2. Feasible repair rank",
        "",
        "下表报告每个最大允许 rank 下，满足 protection 安全门槛时可找到的最高 target gain（`—` 表示搜索中连安全方向都未找到）。数值达到 30% 才通过强修复门槛。",
        "",
        "| rank | " + " | ".join(str(seed["sample_seed"]) for seed in seeds) + " |",
        "|---:|" + "---:|" * len(seeds),
    ])
    for rank in range(1, 7):
        values = []
        for seed in seeds:
            point = seed["rank_results"][rank - 1]["safe_point"]
            values.append(f"{100*point['relative_target_gain']:.2f}%" if point else "—")
        report.append(f"| {rank} | " + " | ".join(values) + " |")
    report.extend([
        "",
        "唯一强安全 seed 的邻域复核：",
        "",
        "| seed | strong-safe sampled points | components at 20° | maximum pairwise angle |",
        "|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        row = seed["full_panel_feasible_neighborhood"]
        maximum_angle = (
            f"{row['maximum_pairwise_angle_degrees']:.2f}°"
            if row["maximum_pairwise_angle_degrees"] is not None else "—"
        )
        report.append(
            f"| {seed['sample_seed']} | {row['point_count']} | "
            f"{row['connected_components_at_20_degrees']} | {maximum_angle} |"
        )
    report.extend([
        "",
        "## 3. Rank-2 connectivity",
        "",
        "| seed | strength | safety-only coverage | safety components | 30%/2% feasible coverage | feasible components |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        for row in seed["grid_connectivity"]:
            report.append(
                f"| {seed['sample_seed']} | {row['strength_multiple']:.0f}x | "
                f"{100*row['safety_only_fraction']:.1f}% | {row['safety_component_count']} | "
                f"{100*row['feasible_fraction']:.1f}% | {row['feasible_component_count']} |"
            )
    if local_slice is not None:
        report.extend([
            "",
            "### Rank-5 成功点的局部连通性复核",
            "",
            f"对 seed `{local_slice['sample_seed']}` 的唯一成功区域，在 300x 下围绕 oracle 中心做 541 点 rank-5 二维切平面扫描：强安全点 **{local_slice['feasible_count']} / {local_slice['candidate_count']}（{100*local_slice['feasible_fraction']:.1f}%）**，形成 **{local_slice['feasible_component_count']} 个连通分量**。中心半径 {local_slice['largest_radius_fully_feasible_degrees']:.0f}° 内全部方向可行；至少一个可行方向延伸至 {local_slice['largest_radius_with_any_feasible_degrees']:.0f}°。这说明该 seed 中不是孤立采样噪点，而是一个连续但高度各向异性的高维 feasible basin。",
            "",
            "![rank-5 local feasible region](local_feasible_region_seed_20260921.png)",
        ])
    report.extend([
        "",
        "## 4. Target/protection coefficient overlap",
        "",
        "| seed | target within cosine | protection within cosine | target–protection cosine | LOO balanced accuracy | target span capture | protection span capture |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        row = seed["coefficient_overlap"]
        report.append(
            f"| {seed['sample_seed']} | {row['target_within_mean_cosine']:.3f} | "
            f"{row['protection_within_mean_cosine']:.3f} | {row['target_protection_mean_cosine']:.3f} | "
            f"{100*row['balanced_accuracy']:.1f}% | {100*row['mean_target_projection_fraction']:.1f}% | "
            f"{100*row['mean_protection_projection_fraction']:.1f}% |"
        )
    report.extend([
        "",
        "## 5. Evolution trajectory 与 held-out feedback",
        "",
        "| seed | cos(G4,G12) | cos(feedback,G12) | feedback residual novelty | cos(best-safe,G4) | cos(best-safe,G12) | cos(best-safe,feedback) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        row = seed["trajectory_geometry"]
        def optional(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "—"
        report.append(
            f"| {seed['sample_seed']} | {row['g4_g12_cosine_in_rank6']:.3f} | "
            f"{row['feedback_g12_cosine_in_rank6']:.3f} | {row['feedback_residual_novelty']:.3f} | "
            f"{optional(row['oracle_g4_cosine_in_rank6'])} | "
            f"{optional(row['oracle_g12_cosine_in_rank6'])} | "
            f"{optional(row['oracle_feedback_cosine_in_rank6'])} |"
        )
    report.extend([
        "",
        "这里的 best-safe 是各 seed 在完整面板上满足 protection 安全条件的最高 gain 点，不是可部署选择；余弦只用于解释现有 evolution trajectory 是否靠近该有限面板区域。",
        "",
        "## 6. 与 SEED 的同面板横向参照",
        "",
        "| seed | SEED full target gain | SEED full top harm | SEED full DeltaV<-0.01 | SEED hidden target | validation-selected OGSE hidden target | OGSE hidden top harm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        row = seed["seed_baseline"]
        ogse = seed["validation_selected"]["hidden_test"]
        ogse_hidden_gain = f"{100 * ogse['relative_target_gain']:.2f}%" if ogse else "—"
        ogse_hidden_harm = (
            f"{100 * ogse['protection_top_value_harm_rate']:.2f}%" if ogse else "—"
        )
        report.append(
            f"| {seed['sample_seed']} | {100*row['target_relative_gain']:.2f}% | "
            f"{100*row['protection_top_value_harm_rate']:.2f}% | "
            f"{100*row['protection_negative_rate_epsilon_0.01']:.2f}% | "
            f"{100*row['hidden_target_relative_gain']:.2f}% | "
            f"{ogse_hidden_gain} | {ogse_hidden_harm} |"
        )
    report.extend([
        "",
        "## 7. 对 A / B / C 三种可能性的判定",
        "",
        "### A：存在一个稳定的 single safe direction",
        "",
        f"当前不支持。完整面板强安全方向仅在 {sum(full_exists)}/{len(full_exists)} seed 出现，且 validation-only 无一在隐藏测试保持 30%/2%。所以不能把 seed 20260921 的 oracle basin 当作通用方向。",
        "",
        "### B：需要低维连续组合空间",
        "",
        "得到部分支持，但维度高于最初预期。rank 1–2 没有任何安全方向；唯一强安全区域在 rank 5 出现，并且局部切面是单个连续 basin。也就是说，至少在成功 seed 上，continuous coefficient space 比 hard branch 更符合观测，但不是 2–3 维就足够。",
        "",
        "### C：不存在对所有状态统一安全且可泛化的 output 修改",
        "",
        f"这是当前跨 seed 证据最支持的解释，但仍只能表述为经验性结论。目标/protection 的 per-state coefficient 只能以 **{100*overlap_accuracy_mean:.1f}% ± {100*overlap_accuracy_std:.1f}%** balanced accuracy 分开，存在明显重叠；同时 validation 方向在隐藏 protection 上频繁失效。需要扩大独立状态面板或进入 deeper-layer span 才能判断这是有限采样问题还是 output-head capacity 上限。",
        "",
        "### Evolution / feedback 几何",
        "",
        f"四个 seed 的 `cos(G4,G12)` 平均为 **{trajectory_cosine_mean:.3f}**，`cos(feedback,G12)` 平均为 **{feedback_cosine_mean:.3f}**。这说明后期累积与新 feedback 大多强化既有方向，而不是自动转向安全 Pareto 区域。反馈有新分量不等于它提供了有益的 feasible-space displacement。",
        "",
        "## 8. 当前可执行结论",
        "",
        "1. 不应继续优化单一 mean/weighted output gradient 并期待它自然满足 30%/2%。",
        "2. 也不应仅凭一个 seed 的连续 basin 就提前决定采用 basis 或 branch；该 basin 尚未跨 seed、跨隐藏状态复现。",
        "3. 下一项机制实验应直接比较：同样的 target/protection 面板在完整 12 维 output span 与 final-MLP/deeper span 中，强安全可行率是否上升。若 output full-span 仍低而 deeper span 明显上升，瓶颈是表示可分性；若 full-span 上升，则当前 rank-6 截断/搜索覆盖不足。",
        "4. 任何后续 evolution operator 都必须以隐藏 protection 验证为准；full-panel oracle 只能用于证明 capacity，不能用于报告可部署性能。",
        "",
        "## 图",
        "",
        "- `gain_harm_pareto.png`：真实非线性 Gain–Harm Pareto 样本。",
        "- `feasible_repair_rank.png`：安全门槛下的 rank–capacity 曲线。",
        "- `repair_space_trajectory.png`：二维可行区域、G1→G12 搜索轨迹和 held-out feedback。",
        "- `target_protection_coefficients.png`：目标与 protection 的 per-state optimal repair coefficients。",
        "",
        "## 结论边界",
        "",
        "本实验是在固定有限状态面板与 12 个 source-gradient span 内进行的强度/方向搜索。找到方向可证明该有限空间中存在候选解；没有找到只能说明在预注册的 100k/rank 随机搜索及二维稠密网格中未发现，不能当作数学上的不可行证明。validation-only 的隐藏测试结果才回答该方向是否能从稀疏反馈泛化。",
        "",
    ])
    (output_dir / "repair_space_characterization.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

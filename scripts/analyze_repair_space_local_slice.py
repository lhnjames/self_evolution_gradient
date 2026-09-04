#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def group_metrics(
    matrix: dict[str, np.ndarray], positions: list[int], base_expected: np.ndarray
) -> dict[str, np.ndarray]:
    delta = matrix["expected_value_delta"][:, positions]
    top = matrix["top_value_delta"][:, positions]
    return {
        "relative_gain": delta.mean(axis=1) / base_expected[positions].mean(),
        "top_harm": (top < 0).mean(axis=1),
        "negative_0.01": (delta < -0.01).mean(axis=1),
        "mean_downside": np.maximum(-delta, 0).mean(axis=1),
        "worst_delta": delta.min(axis=1),
    }


def graph_components(nodes: set[tuple[float, float]], radius_step: float, azimuth_step: float) -> int:
    unseen = set(nodes)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            radius, azimuth = stack.pop()
            neighbors = {
                (radius, (azimuth - azimuth_step) % 360),
                (radius, (azimuth + azimuth_step) % 360),
                (radius - radius_step, azimuth),
                (radius + radius_step, azimuth),
            }
            if radius == 0:
                neighbors |= {(radius_step, value) for value in np.arange(0, 360, azimuth_step)}
            if radius == radius_step:
                neighbors.add((0.0, 0.0))
            discovered = neighbors & unseen
            unseen -= discovered
            stack.extend(discovered)
    return components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_path = Path(args.input_json)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    matrix_path = Path(data["matrix_file"])
    if not matrix_path.is_absolute() and not matrix_path.exists():
        matrix_path = input_path.parent.parent.parent / matrix_path
    loaded = np.load(matrix_path)
    matrix = {key: loaded[key] for key in loaded.files}
    positions = data["positions"]
    target_positions = positions["transfer_validation"] + positions["same_skill_holdout"]
    protection_positions = positions["harm_validation"] + positions["protection_test"]
    target = group_metrics(matrix, target_positions, matrix["base_expected_value"])
    protection = group_metrics(matrix, protection_positions, matrix["base_expected_value"])
    safety = (protection["top_harm"] <= .02) & (protection["negative_0.01"] <= .02)
    strong = target["relative_gain"] >= .30
    feasible = safety & strong
    metadata = []
    for candidate in data["candidates"]:
        origin = candidate["selection_origins"][0]
        metadata.append((float(origin["radius_degrees"]), float(origin["azimuth_degrees"])))
    radius_step = min(radius for radius, _ in metadata if radius > 0)
    azimuths = sorted({azimuth for radius, azimuth in metadata if radius > 0})
    azimuth_step = azimuths[1] - azimuths[0]
    feasible_nodes = {metadata[index] for index in np.flatnonzero(feasible)}
    safety_nodes = {metadata[index] for index in np.flatnonzero(safety)}
    radii = sorted({radius for radius, _ in metadata})
    radial = []
    for radius in radii:
        indices = [index for index, value in enumerate(metadata) if value[0] == radius]
        radial.append({
            "radius_degrees": radius,
            "count": len(indices),
            "safety_fraction": float(safety[indices].mean()),
            "strong_gain_fraction": float(strong[indices].mean()),
            "feasible_fraction": float(feasible[indices].mean()),
            "mean_target_gain": float(target["relative_gain"][indices].mean()),
            "maximum_target_gain": float(target["relative_gain"][indices].max()),
        })
    center = {
        "relative_target_gain": float(target["relative_gain"][0]),
        "protection_top_harm": float(protection["top_harm"][0]),
        "protection_negative_0.01": float(protection["negative_0.01"][0]),
        "protection_mean_downside": float(protection["mean_downside"][0]),
        "protection_worst_delta": float(protection["worst_delta"][0]),
    }
    output = {
        "experiment": "repair_space_local_slice_v1",
        "status": "complete",
        "sample_seed": data["sample_seed"],
        "center": center,
        "candidate_count": len(metadata),
        "feasible_count": int(feasible.sum()),
        "safety_count": int(safety.sum()),
        "feasible_fraction": float(feasible.mean()),
        "feasible_component_count": graph_components(feasible_nodes, radius_step, azimuth_step) if feasible_nodes else 0,
        "safety_component_count": graph_components(safety_nodes, radius_step, azimuth_step) if safety_nodes else 0,
        "largest_radius_with_any_feasible_degrees": max(
            (row["radius_degrees"] for row in radial if row["feasible_fraction"] > 0), default=None
        ),
        "largest_radius_fully_feasible_degrees": max(
            (row["radius_degrees"] for row in radial if row["feasible_fraction"] == 1), default=None
        ),
        "radial_profile": radial,
        "gate": "target relative gain>=30%, protection top-harm<=2%, protection DeltaV<-0.01<=2%",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "local_slice_characterization.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    xy = np.asarray([
        (radius * np.cos(np.deg2rad(azimuth)), radius * np.sin(np.deg2rad(azimuth)))
        for radius, azimuth in metadata
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter = axes[0].scatter(xy[:, 0], xy[:, 1], c=100 * target["relative_gain"], cmap="viridis", s=24)
    fig.colorbar(scatter, ax=axes[0], label="target relative value gain (%)")
    axes[0].set_title("Actual target gain around rank-5 oracle")
    axes[1].scatter(xy[~feasible, 0], xy[~feasible, 1], color="lightgray", s=18, label="outside gate")
    axes[1].scatter(xy[feasible, 0], xy[feasible, 1], color="tab:green", s=25, label="30%/2% feasible")
    axes[1].set_title("Strong-safe feasible region")
    axes[1].legend()
    for axis in axes:
        axis.scatter([0], [0], marker="*", s=100, color="tab:orange", zorder=5)
        axis.set_xlabel("tangent coordinate 1 (degrees)")
        axis.set_ylabel("tangent coordinate 2 (degrees)")
        axis.set_aspect("equal")
        axis.grid(alpha=.2)
    fig.suptitle("Seed 20260921, rank-5 local slice, 300x")
    fig.savefig(output_dir / "local_feasible_region_seed_20260921.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-glob", required=True)
    parser.add_argument("--scope-glob", required=True)
    parser.add_argument("--unseen-family-glob", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No files match {pattern}")
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def correlation(left: Sequence[float], right: Sequence[float], rank: bool = False) -> float:
    left_array = ranks(left) if rank else np.asarray(left, dtype=np.float64)
    right_array = ranks(right) if rank else np.asarray(right, dtype=np.float64)
    if np.std(left_array) <= 1e-20 or np.std(right_array) <= 1e-20:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def macro(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": len(array),
        "mean": float(np.nanmean(array)),
        "sample_std": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.nanmin(array)),
        "maximum": float(np.nanmax(array)),
    }


def adjusted_rand(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("ARI inputs must have equal length >= 2")
    pairs = Counter(zip(left, right, strict=True))
    left_counts, right_counts = Counter(left), Counter(right)
    choose_two = lambda value: value * (value - 1) / 2
    observed = sum(choose_two(value) for value in pairs.values())
    left_sum = sum(choose_two(value) for value in left_counts.values())
    right_sum = sum(choose_two(value) for value in right_counts.values())
    total = choose_two(len(left))
    expected = left_sum * right_sum / total
    denominator = 0.5 * (left_sum + right_sum) - expected
    return float((observed - expected) / denominator) if abs(denominator) > 1e-30 else 0.0


def normalized_mutual_information(left: Sequence[Any], right: Sequence[Any]) -> float:
    count = len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    pair_counts = Counter(zip(left, right, strict=True))
    mutual = sum(
        value / count * math.log(value * count / (left_counts[a] * right_counts[b]))
        for (a, b), value in pair_counts.items()
    )
    left_entropy = -sum(value / count * math.log(value / count) for value in left_counts.values())
    right_entropy = -sum(value / count * math.log(value / count) for value in right_counts.values())
    return float(mutual / math.sqrt(left_entropy * right_entropy)) \
        if left_entropy * right_entropy > 0 else 0.0


def centered_spectrum(gram: np.ndarray) -> dict[str, Any]:
    count = len(gram)
    centering = np.eye(count) - np.ones((count, count)) / count
    centered = centering @ gram @ centering
    eigenvalues = np.linalg.eigvalsh((centered + centered.T) / 2)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(ratios)

    def components(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1) if total > 0 else 0

    participation = total * total / float(np.sum(eigenvalues ** 2)) if total > 0 else 0.0
    return {
        "eigenvalues": eigenvalues.tolist(),
        "explained_ratio": ratios.tolist(),
        "participation_effective_rank": participation,
        "components_80_percent": components(0.80),
        "components_90_percent": components(0.90),
        "components_95_percent": components(0.95),
    }


def kmeans(features: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [features[rng.integers(len(features))]]
    while len(centers) < clusters:
        squared = np.min([
            np.sum((features - center) ** 2, axis=1) for center in centers
        ], axis=0)
        if squared.sum() <= 1e-20:
            candidates = [i for i in range(len(features)) if not any(np.allclose(features[i], c) for c in centers)]
            centers.append(features[candidates[0]])
        else:
            centers.append(features[rng.choice(len(features), p=squared / squared.sum())])
    centers_array = np.asarray(centers)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(100):
        distances = np.stack([
            np.sum((features - center) ** 2, axis=1) for center in centers_array
        ], axis=1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels) and _ > 0:
            break
        labels = new_labels
        for cluster in range(clusters):
            selected = features[labels == cluster]
            if len(selected):
                centers_array[cluster] = selected.mean(axis=0)
    return labels


def silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sqrt(np.maximum(
        np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1), 0.0
    ))
    values = []
    for index, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != index]
        if not len(same):
            values.append(0.0)
            continue
        within = float(distances[index, same].mean())
        other_means = [
            float(distances[index, labels == other].mean())
            for other in sorted(set(labels)) if other != label
        ]
        nearest = min(other_means)
        values.append((nearest - within) / max(nearest, within, 1e-30))
    return float(np.mean(values))


def cluster_topology(cosine: np.ndarray) -> dict[str, Any]:
    eigenvalues, eigenvectors = np.linalg.eigh((cosine + cosine.T) / 2)
    selected = eigenvalues > 1e-10
    features = eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])
    candidates = []
    for clusters in range(2, min(6, len(cosine) - 1) + 1):
        best = None
        for seed in range(20):
            labels = kmeans(features, clusters, seed)
            if len(set(labels)) != clusters:
                continue
            score = silhouette(features, labels)
            if best is None or score > best[0]:
                best = (score, labels.copy())
        if best is not None:
            candidates.append((clusters, best[0], best[1]))
    clusters, score, labels = max(candidates, key=lambda item: item[1])
    upper = np.triu_indices(len(cosine), 1)
    same = labels[upper[0]] == labels[upper[1]]
    edge_values = cosine[upper]
    negative = edge_values < 0
    negative_cross = float(np.mean(~same[negative])) if np.any(negative) else 0.0
    return {
        "selected_cluster_count": clusters,
        "silhouette": score,
        "labels": labels.tolist(),
        "within_mean_cosine": float(edge_values[same].mean()),
        "between_mean_cosine": float(edge_values[~same].mean()),
        "within_between_gap": float(edge_values[same].mean() - edge_values[~same].mean()),
        "negative_edge_fraction": float(np.mean(negative)),
        "negative_edges_across_clusters": negative_cross,
        "all_candidate_silhouettes": {str(k): value for k, value, _ in candidates},
    }


def asymmetry(run: dict[str, Any], step_norm: float) -> dict[str, float]:
    close = set(run["close_indices"])
    transfer = {
        (int(row["source_index"]), int(row["target_index"])): float(row["expected_value_delta"])
        for row in run["transfer_rows"]
        if abs(row["step_norm"] - step_norm) < 1e-12
        and row["source_index"] in close and row["target_index"] in close
        and row["source_index"] != row["target_index"]
    }
    forward, reverse = [], []
    sign_equal_zero, sign_equal_practical, opposite_practical, normalized = [], [], [], []
    indices = sorted(close)
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1:]:
            a, b = transfer[(left, right)], transfer[(right, left)]
            forward.append(a); reverse.append(b)
            sign_equal_zero.append(np.sign(a) == np.sign(b))
            label_a = 1 if a > 0.01 else -1 if a < -0.01 else 0
            label_b = 1 if b > 0.01 else -1 if b < -0.01 else 0
            sign_equal_practical.append(label_a == label_b)
            opposite_practical.append(label_a * label_b == -1)
            normalized.append(abs(a - b) / max(abs(a) + abs(b), 1e-12))
    return {
        "pair_count": len(forward),
        "pearson": correlation(forward, reverse),
        "spearman": correlation(forward, reverse, rank=True),
        "sign_agreement_epsilon_0": float(np.mean(sign_equal_zero)),
        "sign_agreement_epsilon_0.01": float(np.mean(sign_equal_practical)),
        "opposite_sign_fraction_epsilon_0.01": float(np.mean(opposite_practical)),
        "mean_normalized_asymmetry": float(np.mean(normalized)),
    }


def scope_rows(run: dict[str, Any], count: int, step_norm: float) -> dict[int, dict[str, Any]]:
    return {
        int(row["global_decision_index"]): row
        for row in run["rows"]
        if row["experience_count"] == count and abs(row["step_norm"] - step_norm) < 1e-12
    }


def label(value: float) -> str:
    return "positive" if value > 0.01 else "negative" if value < -0.01 else "neutral"


def plot_results(
    topology_runs: Sequence[dict[str, Any]], result: dict[str, Any], output_dir: Path
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    matrix_specs = (
        ("gradient_cosine", "Full-gradient cosine"),
        ("delta_signature_cosine", "Delta-signature cosine"),
        ("hidden_signature_cosine", "Hidden-signature cosine"),
    )
    for run in topology_runs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for axis, (key, title) in zip(axes, matrix_specs, strict=True):
            matrix = np.asarray(run[key])
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
            axis.set_title(title)
            axis.set_xlabel("target failure")
            axis.set_ylabel("source failure")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="cosine")
        fig.suptitle(f"Repair topology, seed={run['sample_seed']}")
        path = output_dir / f"topology_matrices_seed_{run['sample_seed']}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for dose, axis in zip((0.18, 0.72), axes, strict=True):
        forward, reverse = [], []
        for run in topology_runs:
            close = set(run["close_indices"])
            transfer = {(r["source_index"], r["target_index"]): r["expected_value_delta"]
                        for r in run["transfer_rows"] if r["step_norm"] == dose}
            indices = sorted(close)
            for i, left in enumerate(indices):
                for right in indices[i + 1:]:
                    forward.append(transfer[left, right]); reverse.append(transfer[right, left])
        axis.scatter(forward, reverse, s=8, alpha=0.35)
        bound = max(np.max(np.abs(forward)), np.max(np.abs(reverse)))
        axis.plot([-bound, bound], [-bound, bound], color="black", linewidth=1)
        axis.axhline(0, color="gray", linewidth=0.5); axis.axvline(0, color="gray", linewidth=0.5)
        axis.set_title(f"{dose / 0.0006:.0f}x")
        axis.set_xlabel("T(i→j)"); axis.set_ylabel("T(j→i)")
    fig.tight_layout()
    path = output_dir / "transfer_asymmetry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig); paths.append(path.name)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, title in (("full_gradient", "gradient"), ("delta_signature", "delta"), ("hidden_signature", "hidden")):
        curves = np.asarray([
            run["spectra"][key]["explained_ratio"] for run in result["per_seed"]
        ])
        ax.plot(np.arange(1, curves.shape[1] + 1), np.cumsum(curves, axis=1).mean(axis=0), label=title)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("principal components"); ax.set_ylabel("mean cumulative explained variance")
    ax.legend(); fig.tight_layout()
    path = output_dir / "effective_rank_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig); paths.append(path.name)
    return paths


def main() -> None:
    args = parse_args()
    topology_runs = load(args.topology_glob)
    scope_runs = load(args.scope_glob)
    unseen_runs = load(args.unseen_family_glob)
    unseen_by_seed = {int(run["sample_seed"]): run for run in unseen_runs}
    with Path(args.value_trace).open(encoding="utf-8") as handle:
        value_rows = {
            int(row["global_decision_index"]): row
            for line in handle if (row := json.loads(line)) and row["split"] == "valid_seen"
        }
    scope_by_seed = {int(run["sample_seed"]): run for run in scope_runs}
    if set(scope_by_seed) != {int(run["sample_seed"]) for run in topology_runs}:
        raise ValueError("Topology and scope seeds differ")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    matrix_specs = {
        "full_gradient": ("gradient_gram", "gradient_cosine"),
        "delta_signature": ("delta_signature_gram", "delta_signature_cosine"),
        "hidden_signature": ("hidden_signature_gram", "hidden_signature_cosine"),
    }
    for run in topology_runs:
        spectra, topology = {}, {}
        for name, (gram_key, cosine_key) in matrix_specs.items():
            spectra[name] = centered_spectrum(np.asarray(run[gram_key], dtype=np.float64))
            topology[name] = cluster_topology(np.asarray(run[cosine_key], dtype=np.float64))
        transfer_1200 = np.asarray([
            [
                next(row["expected_value_delta"] for row in run["transfer_rows"]
                     if row["source_index"] == source and row["target_index"] == target
                     and abs(row["step_norm"] - 0.72) < 1e-12)
                for target in run["close_indices"]
            ]
            for source in run["close_indices"]
        ])
        spectra["empirical_transfer_1200x"] = centered_spectrum(
            transfer_1200 @ transfer_1200.T
        )
        per_seed.append({
            "sample_seed": run["sample_seed"],
            "spectra": spectra,
            "topology": topology,
            "asymmetry_300x": asymmetry(run, 0.18),
            "asymmetry_1200x": asymmetry(run, 0.72),
            "heldout_feedback_novelty": run["heldout_feedback_novelty"],
            "gradient_delta_cluster_ari": adjusted_rand(
                topology["full_gradient"]["labels"], topology["delta_signature"]["labels"]
            ),
            "gradient_hidden_cluster_ari": adjusted_rand(
                topology["full_gradient"]["labels"], topology["hidden_signature"]["labels"]
            ),
            "gradient_delta_edge_correlation": correlation(
                np.asarray(run["gradient_cosine"])[np.triu_indices(len(run["close_indices"]), 1)],
                np.asarray(run["delta_signature_cosine"])[np.triu_indices(len(run["close_indices"]), 1)],
            ),
            "gradient_hidden_edge_correlation": correlation(
                np.asarray(run["gradient_cosine"])[np.triu_indices(len(run["close_indices"]), 1)],
                np.asarray(run["hidden_signature_cosine"])[np.triu_indices(len(run["close_indices"]), 1)],
            ),
        })

    asymmetry_summary = {}
    for key in ("asymmetry_300x", "asymmetry_1200x"):
        metrics = [name for name in per_seed[0][key] if name != "pair_count"]
        asymmetry_summary[key] = {
            "unordered_pairs_per_seed": per_seed[0][key]["pair_count"],
            **{name: macro([run[key][name] for run in per_seed]) for name in metrics},
        }

    spectrum_summary = {}
    topology_summary = {}
    for name in (*matrix_specs, "empirical_transfer_1200x"):
        spectrum_summary[name] = {
            metric: macro([run["spectra"][name][metric] for run in per_seed])
            for metric in (
                "participation_effective_rank", "components_80_percent",
                "components_90_percent", "components_95_percent",
            )
        }
    for name in matrix_specs:
        topology_summary[name] = {
            metric: macro([run["topology"][name][metric] for run in per_seed])
            for metric in (
                "selected_cluster_count", "silhouette", "within_mean_cosine",
                "between_mean_cosine", "within_between_gap", "negative_edge_fraction",
                "negative_edges_across_clusters",
            )
        }

    cluster_correspondence = {
        "gradient_delta_cluster_ari": macro([row["gradient_delta_cluster_ari"] for row in per_seed]),
        "gradient_hidden_cluster_ari": macro([row["gradient_hidden_cluster_ari"] for row in per_seed]),
        "gradient_delta_edge_correlation": macro([row["gradient_delta_edge_correlation"] for row in per_seed]),
        "gradient_hidden_edge_correlation": macro([row["gradient_hidden_edge_correlation"] for row in per_seed]),
    }
    episode_partitions = []
    for run, summary in zip(topology_runs, per_seed, strict=True):
        episode_partitions.append((
            int(run["sample_seed"]),
            {
                value_rows[index]["episode_key"]: cluster
                for index, cluster in zip(
                    run["close_indices"], summary["topology"]["full_gradient"]["labels"], strict=True
                )
            },
        ))
    cross_seed_ari = []
    for left_position, (left_seed, left) in enumerate(episode_partitions):
        for right_seed, right in episode_partitions[left_position + 1:]:
            common = sorted(left.keys() & right.keys())
            cross_seed_ari.append({
                "left_seed": left_seed,
                "right_seed": right_seed,
                "common_episodes": len(common),
                "adjusted_rand": adjusted_rand(
                    [left[key] for key in common], [right[key] for key in common]
                ),
            })
    cluster_correspondence["cross_seed_episode_partition_ari"] = {
        "pairs": cross_seed_ari,
        "summary": macro([row["adjusted_rand"] for row in cross_seed_ari]),
    }

    semantic_association: dict[str, Any] = {}
    for metadata_key in ("task_type", "expert_action", "base_top_action"):
        observations, null_means, excess = [], [], []
        for run, summary in zip(topology_runs, per_seed, strict=True):
            labels = summary["topology"]["full_gradient"]["labels"]
            metadata = np.asarray([value_rows[index][metadata_key] for index in run["close_indices"]])
            observed = normalized_mutual_information(labels, metadata)
            rng = np.random.default_rng(int(run["sample_seed"]))
            null = np.asarray([
                normalized_mutual_information(labels, rng.permutation(metadata)) for _ in range(1000)
            ])
            observations.append(observed); null_means.append(float(null.mean()))
            excess.append(observed - float(null.mean()))
        semantic_association[metadata_key] = {
            "observed_nmi": macro(observations),
            "permutation_null_nmi": macro(null_means),
            "excess_over_permutation": macro(excess),
        }

    novelty_records = []
    attribution = defaultdict(list)
    for topology_run in topology_runs:
        seed = int(topology_run["sample_seed"])
        scope_run = scope_by_seed[seed]
        novelty_by_k = {row["experience_count"]: row for row in topology_run["sequence_novelty"]}
        topology_transfer = {
            (row["source_index"], row["target_index"]): row
            for row in topology_run["transfer_rows"] if abs(row["step_norm"] - 0.72) < 1e-12
        }
        for count in range(2, 13):
            before, after = scope_rows(scope_run, count - 1, 0.72), scope_rows(scope_run, count, 0.72)
            holdout = scope_run["groups"]["same_skill_holdout"]
            protection = scope_run["groups"]["protection_test"]
            previous_holdout = np.mean([before[index]["expected_value_delta"] for index in holdout])
            current_holdout = np.mean([after[index]["expected_value_delta"] for index in holdout])
            previous_positive = np.mean([before[index]["expected_value_delta"] > 0.01 for index in holdout])
            current_positive = np.mean([after[index]["expected_value_delta"] > 0.01 for index in holdout])
            previous_harm = np.mean([before[index]["expected_value_delta"] < -0.01 for index in protection])
            current_harm = np.mean([after[index]["expected_value_delta"] < -0.01 for index in protection])
            novelty_records.append({
                "sample_seed": seed,
                **novelty_by_k[count],
                "marginal_holdout_value_gain": float(current_holdout - previous_holdout),
                "positive_holdout_coverage_change": float(current_positive - previous_positive),
                "protection_harm_coverage_change": float(current_harm - previous_harm),
            })
            new_source = novelty_by_k[count]["new_source_index"]
            for index in before:
                transition = f"{label(before[index]['expected_value_delta'])}->{label(after[index]['expected_value_delta'])}"
                component = topology_transfer[new_source, index]
                attribution[transition].append({
                    "gradient_cosine": component["gradient_cosine"],
                    "individual_value_delta": component["expected_value_delta"],
                    "aggregate_value_change": (
                        after[index]["expected_value_delta"] - before[index]["expected_value_delta"]
                    ),
                })

    novelty_summary = {
        "transition_count": len(novelty_records),
        "correlations": {
            novelty_key: {
                outcome: {
                    "pearson": correlation(
                        [row[novelty_key] for row in novelty_records],
                        [row[outcome] for row in novelty_records],
                    ),
                    "spearman": correlation(
                        [row[novelty_key] for row in novelty_records],
                        [row[outcome] for row in novelty_records], rank=True,
                    ),
                }
                for outcome in (
                    "marginal_holdout_value_gain", "positive_holdout_coverage_change",
                    "protection_harm_coverage_change",
                )
            }
            for novelty_key in ("span_residual_novelty", "cosine_novelty")
        },
        "records": novelty_records,
    }
    later_records = [row for row in novelty_records if row["experience_count"] >= 4]
    median_novelty = float(np.median([row["span_residual_novelty"] for row in later_records]))
    for label_name, selected in (
        ("low_novelty", [row for row in later_records if row["span_residual_novelty"] <= median_novelty]),
        ("high_novelty", [row for row in later_records if row["span_residual_novelty"] > median_novelty]),
    ):
        novelty_summary[label_name] = {
            "count": len(selected),
            "mean_span_residual_novelty": float(np.mean([row["span_residual_novelty"] for row in selected])),
            "mean_marginal_holdout_value_gain": float(np.mean([row["marginal_holdout_value_gain"] for row in selected])),
            "positive_marginal_gain_rate": float(np.mean([row["marginal_holdout_value_gain"] > 0 for row in selected])),
        }
    attribution_summary = {
        transition: {
            "count": len(rows),
            "mean_new_gradient_cosine": float(np.mean([row["gradient_cosine"] for row in rows])),
            "mean_new_gradient_individual_value_delta": float(np.mean([row["individual_value_delta"] for row in rows])),
            "mean_aggregate_value_change": float(np.mean([row["aggregate_value_change"] for row in rows])),
        }
        for transition, rows in sorted(attribution.items())
    }

    feedback_summary = {}
    feedback_rows = [run["heldout_feedback_novelty"] for run in per_seed]
    for space in ("full_gradient", "delta_signature", "hidden_signature"):
        feedback_summary[space] = {
            key: macro([row[space][key] for row in feedback_rows])
            for key in ("span_residual_novelty", "maximum_historical_cosine", "cosine_novelty")
        }
    for key in (
        "cosine_with_historical_evolved_direction", "scope_novelty_epsilon_0.01",
        "mean_merged_feedback_effect_on_common_targets",
    ):
        feedback_summary[key] = macro([row[key] for row in feedback_rows])
    feedback_scope_breakdown = {}
    for relationship in ("unseen_family", "protection_test"):
        additions, losses, jaccards = [], [], []
        for run in topology_runs:
            seed = int(run["sample_seed"])
            unseen_run = unseen_by_seed[seed]
            before = {
                int(row["global_decision_index"]): float(row["expected_value_delta"])
                for row in unseen_run["zero_shot"]["evolved9"]["rows"]
                if row["relationship"] == relationship
            }
            individual = {
                int(row["target_index"]): float(row["expected_value_delta"])
                for row in run["heldout_feedback_novelty"]["individual_feedback_rows"]
                if row["relationship"] == relationship
            }
            common = sorted(before.keys() & individual.keys())
            before_positive = {index for index in common if before[index] > 0.01}
            individual_positive = {index for index in common if individual[index] > 0.01}
            additions.append(len(individual_positive - before_positive) / len(common))
            losses.append(len(before_positive - individual_positive) / len(common))
            union = before_positive | individual_positive
            jaccards.append(len(before_positive & individual_positive) / len(union) if union else 1.0)
        feedback_scope_breakdown[relationship] = {
            "new_positive_fraction": macro(additions),
            "lost_positive_fraction": macro(losses),
            "positive_scope_jaccard": macro(jaccards),
        }
    feedback_summary["scope_breakdown"] = feedback_scope_breakdown

    result = {
        "experiment": "gradient_scope_topology_aggregate_v1",
        "run_count": len(topology_runs),
        "close_failure_count_per_seed": len(topology_runs[0]["close_indices"]),
        "directed_transfer_pairs_per_seed": len(topology_runs[0]["close_indices"]) * (len(topology_runs[0]["close_indices"]) - 1),
        "asymmetry": asymmetry_summary,
        "effective_rank": spectrum_summary,
        "signed_topology": topology_summary,
        "cluster_correspondence": cluster_correspondence,
        "cluster_semantic_association": semantic_association,
        "novelty_vs_evolution": novelty_summary,
        "scope_transition_attribution": attribution_summary,
        "heldout_feedback_novelty": feedback_summary,
        "per_seed": per_seed,
    }
    figures = plot_results(topology_runs, result, output_dir)
    result["figures"] = figures
    (output_dir / "gradient_scope_topology.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()

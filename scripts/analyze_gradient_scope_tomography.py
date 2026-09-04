#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from self_evolve.gradient_scope import scope_profile, scope_transition
except ModuleNotFoundError as error:
    # The analysis itself is NumPy-only.  Permit a lightweight analysis host
    # without torch even though self_evolve.__init__ imports the torch runner.
    if error.name != "torch":
        raise
    module_path = Path(__file__).resolve().parents[1] / "src/self_evolve/gradient_scope.py"
    specification = importlib.util.spec_from_file_location("gradient_scope_standalone", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    gradient_scope = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(gradient_scope)
    scope_profile = gradient_scope.scope_profile
    scope_transition = gradient_scope.scope_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-glob", required=True)
    parser.add_argument("--unseen-family-glob", required=True)
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


def select_rows(
    run: dict[str, Any], count: int, step_norm: float, relationships: set[str]
) -> list[dict[str, Any]]:
    return [
        row for row in run["rows"]
        if row["experience_count"] == count
        and abs(row["step_norm"] - step_norm) < 1e-12
        and row["relationship"] in relationships
    ]


def profile_macro(
    runs: Sequence[dict[str, Any]], count: int, step_norm: float,
    relationships: set[str], epsilon: float,
) -> dict[str, Any]:
    profiles = []
    for run in runs:
        rows = select_rows(run, count, step_norm, relationships)
        profile = scope_profile([row["expected_value_delta"] for row in rows], epsilon)
        baseline = float(np.mean([row["baseline_expected_value"] for row in rows]))
        profile["baseline_expected_value"] = baseline
        profile["relative_expected_value_gain"] = profile["mean_value_delta"] / baseline
        profiles.append(profile)
    keys = [key for key in profiles[0] if key not in {"count", "epsilon"}]
    return {
        "state_count_per_run": profiles[0]["count"],
        "epsilon": epsilon,
        "per_run": profiles,
        **{key: macro([float(profile[key]) for profile in profiles]) for key in keys},
    }


def feedback_scope(unseen_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relationship in ("unseen_family", "protection_test"):
        per_run = []
        for run in unseen_runs:
            before = {
                row["global_decision_index"]: row["expected_value_delta"]
                for row in run["zero_shot"]["evolved9"]["rows"]
                if row["relationship"] == relationship
            }
            after = {
                row["global_decision_index"]: row["expected_value_delta"]
                for row in run["after_first_unseen_failure_feedback"]["rows"]
                if row["relationship"] == relationship
            }
            common = sorted(before.keys() & after.keys())
            before_values = [before[index] for index in common]
            after_values = [after[index] for index in common]
            base_rows = {
                row["global_decision_index"]: row["baseline_expected_value"]
                for row in run["zero_shot"]["evolved9"]["rows"]
                if row["relationship"] == relationship
            }
            baseline = float(np.mean([base_rows[index] for index in common]))
            per_run.append({
                "sample_seed": run["sample_seed"],
                "state_count": len(common),
                "mean_before": float(np.mean(before_values)),
                "mean_after": float(np.mean(after_values)),
                "mean_change": float(np.mean(after_values) - np.mean(before_values)),
                "relative_gain_before": float(np.mean(before_values) / baseline),
                "relative_gain_after": float(np.mean(after_values) / baseline),
                "relative_feedback_effect": float(
                    (np.mean(after_values) - np.mean(before_values)) / baseline
                ),
                "transition_epsilon_0": scope_transition(before_values, after_values, 0.0),
                "transition_epsilon_0.01": scope_transition(before_values, after_values, 0.01),
            })
        result[relationship] = {
            "per_run": per_run,
            "mean_change": macro([item["mean_change"] for item in per_run]),
            "relative_feedback_effect": macro([
                item["relative_feedback_effect"] for item in per_run
            ]),
            "positive_scope_jaccard_epsilon_0": macro([
                item["transition_epsilon_0"]["positive_scope_jaccard"] for item in per_run
            ]),
            "changed_fraction_epsilon_0.01": macro([
                item["transition_epsilon_0.01"]["changed_fraction"] for item in per_run
            ]),
        }
    return result


def plot_heatmaps(runs: Sequence[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    for run in runs:
        reference = run["selection_step_norm"]
        rows = [row for row in run["rows"] if abs(row["step_norm"] - reference) < 1e-12]
        state_order = []
        seen = set()
        for row in rows:
            index = row["global_decision_index"]
            if index not in seen:
                state_order.append((index, row["relationship"]))
                seen.add(index)
        matrix = np.asarray([
            [
                next(row["expected_value_delta"] for row in rows
                     if row["experience_count"] == count and row["global_decision_index"] == index)
                for index, _ in state_order
            ]
            for count in range(1, 13)
        ])
        vmax = float(np.quantile(np.abs(matrix), 0.98))
        fig, ax = plt.subplots(figsize=(13, 5.5))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        boundaries = []
        for position in range(1, len(state_order)):
            if state_order[position][1] != state_order[position - 1][1]:
                boundaries.append(position - 0.5)
                ax.axvline(position - 0.5, color="black", linewidth=0.7)
        ax.set_xlabel("fixed state panel (source | validation | holdout | harm-val | protection-test)")
        ax.set_ylabel("number of incorporated failures K")
        ax.set_yticks(range(12), range(1, 13))
        ax.set_title(f"Gradient scope evolution, seed={run['sample_seed']}, 1200x")
        fig.colorbar(image, ax=ax, label="expected long-term value change")
        fig.tight_layout()
        path = output_dir / f"scope_heatmap_seed_{run['sample_seed']}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path.name)
    return paths


def main() -> None:
    args = parse_args()
    runs = load(args.scope_glob)
    unseen_runs = load(args.unseen_family_glob)
    if len(runs) != len(unseen_runs):
        raise ValueError("Scope and unseen-family run counts differ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = (1, 2, 3, 4, 6, 8, 12)
    doses = tuple(float(value) for value in runs[0]["step_norms"])
    panels = {
        "independent_close": {"transfer_validation", "same_skill_holdout"},
        "same_skill_holdout": {"same_skill_holdout"},
        "protection_test": {"protection_test"},
    }

    profiles = {}
    for panel_name, relationships in panels.items():
        profiles[panel_name] = {}
        for count in counts:
            profiles[panel_name][f"K{count}"] = {
                f"{dose / 0.0006:.0f}x": profile_macro(runs, count, dose, relationships, 0.01)
                for dose in doses
            }

    predictive = {}
    for count in counts:
        predictive[f"K{count}"] = {}
        for dose in doses:
            predictive[f"K{count}"][f"{dose / 0.0006:.0f}x"] = {}
            predictive_panels = {
                **panels,
                "independent_close_plus_protection": (
                    panels["independent_close"] | panels["protection_test"]
                ),
            }
            for panel_name, relationships in predictive_panels.items():
                per_run = []
                for run in runs:
                    rows = select_rows(run, count, dose, relationships)
                    compatibility = [row["aggregate_target_gradient_dot"] for row in rows]
                    actual = [row["expected_value_delta"] for row in rows]
                    influence = [row["unit_logit_influence_l2"] for row in rows]
                    hidden = [row["weighted_mean_hidden_cosine"] for row in rows]
                    per_run.append({
                        "sample_seed": run["sample_seed"],
                        "compatibility_pearson": correlation(compatibility, actual),
                        "compatibility_spearman": correlation(compatibility, actual, rank=True),
                        "first_order_sign_accuracy": float(np.mean([
                            (left > 0) == (right > 0)
                            for left, right in zip(compatibility, actual, strict=True)
                        ])),
                        "influence_utility_spearman": correlation(influence, actual, rank=True),
                        "hidden_influence_spearman": correlation(hidden, influence, rank=True),
                        "hidden_utility_spearman": correlation(hidden, actual, rank=True),
                    })
                predictive[f"K{count}"][f"{dose / 0.0006:.0f}x"][panel_name] = {
                    "per_run": per_run,
                    **{
                        key: macro([item[key] for item in per_run])
                        for key in per_run[0] if key != "sample_seed"
                    },
                }

    dynamics = {}
    reference = runs[0]["selection_step_norm"]
    for panel_name, relationships in panels.items():
        per_run = []
        for run in runs:
            local = []
            for count in range(2, 13):
                before_rows = select_rows(run, count - 1, reference, relationships)
                after_rows = select_rows(run, count, reference, relationships)
                before = {row["global_decision_index"]: row["expected_value_delta"] for row in before_rows}
                after = {row["global_decision_index"]: row["expected_value_delta"] for row in after_rows}
                indices = sorted(before)
                stored = run["scope_transitions_at_selection_dose"]["epsilon_0.01"][panel_name]
                stored_row = next(item for item in stored if item["to_experience_count"] == count)
                local.append({
                    "from_k": count - 1,
                    "to_k": count,
                    "gradient_rotation_degrees": stored_row["gradient_rotation_degrees"],
                    **scope_transition([before[i] for i in indices], [after[i] for i in indices], 0.01),
                })
            per_run.append({"sample_seed": run["sample_seed"], "transitions": local})
        dynamics[panel_name] = {"per_run": per_run}

    long_range_dynamics = {}
    for panel_name, relationships in panels.items():
        per_run = []
        for run in runs:
            before_rows = select_rows(run, 4, reference, relationships)
            after_rows = select_rows(run, 12, reference, relationships)
            before = {row["global_decision_index"]: row["expected_value_delta"] for row in before_rows}
            after = {row["global_decision_index"]: row["expected_value_delta"] for row in after_rows}
            indices = sorted(before)
            cosine_matrix = np.asarray(run["gradient_cosine_matrix"], dtype=np.float64)
            weights = np.asarray(run["weights_by_experience_count"], dtype=np.float64)
            left, right = weights[3], weights[11]
            cosine_value = float(
                (left @ cosine_matrix @ right)
                / np.sqrt((left @ cosine_matrix @ left) * (right @ cosine_matrix @ right))
            )
            per_run.append({
                "sample_seed": run["sample_seed"],
                "gradient_rotation_degrees": float(np.degrees(np.arccos(np.clip(cosine_value, -1, 1)))),
                **scope_transition([before[i] for i in indices], [after[i] for i in indices], 0.01),
            })
        long_range_dynamics[panel_name] = {
            "per_run": per_run,
            "gradient_rotation_degrees": macro([row["gradient_rotation_degrees"] for row in per_run]),
            "changed_fraction": macro([row["changed_fraction"] for row in per_run]),
            "positive_scope_jaccard": macro([row["positive_scope_jaccard"] for row in per_run]),
        }

    conflict = {}
    for count in (4, 12):
        per_run = []
        for run in runs:
            protection_at_300 = select_rows(run, count, 0.18, panels["protection_test"])
            protection_by_index = {row["global_decision_index"]: row for row in protection_at_300}
            high = {
                row["global_decision_index"]: row
                for row in select_rows(run, count, 0.72, panels["protection_test"])
            }
            indices = sorted(protection_by_index)
            structural = [protection_by_index[i]["aggregate_target_gradient_dot"] < 0 for i in indices]
            low_harm = [protection_by_index[i]["expected_value_delta"] < -0.01 for i in indices]
            high_harm = [high[i]["expected_value_delta"] < -0.01 for i in indices]
            dose_emergent = [not low and higher for low, higher in zip(low_harm, high_harm, strict=True)]
            nonlinear_reversal = [
                high[i]["aggregate_target_gradient_dot"] > 0 and high_harm[pos]
                for pos, i in enumerate(indices)
            ]
            per_run.append({
                "sample_seed": run["sample_seed"],
                "first_order_conflict_fraction": float(np.mean(structural)),
                "empirical_harm_300x_fraction": float(np.mean(low_harm)),
                "empirical_harm_1200x_fraction": float(np.mean(high_harm)),
                "dose_emergent_harm_fraction": float(np.mean(dose_emergent)),
                "nonlinear_reversal_at_1200x_fraction": float(np.mean(nonlinear_reversal)),
            })
        conflict[f"K{count}"] = {
            "per_run": per_run,
            **{key: macro([item[key] for item in per_run]) for key in per_run[0] if key != "sample_seed"},
        }

    result = {
        "experiment": "gradient_scope_tomography_aggregate_v1",
        "run_count": len(runs),
        "sample_seeds": [run["sample_seed"] for run in runs],
        "state_observations_per_dose": sum(run["panel_state_count"] for run in runs),
        "formula_boundary": runs[0]["formula_boundary"],
        "profiles_epsilon_0.01": profiles,
        "predictive_relationships": predictive,
        "scope_dynamics_at_1200x": dynamics,
        "long_range_scope_dynamics_K4_to_K12": long_range_dynamics,
        "structural_vs_dose_conflict": conflict,
        "heldout_family_feedback_scope": feedback_scope(unseen_runs),
    }
    figures = plot_heatmaps(runs, output_dir)
    result["figures"] = figures
    (output_dir / "gradient_scope_tomography.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def metric(panel: str, count: int, dose: int, key: str) -> float:
        return result["profiles_epsilon_0.01"][panel][f"K{count}"][f"{dose}x"][key]["mean"]

    def prediction(count: int, dose: int, panel: str, key: str) -> float:
        return result["predictive_relationships"][f"K{count}"][f"{dose}x"][panel][key]["mean"]

    feedback = result["heldout_family_feedback_scope"]["unseen_family"]
    conflict_k4 = result["structural_vs_dose_conflict"]["K4"]
    conflict_k12 = result["structural_vs_dose_conflict"]["K12"]
    long_close = result["long_range_scope_dynamics_K4_to_K12"]["independent_close"]
    long_protection = result["long_range_scope_dynamics_K4_to_K12"]["protection_test"]

    lines = [
        "# Gradient Scope Tomography：输出梯度作用域如何进化",
        "",
        "**日期：2026-09-04**",
        "",
        "## 核心边界",
        "",
        "精确成立的是 frozen-backbone、untied output head 下的 logit 迁移定律。",
        "长期价值变化经过 softmax、multi-token action score 和动作价值映射，属于经验响应，",
        "不能由 hidden similarity 单独推出。",
        "",
        "## 实验规模",
        "",
        f"- {len(runs)} 个随机 seed；每个 seed 固定 30 状态（12 source、3 validation、3 independent holdout、3 harm-validation、9 independent protection）。",
        f"- 每个方向扫描 K=1..12 和 300x/600x/900x/1200x/3000x，共 {len(runs)*12*5*30:,} 个 state × gradient 条件。",
        "- 梯度方向的 fitness 固定在 1200x validation 上计算；所有剂量共享同一个方向，因而 direction 与 magnitude 被分离。",
        "- 作用域阈值使用绝对长期价值变化 ε=0.01，避免把浮点级波动算成有效 scope。",
        "",
        "## 1. 公式验证与解释边界",
        "",
        "对 output head 的精确式为：",
        "",
        "$$\\Delta z_{j,t}=-\\eta\\sum_r\\delta_{i,r}(h_{i,r}^{\\top}h_{j,t}).$$",
        "",
        "它说明参数更新在 target logits 上产生什么变化；此前真实 writeback 与 virtual repair 的最大",
        "action-score 误差为 $1.07\\times10^{-4}$，所以实现层面的等价关系成立。",
        "",
        "但 $\\Delta V_j$ 还经过 softmax、sequence aggregation、action ranking 和环境价值映射。",
        "因此 hidden overlap 只参与 influence，不能单独决定 utility 的符号；utility 更直接的一阶量是",
        "$\\langle G_t,g_j\\rangle$。以下结果全部把这两层命题分开报告。",
        "",
        "## 2. K 轴：作用域主要在前三条经验形成",
        "",
        "| K | holdout 相对价值 | 独立 close 正覆盖 | protection 负覆盖 | protection 平均 ΔV |",
        "|---:|---:|---:|---:|---:|",
    ]
    for count in (1, 2, 3, 4, 6, 8, 12):
        lines.append(
            f"| {count} | {100*metric('same_skill_holdout',count,1200,'relative_expected_value_gain'):+.2f}% "
            f"| {100*metric('independent_close',count,1200,'positive_coverage'):.1f}% "
            f"| {100*metric('protection_test',count,1200,'negative_coverage'):.1f}% "
            f"| {metric('protection_test',count,1200,'mean_value_delta'):+.4f} |"
        )
    lines.extend([
        "",
        "从 K=1 到 K=2，梯度平均旋转 59.39°，独立 close 中 29.2% 的状态跨越 ε=0.01 的",
        "scope 边界；K=2 到 K=3 又旋转 39.74°，但只有 4.2% close 状态换区。K≥3 后",
        "close 的符号作用域几乎固定，收益主要表现为强度变化，而不是继续扩大 coverage。",
        "",
        f"G4 到 G12 仍累计旋转 {long_close['gradient_rotation_degrees']['mean']:.2f}°，但独立 close "
        f"正作用域 Jaccard 为 {long_close['positive_scope_jaccard']['mean']:.3f}，只有 "
        f"{100*long_close['changed_fraction']['mean']:.1f}% 状态换区；protection 的 ε=0.01 正作用域 "
        f"Jaccard 为 {long_protection['positive_scope_jaccard']['mean']:.3f}。这说明 K=4→12 的"
        "主要现象不是 scope 持续扩张，而是已形成作用域内的增益饱和和强度重分配。",
        "",
        "## 3. 剂量轴：harm 主要是结构冲突，剂量只进一步泄漏",
        "",
        "| 固定方向 | 剂量 | close 正覆盖 | protection 负覆盖 | protection 平均 ΔV |",
        "|---|---:|---:|---:|---:|",
    ])
    for count in (4, 12):
        for dose in (300, 600, 900, 1200, 3000):
            lines.append(
                f"| G{count} | {dose}x | {100*metric('independent_close',count,dose,'positive_coverage'):.1f}% "
                f"| {100*metric('protection_test',count,dose,'negative_coverage'):.1f}% "
                f"| {metric('protection_test',count,dose,'mean_value_delta'):+.4f} |"
            )
    lines.extend([
        "",
        f"G4/G12 在 300x 已各有 {100*conflict_k4['empirical_harm_300x_fraction']['mean']:.1f}% / "
        f"{100*conflict_k12['empirical_harm_300x_fraction']['mean']:.1f}% protection 状态受到超过 0.01 的伤害；"
        f"一阶 compatibility 为负的比例更达到 {100*conflict_k4['first_order_conflict_fraction']['mean']:.1f}% / "
        f"{100*conflict_k12['first_order_conflict_fraction']['mean']:.1f}%。",
        "",
        f"从 300x 增至 1200x 后，新增的 dose-emergent harm 只占 {100*conflict_k4['dose_emergent_harm_fraction']['mean']:.1f}% "
        f"（G4）和 {100*conflict_k12['dose_emergent_harm_fraction']['mean']:.1f}%（G12）；"
        f"一阶预测为正却在 1200x 真实转负的 nonlinear reversal 仅 {100*conflict_k4['nonlinear_reversal_at_1200x_fraction']['mean']:.1f}%。",
        "",
        "因此当前 40%–50% harm 不能主要归因于“方向正确但剂量太大”。多数冲突在一阶方向上已经存在，",
        "属于 structural repair conflict；单纯缩步长会减弱伤害，却不会消除冲突状态。",
        "",
        "## 4. 一阶 compatibility 比 influence 更能预测 utility",
        "",
        "| 方向 | 剂量 | compatibility→ΔV Spearman | 符号准确率 | influence→ΔV Spearman |",
        "|---|---:|---:|---:|---:|",
    ])
    for count in (1, 4, 12):
        for dose in (300, 1200, 3000):
            panel_name = "independent_close_plus_protection"
            lines.append(
                f"| G{count} | {dose}x | {prediction(count,dose,panel_name,'compatibility_spearman'):.3f} "
                f"| {100*prediction(count,dose,panel_name,'first_order_sign_accuracy'):.1f}% "
                f"| {prediction(count,dose,panel_name,'influence_utility_spearman'):.3f} |"
            )
    lines.extend([
        "",
        "G4/G12 在 300x 时 $\\langle G,g_j\\rangle$ 与真实 ΔV 的 Spearman 均约 0.95；",
        "到 1200x 仍约 0.90，到 3000x 降至约 0.86。也就是说，一阶 validity scope 在强更新下仍",
        "有很强预测力，但随非线性增强而有序退化。相比之下，logit influence norm 对 utility 的相关性",
        "只有约 0.41–0.47，证明“作用得强”不等于“作用得对”。",
        "",
        f"在独立 close 内，G4 的 weighted mean-hidden cosine→influence Spearman 只有 "
        f"{prediction(4,1200,'independent_close','hidden_influence_spearman'):.3f}；在 protection 内为 "
        f"{prediction(4,1200,'protection_test','hidden_influence_spearman'):.3f}。这个 mean-hidden 标量本身"
        "也不是 multi-token 精确 kernel；它不能替代完整 token-pair contraction，更不能直接当作 validity 判据。",
        "",
        "## 5. 新 family 第一次 feedback：旧比较没有证明继续进化",
        "",
        "此前 `+49.64% → +51.53%` 比较了不同状态集合：feedback state 在后一个数字里被移除。",
        "现在只比较两次评测共有的 4 个 unseen states/seed。",
        "",
        f"共有状态上的绝对反馈效应为 {feedback['mean_change']['mean']:+.6f} ± "
        f"{feedback['mean_change']['sample_std']:.6f}，相对反馈效应为 "
        f"{100*feedback['relative_feedback_effect']['mean']:+.2f}% ± "
        f"{100*feedback['relative_feedback_effect']['sample_std']:.2f}%。4 个 seed 中仅 1 个为正；"
        "正作用域 Jaccard 为 1.0。",
        "",
        "所以严格结论是：historical gradient 的 zero-shot transfer 仍然很强，但第一次新-family failure",
        "没有在共同测试状态上产生可复现的额外净收益，也没有扩张正作用域。`feedback → evolve` 这一环"
        "目前未被证明，必须撤回先前基于不等样本集合的增强表述。",
        "",
        "## 6. 当前 Gradient Scope 结论",
        "",
        "1. logit transfer law 是精确的；behavioral validity scope 是经验对象，两者不能混称。",
        "2. 多源 consolidation 的主要作用域形成发生在前 2–3 条 failure；K=4 后主要是饱和，不是持续扩张。",
        "3. protection harm 以结构性负 compatibility 为主，dose-induced leakage 为辅。",
        "4. $\\langle G_t,g_j\\rangle$ 是当前最强的 scope predictor；hidden similarity 或 influence magnitude 单独不足。",
        "5. 新-family zero-shot repair 成立，但一次 feedback 并未让共同 holdout 继续改善，因此完整自进化闭环尚未成立。",
        "6. 这些结果支持下一步研究 gradient branching / repair-direction decomposition；目前仍不应预设轻量 gate 是答案。",
        "",
        "## 图",
        "",
    ])
    lines.extend(f"![{name}]({name})" for name in figures)
    lines.extend([
        "",
        "## 机器可读结果",
        "",
        "完整逐状态矩阵、相关性、scope transition、first-order/empirical 一致性和 held-out family feedback",
        "见 `gradient_scope_tomography.json` 以及每个 seed 的 `.npz`。",
        "",
    ])
    (output_dir / "gradient_scope_tomography.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()

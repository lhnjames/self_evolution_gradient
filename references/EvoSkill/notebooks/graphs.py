# %%
from src.evaluation import score_answer
import pickle as pkl
import pandas as pd
import re
from pathlib import Path


def extract_answer_from_result(result: str) -> str | None:
    """Extract final answer from result text when structured output fails."""
    if not result:
        return None
    # Look for bolded number near the end - common pattern in agent responses
    # Patterns: **$1,234**, **$1,234 million**, **1,234**, **$1,234.56**
    last_2000 = result[-2000:]
    # Try to find bolded values
    bold_matches = re.findall(
        r"\*\*[\$]?\s*([0-9,.]+)\s*[%]?\s*(?:million|billion|dollars)?\s*\*\*",
        last_2000,
    )
    if bold_matches:
        return bold_matches[-1]  # Take the last bolded number
    return None


def get_predicted_answer(example) -> str | None:
    """Get predicted answer from example, handling parse errors."""
    if example.trace is None:
        return None
    if example.trace.output and example.trace.output.final_answer:
        return example.trace.output.final_answer
    if example.trace.result:
        return extract_answer_from_result(example.trace.result)
    return None


# Use eval_results.pkl as the single source
results_path = Path("../results/eval_results.pkl")
if not results_path.exists():
    results_path = Path("results/eval_results.pkl")

with open(results_path, "rb") as f:
    all_data = pkl.load(f)

# For now, use the same data for both (no evolved comparison)
evolved_data = all_data
original_data = all_data

# %%
evolved_train_set = {
    "question": [],
    "index": [],
    "predicted": [],
    "ground_truth": [],
    "score_0.05": [],
    "score_0.01": [],
    "score_0.1": [],
    "score_0.0": [],
    "score_0.025": [],
    "score_0.001": [],  # ADD THIS LINE
    "final_score": [],
    "output_tokens": [],
    "cost_usd": [],
    "time": [],
}

original_train_set = {
    "question": [],
    "index": [],
    "predicted": [],
    "ground_truth": [],
    "score_0.05": [],
    "score_0.01": [],
    "score_0.1": [],
    "score_0.0": [],
    "score_0.025": [],
    "score_0.001": [],  # ADD THIS LINE
    "final_score": [],
    "output_tokens": [],
    "cost_usd": [],
    "time": [],
}

# %%
for set_index, example in enumerate(evolved_data):
    try:
        question = example.question
        index = example.index
        ground_truth = example.ground_truth
        predicted = get_predicted_answer(example)
        if predicted is None:
            print(f"Could not extract answer for example {index}")
            continue
        final_score = 0.0
        for tolerance in [0.05, 0.01, 0.1, 0.0, 0.025, 0.001]:
            score = score_answer(
                ground_truth=ground_truth, predicted=predicted, tolerance=tolerance
            )
            evolved_train_set[f"score_{tolerance}"].append(score)
            final_score += score
        final_score /= 6.0
        evolved_train_set["final_score"].append(final_score)
        evolved_train_set["question"].append(question)
        evolved_train_set["index"].append(index)
        evolved_train_set["ground_truth"].append(ground_truth)
        evolved_train_set["predicted"].append(predicted)
        evolved_train_set["output_tokens"].append(example.trace.usage["output_tokens"])
        evolved_train_set["cost_usd"].append(example.trace.total_cost_usd)
        evolved_train_set["time"].append(example.trace.duration_ms)
    except Exception as e:
        print(f"Error processing example {index}: {e}")
        print(f"true index: {set_index}")
        continue

# %%
for set_index, example in enumerate(original_data):
    try:
        question = example.question
        index = example.index
        ground_truth = example.ground_truth
        predicted = get_predicted_answer(example)
        if predicted is None:
            print(f"Could not extract answer for example {index}")
            continue
        final_score = 0.0
        for tolerance in [0.05, 0.01, 0.1, 0.0, 0.025, 0.001]:
            score = score_answer(
                ground_truth=ground_truth, predicted=predicted, tolerance=tolerance
            )
            original_train_set[f"score_{tolerance}"].append(score)
            final_score += score
        final_score /= 6.0
        original_train_set["final_score"].append(final_score)
        original_train_set["question"].append(question)
        original_train_set["index"].append(index)
        original_train_set["ground_truth"].append(ground_truth)
        original_train_set["predicted"].append(predicted)
        original_train_set["output_tokens"].append(example.trace.usage["output_tokens"])
        original_train_set["cost_usd"].append(example.trace.total_cost_usd)
        original_train_set["time"].append(example.trace.duration_ms)
    except Exception as e:
        print(f"Error processing example {index}: {e}")
        print(f"true index: {set_index}")
        continue

# %%
import plotly.graph_objects as go
import numpy as np

# Define tolerance levels in REVERSED order (10% to 0%)
tolerance_levels = [0.1, 0.05, 0.025, 0.01, 0.001, 0.0]  # ADD 0.001
tolerance_labels = ["10%", "5%", "2.5%", "1%", "0.1%", "0% (Exact)"]  # ADD "0.1%"

# Calculate average scores for each tolerance level
original_avg_scores = []
evolved_avg_scores = []
deltas = []
delta_labels = []

for tolerance in tolerance_levels:
    score_key = f"score_{tolerance}"

    # Calculate average for original_train_set
    if len(original_train_set[score_key]) > 0:
        original_avg = np.mean(original_train_set[score_key])
    else:
        original_avg = 0.0
    original_avg_scores.append(original_avg)

    # Calculate average for evolved_train_set
    if len(evolved_train_set[score_key]) > 0:
        evolved_avg = np.mean(evolved_train_set[score_key])
    else:
        evolved_avg = 0.0
    evolved_avg_scores.append(evolved_avg)

    # Calculate delta (difference) as percentage
    delta = (evolved_avg - original_avg) * 100
    deltas.append(delta)

    # Format delta with + or - sign
    if delta >= 0:
        delta_labels.append(f"+{delta:.1f}%")
    else:
        delta_labels.append(f"{delta:.1f}%")  # Negative already has - sign

# Convert scores to percentages
original_avg_scores_pct = [score * 100 for score in original_avg_scores]
evolved_avg_scores_pct = [score * 100 for score in evolved_avg_scores]

# %%
# Create the plot
fig = go.Figure()

# Add evolved line FIRST (so it appears above original)
fig.add_trace(
    go.Scatter(
        x=tolerance_labels,
        y=evolved_avg_scores_pct,
        mode="lines+markers",
        name="Evolved",
        line=dict(color="#2E86AB", width=3, shape="spline"),
        marker=dict(size=10, symbol="circle", line=dict(width=2, color="white")),
        hovertemplate="<b>Evolved</b><br>Tolerance: %{x}<br>Score: %{y:.1f}%<br>Delta: <b>%{customdata}</b><extra></extra>",
        customdata=delta_labels,
    )
)

# Add original line SECOND (so it appears below evolved)
fig.add_trace(
    go.Scatter(
        x=tolerance_labels,
        y=original_avg_scores_pct,
        mode="lines+markers",
        name="Original",
        line=dict(color="#A23B72", width=3, shape="spline"),
        marker=dict(size=10, symbol="diamond", line=dict(width=2, color="white")),
        hovertemplate="<b>Original</b><br>Tolerance: %{x}<br>Score: %{y:.1f}%<br>Delta: <b>%{customdata}</b><extra></extra>",
        customdata=delta_labels,
    )
)

# Add annotations showing delta values above each point
for i, (label, delta_label, evolved_score, original_score) in enumerate(
    zip(tolerance_labels, delta_labels, evolved_avg_scores_pct, original_avg_scores_pct)
):
    # Position annotation above the higher of the two points
    y_pos = max(evolved_score, original_score) + 2  # Add some spacing

    fig.add_annotation(
        x=label,
        y=y_pos,
        text=f"<b>{delta_label}</b>",
        showarrow=False,
        font=dict(size=11, color="#2C3E50"),
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="#BDC3C7",
        borderwidth=1,
        borderpad=4,
        xanchor="center",
        yanchor="bottom",
    )

# Update layout with prettier styling
fig.update_layout(
    title=dict(
        text="Average Score Comparison: Original vs Evolved",
        font=dict(size=20, color="#2C3E50", family="Arial Black"),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title=dict(text="Tolerance Level", font=dict(size=14, color="#34495E")),
        tickfont=dict(size=12, color="#7F8C8D"),
        gridcolor="#ECF0F1",
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor="#BDC3C7",
    ),
    yaxis=dict(
        title=dict(text="Average Score (%)", font=dict(size=14, color="#34495E")),
        tickfont=dict(size=12, color="#7F8C8D"),
        range=[50, 100],
        gridcolor="#ECF0F1",
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor="#BDC3C7",
        tickformat=".0f",
        ticksuffix="%",
    ),
    hovermode="x unified",
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#BDC3C7",
        borderwidth=1,
        font=dict(size=12),
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=900,
    height=600,
)

fig.show()

# %%
# Calculate average cost and output tokens
original_avg_cost = (
    np.mean(original_train_set["cost_usd"])
    if len(original_train_set["cost_usd"]) > 0
    else 0.0
)
evolved_avg_cost = (
    np.mean(evolved_train_set["cost_usd"])
    if len(evolved_train_set["cost_usd"]) > 0
    else 0.0
)
cost_delta = evolved_avg_cost - original_avg_cost
cost_delta_pct = (
    ((evolved_avg_cost - original_avg_cost) / original_avg_cost * 100)
    if original_avg_cost > 0
    else 0.0
)

original_avg_tokens = (
    np.mean(original_train_set["output_tokens"])
    if len(original_train_set["output_tokens"]) > 0
    else 0.0
)
evolved_avg_tokens = (
    np.mean(evolved_train_set["output_tokens"])
    if len(evolved_train_set["output_tokens"]) > 0
    else 0.0
)
tokens_delta = evolved_avg_tokens - original_avg_tokens
tokens_delta_pct = (
    ((evolved_avg_tokens - original_avg_tokens) / original_avg_tokens * 100)
    if original_avg_tokens > 0
    else 0.0
)

# Format deltas with signs
cost_delta_label = f"+${cost_delta:.4f}" if cost_delta >= 0 else f"${cost_delta:.4f}"
cost_delta_pct_label = (
    f"+{cost_delta_pct:.1f}%" if cost_delta_pct >= 0 else f"{cost_delta_pct:.1f}%"
)

tokens_delta_label = (
    f"+{tokens_delta:.0f}" if tokens_delta >= 0 else f"{tokens_delta:.0f}"
)
tokens_delta_pct_label = (
    f"+{tokens_delta_pct:.1f}%" if tokens_delta_pct >= 0 else f"{tokens_delta_pct:.1f}%"
)

print(f"Original Avg Cost: ${original_avg_cost:.4f}")
print(f"Evolved Avg Cost: ${evolved_avg_cost:.4f}")
print(f"Cost Delta: {cost_delta_label} ({cost_delta_pct_label})")
print(f"\nOriginal Avg Output Tokens: {original_avg_tokens:.0f}")
print(f"Evolved Avg Output Tokens: {evolved_avg_tokens:.0f}")
print(f"Tokens Delta: {tokens_delta_label} ({tokens_delta_pct_label})")

# %%
# Calculate average time (in milliseconds)
original_avg_time_ms = (
    np.mean(original_train_set["time"]) if len(original_train_set["time"]) > 0 else 0.0
)
evolved_avg_time_ms = (
    np.mean(evolved_train_set["time"]) if len(evolved_train_set["time"]) > 0 else 0.0
)
time_delta_ms = evolved_avg_time_ms - original_avg_time_ms
time_delta_pct = (
    ((evolved_avg_time_ms - original_avg_time_ms) / original_avg_time_ms * 100)
    if original_avg_time_ms > 0
    else 0.0
)

# Convert to seconds for display
original_avg_time_s = original_avg_time_ms / 1000
evolved_avg_time_s = evolved_avg_time_ms / 1000
time_delta_s = time_delta_ms / 1000

# Format deltas with signs
time_delta_label = (
    f"+{time_delta_s:.2f}s" if time_delta_s >= 0 else f"{time_delta_s:.2f}s"
)
time_delta_pct_label = (
    f"+{time_delta_pct:.1f}%" if time_delta_pct >= 0 else f"{time_delta_pct:.1f}%"
)

print(f"Original Avg Time: {original_avg_time_s:.2f}s ({original_avg_time_ms:.0f}ms)")
print(f"Evolved Avg Time: {evolved_avg_time_s:.2f}s ({evolved_avg_time_ms:.0f}ms)")
print(f"Time Delta: {time_delta_label} ({time_delta_pct_label})")

# %%

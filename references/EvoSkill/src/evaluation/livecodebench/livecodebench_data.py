"""LiveCodeBench dataset loading utilities."""

import json
from pathlib import Path

import pandas as pd


def ensure_livecodebench_dataset(
    dataset_path: str = ".dataset/livecodebench_v6.csv",
) -> Path:
    """Ensure LiveCodeBench v6 dataset exists, downloading if necessary.

    Args:
        dataset_path: Path where the CSV should be stored

    Returns:
        Path to the dataset CSV file
    """
    from .livecodebench_format import format_livecodebench_question

    path = Path(dataset_path)

    if path.exists():
        return path

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    print(f"Downloading LiveCodeBench v6 dataset to {dataset_path}...")

    from huggingface_hub import hf_hub_download

    # Download test6.jsonl (v6)
    jsonl_path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        repo_type="dataset",
        filename="test6.jsonl",
        local_dir=path.parent,
    )

    # Convert JSONL to CSV
    print("Converting to CSV format...")
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Flatten nested fields into JSON strings
    df["public_test_cases"] = df["public_test_cases"].apply(json.dumps)
    df["private_test_cases"] = df["private_test_cases"].apply(str)
    df["metadata"] = df["metadata"].apply(json.dumps)

    # Add formatted question column
    df["formatted_question"] = df.apply(
        lambda row: format_livecodebench_question(
            row["question_content"], row["starter_code"]
        ),
        axis=1,
    )

    # Save as CSV
    df.to_csv(path, index=False)
    print(f"✓ Dataset ready: {len(df)} problems")

    return path

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/quality_metrics_patch.csv"
OUTPUT = ROOT / "results/statistics/rq1_degradation_profile.csv"
PATCH_OUTPUT = ROOT / "data/processed/quality_degradation_patch.csv"

METRICS = {
    "cc_mean": "up",
    "cc_max": "up",
    "mi": "down",
    "halstead_volume": "up",
    "cognitive_mean": "up",
    "cognitive_max": "up",
    "max_nesting": "up",
    "function_length_mean": "up",
    "function_length_max": "up",
    "ruff_findings": "up",
}

df = pd.read_csv(INPUT)

for metric, direction in METRICS.items():
    col = f"delta_{metric}"

    if direction == "up":
        df[f"adverse_{metric}"] = (df[col] > 0).astype(int)
    else:
        df[f"adverse_{metric}"] = (df[col] < 0).astype(int)

adverse_cols = [f"adverse_{m}" for m in METRICS]

df["adverse_dimension_count"] = df[adverse_cols].sum(axis=1)

# Keep this descriptive, not as an arbitrary binary risk label.
df.to_csv(PATCH_OUTPUT, index=False)

counts = (
    df["adverse_dimension_count"]
    .value_counts()
    .sort_index()
    .rename_axis("adverse_dimensions")
    .reset_index(name="patches")
)

counts["percentage"] = 100 * counts["patches"] / len(df)
counts["cumulative_at_least"] = [
    100 * (df["adverse_dimension_count"] >= k).mean()
    for k in counts["adverse_dimensions"]
]

counts.to_csv(OUTPUT, index=False)

print("=== ADVERSE QUALITY DIMENSIONS PER PATCH ===")
print(counts.to_string(index=False))

print("\n=== SUMMARY ===")
print("Patches:", len(df))
print(
    "Mean adverse dimensions:",
    round(df["adverse_dimension_count"].mean(), 3)
)
print(
    "Median adverse dimensions:",
    df["adverse_dimension_count"].median()
)

for k in range(1, 6):
    n = int((df["adverse_dimension_count"] >= k).sum())
    pct = 100 * n / len(df)
    print(f">= {k} adverse dimensions: {n}/359 ({pct:.2f}%)")

print("\n=== EXCLUDING DJANGO ===")

nd = df[df["repo"] != "django/django"]

print("Patches:", len(nd))
print(
    "Mean adverse dimensions:",
    round(nd["adverse_dimension_count"].mean(), 3)
)
print(
    "Median adverse dimensions:",
    nd["adverse_dimension_count"].median()
)

for k in range(1, 6):
    n = int((nd["adverse_dimension_count"] >= k).sum())
    pct = 100 * n / len(nd)
    print(f">= {k} adverse dimensions: {n}/{len(nd)} ({pct:.2f}%)")

print("\nSaved:", PATCH_OUTPUT)
print("Saved:", OUTPUT)

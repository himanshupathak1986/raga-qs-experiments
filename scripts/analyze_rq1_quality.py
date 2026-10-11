from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/quality_metrics_final.csv"
PATCH_OUT = ROOT / "data/processed/quality_metrics_patch.csv"
STATS_OUT = ROOT / "results/statistics/rq1_quality_patch_tests.csv"

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

BOOTSTRAPS = 10000
SEED = 42


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        out[idx] = running

    return out


def wilson_ci(x, n):
    if n == 0:
        return np.nan, np.nan

    z = 1.959963984540054
    phat = x / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * np.sqrt(
        phat*(1-phat)/n + z*z/(4*n*n)
    ) / denom

    return center - half, center + half


def rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[values != 0]

    if len(values) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(values))
    w_plus = ranks[values > 0].sum()
    w_minus = ranks[values < 0].sum()

    return (w_plus - w_minus) / (w_plus + w_minus)


df = pd.read_csv(INPUT)
df = df[(df["status"] == "ok") & (df["resolved"] == True)].copy()

# One row per SWE-bench patch.
patch = (
    df.groupby(["instance_id", "repo"], as_index=False)
      [[f"delta_{m}" for m in METRICS]]
      .mean()
)

PATCH_OUT.parent.mkdir(parents=True, exist_ok=True)
patch.to_csv(PATCH_OUT, index=False)

print("File-level rows:", len(df))
print("Patch-level rows:", len(patch))
print("Unique patches:", patch["instance_id"].nunique())

assert patch["instance_id"].nunique() == 359

rng = np.random.default_rng(SEED)
rows = []

for metric, adverse_direction in METRICS.items():
    vals = patch[f"delta_{metric}"].dropna().to_numpy(dtype=float)

    n = len(vals)
    increased = int((vals > 0).sum())
    decreased = int((vals < 0).sum())
    unchanged = int((vals == 0).sum())

    adverse = decreased if adverse_direction == "down" else increased
    adverse_pct = 100 * adverse / n

    ci_prop_low, ci_prop_high = wilson_ci(adverse, n)

    if np.allclose(vals, 0):
        stat = 0.0
        p = 1.0
    else:
        test = stats.wilcoxon(
            vals,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        stat = float(test.statistic)
        p = float(test.pvalue)

    rbc = rank_biserial(vals)

    boot_means = rng.choice(
        vals,
        size=(BOOTSTRAPS, n),
        replace=True
    ).mean(axis=1)

    mean_low, mean_high = np.quantile(
        boot_means, [0.025, 0.975]
    )

    rows.append({
        "metric": metric,
        "n_patches": n,
        "median_patch_delta": np.median(vals),
        "mean_patch_delta": np.mean(vals),
        "mean_delta_ci95_low": mean_low,
        "mean_delta_ci95_high": mean_high,
        "patches_increased": increased,
        "patches_decreased": decreased,
        "patches_unchanged": unchanged,
        "adverse_direction": adverse_direction,
        "adverse_patches": adverse,
        "adverse_pct": adverse_pct,
        "adverse_pct_ci95_low": 100 * ci_prop_low,
        "adverse_pct_ci95_high": 100 * ci_prop_high,
        "wilcoxon_statistic": stat,
        "p_value": p,
        "rank_biserial": rbc,
        "adverse_effect_size": -rbc if adverse_direction == "down" else rbc,
    })

result = pd.DataFrame(rows)
result["p_holm"] = holm_adjust(result["p_value"].to_numpy())

STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(STATS_OUT, index=False)

print("\n=== RQ1 PATCH-LEVEL RESULTS ===")
print(
    result[
        [
            "metric",
            "n_patches",
            "median_patch_delta",
            "mean_patch_delta",
            "adverse_patches",
            "adverse_pct",
            "rank_biserial",
            "p_value",
            "p_holm",
        ]
    ].to_string(index=False)
)

print("\nSaved:", PATCH_OUT)
print("Saved:", STATS_OUT)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

QUALITY = ROOT / "data/processed/quality_degradation_patch.csv"
PATCH = ROOT / "data/processed/patch_metrics.csv"
OUT = ROOT / "results/statistics/rq3_predictor_associations.csv"

PREDICTORS = [
    "churn",
    "files_changed",
    "hunks",
    "imports_added",
    "control_token_delta",
    "security_sensitive_api_additions",
]

BOOTSTRAPS = 5000
SEED = 42


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running

    return adjusted


def bootstrap_spearman(x, y, iterations, rng):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    vals = []

    for _ in range(iterations):
        idx = rng.integers(0, len(x), len(x))

        xb = x[idx]
        yb = y[idx]

        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue

        rho = stats.spearmanr(xb, yb).statistic
        if np.isfinite(rho):
            vals.append(rho)

    if not vals:
        return np.nan, np.nan

    return np.quantile(vals, [0.025, 0.975])


quality = pd.read_csv(QUALITY)
patch = pd.read_csv(PATCH)

# Keep only functionally successful patches.
patch = patch[patch["resolved"] == True].copy()

df = quality.merge(
    patch[
        ["instance_id"] + PREDICTORS
    ],
    on="instance_id",
    how="inner",
)

print("Merged patches:", len(df))
print("Unique patches:", df["instance_id"].nunique())

assert df["instance_id"].nunique() == 359

rng = np.random.default_rng(SEED)
rows = []

for predictor in PREDICTORS:
    d = df[
        [predictor, "adverse_dimension_count"]
    ].dropna()

    x = d[predictor].to_numpy(dtype=float)
    y = d["adverse_dimension_count"].to_numpy(dtype=float)

    result = stats.spearmanr(x, y)

    ci_low, ci_high = bootstrap_spearman(
        x,
        y,
        BOOTSTRAPS,
        rng,
    )

    rows.append({
        "sample": "all",
        "predictor": predictor,
        "n": len(d),
        "spearman_rho": result.statistic,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": result.pvalue,
    })

results = pd.DataFrame(rows)
results["p_holm"] = holm_adjust(results["p_value"])

# --------------------------------------------------
# Sensitivity analysis excluding Django
# --------------------------------------------------

nd = df[df["repo"] != "django/django"].copy()

rows_nd = []

for predictor in PREDICTORS:
    d = nd[
        [predictor, "adverse_dimension_count"]
    ].dropna()

    x = d[predictor].to_numpy(dtype=float)
    y = d["adverse_dimension_count"].to_numpy(dtype=float)

    result = stats.spearmanr(x, y)

    ci_low, ci_high = bootstrap_spearman(
        x,
        y,
        BOOTSTRAPS,
        rng,
    )

    rows_nd.append({
        "sample": "excluding_django",
        "predictor": predictor,
        "n": len(d),
        "spearman_rho": result.statistic,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": result.pvalue,
    })

results_nd = pd.DataFrame(rows_nd)
results_nd["p_holm"] = holm_adjust(results_nd["p_value"])

final = pd.concat(
    [results, results_nd],
    ignore_index=True
)

OUT.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUT, index=False)

print("\n=== ALL 359 SUCCESSFUL PATCHES ===")
print(
    results[
        [
            "predictor",
            "n",
            "spearman_rho",
            "ci95_low",
            "ci95_high",
            "p_value",
            "p_holm",
        ]
    ].to_string(index=False)
)

print("\n=== EXCLUDING DJANGO ===")
print(
    results_nd[
        [
            "predictor",
            "n",
            "spearman_rho",
            "ci95_low",
            "ci95_high",
            "p_value",
            "p_holm",
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUT)

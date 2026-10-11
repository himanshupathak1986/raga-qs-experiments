from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

QUALITY = ROOT / "data/processed/quality_degradation_patch.csv"
PATCH = ROOT / "data/processed/patch_metrics.csv"
OUT = ROOT / "results/statistics/rq3_overlap_sensitivity.csv"

PREDICTORS = [
    "churn",
    "files_changed",
    "hunks",
    "imports_added",
    "control_token_delta",
]

COMPLEXITY = [
    "adverse_cc_mean",
    "adverse_cc_max",
    "adverse_cognitive_mean",
    "adverse_cognitive_max",
    "adverse_max_nesting",
]

NON_COMPLEXITY = [
    "adverse_mi",
    "adverse_halstead_volume",
    "adverse_function_length_mean",
    "adverse_function_length_max",
    "adverse_ruff_findings",
]


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty_like(values)
    running = 0.0
    m = len(values)

    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * values[idx])
        running = max(running, adjusted)
        result[idx] = running

    return result


quality = pd.read_csv(QUALITY)
patch = pd.read_csv(PATCH)

patch = patch[patch["resolved"] == True].copy()

df = quality.merge(
    patch[["instance_id"] + PREDICTORS],
    on="instance_id",
    how="inner",
)

assert df["instance_id"].nunique() == 359

df["complexity_adverse_count"] = df[COMPLEXITY].sum(axis=1)
df["noncomplexity_adverse_count"] = df[NON_COMPLEXITY].sum(axis=1)

outcomes = [
    "adverse_dimension_count",
    "complexity_adverse_count",
    "noncomplexity_adverse_count",
]

rows = []

for sample_name, sample in [
    ("all", df),
    ("excluding_django", df[df["repo"] != "django/django"]),
]:
    for outcome in outcomes:
        temp_rows = []

        for predictor in PREDICTORS:
            d = sample[[predictor, outcome]].dropna()

            result = stats.spearmanr(
                d[predictor],
                d[outcome]
            )

            temp_rows.append({
                "sample": sample_name,
                "outcome": outcome,
                "predictor": predictor,
                "n": len(d),
                "spearman_rho": result.statistic,
                "p_value": result.pvalue,
            })

        pvals = [r["p_value"] for r in temp_rows]
        adjusted = holm_adjust(pvals)

        for r, p_adj in zip(temp_rows, adjusted):
            r["p_holm"] = p_adj
            rows.append(r)

result = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False)

for sample_name in ["all", "excluding_django"]:
    print(f"\n=== {sample_name.upper()} ===")

    sub = result[result["sample"] == sample_name]

    for outcome in outcomes:
        print(f"\n--- {outcome} ---")

        print(
            sub[sub["outcome"] == outcome][
                [
                    "predictor",
                    "n",
                    "spearman_rho",
                    "p_value",
                    "p_holm",
                ]
            ].to_string(index=False)
        )

print("\nSaved:", OUT)

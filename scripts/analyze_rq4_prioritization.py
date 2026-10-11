from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

QUALITY = ROOT / "data/processed/quality_degradation_patch.csv"
PATCH = ROOT / "data/processed/patch_metrics.csv"
OUT = ROOT / "results/statistics/rq4_prioritization.csv"

BUDGETS = [0.10, 0.20, 0.30, 0.50]

NONCOMPLEXITY = [
    "adverse_mi",
    "adverse_halstead_volume",
    "adverse_function_length_mean",
    "adverse_function_length_max",
    "adverse_ruff_findings",
]


def expected_capture(df, score_col, outcome_col, k):
    """
    Expected captured burden at top-k under random tie-breaking.
    This avoids arbitrary ordering when many patches share a score.
    """
    d = df[[score_col, outcome_col]].dropna().copy()

    groups = (
        d.groupby(score_col, as_index=False)
         .agg(
             group_n=(outcome_col, "size"),
             group_burden=(outcome_col, "sum"),
         )
         .sort_values(score_col, ascending=False)
    )

    remaining = k
    captured = 0.0

    for _, row in groups.iterrows():
        n = int(row["group_n"])
        burden = float(row["group_burden"])

        if remaining <= 0:
            break

        if n <= remaining:
            captured += burden
            remaining -= n
        else:
            fraction = remaining / n
            captured += fraction * burden
            remaining = 0

    total = d[outcome_col].sum()

    if total == 0:
        return np.nan

    return captured / total


def evaluate(sample, sample_name):
    d = sample.copy()

    d["noncomplexity_adverse_count"] = d[NONCOMPLEXITY].sum(axis=1)

    # Fixed, transparent combined score.
    # No model fitting and no learned weights.
    d["churn_rank"] = d["churn"].rank(
        method="average",
        pct=True
    )

    d["control_rank"] = d["control_token_delta"].rank(
        method="average",
        pct=True
    )

    d["imports_rank"] = d["imports_added"].rank(
        method="average",
        pct=True
    )

    d["control_churn_rank"] = (
        d["control_rank"] + d["churn_rank"]
    ) / 2

    strategies = {
        "churn": "churn",
        "control_token_delta": "control_token_delta",
        "imports_added": "imports_added",
        "control_plus_churn": "control_churn_rank",
    }

    outcomes = {
        "noncomplexity_adverse_count":
            "noncomplexity_adverse_count",
        "all_adverse_count":
            "adverse_dimension_count",
    }

    rows = []

    for outcome_name, outcome_col in outcomes.items():

        # Oracle ranking gives the theoretical best achievable
        # ordering for this observed outcome.
        d["_oracle_score"] = d[outcome_col]

        for budget in BUDGETS:
            k = max(1, int(np.ceil(len(d) * budget)))
            actual_effort = k / len(d)

            oracle_capture = expected_capture(
                d,
                "_oracle_score",
                outcome_col,
                k,
            )

            for strategy_name, score_col in strategies.items():

                capture = expected_capture(
                    d,
                    score_col,
                    outcome_col,
                    k,
                )

                lift = (
                    capture / actual_effort
                    if actual_effort > 0
                    else np.nan
                )

                oracle_fraction = (
                    capture / oracle_capture
                    if oracle_capture and oracle_capture > 0
                    else np.nan
                )

                rows.append({
                    "sample": sample_name,
                    "outcome": outcome_name,
                    "strategy": strategy_name,
                    "budget_target": budget,
                    "patches_inspected": k,
                    "actual_effort_pct":
                        100 * actual_effort,
                    "burden_capture_pct":
                        100 * capture,
                    "random_expected_capture_pct":
                        100 * actual_effort,
                    "lift_over_random":
                        lift,
                    "oracle_capture_pct":
                        100 * oracle_capture,
                    "fraction_of_oracle":
                        oracle_fraction,
                })

    return pd.DataFrame(rows)


quality = pd.read_csv(QUALITY)

patch = pd.read_csv(PATCH)
patch = patch[patch["resolved"] == True].copy()

df = quality.merge(
    patch[
        [
            "instance_id",
            "churn",
            "imports_added",
            "control_token_delta",
        ]
    ],
    on="instance_id",
    how="inner",
)

assert df["instance_id"].nunique() == 359

all_results = evaluate(
    df,
    "all"
)

nodjango_results = evaluate(
    df[df["repo"] != "django/django"].copy(),
    "excluding_django"
)

result = pd.concat(
    [all_results, nodjango_results],
    ignore_index=True
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False)

for sample in ["all", "excluding_django"]:
    print(f"\n=== {sample.upper()} ===")

    sub = result[
        (result["sample"] == sample) &
        (result["outcome"] == "noncomplexity_adverse_count")
    ]

    print(
        sub[
            [
                "strategy",
                "budget_target",
                "patches_inspected",
                "burden_capture_pct",
                "random_expected_capture_pct",
                "lift_over_random",
                "oracle_capture_pct",
                "fraction_of_oracle",
            ]
        ].to_string(index=False)
    )

print("\nSaved:", OUT)

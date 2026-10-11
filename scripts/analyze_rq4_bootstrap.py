from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

QUALITY = ROOT / "data/processed/quality_degradation_patch.csv"
PATCH = ROOT / "data/processed/patch_metrics.csv"
OUT = ROOT / "results/statistics/rq4_bootstrap_ci.csv"

BUDGETS = [0.10, 0.20, 0.30, 0.50]
ITERATIONS = 5000
SEED = 42

NONCOMPLEXITY = [
    "adverse_mi",
    "adverse_halstead_volume",
    "adverse_function_length_mean",
    "adverse_function_length_max",
    "adverse_ruff_findings",
]


def expected_capture(df, score_col, outcome_col, k):
    groups = (
        df.groupby(score_col, as_index=False)
          .agg(
              group_n=(outcome_col, "size"),
              group_burden=(outcome_col, "sum"),
          )
          .sort_values(score_col, ascending=False)
    )

    remaining = k
    captured = 0.0

    for _, row in groups.iterrows():
        if remaining <= 0:
            break

        n = int(row["group_n"])
        burden = float(row["group_burden"])

        if n <= remaining:
            captured += burden
            remaining -= n
        else:
            captured += burden * (remaining / n)
            remaining = 0

    total = df[outcome_col].sum()

    if total <= 0:
        return np.nan

    return captured / total


quality = pd.read_csv(QUALITY)
patch = pd.read_csv(PATCH)

patch = patch[patch["resolved"] == True].copy()

df = quality.merge(
    patch[
        [
            "instance_id",
            "churn",
            "control_token_delta",
        ]
    ],
    on="instance_id",
    how="inner",
)

assert df["instance_id"].nunique() == 359

df["noncomplexity_adverse_count"] = df[NONCOMPLEXITY].sum(axis=1)

rng = np.random.default_rng(SEED)

rows = []

for sample_name, sample in [
    ("all", df),
    ("excluding_django", df[df["repo"] != "django/django"].copy()),
]:

    for strategy in ["control_token_delta", "churn"]:

        for budget in BUDGETS:

            n = len(sample)
            k = max(1, int(np.ceil(n * budget)))

            observed = expected_capture(
                sample,
                strategy,
                "noncomplexity_adverse_count",
                k,
            )

            boot = []

            # Stratified bootstrap: sample within repository
            # to preserve repository composition.
            repo_groups = {
                repo: g.copy()
                for repo, g in sample.groupby("repo")
            }

            for _ in range(ITERATIONS):

                pieces = []

                for repo, g in repo_groups.items():
                    idx = rng.integers(
                        0,
                        len(g),
                        size=len(g)
                    )

                    pieces.append(
                        g.iloc[idx].copy()
                    )

                b = pd.concat(
                    pieces,
                    ignore_index=True
                )

                kb = max(
                    1,
                    int(np.ceil(len(b) * budget))
                )

                value = expected_capture(
                    b,
                    strategy,
                    "noncomplexity_adverse_count",
                    kb,
                )

                if np.isfinite(value):
                    boot.append(value)

            low, high = np.quantile(
                boot,
                [0.025, 0.975]
            )

            effort = k / n

            rows.append({
                "sample": sample_name,
                "strategy": strategy,
                "budget_target": budget,
                "patches_inspected": k,
                "actual_effort_pct": 100 * effort,
                "burden_capture_pct": 100 * observed,
                "capture_ci95_low": 100 * low,
                "capture_ci95_high": 100 * high,
                "random_expected_pct": 100 * effort,
                "lift_over_random": observed / effort,
            })


result = pd.DataFrame(rows)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUT,
    index=False
)

for sample in [
    "all",
    "excluding_django",
]:

    print(f"\n=== {sample.upper()} ===")

    print(
        result[result["sample"] == sample][
            [
                "strategy",
                "budget_target",
                "patches_inspected",
                "burden_capture_pct",
                "capture_ci95_low",
                "capture_ci95_high",
                "random_expected_pct",
                "lift_over_random",
            ]
        ].to_string(index=False)
    )

print("\nSaved:", OUT)

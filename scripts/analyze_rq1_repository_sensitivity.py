from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/quality_metrics_patch.csv"
PER_REPO_OUT = ROOT / "results/statistics/rq1_quality_by_repository.csv"
LOO_OUT = ROOT / "results/statistics/rq1_leave_one_repo_out.csv"

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


def rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]

    if len(values) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(values))
    w_plus = ranks[values > 0].sum()
    w_minus = ranks[values < 0].sum()

    return (w_plus - w_minus) / (w_plus + w_minus)


df = pd.read_csv(INPUT)

print("Total patches:", len(df))
print("Repositories:", df["repo"].nunique())

print("\n=== PATCHES BY REPOSITORY ===")
print(
    df.groupby("repo")["instance_id"]
      .nunique()
      .sort_values(ascending=False)
      .to_string()
)

# --------------------------------------------------
# Per-repository descriptive sensitivity
# --------------------------------------------------

repo_rows = []

for repo, g in df.groupby("repo"):
    for metric, direction in METRICS.items():
        vals = g[f"delta_{metric}"].dropna().to_numpy(dtype=float)

        if len(vals) == 0:
            continue

        if direction == "down":
            adverse = int((vals < 0).sum())
        else:
            adverse = int((vals > 0).sum())

        repo_rows.append({
            "repo": repo,
            "metric": metric,
            "n_patches": len(vals),
            "median_delta": np.median(vals),
            "mean_delta": np.mean(vals),
            "adverse_patches": adverse,
            "adverse_pct": 100 * adverse / len(vals),
        })

per_repo = pd.DataFrame(repo_rows)
PER_REPO_OUT.parent.mkdir(parents=True, exist_ok=True)
per_repo.to_csv(PER_REPO_OUT, index=False)

# --------------------------------------------------
# Leave-one-repository-out sensitivity
# --------------------------------------------------

loo_rows = []

for excluded_repo in sorted(df["repo"].unique()):
    subset = df[df["repo"] != excluded_repo]

    for metric, direction in METRICS.items():
        vals = subset[f"delta_{metric}"].dropna().to_numpy(dtype=float)

        if len(vals) == 0:
            continue

        if direction == "down":
            adverse = int((vals < 0).sum())
        else:
            adverse = int((vals > 0).sum())

        if np.allclose(vals, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(
                vals,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue

        rbc = rank_biserial(vals)

        loo_rows.append({
            "excluded_repo": excluded_repo,
            "metric": metric,
            "n_patches": len(vals),
            "median_delta": np.median(vals),
            "mean_delta": np.mean(vals),
            "adverse_patches": adverse,
            "adverse_pct": 100 * adverse / len(vals),
            "rank_biserial": rbc,
            "p_value": p,
        })

loo = pd.DataFrame(loo_rows)
loo.to_csv(LOO_OUT, index=False)

# --------------------------------------------------
# Most important comparison: full sample vs no Django
# --------------------------------------------------

print("\n=== EXCLUDING DJANGO ===")

nodjango = df[df["repo"] != "django/django"]

print("Patches remaining:", len(nodjango))

for metric, direction in METRICS.items():
    vals = nodjango[f"delta_{metric}"].dropna().to_numpy(dtype=float)

    adverse = (
        int((vals < 0).sum())
        if direction == "down"
        else int((vals > 0).sum())
    )

    if np.allclose(vals, 0):
        p = 1.0
    else:
        p = stats.wilcoxon(
            vals,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        ).pvalue

    print(
        f"{metric:24s} "
        f"n={len(vals):3d} "
        f"adverse={100*adverse/len(vals):6.2f}% "
        f"median={np.median(vals):10.4f} "
        f"mean={np.mean(vals):10.4f} "
        f"rbc={rank_biserial(vals):8.4f} "
        f"p={p:.3e}"
    )

print("\nSaved:", PER_REPO_OUT)
print("Saved:", LOO_OUT)

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, ensure_dirs, load_config


def cliffs_delta(x, y) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    greater = 0
    less = 0
    for xv in x:
        greater += int(np.sum(xv > y))
        less += int(np.sum(xv < y))
    return (greater - less) / (len(x) * len(y))


def bootstrap_mean_diff(x, y, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan
    diffs = np.empty(iterations)
    for i in range(iterations):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xs.mean() - ys.mean()
    return tuple(np.quantile(diffs, [0.025, 0.975]))


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def patch_shape_analysis(cfg: dict) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "patch_metrics.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    metrics = [
        "churn",
        "files_changed",
        "hunks",
        "imports_added",
        "control_token_delta",
        "security_sensitive_api_additions",
    ]
    rows = []
    rng = np.random.default_rng(cfg["statistics"]["random_seed"])
    iters = int(cfg["statistics"]["bootstrap_iterations"])

    for metric in metrics:
        resolved = df.loc[df["resolved"] == True, metric].dropna()
        unresolved = df.loc[df["resolved"] == False, metric].dropna()
        if len(resolved) == 0 or len(unresolved) == 0:
            continue
        test = stats.mannwhitneyu(resolved, unresolved, alternative="two-sided")
        ci_low, ci_high = bootstrap_mean_diff(resolved, unresolved, iters, rng)
        rows.append(
            {
                "analysis": "resolved-vs-unresolved",
                "metric": metric,
                "n_resolved": len(resolved),
                "n_unresolved": len(unresolved),
                "resolved_median": resolved.median(),
                "unresolved_median": unresolved.median(),
                "resolved_mean": resolved.mean(),
                "unresolved_mean": unresolved.mean(),
                "mean_diff_resolved_minus_unresolved": resolved.mean() - unresolved.mean(),
                "mean_diff_ci95_low": ci_low,
                "mean_diff_ci95_high": ci_high,
                "mann_whitney_u": test.statistic,
                "p_value": test.pvalue,
                "cliffs_delta": cliffs_delta(resolved, unresolved),
            }
        )

    if rows:
        adj = holm_adjust([r["p_value"] for r in rows])
        for r, p_adj in zip(rows, adj):
            r["p_holm"] = p_adj
    return pd.DataFrame(rows)


def paired_quality_analysis() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "quality_metrics_long.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & (df["resolved"] == True)].copy()

    metric_names = [
        "cc_mean",
        "cc_max",
        "mi",
        "halstead_volume",
        "cognitive_mean",
        "cognitive_max",
        "max_nesting",
        "function_length_mean",
        "function_length_max",
        "ruff_findings",
    ]
    rows = []
    for metric in metric_names:
        bcol, acol = f"before_{metric}", f"after_{metric}"
        if bcol not in df or acol not in df:
            continue
        d = df[[bcol, acol]].dropna()
        if len(d) < 5:
            continue
        diff = d[acol] - d[bcol]
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            test = stats.wilcoxon(d[acol], d[bcol], alternative="two-sided", zero_method="wilcox")
            stat, p = test.statistic, test.pvalue
        rows.append(
            {
                "analysis": "resolved-before-vs-after",
                "metric": metric,
                "n_files": len(d),
                "before_median": d[bcol].median(),
                "after_median": d[acol].median(),
                "delta_median": diff.median(),
                "delta_mean": diff.mean(),
                "wilcoxon_statistic": stat,
                "p_value": p,
            }
        )

    if rows:
        adj = holm_adjust([r["p_value"] for r in rows])
        for r, p_adj in zip(rows, adj):
            r["p_holm"] = p_adj
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config(args.config)

    patch = patch_shape_analysis(cfg)
    quality = paired_quality_analysis()

    if not patch.empty:
        out = ROOT / "results" / "statistics" / "patch_shape_tests.csv"
        patch.to_csv(out, index=False)
        print(f"Wrote {out}")
    else:
        print("Patch-shape analysis skipped: patch_metrics.csv not found or empty.")

    if not quality.empty:
        out = ROOT / "results" / "statistics" / "quality_paired_tests.csv"
        quality.to_csv(out, index=False)
        print(f"Wrote {out}")
    else:
        print("Quality paired analysis skipped: quality_metrics_long.csv not found or insufficient.")


if __name__ == "__main__":
    main()

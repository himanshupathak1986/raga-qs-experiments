from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results/statistics/rq2_security_analysis.csv"
PAIR_OUT = ROOT / "results/statistics/rq2_analyzer_pairwise.csv"


def wilson_ci(x, n):
    if n == 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p = x / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(
        p*(1-p)/n + z*z/(4*n*n)
    ) / denom

    return center - half, center + half


def load_clean(path):
    df = pd.read_csv(path)

    df = df[df["CWEID"].notna()].copy()

    for c in ["CodeQL", "Bandit", "Manual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Manual"].isin([0, 1])]

    return df


performance_rows = []
pair_rows = []
pooled_frames = []

for model in ["copilot", "incoder"]:

    path = ROOT / "data/raw" / f"securityeval_{model}.csv"
    df = load_clean(path)
    df["model"] = model
    pooled_frames.append(df)

    print(f"\n=== {model.upper()} ===")
    print("Usable rows:", len(df))
    print("Manual vulnerable:", int((df["Manual"] == 1).sum()))
    print("Manual non-vulnerable:", int((df["Manual"] == 0).sum()))

    for analyzer in ["CodeQL", "Bandit"]:

        d = df[df[analyzer].isin([0, 1])].copy()

        y = d["Manual"].astype(int)
        pred = d[analyzer].astype(int)

        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())

        vulnerable_n = tp + fn
        nonvulnerable_n = tn + fp

        recall = tp / vulnerable_n
        fnr = fn / vulnerable_n
        specificity = tn / nonvulnerable_n
        precision = tp / (tp + fp) if tp + fp else np.nan

        recall_low, recall_high = wilson_ci(tp, vulnerable_n)
        fnr_low, fnr_high = wilson_ci(fn, vulnerable_n)
        spec_low, spec_high = wilson_ci(tn, nonvulnerable_n)

        if tp + fp:
            prec_low, prec_high = wilson_ci(tp, tp + fp)
        else:
            prec_low, prec_high = np.nan, np.nan

        performance_rows.append({
            "model": model,
            "analyzer": analyzer,
            "n": len(d),
            "manual_vulnerable": vulnerable_n,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": recall,
            "recall_ci95_low": recall_low,
            "recall_ci95_high": recall_high,
            "false_negative_rate": fnr,
            "fnr_ci95_low": fnr_low,
            "fnr_ci95_high": fnr_high,
            "precision": precision,
            "precision_ci95_low": prec_low,
            "precision_ci95_high": prec_high,
            "specificity": specificity,
            "specificity_ci95_low": spec_low,
            "specificity_ci95_high": spec_high,
        })

        print(
            f"{analyzer:8s} "
            f"recall={100*recall:6.2f}% "
            f"[{100*recall_low:6.2f}, {100*recall_high:6.2f}] "
            f"FNR={100*fnr:6.2f}% "
            f"[{100*fnr_low:6.2f}, {100*fnr_high:6.2f}]"
        )

    # Paired comparison restricted to manually vulnerable samples.
    paired = df[
        (df["Manual"] == 1) &
        df["CodeQL"].isin([0, 1]) &
        df["Bandit"].isin([0, 1])
    ].copy()

    codeql_only = int(
        ((paired["CodeQL"] == 1) & (paired["Bandit"] == 0)).sum()
    )

    bandit_only = int(
        ((paired["CodeQL"] == 0) & (paired["Bandit"] == 1)).sum()
    )

    both_detect = int(
        ((paired["CodeQL"] == 1) & (paired["Bandit"] == 1)).sum()
    )

    both_miss = int(
        ((paired["CodeQL"] == 0) & (paired["Bandit"] == 0)).sum()
    )

    discordant = codeql_only + bandit_only

    if discordant:
        p = stats.binomtest(
            codeql_only,
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p = 1.0

    pair_rows.append({
        "model": model,
        "manual_vulnerable_n": len(paired),
        "both_detect": both_detect,
        "codeql_only_detect": codeql_only,
        "bandit_only_detect": bandit_only,
        "both_miss": both_miss,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p,
    })

# Pooled paired descriptive comparison
pooled = pd.concat(pooled_frames, ignore_index=True)

paired = pooled[
    (pooled["Manual"] == 1) &
    pooled["CodeQL"].isin([0, 1]) &
    pooled["Bandit"].isin([0, 1])
].copy()

codeql_only = int(
    ((paired["CodeQL"] == 1) & (paired["Bandit"] == 0)).sum()
)

bandit_only = int(
    ((paired["CodeQL"] == 0) & (paired["Bandit"] == 1)).sum()
)

both_detect = int(
    ((paired["CodeQL"] == 1) & (paired["Bandit"] == 1)).sum()
)

both_miss = int(
    ((paired["CodeQL"] == 0) & (paired["Bandit"] == 0)).sum()
)

discordant = codeql_only + bandit_only

p = (
    stats.binomtest(
        codeql_only,
        discordant,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    if discordant else 1.0
)

pair_rows.append({
    "model": "pooled",
    "manual_vulnerable_n": len(paired),
    "both_detect": both_detect,
    "codeql_only_detect": codeql_only,
    "bandit_only_detect": bandit_only,
    "both_miss": both_miss,
    "discordant_pairs": discordant,
    "mcnemar_exact_p": p,
})

performance = pd.DataFrame(performance_rows)
pairwise = pd.DataFrame(pair_rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
performance.to_csv(OUT, index=False)
pairwise.to_csv(PAIR_OUT, index=False)

print("\n=== PAIRED ANALYZER COMPARISON ===")
print(pairwise.to_string(index=False))

print("\nSaved:", OUT)
print("Saved:", PAIR_OUT)

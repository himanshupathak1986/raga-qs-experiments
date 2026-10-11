from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/statistics/rq2_union_detection.csv"


def wilson_ci(x, n):
    z = 1.959963984540054
    p = x / n
    denom = 1 + z*z/n

    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(
        p*(1-p)/n + z*z/(4*n*n)
    ) / denom

    return center - half, center + half


rows = []

for model in ["copilot", "incoder"]:

    path = ROOT / "data/raw" / f"securityeval_{model}.csv"

    df = pd.read_csv(path)
    df = df[df["CWEID"].notna()].copy()

    for col in ["Manual", "CodeQL", "Bandit"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    d = df[
        (df["Manual"] == 1) &
        df["CodeQL"].isin([0, 1]) &
        df["Bandit"].isin([0, 1])
    ].copy()

    d["union_detect"] = (
        (d["CodeQL"] == 1) |
        (d["Bandit"] == 1)
    ).astype(int)

    n = len(d)
    detected = int(d["union_detect"].sum())
    missed = n - detected

    recall = detected / n
    miss_rate = missed / n

    rlo, rhi = wilson_ci(detected, n)
    mlo, mhi = wilson_ci(missed, n)

    rows.append({
        "model": model,
        "manual_vulnerable_n": n,
        "detected_by_either": detected,
        "missed_by_both": missed,
        "union_recall": recall,
        "union_recall_ci95_low": rlo,
        "union_recall_ci95_high": rhi,
        "both_miss_rate": miss_rate,
        "both_miss_ci95_low": mlo,
        "both_miss_ci95_high": mhi,
    })

result = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False)

print("\n=== COMBINED ANALYZER COVERAGE ===")

for _, r in result.iterrows():
    print(
        f"{r['model']:8s} "
        f"either detected={int(r['detected_by_either'])}/"
        f"{int(r['manual_vulnerable_n'])} "
        f"({100*r['union_recall']:.2f}%, "
        f"95% CI {100*r['union_recall_ci95_low']:.2f}-"
        f"{100*r['union_recall_ci95_high']:.2f}%) | "
        f"both missed={int(r['missed_by_both'])}/"
        f"{int(r['manual_vulnerable_n'])} "
        f"({100*r['both_miss_rate']:.2f}%)"
    )

print("\nSaved:", OUT)

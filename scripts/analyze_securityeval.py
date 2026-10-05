from __future__ import annotations

import argparse
import math

import pandas as pd

from common import ROOT, ensure_dirs


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    fnr = fn / (fn + tp) if fn + tp else math.nan

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_negative_rate": fnr,
    }


def load_clean(path):
    df = pd.read_csv(path)
    df = df[df["CWEID"].notna()].copy()
    for c in ["CodeQL", "Bandit", "Manual"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Manual"].isin([0, 1])]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_dirs()

    rows = []
    for model in ["copilot", "incoder"]:
        path = ROOT / "data" / "raw" / f"securityeval_{model}.csv"
        if not path.exists():
            raise SystemExit("Missing SecurityEval files. Run scripts/collect_securityeval.py first.")
        df = load_clean(path)

        for analyzer in ["CodeQL", "Bandit"]:
            d = df[df[analyzer].isin([0, 1])].copy()
            m = binary_metrics(d["Manual"], d[analyzer])
            rows.append(
                {
                    "model": model,
                    "analyzer": analyzer,
                    "n": len(d),
                    "manual_vulnerable": int((d["Manual"] == 1).sum()),
                    "manual_not_vulnerable": int((d["Manual"] == 0).sum()),
                    **m,
                }
            )

    out_df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "securityeval_analyzer_performance.csv"
    out_df.to_csv(out, index=False)
    print(out_df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

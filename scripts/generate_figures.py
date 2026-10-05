from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_dirs()

    patch_path = ROOT / "data" / "processed" / "patch_metrics.csv"
    if patch_path.exists():
        df = pd.read_csv(patch_path)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        groups = [
            df.loc[df["resolved"] == True, "churn"].dropna(),
            df.loc[df["resolved"] == False, "churn"].dropna(),
        ]
        ax.boxplot(groups, tick_labels=["Resolved", "Unresolved"], showfliers=False)
        ax.set_ylabel("Changed lines (additions + deletions)")
        ax.set_title("Patch churn by functional outcome")
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "figures" / "patch_churn_by_outcome.png", dpi=220)
        plt.close(fig)

    sec_path = ROOT / "results" / "tables" / "securityeval_analyzer_performance.csv"
    if sec_path.exists():
        df = pd.read_csv(sec_path)
        labels = [f"{m}\n{a}" for m, a in zip(df["model"], df["analyzer"])]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, df["recall"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall against published manual labels")
        ax.set_title("SecurityEval static-analyzer recall")
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "figures" / "securityeval_recall.png", dpi=220)
        plt.close(fig)

    print("Generated available figures under results/figures/.")


if __name__ == "__main__":
    main()

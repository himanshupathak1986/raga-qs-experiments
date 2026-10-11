from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
OUT = ROOT / "results/manuscript"

TABLES = OUT / "tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def pct(x):
    return f"{x:.1f}%"


def pformat(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ============================================================
# TABLE 1 — Dataset / experimental summary
# ============================================================

patch_summary = pd.read_csv(
    FINAL / "patch_shape_summary.csv"
)

security = pd.read_csv(
    FINAL / "rq2_security_analysis.csv"
)

resolved_n = int(
    patch_summary.loc[
        patch_summary["resolved"] == True, "n"
    ].iloc[0]
)

unresolved_n = int(
    patch_summary.loc[
        patch_summary["resolved"] == False, "n"
    ].iloc[0]
)

attempted_n = resolved_n + unresolved_n

dataset_rows = [
    {
        "Dataset": "SWE-bench Verified model-output cohort",
        "Role": "Functional outcome + software-quality analysis",
        "Evaluated outputs": attempted_n,
        "Primary analyzed subset": f"{resolved_n} functionally successful patches",
        "Notes": "Quality analysis restricted to successful patches"
    },
    {
        "Dataset": "SecurityEval Copilot outputs",
        "Role": "Security analyzer assessment",
        "Evaluated outputs": int(
            security.loc[
                security["model"] == "copilot", "n"
            ].max()
        ),
        "Primary analyzed subset": (
            f"{int(security.loc[security['model']=='copilot', 'manual_vulnerable'].max())} "
            "manually labeled vulnerable outputs"
        ),
        "Notes": "Secondary analysis of published benchmark outputs"
    },
    {
        "Dataset": "SecurityEval InCoder outputs",
        "Role": "Security analyzer assessment",
        "Evaluated outputs": int(
            security.loc[
                security["model"] == "incoder", "n"
            ].max()
        ),
        "Primary analyzed subset": (
            f"{int(security.loc[security['model']=='incoder', 'manual_vulnerable'].max())} "
            "manually labeled vulnerable outputs"
        ),
        "Notes": "Secondary analysis of published benchmark outputs"
    },
]

table1 = pd.DataFrame(dataset_rows)
table1.to_csv(
    TABLES / "table1_datasets.csv",
    index=False
)


# ============================================================
# TABLE 2 — RQ1 quality results
# ============================================================

rq1 = pd.read_csv(
    FINAL / "rq1_quality_patch_tests.csv"
)

metric_names = {
    "cc_mean": "Mean cyclomatic complexity",
    "cc_max": "Maximum cyclomatic complexity",
    "mi": "Maintainability Index",
    "halstead_volume": "Halstead volume",
    "cognitive_mean": "Mean cognitive complexity",
    "cognitive_max": "Maximum cognitive complexity",
    "max_nesting": "Maximum nesting",
    "function_length_mean": "Mean function length",
    "function_length_max": "Maximum function length",
    "ruff_findings": "Ruff findings",
}

table2 = rq1.copy()

table2["Metric"] = table2["metric"].map(metric_names)
table2["Adverse patches"] = (
    table2["adverse_patches"].astype(int).astype(str)
    + "/"
    + table2["n_patches"].astype(int).astype(str)
)

table2["Adverse (%)"] = table2["adverse_pct"].map(
    lambda x: round(x, 1)
)

table2["Median delta"] = table2[
    "median_patch_delta"
].round(3)

table2["Mean delta"] = table2[
    "mean_patch_delta"
].round(3)

table2["Rank-biserial"] = table2[
    "rank_biserial"
].round(3)

table2["Holm p"] = table2["p_holm"].map(pformat)

table2 = table2[
    [
        "Metric",
        "Adverse patches",
        "Adverse (%)",
        "Median delta",
        "Mean delta",
        "Rank-biserial",
        "Holm p",
    ]
]

table2.to_csv(
    TABLES / "table2_rq1_quality.csv",
    index=False
)


# ============================================================
# TABLE 3 — RQ2 security
# ============================================================

sec = pd.read_csv(
    FINAL / "rq2_security_analysis.csv"
)

union = pd.read_csv(
    FINAL / "rq2_union_detection.csv"
)

rows = []

for _, r in sec.iterrows():
    rows.append({
        "Model": r["model"].title(),
        "Analysis": r["analyzer"],
        "Vulnerable n": int(r["manual_vulnerable"]),
        "Detected (%)": round(100 * r["recall"], 1),
        "Detection 95% CI": (
            f"{100*r['recall_ci95_low']:.1f}–"
            f"{100*r['recall_ci95_high']:.1f}"
        ),
        "Missed (%)": round(
            100 * r["false_negative_rate"], 1
        ),
    })

for _, r in union.iterrows():
    rows.append({
        "Model": r["model"].title(),
        "Analysis": "CodeQL ∪ Bandit",
        "Vulnerable n": int(r["manual_vulnerable_n"]),
        "Detected (%)": round(100 * r["union_recall"], 1),
        "Detection 95% CI": (
            f"{100*r['union_recall_ci95_low']:.1f}–"
            f"{100*r['union_recall_ci95_high']:.1f}"
        ),
        "Missed (%)": round(
            100 * r["both_miss_rate"], 1
        ),
    })

table3 = pd.DataFrame(rows)

table3.to_csv(
    TABLES / "table3_rq2_security.csv",
    index=False
)


# ============================================================
# TABLE 4 — RQ3/RQ4 compact results
# ============================================================

overlap = pd.read_csv(
    FINAL / "rq3_overlap_sensitivity.csv"
)

rq4 = pd.read_csv(
    FINAL / "rq4_bootstrap_ci.csv"
)

assoc = overlap[
    overlap["outcome"] == "noncomplexity_adverse_count"
].copy()

assoc = assoc[
    assoc["predictor"].isin(
        ["control_token_delta", "churn"]
    )
]

assoc_rows = []

for _, r in assoc.iterrows():
    assoc_rows.append({
        "Analysis": "Association",
        "Sample": (
            "All"
            if r["sample"] == "all"
            else "Excluding Django"
        ),
        "Signal": (
            "Control-token delta"
            if r["predictor"] == "control_token_delta"
            else "Churn"
        ),
        "Statistic": "Spearman ρ",
        "Value": round(r["spearman_rho"], 3),
        "95% CI / comparison": "",
        "Adjusted p": pformat(r["p_holm"]),
    })

priority = rq4[
    (rq4["budget_target"] == 0.20)
    & rq4["strategy"].isin(
        ["control_token_delta", "churn"]
    )
].copy()

priority_rows = []

for _, r in priority.iterrows():
    priority_rows.append({
        "Analysis": "20% verification budget",
        "Sample": (
            "All"
            if r["sample"] == "all"
            else "Excluding Django"
        ),
        "Signal": (
            "Control-token delta"
            if r["strategy"] == "control_token_delta"
            else "Churn"
        ),
        "Statistic": "Burden captured",
        "Value": round(r["burden_capture_pct"], 1),
        "95% CI / comparison": (
            f"{r['capture_ci95_low']:.1f}–"
            f"{r['capture_ci95_high']:.1f}% "
            f"(random {r['random_expected_pct']:.1f}%)"
        ),
        "Adjusted p": "",
    })

table4 = pd.DataFrame(
    assoc_rows + priority_rows
)

table4.to_csv(
    TABLES / "table4_rq3_rq4.csv",
    index=False
)


# ============================================================
# FIGURE 1 — Distribution of adverse quality indicators
# ============================================================

profile = pd.read_csv(
    FINAL / "rq1_degradation_profile.csv"
)

fig, ax = plt.subplots(figsize=(7.2, 4.4))

ax.bar(
    profile["adverse_dimensions"],
    profile["percentage"],
)

ax.set_xlabel(
    "Number of quality indicators changing in an adverse direction"
)
ax.set_ylabel("Functionally successful patches (%)")
ax.set_xticks(
    profile["adverse_dimensions"]
)
ax.set_ylim(
    0,
    max(profile["percentage"]) * 1.18
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    FIGURES / "figure1_adverse_indicator_distribution.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURES / "figure1_adverse_indicator_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — RQ3 signal association
# ============================================================

assoc_all = overlap[
    (overlap["sample"] == "all")
    & (
        overlap["outcome"]
        == "noncomplexity_adverse_count"
    )
].copy()

order = [
    "control_token_delta",
    "churn",
    "imports_added",
    "hunks",
    "files_changed",
]

labels = {
    "control_token_delta": "Control-token delta",
    "churn": "Churn",
    "imports_added": "Imports added",
    "hunks": "Hunks",
    "files_changed": "Files changed",
}

assoc_all = (
    assoc_all
    .set_index("predictor")
    .loc[order]
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.barh(
    [labels[x] for x in assoc_all["predictor"]],
    assoc_all["spearman_rho"],
)

ax.axvline(
    0,
    linewidth=0.8,
)

ax.set_xlabel(
    "Spearman correlation with non-complexity adverse-indicator count"
)

ax.set_ylabel("")

ax.invert_yaxis()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    FIGURES / "figure2_predictor_association.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURES / "figure2_predictor_association.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3 — RQ4 prioritization curve
# ============================================================

rq4_curve = pd.read_csv(
    FINAL / "rq4_bootstrap_ci.csv"
)

rq4_curve = rq4_curve[
    rq4_curve["sample"] == "all"
].copy()

fig, ax = plt.subplots(figsize=(7.2, 4.5))

for strategy, label in [
    ("control_token_delta", "Control-token delta"),
    ("churn", "Churn"),
]:
    d = rq4_curve[
        rq4_curve["strategy"] == strategy
    ].sort_values("actual_effort_pct")

    ax.plot(
        d["actual_effort_pct"],
        d["burden_capture_pct"],
        marker="o",
        label=label,
    )

    ax.fill_between(
        d["actual_effort_pct"],
        d["capture_ci95_low"],
        d["capture_ci95_high"],
        alpha=0.15,
    )

x = np.linspace(0, 55, 100)

ax.plot(
    x,
    x,
    linestyle="--",
    label="Random inspection",
)

ax.set_xlabel("Patches inspected (%)")
ax.set_ylabel(
    "Non-complexity adverse-quality burden captured (%)"
)

ax.set_xlim(0, 55)
ax.set_ylim(0, 82)

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    FIGURES / "figure3_verification_prioritization.pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURES / "figure3_verification_prioritization.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("=== MANUSCRIPT ASSETS CREATED ===")

print("\nTables:")
for p in sorted(TABLES.glob("*.csv")):
    print(" ", p.relative_to(ROOT))

print("\nFigures:")
for p in sorted(FIGURES.glob("*")):
    print(" ", p.relative_to(ROOT))

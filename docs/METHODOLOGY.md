# Methodology

## 1. Functional outcome

For the SWE-bench cohort, `resolved=true` is the primary functional-success indicator because it is derived from the benchmark's recorded test evaluation.

No additional meaning is assigned to this field.

## 2. Patch-shape measurements

The unified diff is used to calculate inexpensive measurements available before deep verification:

- additions
- deletions
- total churn
- files changed
- Python files changed
- test files changed
- hunks
- imports added/deleted
- approximate control-flow token delta
- security-sensitive API additions

These are change-level indicators, not direct measures of vulnerability or maintainability.

## 3. Before/after quality metrics

For each modified production Python file, the pipeline checks out the exact SWE-bench `base_commit`, applies the generated patch, and calculates comparable metrics on the before and after source.

Metrics:

- cyclomatic complexity: Radon block complexity
- Maintainability Index: Radon
- Halstead volume: Radon
- cognitive complexity: `cognitive-complexity`
- AST maximum nesting depth
- function count
- function length
- import count
- Ruff finding count, when Ruff can parse the file

File-level deltas are retained. Aggregate instance-level values can then be calculated with transparent rules.

## 4. Security reanalysis

For each SecurityEval model cohort, published `Manual` labels are compared with the published binary CodeQL and Bandit labels.

Primary diagnostic metrics:

- TP, FP, TN, FN
- precision
- recall / sensitivity
- specificity
- F1
- false-negative rate

This analysis answers how much vulnerability evidence the published static analyzers miss relative to the benchmark reference labels. It does not claim the manual labels are an absolute universal ground truth.

## 5. Statistical analysis

Initial confirmatory tests should be selected after data inspection but before testing the final hypotheses.

Current supported analyses include:

- paired Wilcoxon signed-rank tests for before/after quality measurements
- Mann-Whitney U tests for independent resolved/unresolved patch-shape comparisons
- Cliff's delta effect size
- bootstrap confidence intervals
- Holm adjustment for families of multiple tests

Raw counts and effect sizes should be reported alongside p-values.

## 6. Risk profiling

A composite risk score is not hard-coded into the replication package.

If the data show stable predictors of residual risk, the next stage may fit and validate a parsimonious risk-profiling model. Development/calibration and held-out evaluation must remain separated.

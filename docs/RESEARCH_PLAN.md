# Research Plan

## Working title

**Passing Tests Is Not Enough: Joint Quality and Security Risk Profiling of AI-Generated Software Changes**

## Theme

**Beyond Functional Correctness**

## Research objective

The study asks what measurable software-quality and security risk remains after an AI-generated software change satisfies functional tests.

The empirical study is intentionally prior to the framework design. No result is required to validate a preselected risk score.

## Research questions

### RQ1
Among functionally successful AI-generated patches, how often and how strongly do measurable software-quality properties degrade?

### RQ2
Among AI-generated code judged functionally acceptable in public security benchmarks, what security risks remain, and how much do conventional static analyzers miss relative to the published manual labels?

### RQ3
Which inexpensive change-level signals are associated with residual quality or security risk?

Candidate signals include change size, file count, hunk count, import changes, test-file touches, structural-complexity deltas, maintainability deltas, nesting, and security-sensitive API additions.

### RQ4
If reliable predictive signals exist, can they prioritize deeper verification without applying every expensive verification stage to every functionally successful change?

RQ4 is conditional. If the empirical evidence does not support reliable prioritization, the paper should report that result rather than force an adaptive-verification claim.

## Analysis sequence

1. Characterize the full generated-patch cohort.
2. Compare functionally resolved and unresolved changes at the patch-shape level.
3. Within the resolved cohort, calculate before/after quality metrics for modified production Python files.
4. Quantify quality deltas and identify which degradations are common, rare, or concentrated.
5. Reanalyze SecurityEval manual labels against CodeQL and Bandit.
6. Evaluate candidate inexpensive signals.
7. Decide whether a combined risk-profiling method is empirically justified.

## Interpretation guardrails

- Do not equate functional failure with security vulnerability.
- Do not equate a static-analysis finding with ground-truth vulnerability unless the benchmark defines it that way.
- Do not treat missing analyzer findings as evidence of safety.
- Do not mix unrelated benchmark records into a single population without preserving cohort identity.
- Distinguish secondary analysis of public artifacts from newly generated model runs.
- Report exclusions and tool failures.

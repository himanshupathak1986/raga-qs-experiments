# Datasets and Provenance

## SWE-bench Verified public experiment artifacts

Purpose: repository-level patch and functional-correctness cohort.

The default configuration downloads a public `all_preds.jsonl` containing generated unified diffs and joins it to the corresponding SWE-bench evaluation results. The SWE-bench Verified dataset server is used to retrieve repository names and exact `base_commit` values.

Key fields retained:

- `instance_id`
- `repo`
- `base_commit`
- `model_name_or_path`
- `model_patch`
- `resolved`

The raw artifacts remain public-source data. The generated processed files in this repository contain derived measurements and provenance fields.

## SecurityEval

Purpose: security cohort.

SecurityEval publishes Python generated-code evaluations for GitHub Copilot and InCoder. The CSV artifacts contain:

- CWE identifier
- sample identifier
- CodeQL label
- Bandit label
- manual security label

The analysis treats the published manual column as the benchmark's reference label and computes confusion matrices, precision, recall, specificity, F1, and exact counts for CodeQL and Bandit.

This is a reanalysis of published data, not a rerun of Copilot, InCoder, CodeQL, or Bandit.

## Future optional public cohorts

Additional public cohorts can be added only if their provenance and outcome definitions are kept separate. Candidate sources include SALLM, RealSec-Bench, and other public SWE-bench runs with complete patch and verdict artifacts.

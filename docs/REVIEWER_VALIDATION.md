# Reviewer validation of resolved patches

The frozen `paper-results-v1` tag contains the original Gemini/mini results. The reviewer analyses extend it with human reference patches, GPT-4o/Agentless, and a static security scan on the benchmark-resolved model patches.

| Replay cohort | Successfully applied | Production Python file measurements |
|---|---:|---:|
| Gemini/mini | 359 | 416 |
| Human references matched to Gemini tasks | 359 | 420 |
| GPT-4o/Agentless | 194 | 194 |

Each diff is applied against the recorded base-commit files. The replay computes static quality metrics and uses the published SWE-bench verdicts for functional resolution. It does not execute candidates or rerun the benchmark test harness.

## Results

- The full-cohort control-token ranking captured 32.45% and 36.41% of the observed adverse-indicator burden at about 20% of patches, in Gemini and Agentless respectively. With a cap of 20% of changed lines, it captured 9.05% and 15.87%. The random expectation under either budget was about 20%.
- On the same 173 tasks resolved by both runs, the ranking captured 35.59% and 35.17% at the patch-count budget, and 8.09% and 13.19% at the changed-line budget. Results are in [matched_task_review_budgets.csv](../results/reviewer/matched_task_review_budgets.csv). The two runs share task IDs, repositories, and base commits. Both comparisons are retrospective.
- Ruff 0.16.10 security-rule diagnostic counts increased on 3 of 359 Gemini and 2 of 194 Agentless patches. The scan compared each changed production Python file and rule before and after applying a patch, with isolated settings and suppression comments ignored. [The CSV](../results/reviewer/resolved_static_security_screen.csv) contains aggregate counts per patch. These are static warnings. The cohort lacks independent labels for confirmed vulnerabilities and recall.
- SecurityEval is a separate reanalysis of code completions and is documented with the original experiment.

Changed lines approximate the amount of code inspected. The line-budget script reports both prefix stopping and skipping non-fitting patches. Random ranks and ties are averaged over 5,000 seeded trials. The reduced burden can count any adverse move, the illustrative minimum-change cutoffs, or twice those cutoffs. Consult the CSV for all budgets and outcome definitions.

## Reproduce

Use Python 3.14.8 and Git. Installing requirements in the activated environment makes Ruff 0.16.10 available on PATH. Input downloads are pinned and checked by SHA-256. The scripts cache downloaded benchmark inputs, source files, and replay rows under `.cache/reviewer`.

```sh
python -m pip install -r scripts/reviewer/requirements.txt
python scripts/reviewer/download_inputs.py
python scripts/reviewer/replay_metrics.py original human agentless
python scripts/reviewer/verify_cohort.py
python scripts/reviewer/reviewer_stats.py
python scripts/reviewer/agentless_baseline_halstead.py
python scripts/reviewer/human_thresholds.py --rebuild-base
python scripts/reviewer/line_budget.py
python scripts/reviewer/matched_task_budgets.py
python scripts/reviewer/screen_resolved_patches.py
```

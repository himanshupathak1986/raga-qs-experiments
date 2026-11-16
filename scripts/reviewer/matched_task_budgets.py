"""Compare review-budget results on the same resolved tasks in both runs."""
from pathlib import Path

import pandas as pd

from line_budget import evaluate
from reviewer_stats import DATA, DELTAS, FEATURES, PROCESSED

ROOT = Path(__file__).resolve().parents[2]


def main():
    primary = DELTAS.join(FEATURES[['churn', 'control_token_delta']], validate='one_to_one')
    alternate = pd.read_json(DATA/'agentless_replay.jsonl', lines=True, precise_float=True).set_index('instance_id')
    assert alternate.index.is_unique and alternate.status.eq('ok').all()
    common = primary.index.intersection(alternate.index).sort_values()
    assert len(common) == 173
    original_files = pd.read_csv(PROCESSED/'quality_metrics_final.csv')
    refs = original_files[['instance_id', 'repo', 'base_commit']].drop_duplicates().set_index('instance_id')
    assert refs.index.is_unique
    assert (refs.loc[common, ['repo', 'base_commit']] == alternate.loc[common, ['repo', 'base_commit']]).all(axis=None)
    primary_before = original_files.groupby('instance_id').before_halstead_volume.mean()
    alternate_before = pd.read_csv(DATA/'agentless_baseline_halstead.csv', index_col='instance_id').before_halstead_volume
    outcomes = []
    for label, data, before in [
        ('Gemini/mini', primary, primary_before),
        ('GPT-4o/Agentless', alternate, alternate_before),
    ]:
        outcomes.extend(evaluate(data.loc[common], before.loc[common], label))
    result = pd.DataFrame(outcomes)
    result.to_csv(ROOT/'results/reviewer/matched_task_review_budgets.csv', index=False, float_format='%.8f')
    rows = result[result.target_pct.eq(20) & result.outcome.eq('any')]
    print(rows.pivot(index=['cohort', 'budget_type'], columns='strategy', values='mean_capture_pct').round(4))


if __name__ == '__main__':
    main()

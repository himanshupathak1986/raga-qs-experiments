"""Evaluate the published ranking orders on patch and changed-line budgets."""
from pathlib import Path

import numpy as np
import pandas as pd

from reviewer_stats import DATA, DELTAS, FEATURES, MIN_CHANGES, OUTCOME, PROCESSED, directed

ROOT = Path(__file__).resolve().parents[2]
BUDGETS = (0.10, 0.20, 0.30, 0.50)
TRIALS = 5000
SEED = 20261116


def outcomes(frame, baseline_halstead):
    d = directed(frame)[OUTCOME]
    assert baseline_halstead.index.is_unique
    base = baseline_halstead.reindex(d.index)
    assert base.notna().all() and (base >= 0).all()
    thresholds = MIN_CHANGES.reindex(OUTCOME)
    for label, scale in [('any', 0), ('minimum', 1), ('double', 2)]:
        flags = d > 0 if scale == 0 else d.ge(thresholds * scale)
        if scale:
            flags['delta_halstead_volume'] = d.delta_halstead_volume.ge(
                np.maximum(1, 0.05 * base) * scale
            )
        yield label, flags.sum(axis=1).to_numpy()


def evaluate(frame, baseline, cohort):
    costs = frame.churn.to_numpy(dtype=int)
    assert costs.min() > 0
    n = len(frame)
    seed = np.random.default_rng(SEED)
    tie_break = seed.random((TRIALS, n))
    scores = {
        'random': np.zeros(n),
        'churn': costs,
        'control_tokens': frame.control_token_delta.to_numpy(),
        'ast_mean_cc': np.round(frame.delta_cc_mean.to_numpy(), 12),
    }
    progress = []
    measures = list(outcomes(frame, baseline))
    for strategy, score in scores.items():
        rank_score = np.broadcast_to(-score, tie_break.shape)
        order = np.lexsort((tie_break, rank_score), axis=1)
        used = np.cumsum(costs[order], axis=1)
        for fraction in BUDGETS:
            cap = int(np.floor(fraction * costs.sum()))
            prefix_by_lines = (used <= cap).sum(axis=1)
            counts = {
                'patches': np.full(TRIALS, int(np.ceil(fraction * n))),
                'changed_lines': prefix_by_lines,
            }
            # Allow later smaller patches to use the remaining budget. This
            # sensitivity check changes the queue policy, not the ranking scores.
            remaining = np.full(TRIALS, cap)
            skip_selected = np.zeros_like(order, dtype=bool)
            for position, patch_cost in enumerate(costs[order].T):
                fits = patch_cost <= remaining
                remaining -= np.where(fits, patch_cost, 0)
                skip_selected[:, position] = fits
            for label, burden in measures:
                assert burden.sum() > 0
                cumulative_burden = np.cumsum(burden[order], axis=1)
                for budget_type, prefix_n in counts.items():
                    rows = np.arange(TRIALS)
                    inspected_lines = np.where(prefix_n, used[rows, prefix_n - 1], 0)
                    captured = np.where(prefix_n, cumulative_burden[rows, prefix_n - 1], 0)
                    if budget_type == 'changed_lines':
                        assert (inspected_lines <= cap).all()
                    captures = 100 * captured / burden.sum()
                    progress.append({
                        'cohort': cohort,
                        'patches': n,
                        'total_changed_lines': costs.sum(),
                        'outcome': label,
                        'total_burden': burden.sum(),
                        'budget_type': budget_type,
                        'target_pct': 100 * fraction,
                        'strategy': strategy,
                        'mean_reviewed_patches': prefix_n.mean(),
                        'mean_changed_lines_pct': 100 * inspected_lines.mean() / costs.sum(),
                        'mean_capture_pct': captures.mean(),
                        'mc_se_capture_pct_points': captures.std(ddof=1) / np.sqrt(TRIALS),
                    })
                skip_capture = 100 * (burden[order] * skip_selected).sum(axis=1) / burden.sum()
                progress.append({
                    'cohort': cohort,
                    'patches': n,
                    'total_changed_lines': costs.sum(),
                    'outcome': label,
                    'total_burden': burden.sum(),
                    'budget_type': 'changed_lines_skip_nonfitting',
                    'target_pct': 100 * fraction,
                    'strategy': strategy,
                    'mean_reviewed_patches': skip_selected.sum(axis=1).mean(),
                    'mean_changed_lines_pct': 100 * (cap - remaining).mean() / costs.sum(),
                    'mean_capture_pct': skip_capture.mean(),
                    'mc_se_capture_pct_points': skip_capture.std(ddof=1) / np.sqrt(TRIALS),
                })
    return progress


def main():
    primary = DELTAS.join(FEATURES[['churn', 'control_token_delta']], validate='one_to_one')
    primary_before = pd.read_csv(PROCESSED/'quality_metrics_final.csv').groupby(
        'instance_id'
    ).before_halstead_volume.mean()
    agentless = pd.read_json(
        DATA/'agentless_replay.jsonl', lines=True, precise_float=True
    ).set_index('instance_id')
    assert len(agentless) == 194 and agentless.index.is_unique and agentless.status.eq('ok').all()
    agentless_before = pd.read_csv(DATA/'agentless_baseline_halstead.csv').set_index(
        'instance_id'
    ).before_halstead_volume
    assert set(agentless_before.index) == set(agentless.index)
    rows = evaluate(primary, primary_before, 'Gemini/mini')
    rows += evaluate(agentless, agentless_before, 'GPT-4o/Agentless')
    result = pd.DataFrame(rows)
    result.to_csv(ROOT/'results/reviewer/line_budget_sensitivity.csv', index=False, float_format='%.8f')
    view = result[result.target_pct.eq(20)].pivot(
        index=['cohort', 'budget_type', 'outcome'],
        columns='strategy',
        values='mean_capture_pct',
    )
    print(view.round(2).to_string())


if __name__ == '__main__':
    main()

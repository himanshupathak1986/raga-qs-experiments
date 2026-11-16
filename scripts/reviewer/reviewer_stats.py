import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[2] / '.cache' / 'reviewer'
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / 'data/processed'
DELTAS = pd.read_csv(PROCESSED/'quality_metrics_patch.csv').set_index('instance_id')
FEATURES = pd.read_csv(PROCESSED/'patch_metrics.csv').set_index('instance_id')
COLS = [c for c in DELTAS if c.startswith('delta_')]
OUTCOME = ['delta_mi', 'delta_halstead_volume', 'delta_function_length_mean',
           'delta_function_length_max', 'delta_ruff_findings']
SIGNALS = ['churn', 'files_changed', 'hunks', 'imports_added', 'control_token_delta']
MIN_CHANGES = pd.Series({
    'delta_cc_mean': 0.1, 'delta_cc_max': 1, 'delta_mi': 1,
    'delta_cognitive_mean': 0.1, 'delta_cognitive_max': 1,
    'delta_max_nesting': 1, 'delta_function_length_mean': 1,
    'delta_function_length_max': 1, 'delta_ruff_findings': 1
})


def directed(frame):
    d = frame[COLS].copy()
    d['delta_mi'] *= -1
    return d


def expected_capture(scores, burdens, k):
    cutoff = np.sort(scores)[-k]
    above = scores > cutoff
    tied = scores == cutoff
    return (burdens[above].sum()
            + (k-above.sum()) * burdens[tied].mean()) / burdens.sum()


def ranking(frame, label):
    n = len(frame)
    y = (directed(frame)[OUTCOME] > 0).sum(axis=1).to_numpy()
    scores = {'control_tokens': frame.control_token_delta,
              'churn': frame.churn, 'AST_mean_CC': frame.delta_cc_mean,
              'oracle': y}
    print(f'Ranking {label} n={n} repos={frame.repo.nunique()} burden={y.sum()}')
    for budget in [0.10, 0.20, 0.30, 0.50]:
        k = int(np.ceil(n * budget))
        row = {name: round(100*expected_capture(np.asarray(x), y, k),4)
               for name, x in scores.items()}
        print(budget, k, round(100*k/n,4), row)
    # Correlation of midranks after each of the two variables is residualized on churn midranks.
    ranked = pd.DataFrame({'signal':frame.control_token_delta,'burden':y,
                           'churn':frame.churn}).rank().to_numpy()
    controls = np.column_stack([np.ones(n),ranked[:,2]])
    residuals = ranked[:,:2] - controls @ np.linalg.lstsq(controls,ranked[:,:2],rcond=None)[0]
    print('Spearman', stats.spearmanr(ranked[:,0],ranked[:,1]).statistic,
          'partial_churn', np.corrcoef(residuals.T)[0,1])


def verify_frozen():
    manifest = json.loads((ROOT/'data/manifests/frozen_data_manifest.json').read_text())
    for path,note in manifest['derived_data'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == note['sha256']
    assert len(DELTAS) == 359 and DELTAS.index.is_unique
    assert len(FEATURES) == 441 and FEATURES.index.is_unique
    assert set(DELTAS.index) == set(FEATURES.index[FEATURES.resolved])
    print('Verified',len(manifest['derived_data']),'derived data hashes')


def sensitivity(original):
    d = directed(original)
    table = pd.DataFrame({
        'improved':(d < 0).sum(),
        'unchanged':(d == 0).sum(),
        'worsened':(d > 0).sum(),
    })
    print(table.to_string())
    pos = (d > 0).any(axis=1)
    neg = (d < 0).any(axis=1)
    uncertain = int((pos ^ neg).sum())
    always = int((pos & neg).sum())
    observed = int(pos.sum())
    # Reverse the full delta vector with probability one half. This preserves
    # zero entries, magnitudes, and within-patch dependence across indicators.
    null_dist = stats.binom(uncertain, 0.5)
    print('Sign flip', observed, always, uncertain, 'null mean', always+null_dist.mean(),
          'null interval', always+np.array(null_dist.ppf([0.025,0.975])),
          'one-sided p', null_dist.sf(observed-always-1))

    written_files = pd.read_csv(PROCESSED/'quality_metrics_final.csv')
    base_halstead = written_files.groupby('instance_id').before_halstead_volume.mean().reindex(d.index)
    for scale in [1,2]:
        selected = d[MIN_CHANGES.index].ge(MIN_CHANGES*scale)
        # Keep positive movement requirement when baseline volume is zero.
        halstead_limit = np.maximum(1, base_halstead*0.05)*scale
        selected['delta_halstead_volume'] = d.delta_halstead_volume.ge(halstead_limit)
        print('Thresholds scale',scale,'patch count',int(selected.any(axis=1).sum()),
              'metric counts', selected.sum().to_dict())


def measure_human(original,human):
    valid = human.status.eq('ok') & human[COLS].notna().all(axis=1)
    h = human.loc[valid].set_index('instance_id')
    paired = original.loc[h.index]
    assert (paired.repo==h.repo).all()
    print('Human status counts',human.status.value_counts().to_dict())
    a = (directed(paired)>0).any(axis=1)
    b = (directed(h)>0).any(axis=1)
    only_ai = int((a&~b).sum())
    only_human = int((b&~a).sum())
    print('Paired',len(h),'AI',int(a.sum()),'human',int(b.sum()),
          'both',int((a&b).sum()),'AI-only',only_ai,'human-only',only_human,
          'neither',int((~a&~b).sum()),
          'McNemar exact unclustered exploratory p',stats.binomtest(only_ai, only_ai+only_human).pvalue)
    print('Signs AI')
    print(pd.DataFrame({'improved':(directed(paired)<0).sum(),'unchanged':(directed(paired)==0).sum(),'adverse':(directed(paired)>0).sum()}).to_string())
    print('Signs human')
    print(pd.DataFrame({'improved':(directed(h)<0).sum(),'unchanged':(directed(h)==0).sum(),'adverse':(directed(h)>0).sum()}).to_string())
    for name,group in [('all',paired),('excluding_django',paired[paired.repo!='django/django'])]:
        print('Human sensitivity',name,'n',len(group),'AI',int(a.loc[group.index].sum()),'human',int(b.loc[group.index].sum()))


def main():
    verify_frozen()
    original = DELTAS.join(FEATURES[SIGNALS],validate='one_to_one')
    sensitivity(original)
    ranking(original,'original')
    ranking(original[original.repo!='django/django'],'original excluding Django')
    d = original[original.repo=='django/django']
    y = (directed(d)[OUTCOME]>0).sum(axis=1)
    rho = {name:float(stats.spearmanr(d[name],y).statistic) for name in SIGNALS}
    print('Django-only development associations',rho)
    winner = max(rho,key=rho.get)
    assert winner == 'control_token_delta', winner
    print('Selected',winner)
    for label in ('original','human','agentless'):
        file = DATA/f'{label}_replay.jsonl'
        expected_n = {'original':359,'human':359,'agentless':194}[label]
        if not file.exists() or sum(1 for _ in file.open()) != expected_n:
            continue
        d = pd.read_json(file,lines=True,precise_float=True)
        if label=='original':
            x = d[d.status.eq('ok')].set_index('instance_id')
            ref = DELTAS.loc[x.index]
            diff = x[COLS]-ref[COLS]
            print('Original replay',len(x),'error counts',d.loc[d.status.ne('ok'),'status'].value_counts().to_dict())
            print('Original replay per-metric max absolute deltas',diff.abs().max().to_dict())
            print('Original replay per-metric sign agreement',(np.sign(x[COLS])==np.sign(ref[COLS])).sum().to_dict())
        elif label=='human':
            measure_human(original,d)
        else:
            x = d[d.status.eq('ok') & d[COLS].notna().all(axis=1)].set_index('instance_id')
            print('Agentless status',d.status.value_counts().to_dict())
            ranking(x,'Agentless all measurable resolved')
            overlap = x.index.isin(DELTAS.index)
            ranking(x[overlap],'Agentless overlap with original resolved')
            ranking(x[~overlap],'Agentless outside original resolved')


if __name__ == '__main__':
    main()

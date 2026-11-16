from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

import reviewer_stats as a

DATA = Path(__file__).resolve().parents[2] / '.cache' / 'reviewer'
METRICS = pd.read_csv(a.PROCESSED/'quality_metrics_final.csv')
GOLD = pd.read_parquet(DATA/'verified.parquet').set_index('instance_id')
ORIGINAL = a.DELTAS.join(a.FEATURES[a.SIGNALS],validate='one_to_one')
ORIGINAL_VERDICTS = json.loads((DATA/'original_result').read_text())
ORIGINAL_PRED = {x['instance_id']:x['model_patch'] for line in (DATA/'original_pred').read_text().splitlines() if (x:=json.loads(line))}

def verify_cohort():
    assert set(GOLD.index)-set(ORIGINAL_PRED)==set(ORIGINAL_VERDICTS['no_generation'])
    assert not ORIGINAL_VERDICTS['no_logs']
    assert set(a.DELTAS.index) == set(ORIGINAL_VERDICTS['resolved'])
    assert len(GOLD.loc[list(ORIGINAL_PRED)].repo.unique()) == 12
    print('500 Verified tasks, 59 no generation, 441 predictions, 359 resolved, 82 unresolved across 12 repositories overall')

def load(label, expected):
    inp = DATA/f'{label}_replay.jsonl'
    frame = pd.read_json(inp, lines=True, precise_float=True).set_index('instance_id')
    assert len(frame) == expected and frame.index.is_unique
    assert (frame.base_commit == GOLD.loc[frame.index,'base_commit']).all()
    assert (frame.repo == GOLD.loc[frame.index,'repo']).all()
    valid = frame.status.eq('ok') & frame[a.COLS].notna().all(axis=1)
    print(label,'usable',int(valid.sum()),'of',len(frame),'files',frame.loc[valid,'measured_files'].sum())
    return frame.loc[valid]

def threshold_flags(frame):
    directed = a.directed(frame)
    selected = directed[a.MIN_CHANGES.index].ge(a.MIN_CHANGES)
    # A full source replay is only needed for relative Halstead thresholds on the
    # gold/Agentless patches, which may touch different production files.
    return selected

def paired_descriptives(human, agentless):
    matched = ORIGINAL.loc[human.index]
    ai = a.directed(matched).gt(0).any(axis=1)
    h = a.directed(human).gt(0).any(axis=1)
    for cohort, mask in [('all',matched.repo.notna()),('excluding Django',matched.repo.ne('django/django'))]:
        m = mask[mask].index
        print('Paired',cohort,len(m), 'AI',int(ai.loc[m].sum()),'human',int(h.loc[m].sum()))
    print('Agentless adverse at any delta',int(a.directed(agentless).gt(0).any(axis=1).sum()))
    overlap = ORIGINAL.index.intersection(agentless.index)
    print('Agentless and original resolved overlap',len(overlap),
          'Gemini adverse',int(a.directed(ORIGINAL.loc[overlap]).gt(0).any(axis=1).sum()),
          'Agentless adverse',int(a.directed(agentless.loc[overlap]).gt(0).any(axis=1).sum()))

def main():
    verify_cohort()
    h=load('human',359)
    d=load('agentless',194)
    paired_descriptives(h,d)

if __name__ == '__main__':
    main()

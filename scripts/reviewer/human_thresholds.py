from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from radon.metrics import h_visit

import reviewer_stats as a
from replay_metrics import DATA, fetch_base, list_changed_files, is_python_file, is_test_file, as_text

def base_means():
    gold = pd.read_parquet(DATA/'verified.parquet').set_index('instance_id')
    def calculate(instance):
        info=gold.loc[instance]
        paths=[p for p in list_changed_files(info.patch) if is_python_file(p) and not is_test_file(p)]
        volumes=[h_visit(as_text(fetch_base(info.repo,info.base_commit,p))).total.volume for p in paths]
        return instance,float(np.mean(volumes))
    with ThreadPoolExecutor(max_workers=12) as pool:
        return pd.Series(dict(pool.map(calculate,a.DELTAS.index)))

def count(frame, halstead_base):
    d=a.directed(frame)
    for scale in [1,2]:
        selected=d[a.MIN_CHANGES.index].ge(a.MIN_CHANGES*scale)
        selected['delta_halstead_volume']=d.delta_halstead_volume.ge(np.maximum(1, halstead_base*0.05)*scale)
        print('threshold',scale,int(selected.any(axis=1).sum()), selected.sum().to_dict())

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-base', action='store_true')
    args = parser.parse_args()
    stored = DATA/'human_baseline_halstead.csv'
    if args.rebuild_base:
        (DATA/'blob_cache').mkdir(exist_ok=True)
        values = base_means().rename('before_halstead_volume')
        values.rename_axis('instance_id').to_csv(stored)
    base = pd.read_csv(stored, index_col='instance_id').before_halstead_volume
    gold = pd.read_json(DATA/'human_replay.jsonl',lines=True,precise_float=True).set_index('instance_id')
    assert len(gold)==359 and gold.status.eq('ok').all()
    assert set(base.index) == set(gold.index)
    count(gold, base)


if __name__ == '__main__':
    main()

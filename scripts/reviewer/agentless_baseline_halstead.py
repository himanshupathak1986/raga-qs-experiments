"""Re-extract baseline Halstead volumes for the Agentless line-budget check."""
from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd
from radon.metrics import h_visit

from replay_metrics import DATA, as_text, diff_with_headers, fetch_base, is_python_file, is_test_file, list_changed_files


def main():
    (DATA/'blob_cache').mkdir(exist_ok=True)
    verified = pd.read_parquet(DATA/'verified.parquet').set_index('instance_id')
    verdicts = json.loads((DATA/'result').read_text())
    predictions = {record['instance_id']: record['model_patch']
                   for record in map(json.loads, (DATA/'pred').read_text().splitlines())}

    def calculate(instance_id):
        meta = verified.loc[instance_id]
        paths = [p for p in list_changed_files(diff_with_headers(predictions[instance_id]))
                 if is_python_file(p) and not is_test_file(p)]
        volumes = [h_visit(as_text(fetch_base(meta.repo, meta.base_commit, p))).total.volume
                   for p in paths]
        assert volumes
        return instance_id, sum(volumes) / len(volumes)

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(calculate, sorted(verdicts['resolved'])))
    pd.DataFrame(rows,columns=['instance_id', 'before_halstead_volume']).to_csv(
        DATA/'agentless_baseline_halstead.csv', index=False
    )


if __name__ == '__main__':
    main()

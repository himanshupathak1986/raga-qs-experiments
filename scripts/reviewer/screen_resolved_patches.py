"""Compare pre- and post-patch Ruff security-rule counts on resolved Python patches."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import pandas as pd

from replay_metrics import (
    ARTIFACT, DATA, diff_with_headers, fetch_base, is_python_file,
    is_test_file, list_changed_files,
)

RUFF = shutil.which('ruff')


def findings(path):
    scan = subprocess.run(
        [RUFF, 'check', '--isolated', '--no-cache', '--ignore-noqa',
         '--select', 'S', '--output-format', 'json', str(path)],
        capture_output=True, text=True, check=False,
    )
    if scan.returncode not in (0, 1):
        raise RuntimeError(f'Ruff status {scan.returncode}: {scan.stderr.strip()}')
    diagnostics = json.loads(scan.stdout)
    if any(not item['code'].startswith('S') for item in diagnostics):
        raise RuntimeError('Ruff returned a diagnostic outside its security rules')
    return Counter(item['code'] for item in diagnostics)


def measure(job):
    cohort, iid, repo, commit, patch = job
    diff = diff_with_headers(patch)
    files = list_changed_files(diff)
    production = [f for f in files if is_python_file(f) and not is_test_file(f)]
    assert production
    assert all(not Path(f).is_absolute() and '..' not in Path(f).parts for f in files)
    assert not any(line.startswith(('new file mode ', 'deleted file mode ', 'rename from ', 'copy from '))
                   for line in diff.splitlines())
    before, after = Counter(), Counter()
    with TemporaryDirectory(prefix='resolved-screen-') as tmp:
        root = Path(tmp)
        for f in files:
            path = root / f
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch_base(repo, commit, f))
        increases = 0
        by_file = {f: findings(root/f) for f in production}
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', '--whitespace=nowarn', '-'],
                       input=diff.encode(), capture_output=True, cwd=root, check=True)
        for f in production:
            changed = findings(root/f)
            increases += sum((changed - by_file[f]).values())
            before.update(by_file[f])
            after.update(changed)
    return {
        'cohort': cohort, 'instance_id': iid, 'production_files': len(production),
        'pre_flags': before.total(), 'post_flags': after.total(),
        'flag_count_increases_per_file_rule': increases,
        'flag_count_increases_per_patch_rule': sum((after - before).values()),
    }


def load_resolved(metadata, label, predictions_name, verdict_name):
    records = [json.loads(line) for line in (DATA/predictions_name).read_text().splitlines()]
    preds = {record['instance_id']: record['model_patch'] for record in records}
    assert len(preds) == len(records)
    solved = json.loads((DATA/verdict_name).read_text())['resolved']
    return [(label, iid, metadata.at[iid, 'repo'], metadata.at[iid, 'base_commit'], preds[iid])
            for iid in sorted(solved)]


def main():
    if RUFF is None:
        raise RuntimeError('Ruff must be on PATH')
    (DATA/'blob_cache').mkdir(exist_ok=True)
    metadata = pd.read_parquet(DATA/'verified.parquet').set_index('instance_id')
    jobs = load_resolved(metadata, 'Gemini/mini', 'original_pred', 'original_result')
    jobs += load_resolved(metadata, 'GPT-4o/Agentless', 'pred', 'result')
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(measure, jobs))
    out = pd.DataFrame(rows)
    assert not out.duplicated(['cohort', 'instance_id']).any()
    assert out.cohort.value_counts().to_dict() == {'Gemini/mini': 359, 'GPT-4o/Agentless': 194}
    out.to_csv(ARTIFACT/'results/reviewer/resolved_static_security_screen.csv', index=False)
    print(out.groupby('cohort').agg(
        patches=('instance_id', 'size'), files=('production_files', 'sum'),
        pre=('pre_flags', 'sum'), post=('post_flags', 'sum'),
        patches_with_increase=('flag_count_increases_per_file_rule', lambda s: (s > 0).sum()),
        added_counts=('flag_count_increases_per_file_rule', 'sum'),
        patches_with_net_rule_increase=('flag_count_increases_per_patch_rule', lambda s: (s > 0).sum()),
    ).to_string())


if __name__ == '__main__':
    main()

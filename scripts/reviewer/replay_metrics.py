import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import is_python_file, is_test_file
from extract_patch_metrics import patch_metrics
from extract_quality_metrics import delta, list_changed_files, source_metrics

DATA = Path(__file__).resolve().parents[2] / '.cache' / 'reviewer'
ARTIFACT = Path(__file__).resolve().parents[2]
METRICS = ['cc_mean', 'cc_max', 'mi', 'halstead_volume', 'cognitive_mean',
           'cognitive_max', 'max_nesting', 'function_length_mean',
           'function_length_max', 'ruff_findings']


def diff_with_headers(diff):
    # Some Agentless diffs are unified diffs with ---/+++ and no diff --git line.
    # The original AST extractor requires that header. Add it to the diff used
    # for measurement while preserving the actual patch body.
    if list_changed_files(diff):
        return diff
    lines = diff.splitlines(keepends=True)
    output = []
    for i, line in enumerate(lines):
        if line.startswith('--- a/') and i+1 < len(lines) and lines[i+1].startswith('+++ b/'):
            a = line.strip()[4:].split('\t')[0]
            b = lines[i+1].strip()[4:].split('\t')[0]
            output.append(f'diff --git {a} {b}\n')
        output.append(line)
    return ''.join(output)


def fetch_base(repo, commit, path):
    key = hashlib.sha256(f'{repo}/{commit}/{path}'.encode()).hexdigest()
    cache_path = DATA / 'blob_cache' / key
    if cache_path.is_file():
        return cache_path.read_bytes()
    url = 'https://raw.githubusercontent.com/' + quote(f'{repo}/{commit}/{path}', safe='/')
    last_error = None
    for _ in range(3):
        try:
            response = requests.get(url, timeout=(6, 15))
            response.raise_for_status()
            cache_path.write_bytes(response.content)
            return response.content
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f'{repo} {commit} {path}: {last_error}')


def as_text(data):
    return io.TextIOWrapper(io.BytesIO(data), encoding='utf-8', errors='replace').read()


def measure(job):
    label, iid, repo, commit, diff = job
    result = {'sample': label, 'instance_id': iid, 'repo': repo, 'base_commit': commit}
    diff = diff_with_headers(diff)
    files = list_changed_files(diff)
    production = [file for file in files if is_python_file(file) and not is_test_file(file)]
    result.update(patch_metrics(diff, []))
    result['measured_files'] = len(production)
    try:
        assert production, 'no production Python files'
        assert all(not Path(f).is_absolute() and '..' not in Path(f).parts for f in files)
        assert not any(line.startswith(('new file mode ', 'deleted file mode ', 'rename from ', 'copy from ')) for line in diff.splitlines()), 'file operations need full worktree'
        metrics = []
        with TemporaryDirectory(prefix='112f-quality-') as tmp:
            target = Path(tmp)
            base = {}
            for f in files:
                data = fetch_base(repo, commit, f)
                file = target / f
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
                base[f] = data
            applied = subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', '--whitespace=nowarn', '-'],
                                     input=diff.encode(), capture_output=True, cwd=target)
            if applied.returncode:
                raise RuntimeError(applied.stderr.decode(errors='replace').strip()[:350])
            for f in production:
                before = source_metrics(as_text(base[f]))
                after = source_metrics((target / f).read_text(encoding='utf-8', errors='replace'))
                if before['metric_error'] or after['metric_error']:
                    raise RuntimeError(f"{f}: {before['metric_error']} | {after['metric_error']}")
                metrics.append({f'delta_{name}': delta(after[name], before[name]) for name in METRICS})
        for metric, value in pd.DataFrame(metrics).mean().items():
            result[metric] = value
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = str(exc)
    return result


def main():
    (DATA/'blob_cache').mkdir(exist_ok=True)
    verified = pd.read_parquet(DATA/'verified.parquet').set_index('instance_id')
    original_metrics = pd.read_csv(ARTIFACT/'data/processed/quality_metrics_patch.csv').set_index('instance_id')
    original = {x['instance_id']: x['model_patch'] for line in (DATA/'original_pred').read_text().splitlines() if (x:=json.loads(line))}
    alternate = {x['instance_id']: x['model_patch'] for line in (DATA/'pred').read_text().splitlines() if (x:=json.loads(line))}
    alternate_resolved = set(json.loads((DATA/'result').read_text())['resolved'])
    plans = {
        'original': {iid: original[iid] for iid in original_metrics.index},
        'human': verified.loc[original_metrics.index, 'patch'].to_dict(),
        'agentless': {iid: alternate[iid] for iid in sorted(alternate_resolved)},
    }
    labels = sys.argv[1:] or list(plans)
    jobs = []
    for label in labels:
        for iid,diff in plans[label].items():
            info = verified.loc[iid]
            jobs.append((label, iid, info.repo, info.base_commit, diff))
    for label in labels:
        label_jobs = [job for job in jobs if job[0] == label]
        with ThreadPoolExecutor(max_workers=20) as pool:
            with (DATA/f'{label}_replay.jsonl').open('w', encoding='utf-8') as out:
                for i, result in enumerate(pool.map(measure, label_jobs), start=1):
                    out.write(json.dumps(result)+'\n')
                    out.flush()
                    if i % 30 == 0 or result['status'] != 'ok':
                        print(f"{label} {i}/{len(label_jobs)} {result['instance_id']} {result['status'][:300]}", flush=True)
        print('finished',label,len(label_jobs),flush=True)


if __name__ == '__main__':
    main()

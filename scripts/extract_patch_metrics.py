from __future__ import annotations

import argparse
import re

import pandas as pd

from common import ROOT, ensure_dirs, is_python_file, is_test_file, load_config, read_jsonl


CONTROL_RE = re.compile(r"^\s*(if|elif|for|while|except|case)\b|\b(and|or)\b")
IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)")


def changed_file_from_diff_header(line: str) -> str | None:
    if not line.startswith("diff --git "):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    b = parts[3]
    return b[2:] if b.startswith("b/") else b


def patch_metrics(patch: str, sensitive_patterns: list[str]) -> dict:
    compiled_sensitive = [re.compile(p) for p in sensitive_patterns]

    additions = deletions = hunks = 0
    imports_added = imports_deleted = 0
    control_added = control_deleted = 0
    sensitive_added = 0
    changed_files: list[str] = []

    for line in patch.splitlines():
        path = changed_file_from_diff_header(line)
        if path is not None:
            changed_files.append(path)
            continue

        if line.startswith("@@"):
            hunks += 1
            continue

        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
            code = line[1:]
            imports_added += int(bool(IMPORT_RE.search(code)))
            control_added += int(bool(CONTROL_RE.search(code)))
            sensitive_added += sum(bool(p.search(code)) for p in compiled_sensitive)

        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            code = line[1:]
            imports_deleted += int(bool(IMPORT_RE.search(code)))
            control_deleted += int(bool(CONTROL_RE.search(code)))

    unique_files = list(dict.fromkeys(changed_files))
    py_files = [p for p in unique_files if is_python_file(p)]
    test_files = [p for p in py_files if is_test_file(p)]
    prod_py_files = [p for p in py_files if not is_test_file(p)]

    return {
        "additions": additions,
        "deletions": deletions,
        "churn": additions + deletions,
        "hunks": hunks,
        "files_changed": len(unique_files),
        "python_files_changed": len(py_files),
        "production_python_files_changed": len(prod_py_files),
        "test_python_files_changed": len(test_files),
        "touches_tests": int(bool(test_files)),
        "imports_added": imports_added,
        "imports_deleted": imports_deleted,
        "import_delta": imports_added - imports_deleted,
        "control_tokens_added": control_added,
        "control_tokens_deleted": control_deleted,
        "control_token_delta": control_added - control_deleted,
        "security_sensitive_api_additions": sensitive_added,
        "changed_files": ";".join(unique_files),
        "production_python_files": ";".join(prod_py_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)
    cohort = ROOT / "data" / "processed" / "swebench_cohort.jsonl"
    if not cohort.exists():
        raise SystemExit("Missing SWE-bench cohort. Run scripts/collect_swebench.py first.")

    patterns = cfg["quality"]["security_sensitive_patterns"]
    rows = []
    for record in read_jsonl(cohort):
        rows.append(
            {
                "instance_id": record["instance_id"],
                "model_name_or_path": record.get("model_name_or_path"),
                "repo": record.get("repo"),
                "base_commit": record.get("base_commit"),
                "resolved": bool(record.get("resolved")),
                **patch_metrics(record.get("model_patch", ""), patterns),
            }
        )

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "processed" / "patch_metrics.csv"
    df.to_csv(out, index=False)

    summary = (
        df.groupby("resolved")
        .agg(
            n=("instance_id", "size"),
            churn_mean=("churn", "mean"),
            churn_median=("churn", "median"),
            files_mean=("files_changed", "mean"),
            files_median=("files_changed", "median"),
            hunks_mean=("hunks", "mean"),
            hunks_median=("hunks", "median"),
            control_delta_mean=("control_token_delta", "mean"),
            sensitive_addition_rate=("security_sensitive_api_additions", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    summary_path = ROOT / "results" / "tables" / "patch_shape_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(df)} patch rows to {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

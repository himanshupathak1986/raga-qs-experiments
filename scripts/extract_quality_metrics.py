from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit

try:
    from cognitive_complexity.api import get_cognitive_complexity
except Exception:
    get_cognitive_complexity = None

from common import ROOT, ensure_dirs, is_python_file, is_test_file, load_config, read_jsonl, run


def list_changed_files(patch: str) -> list[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                files.append(path)
    return list(dict.fromkeys(files))


def ast_max_nesting(node: ast.AST) -> int:
    nesting_types = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
        ast.With, ast.AsyncWith, ast.Match,
    )

    max_depth = 0

    def walk(n: ast.AST, depth: int) -> None:
        nonlocal max_depth
        next_depth = depth + 1 if isinstance(n, nesting_types) else depth
        max_depth = max(max_depth, next_depth)
        for child in ast.iter_child_nodes(n):
            walk(child, next_depth)

    walk(node, 0)
    return max_depth


def function_lengths(tree: ast.AST) -> list[int]:
    lengths = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            lengths.append(max(1, int(end) - int(node.lineno) + 1))
    return lengths


def cognitive_scores(tree: ast.AST) -> list[float]:
    if get_cognitive_complexity is None:
        return []
    values = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                values.append(float(get_cognitive_complexity(node)))
            except Exception:
                pass
    return values


def ruff_count(source: str) -> tuple[float, str | None]:
    if not shutil.which("ruff"):
        return math.nan, "ruff-not-installed"
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
        f.write(source)
        temp_path = Path(f.name)
    try:
        proc = run(["ruff", "check", "--output-format", "json", str(temp_path)], check=False)
        if proc.returncode not in (0, 1):
            return math.nan, proc.stderr.strip()[:500] or "ruff-error"
        try:
            findings = json.loads(proc.stdout or "[]")
            return float(len(findings)), None
        except json.JSONDecodeError:
            return math.nan, "ruff-json-parse-error"
    finally:
        temp_path.unlink(missing_ok=True)


def source_metrics(source: str) -> dict:
    if not source.strip():
        return {
            "loc": 0.0,
            "cc_mean": math.nan,
            "cc_max": math.nan,
            "mi": math.nan,
            "halstead_volume": math.nan,
            "cognitive_mean": math.nan,
            "cognitive_max": math.nan,
            "max_nesting": math.nan,
            "function_count": 0.0,
            "function_length_mean": math.nan,
            "function_length_max": math.nan,
            "import_count": 0.0,
            "ruff_findings": math.nan,
            "metric_error": None,
        }

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {
            "loc": float(len(source.splitlines())),
            "cc_mean": math.nan,
            "cc_max": math.nan,
            "mi": math.nan,
            "halstead_volume": math.nan,
            "cognitive_mean": math.nan,
            "cognitive_max": math.nan,
            "max_nesting": math.nan,
            "function_count": math.nan,
            "function_length_mean": math.nan,
            "function_length_max": math.nan,
            "import_count": math.nan,
            "ruff_findings": math.nan,
            "metric_error": f"syntax-error: {e}",
        }

    blocks = cc_visit(source)
    cc_values = [float(b.complexity) for b in blocks]
    lengths = function_lengths(tree)
    cog = cognitive_scores(tree)
    imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))

    try:
        mi = float(mi_visit(source, multi=True))
    except Exception:
        mi = math.nan

    try:
        h = h_visit(source)
        volume = float(h.total.volume)
    except Exception:
        volume = math.nan

    ruff, ruff_error = ruff_count(source)

    return {
        "loc": float(len(source.splitlines())),
        "cc_mean": sum(cc_values) / len(cc_values) if cc_values else 0.0,
        "cc_max": max(cc_values) if cc_values else 0.0,
        "mi": mi,
        "halstead_volume": volume,
        "cognitive_mean": sum(cog) / len(cog) if cog else (0.0 if get_cognitive_complexity else math.nan),
        "cognitive_max": max(cog) if cog else (0.0 if get_cognitive_complexity else math.nan),
        "max_nesting": float(ast_max_nesting(tree)),
        "function_count": float(len(lengths)),
        "function_length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "function_length_max": float(max(lengths)) if lengths else 0.0,
        "import_count": float(imports),
        "ruff_findings": ruff,
        "metric_error": ruff_error,
    }


def mirror_path(repo: str) -> Path:
    safe = repo.replace("/", "__")
    return ROOT / ".cache" / "repos" / f"{safe}.git"


def ensure_mirror(repo: str) -> Path:
    mirror = mirror_path(repo)
    if mirror.exists():
        run(["git", "--git-dir", str(mirror), "fetch", "--prune", "origin"], check=False)
        return mirror

    mirror.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--mirror", f"https://github.com/{repo}.git", str(mirror)])
    return mirror


def ensure_commit(mirror: Path, commit: str) -> None:
    check = run(["git", "--git-dir", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    if check.returncode == 0:
        return
    run(["git", "--git-dir", str(mirror), "fetch", "origin", commit])


def show_file(mirror: Path, commit: str, path: str) -> str | None:
    proc = run(["git", "--git-dir", str(mirror), "show", f"{commit}:{path}"], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout


def materialize_after_files(mirror: Path, commit: str, patch: str, paths: list[str]) -> tuple[dict[str, str | None], str | None]:
    with tempfile.TemporaryDirectory(prefix="swepatch-") as td:
        work = Path(td) / "repo"
        clone = run(["git", "clone", "--no-checkout", str(mirror), str(work)], check=False)
        if clone.returncode != 0:
            return {}, f"clone-failed: {clone.stderr[:400]}"

        checkout = run(["git", "checkout", "--detach", commit], cwd=work, check=False)
        if checkout.returncode != 0:
            return {}, f"checkout-failed: {checkout.stderr[:400]}"

        patch_path = Path(td) / "patch.diff"
        patch_path.write_text(patch, encoding="utf-8")
        applied = run(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=work, check=False)
        if applied.returncode != 0:
            return {}, f"git-apply-failed: {applied.stderr[:600]}"

        outputs = {}
        for rel in paths:
            p = work / rel
            outputs[rel] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        return outputs, None


def delta(after: float, before: float) -> float:
    try:
        if math.isnan(after) or math.isnan(before):
            return math.nan
    except TypeError:
        return math.nan
    return after - before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--only-resolved", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)
    exclude_tests = bool(cfg["swebench"].get("exclude_test_files_from_quality", True))

    cohort_path = ROOT / "data" / "processed" / "swebench_cohort.jsonl"
    if not cohort_path.exists():
        raise SystemExit("Missing SWE-bench cohort. Run scripts/collect_swebench.py first.")

    records = list(read_jsonl(cohort_path))
    if args.only_resolved:
        records = [r for r in records if bool(r.get("resolved"))]
    if args.limit is not None:
        records = records[: args.limit]

    metric_names = [
        "loc", "cc_mean", "cc_max", "mi", "halstead_volume",
        "cognitive_mean", "cognitive_max", "max_nesting",
        "function_count", "function_length_mean", "function_length_max",
        "import_count", "ruff_findings",
    ]

    rows = []
    for i, rec in enumerate(records, start=1):
        instance_id = rec["instance_id"]
        repo = rec.get("repo")
        commit = rec.get("base_commit")
        patch = rec.get("model_patch", "")

        if not repo or not commit or not patch.strip():
            rows.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": commit,
                    "resolved": bool(rec.get("resolved")),
                    "file_path": None,
                    "status": "missing-input",
                    "tool_error": "repo/base_commit/patch missing",
                }
            )
            continue

        paths = [p for p in list_changed_files(patch) if is_python_file(p)]
        if exclude_tests:
            paths = [p for p in paths if not is_test_file(p)]

        if not paths:
            rows.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": commit,
                    "resolved": bool(rec.get("resolved")),
                    "file_path": None,
                    "status": "no-production-python-files",
                    "tool_error": None,
                }
            )
            continue

        try:
            mirror = ensure_mirror(repo)
            ensure_commit(mirror, commit)
            after_map, apply_error = materialize_after_files(mirror, commit, patch, paths)
        except Exception as e:
            rows.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": commit,
                    "resolved": bool(rec.get("resolved")),
                    "file_path": None,
                    "status": "repository-error",
                    "tool_error": str(e)[:1000],
                }
            )
            continue

        if apply_error:
            rows.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": commit,
                    "resolved": bool(rec.get("resolved")),
                    "file_path": None,
                    "status": "patch-apply-error",
                    "tool_error": apply_error,
                }
            )
            continue

        for path in paths:
            before_src = show_file(mirror, commit, path) or ""
            after_src = after_map.get(path) or ""
            before = source_metrics(before_src)
            after = source_metrics(after_src)

            row = {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": commit,
                "resolved": bool(rec.get("resolved")),
                "file_path": path,
                "status": "ok",
                "tool_error": " | ".join(
                    x for x in [before.get("metric_error"), after.get("metric_error")] if x
                ) or None,
            }
            for name in metric_names:
                b = before.get(name, math.nan)
                a = after.get(name, math.nan)
                row[f"before_{name}"] = b
                row[f"after_{name}"] = a
                row[f"delta_{name}"] = delta(a, b)

            rows.append(row)

        print(f"[{i}/{len(records)}] {instance_id}")

    out = ROOT / "data" / "processed" / "quality_metrics_long.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} file-level quality rows to {out}")


if __name__ == "__main__":
    main()

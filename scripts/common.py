from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
from typing import Iterable, Iterator, Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "experiment.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for rel in [
        "data/raw",
        "data/processed",
        "data/manifests",
        "results/tables",
        "results/figures",
        "results/statistics",
        ".cache/repos",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(name: str, payload: dict) -> Path:
    ensure_dirs()
    out = ROOT / "data" / "manifests" / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


_TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)(/|_|\.|$)|(^|/)test_.*\.py$|_test\.py$",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE_RE.search(path.replace("\\", "/")))


def is_python_file(path: str) -> bool:
    return path.lower().endswith(".py")

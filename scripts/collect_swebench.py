from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode

import requests

from common import ROOT, ensure_dirs, load_config, sha256_file, write_jsonl, write_manifest


def download(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def fetch_verified_metadata(cfg: dict) -> list[dict]:
    endpoint = cfg["dataset_server"]
    dataset = cfg["dataset_name"]
    config = cfg.get("dataset_config", "default")
    split = cfg.get("dataset_split", "test")
    page_size = 100

    rows: list[dict] = []
    expected_total = None
    while expected_total is None or len(rows) < expected_total:
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": len(rows),
            "length": page_size,
        }
        response = requests.get(f"{endpoint}?{urlencode(params)}", timeout=120)
        response.raise_for_status()
        payload = response.json()
        page = [item["row"] for item in payload["rows"]]
        total = int(payload["num_rows_total"])
        if expected_total is None:
            expected_total = total
        elif expected_total != total:
            raise RuntimeError("Dataset row count changed during pagination.")
        if not page and len(rows) < total:
            raise RuntimeError("Dataset server returned an empty page before completion.")
        rows.extend(page)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)["swebench"]
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"

    predictions_path = raw / "swebench_predictions.jsonl"
    results_path = raw / "swebench_results.json"
    metadata_path = raw / "swebench_verified_metadata.jsonl"

    download(cfg["predictions_url"], predictions_path)
    download(cfg["results_url"], results_path)

    metadata_rows = fetch_verified_metadata(cfg)
    write_jsonl(metadata_path, metadata_rows)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    resolved = set(results.get("resolved", []))
    no_generation = set(results.get("no_generation", []))
    no_logs = set(results.get("no_logs", []))
    meta_by_id = {x["instance_id"]: x for x in metadata_rows}

    joined: list[dict] = []
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pred = json.loads(line)
            instance_id = pred["instance_id"]
            meta = meta_by_id.get(instance_id, {})
            joined.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": pred.get("model_name_or_path"),
                    "model_patch": pred.get("model_patch") or "",
                    "resolved": instance_id in resolved,
                    "no_generation": instance_id in no_generation,
                    "no_logs": instance_id in no_logs,
                    "repo": meta.get("repo"),
                    "base_commit": meta.get("base_commit"),
                    "difficulty": meta.get("difficulty"),
                    "created_at": meta.get("created_at"),
                }
            )

    cohort_path = processed / "swebench_cohort.jsonl"
    write_jsonl(cohort_path, joined)

    retrieved = datetime.now(timezone.utc).isoformat()
    write_manifest(
        "swebench",
        {
            "retrieved_at_utc": retrieved,
            "artifact_repo": cfg["artifact_repo"],
            "predictions_url": cfg["predictions_url"],
            "results_url": cfg["results_url"],
            "dataset_name": cfg["dataset_name"],
            "row_count_predictions": len(joined),
            "row_count_metadata": len(metadata_rows),
            "resolved_predictions": sum(bool(x["resolved"]) for x in joined),
            "files": {
                str(predictions_path.relative_to(ROOT)): sha256_file(predictions_path),
                str(results_path.relative_to(ROOT)): sha256_file(results_path),
                str(metadata_path.relative_to(ROOT)): sha256_file(metadata_path),
                str(cohort_path.relative_to(ROOT)): sha256_file(cohort_path),
            },
        },
    )

    print(f"Wrote {len(joined)} generated patches to {cohort_path}")
    print(f"Resolved: {sum(bool(x['resolved']) for x in joined)}")


if __name__ == "__main__":
    main()

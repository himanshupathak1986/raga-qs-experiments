from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import ROOT, ensure_dirs, load_config, sha256_file, write_manifest


def download(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config(args.config)["securityeval"]
    raw = ROOT / "data" / "raw"

    outputs = {}
    for model, key in [("copilot", "copilot_csv"), ("incoder", "incoder_csv")]:
        path = raw / f"securityeval_{model}.csv"
        download(cfg[key], path)
        outputs[str(path.relative_to(ROOT))] = sha256_file(path)

    write_manifest(
        "securityeval",
        {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": cfg,
            "files": outputs,
        },
    )

    print("Downloaded SecurityEval published result tables.")


if __name__ == "__main__":
    main()

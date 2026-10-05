# Reproducibility

## Zero-cost constraint

The default workflow does not call any paid model API or cloud compute service.

Public HTTP downloads, public Git repositories, and local open-source tools are sufficient.

## Environment

Recommended:

- Python 3.11+
- Git 2.40+
- macOS or Linux
- sufficient disk space for public repository mirrors used by the quality stage

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Provenance

Every processed row retains source identifiers such as the benchmark instance ID, model/run name, repository, and base commit.

The raw-data directory is excluded from version control by default. To freeze an experiment, save:

- source URLs
- retrieval date
- commit SHA or release/tag when available
- processed file checksum
- exact command used

## Smoke testing

Before a full quality extraction, run:

```bash
pytest -q
python scripts/collect_swebench.py
python scripts/extract_patch_metrics.py
python scripts/extract_quality_metrics.py --only-resolved --limit 10
```

Review `data/processed/quality_metrics_long.csv` and the tool-error column before scaling up.

## No silent exclusions

A row is not silently dropped because a parser or tool fails. The quality pipeline records status and error text for later accounting.

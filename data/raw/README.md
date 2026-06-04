# Raw Data

This directory is for downloaded source files such as UniProt, Rhea, and Gene
Ontology exports. Raw downloads are ignored by Git.

Populate this directory with:

```bash
PYTHONPATH=src python scripts/00_download_sources.py --config configs/config.yaml
```

Manual curation tables are tracked separately in `data/manual/`.

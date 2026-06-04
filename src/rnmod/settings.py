from __future__ import annotations

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
# This is a library module. Change routine pipeline parameters in
# configs/config.yaml or in the numbered scripts under scripts/.

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "configs/config.yaml"


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def project_path(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing config paths.{key}") from exc
    return Path(value)


def interim_path(config: dict[str, Any], source_name: str, filename: str) -> Path:
    return project_path(config, "interim_dir") / source_name / filename


def processed_path(config: dict[str, Any], filename: str) -> Path:
    return project_path(config, "processed_dir") / filename

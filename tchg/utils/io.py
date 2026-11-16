from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:

    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:

    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def ensure_dir(path: str | Path) -> Path:

    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:

    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:

    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = deep_update(load_config(parent), cfg)
    return cfg


def save_json(obj: dict[str, Any], path: str | Path) -> None:

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)


def read_csv(path: str | Path) -> pd.DataFrame:

    return pd.read_csv(resolve_path(path))

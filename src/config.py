from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RAW_COLUMNS = ["source", "username", "text", "created_at"]


def load_env() -> dict[str, str]:
    """Load .env if python-dotenv is installed, then return relevant env vars."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    return {
        "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY", ""),
        "X_BEARER_TOKEN": os.getenv("X_BEARER_TOKEN", ""),
        "APIFY_TOKEN": os.getenv("APIFY_TOKEN", ""),
    }


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["env"] = load_env()
    config["project_root"] = PROJECT_ROOT
    ensure_directories(config)
    return config


def project_path(config: dict[str, Any], *parts: str | Path) -> Path:
    root = Path(config.get("project_root", PROJECT_ROOT))
    return root.joinpath(*map(str, parts))


def configured_path(config: dict[str, Any], key: str) -> Path:
    raw_path = config.get("paths", {}).get(key)
    if not raw_path:
        raise KeyError(f"Missing paths.{key} in config.yaml")
    return project_path(config, raw_path)


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ["raw_dir", "manual_raw_dir", "processed_dir", "results_dir", "figures_dir"]:
        raw_path = config.get("paths", {}).get(key)
        if raw_path:
            project_path(config, raw_path).mkdir(parents=True, exist_ok=True)


def output_path(config: dict[str, Any], section: str, filename: str) -> Path:
    key_by_section = {
        "raw": "raw_dir",
        "processed": "processed_dir",
        "results": "results_dir",
        "figures": "figures_dir",
    }
    if section not in key_by_section:
        raise ValueError(f"Unknown output section: {section}")
    return configured_path(config, key_by_section[section]) / filename

"""Locations of data that lives outside the repo."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """$DATA_DIR from the environment, falling back to the repo-root .env."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get("DATA_DIR")
    if not value:
        raise RuntimeError("DATA_DIR is not set (env var or repo-root .env)")
    return Path(value).expanduser()

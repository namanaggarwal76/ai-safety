"""Read API keys from a .env file so they don't have to be exported or typed into the dashboard."""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env", override: bool = False) -> list[str]:
    """Set variables from KEY=value lines. Blank values are skipped, and variables already
    set in the environment win unless ``override``. Returns the names that were set."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not key or not value or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded

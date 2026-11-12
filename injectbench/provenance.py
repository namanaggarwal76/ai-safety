"""What produced a run: code version, scenario and technique content, seed and sampling settings."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .defenses import DEFENSES
from .scenarios import SCENARIOS
from .techniques import Technique


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), *args],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def scenarios_digest() -> str:
    """Hash of everything a model sees or a judge uses that can be serialised."""
    return _sha({
        sc.id: {
            "system": sc.system_prompt, "user": sc.user_message, "data": sc.data,
            "tools": [t.to_api() for t in sc.tools],
            "points": [(p.id, p.channel, list(p.positions or ())) for p in sc.injection_points],
            "surface": {"canary": sc.surface.canary, "hint": sc.surface.secret_hint,
                        "email": sc.surface.attacker_email, "url": sc.surface.attacker_url,
                        "egress": sc.surface.egress_tools,
                        "actions": [(a.id, a.instruction) for a in sc.surface.actions]},
        } for sc in SCENARIOS.values()
    })


def techniques_digest(techniques: dict[str, Technique]) -> str:
    return _sha({t.id: t.render("<<INSTRUCTION>>") for t in techniques.values()})


def defenses_digest() -> str:
    return _sha({d.id: [d().system_prompt("<<S>>"), d().user_message("<<U>>"), d().wrap_tool_result("t", "<<R>>")]
                 for d in DEFENSES.values()})


def provenance(techniques: dict[str, Technique], **settings: Any) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    return {
        "injectbench_version": __version__,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenarios_sha": scenarios_digest(),
        "techniques_sha": techniques_digest(techniques),
        "defenses_sha": defenses_digest(),
        **settings,
    }

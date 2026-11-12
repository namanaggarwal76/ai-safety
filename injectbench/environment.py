"""Sandboxed mock environment: executes tools against scenario data and records calls."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from .models import AgentTask, ToolCall

if TYPE_CHECKING:
    from .defenses import Defense
    from .goals import Goal
    from .scenario import ForbiddenAction, Scenario


class Environment:
    """A private copy of a scenario's data plus a log of every tool call."""

    def __init__(self, scenario: "Scenario", defense: Optional["Defense"] = None,
                 canary_seed: Optional[str] = None) -> None:
        self.scenario = scenario
        self.data: dict = copy.deepcopy(scenario.data)
        # the secret this case uses; a seed gives every case its own, so published
        # scenario files never contain the values models are tested on
        self.canary: str = scenario.surface.canary
        if canary_seed is not None:
            self.canary = fresh_canary(scenario.surface.canary, canary_seed)
            self.data = _replace_all(self.data, scenario.surface.canary, self.canary)
        self.calls: list[ToolCall] = []
        self.defense = defense
        self.attack: Optional[tuple["Goal", Optional["ForbiddenAction"]]] = None
        self.injected_payload: Optional[str] = None
        self.injection_point_id: Optional[str] = None
        # set once a tool result shown to the agent contains the payload
        self.exposed: bool = False
        self.listener: Optional[Callable[["Environment", ToolCall], None]] = None


    def inject(self, point_id: str, payload: str, position: str = "end") -> None:
        point = self.scenario.point(point_id)
        point.place(self.data, payload, position)
        self.injected_payload = payload
        self.injection_point_id = point_id

    def task(self) -> AgentTask:
        system = self.scenario.system_prompt
        user = self.scenario.user_message
        if self.defense is not None:
            system = self.defense.system_prompt(system)
            user = self.defense.user_message(user)
        return AgentTask(system_prompt=system, user_message=user, tools=list(self.scenario.tools))


    def call_tool(self, name: str, args: Optional[dict] = None) -> ToolCall:
        args = dict(args or {})
        handler = self.scenario.handlers.get(name)
        if handler is None:
            call = ToolCall(name=name, args=args, result=f"Error: unknown tool '{name}'", is_error=True)
            self.calls.append(call)
            self._notify(call)
            return call
        try:
            raw = handler(self, args)
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False, default=str)
            is_error = False
        except Exception as exc:
            raw = f"Error: {exc}"
            is_error = True
        # error text can carry attacker content too, so it counts and gets wrapped
        if self.injected_payload and not self.exposed:
            self.exposed = _normalize(self.injected_payload) in _normalize(raw)
        rendered = raw
        if self.defense is not None:
            rendered = self.defense.wrap_tool_result(name, raw)
        call = ToolCall(name=name, args=args, result=rendered, is_error=is_error)
        self.calls.append(call)
        self._notify(call)
        return call

    def _notify(self, call: ToolCall) -> None:
        if self.listener is not None:
            try:
                self.listener(self, call)
            except Exception:
                pass


    @staticmethod
    def require(args: dict, *keys: str) -> None:
        missing = [k for k in keys if k not in args or args[k] in (None, "")]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")

    def dump(self, obj: Any) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _normalize(text: str) -> str:
    text = text.replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")
    return " ".join(text.split())


def fresh_canary(template: str, seed: str) -> str:
    """Same shape as the template, with every letter and digit redrawn except a
    leading lowercase prefix such as ``pat_`` or ``dpl_`` that makes it look real."""
    rng = random.Random(hashlib.sha256(f"{template}|{seed}".encode()).digest())
    m = re.match(r"(?:[a-z]+_)+", template)
    start = m.end() if m else 0
    out = list(template)
    for i in range(start, len(out)):
        c = out[i]
        if c.isdigit():
            out[i] = rng.choice("0123456789")
        elif c.isupper():
            out[i] = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        elif c.islower():
            out[i] = rng.choice("abcdefghijkmnpqrstuvwxyz")
    fresh = "".join(out)
    return fresh if fresh != template else fresh_canary(template, seed + "'")


def _replace_all(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, dict):
        return {k: _replace_all(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_all(v, old, new) for v in obj]
    return obj

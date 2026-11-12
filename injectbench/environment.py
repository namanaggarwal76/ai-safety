"""Sandboxed mock environment: executes tools against scenario data and records calls."""
from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from .models import AgentTask, ToolCall

if TYPE_CHECKING:
    from .defenses import Defense
    from .goals import Goal
    from .scenario import ForbiddenAction, Scenario


class Environment:
    """A private copy of a scenario's data plus a log of every tool call."""

    def __init__(self, scenario: "Scenario", defense: Optional["Defense"] = None) -> None:
        self.scenario = scenario
        self.data: dict = copy.deepcopy(scenario.data)
        self.calls: list[ToolCall] = []
        self.defense = defense
        self.attack: Optional[tuple["Goal", Optional["ForbiddenAction"]]] = None
        self.injected_payload: Optional[str] = None
        self.injection_point_id: Optional[str] = None
        # set once a tool result shown to the agent contains the payload
        self.exposed: bool = False
        self.listener: Optional[Callable[["Environment", ToolCall], None]] = None


    def inject(self, point_id: str, payload: str) -> None:
        point = self.scenario.point(point_id)
        point.apply(self.data, payload)
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
        if self.injected_payload and not is_error and not self.exposed:
            self.exposed = _normalize(self.injected_payload) in _normalize(raw)
        rendered = raw
        if self.defense is not None and not is_error:
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

"""Scenario definitions: the benign task, mock data, tools, and attack surface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .models import AgentTrace, ToolCall, ToolSpec

if TYPE_CHECKING:
    from .environment import Environment

CHANNELS = ("document", "email", "tool_response", "web_page", "calendar", "code_comment", "chat_message")


@dataclass
class InjectionPoint:
    """A place in the scenario's mock data where attacker text can be planted."""

    id: str
    channel: str
    description: str
    apply: Callable[[dict, str], None]  # mutates env data in place

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"unknown channel {self.channel!r}; expected one of {CHANNELS}")


@dataclass
class ForbiddenAction:
    """A side effect the user never asked for. demo() performs it, for the self-check."""

    id: str
    description: str
    instruction: str
    detect: Callable[[AgentTrace], bool]
    demo: Callable[["Environment"], str]


@dataclass
class AttackSurface:
    canary: str  # the secret; it must never show up in agent output
    secret_hint: str
    attacker_email: str
    attacker_url: str
    egress_tools: list[str]  # tools that send data out
    read_secret: Callable[["Environment"], None]
    exfil_via_tool: Callable[["Environment", str], None]
    actions: list[ForbiddenAction] = field(default_factory=list)

    def egress_calls(self, trace: AgentTrace) -> list[ToolCall]:
        return [c for c in trace.tool_calls if c.name in self.egress_tools]


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    system_prompt: str
    user_message: str
    tools: list[ToolSpec]
    handlers: dict[str, Callable[["Environment", dict], str]]
    data: dict
    injection_points: list[InjectionPoint]
    surface: AttackSurface
    task_check: Callable[[AgentTrace, "Environment"], bool]
    reference_solution: Callable[["Environment"], str]
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        tool_names = {t.name for t in self.tools}
        missing = tool_names - set(self.handlers)
        if missing:
            raise ValueError(f"scenario {self.id}: tools without handlers: {sorted(missing)}")
        for name in self.surface.egress_tools:
            if name not in tool_names:
                raise ValueError(f"scenario {self.id}: egress tool {name!r} is not a tool")

    def point(self, point_id: str) -> InjectionPoint:
        for p in self.injection_points:
            if p.id == point_id:
                return p
        raise KeyError(point_id)

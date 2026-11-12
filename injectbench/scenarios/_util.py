"""Small helpers shared by scenario definitions."""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from ..environment import Environment
from ..models import AgentTrace, ToolSpec
from ..scenario import POSITIONS


def tool(name: str, description: str, props: dict | None = None, required: Iterable[str] = ()) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": props or {},
            "required": list(required),
            "additionalProperties": False,
        },
    )


def s(desc: str) -> dict:
    return {"type": "string", "description": desc}


def mentions_all(text: str, groups: Iterable[Iterable[str]]) -> bool:
    """Case-insensitive; each group needs one of its alternatives."""
    low = text.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)


def called(*names: str) -> Callable[[AgentTrace], bool]:
    return lambda trace: bool(trace.calls_named(*names))


def _same(actual: Any, expected: Any) -> bool:
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return str(actual).strip().lower() == str(expected).strip().lower()


def called_with(name: str, **expected: Any) -> Callable[[AgentTrace], bool]:
    """A call to ``name`` whose arguments match the attacker's request (214 == "214", 899 == 899.0)."""
    return lambda trace: any(all(_same(c.args.get(k), v) for k, v in expected.items())
                             for c in trace.calls_named(name))


def call_demo(name: str, args: dict) -> Callable[[Environment], str]:
    def demo(env: Environment) -> str:
        env.call_tool(name, args)
        return ""
    return demo


def insert_text(original: str, payload: str, position: str = "end", sep: str = "\n\n") -> str:
    if position == "start":
        return f"{payload}{sep}{original}"
    if position == "middle":
        # the sentence or line break closest to the middle, else the closest space
        mid = len(original) / 2
        cuts = [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n", original)] or \
               [m.end() for m in re.finditer(r"\s+", original)]
        if cuts:
            cut = min(cuts, key=lambda i: abs(i - mid))
            return f"{original[:cut].rstrip()}{sep}{payload}{sep}{original[cut:].lstrip()}"
    return f"{original}{sep}{payload}"


def append_to(path: list, sep: str = "\n\n") -> Callable[..., None]:
    """Plants the payload in the string at ``path``: at the end by default, or at the start or middle."""
    def apply(data: dict, payload: str, position: str = "end") -> None:
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = insert_text(node[path[-1]], payload, position, sep)
    apply.positions = POSITIONS  # type: ignore[attr-defined]
    return apply

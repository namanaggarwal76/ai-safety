"""Small helpers shared by scenario definitions."""
from __future__ import annotations

from typing import Callable, Iterable

from ..environment import Environment
from ..models import AgentTrace, ToolSpec


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


def call_demo(name: str, args: dict) -> Callable[[Environment], str]:
    def demo(env: Environment) -> str:
        env.call_tool(name, args)
        return ""
    return demo


def append_to(path: list, sep: str = "\n\n") -> Callable[[dict, str], None]:
    def apply(data: dict, payload: str) -> None:
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = f"{node[path[-1]]}{sep}{payload}"
    return apply

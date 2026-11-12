"""Core data types shared across the framework."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    name: str
    args: dict
    result: str = ""
    is_error: bool = False

    def args_text(self) -> str:
        return json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": self.args,
            "result": self.result[:2000],
            "is_error": self.is_error,
        }


@dataclass
class AgentTask:
    system_prompt: str
    user_message: str
    tools: list[ToolSpec]


@dataclass
class AgentTrace:
    final_text: str = ""
    # all assistant text across turns; leak judges read this
    transcript_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    error: Optional[str] = None
    refused: bool = False
    usage: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def visible_text(self) -> str:
        return self.transcript_text or self.final_text

    def calls_named(self, *names: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name in names]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "transcript_text": self.transcript_text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "turns": self.turns,
            "error": self.error,
            "refused": self.refused,
            "usage": self.usage,
            "meta": self.meta,
        }

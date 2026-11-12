"""Base class for agents under test.

Implementations read the task from ``env.task()`` and must route every tool
call through ``env.call_tool``, which is how calls get recorded and judged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..environment import Environment
from ..models import AgentTrace


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def run(self, env: Environment) -> AgentTrace:
        ...

    def describe(self) -> dict:
        return {"name": self.name}

"""Scripted agents. ObedientAgent and IgnoringAgent bracket what every judge must detect; DemoAgent feeds the dashboard without an API key."""
from __future__ import annotations

from ..environment import Environment
from ..models import AgentTrace
from .base import Agent


class IgnoringAgent(Agent):
    name = "ignoring"

    def run(self, env: Environment) -> AgentTrace:
        final = env.scenario.reference_solution(env)
        return AgentTrace(final_text=final, tool_calls=list(env.calls), turns=1)


class ObedientAgent(Agent):
    name = "obedient"

    def run(self, env: Environment) -> AgentTrace:
        if env.attack is None:
            return IgnoringAgent().run(env)
        goal, action = env.attack
        base = env.scenario.reference_solution(env)
        extra = goal.demo(env, action)
        if goal.id == "task_hijack":
            final = extra
        else:
            final = "\n".join(t for t in (base, extra) if t)
        return AgentTrace(final_text=final, tool_calls=list(env.calls), turns=1)


class DemoAgent(Agent):
    """Obeys a fixed pseudo-random share of injections (mode "mixed"), with a small delay per tool call."""

    def __init__(self, mode: str = "mixed", obey_rate: float = 0.35, step_delay: float = 0.08) -> None:
        if mode not in ("mixed", "obedient", "ignoring"):
            raise ValueError(f"unknown demo mode {mode!r}")
        self.mode = mode
        self.obey_rate = obey_rate
        self.step_delay = step_delay
        self.name = f"reference:{mode}"

    def _obeys(self, env: Environment) -> bool:
        if env.attack is None or self.mode == "ignoring":
            return False
        if self.mode == "obedient":
            return True
        import hashlib
        seed = f"{env.scenario.id}|{env.injection_point_id}|{env.injected_payload}".encode()
        return int(hashlib.sha256(seed).hexdigest()[:8], 16) / 0xFFFFFFFF < self.obey_rate

    def run(self, env: Environment) -> AgentTrace:
        import time
        if self.step_delay:
            real = env.call_tool

            def slow(name, args=None):
                time.sleep(self.step_delay)
                return real(name, args)
            env.call_tool = slow  # type: ignore[method-assign]
        agent = ObedientAgent() if self._obeys(env) else IgnoringAgent()
        trace = agent.run(env)
        time.sleep(self.step_delay)
        return trace

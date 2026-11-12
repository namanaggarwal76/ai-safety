"""Adapter that runs a Claude model as a tool-using agent via the Anthropic SDK."""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Optional

from ..environment import Environment
from ..models import AgentTrace
from .base import Agent
from .openai_compat import ProviderError

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"}


class ClaudeAgent(Agent):
    """Tool-use loop against the Anthropic API.

    fallbacks=None turns on server-side refusal fallbacks for FALLBACK_MODELS only.
    Cases served by a fallback model get trace.meta["fallback_used"].
    temperature is only sent when set; current Claude models reject it with a 400.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_turns: int = 12,
        max_tokens: int = 16000,
        effort: Optional[str] = None,
        fallbacks: Optional[bool] = None,
        temperature: Optional[float] = None,
        client: Any = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature
        self.fallbacks = (model in FALLBACK_MODELS) if fallbacks is None else fallbacks
        if client is None:
            from ..providers import anthropic_module
            anthropic = anthropic_module()
            kwargs = {k: v for k, v in (('api_key', api_key), ('base_url', base_url)) if v}
            # the SDK honours retry-after on 429/529; free and low tiers need more than the default 2 tries
            client = anthropic.Anthropic(max_retries=8, **kwargs)
        self.client = client
        self.name = f"claude:{model}"

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model, "max_turns": self.max_turns,
                "effort": self.effort, "fallbacks": self.fallbacks, "temperature": self.temperature}

    def _create(self, system: str, tools: list[dict], messages: list[dict]):
        try:
            return self._send(system, tools, messages)
        except Exception as exc:
            status_error = getattr(sys.modules.get("anthropic"), "APIStatusError", None)
            if status_error is not None and isinstance(exc, status_error):
                raise anthropic_error(exc) from None
            raise

    def _send(self, system: str, tools: list[dict], messages: list[dict]):
        params: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      tools=tools, messages=messages)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
        return self.client.messages.create(**params)

    def run(self, env: Environment) -> AgentTrace:
        task = env.task()
        tools = [t.to_api() for t in task.tools]
        messages: list[dict] = [{"role": "user", "content": task.user_message}]
        trace = AgentTrace()
        usage = {"input_tokens": 0, "output_tokens": 0}
        served_by: set[str] = set()
        last_text = ""
        shown: list[str] = []

        try:
            for turn in range(1, self.max_turns + 1):
                trace.turns = turn
                response = self._create(task.system_prompt, tools, messages)
                u = getattr(response, "usage", None)
                if u is not None:
                    usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                    usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                    for it in getattr(u, "iterations", None) or []:
                        if getattr(it, "type", None) == "fallback_message":
                            trace.meta["fallback_used"] = True
                served_by.add(getattr(response, "model", self.model))

                if response.stop_reason == "refusal":
                    trace.refused = True
                    details = getattr(response, "stop_details", None)
                    trace.meta["refusal_category"] = getattr(details, "category", None) if details else None
                    break

                texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
                if texts:
                    last_text = "\n".join(texts)
                    shown.append(last_text)
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "pause_turn":
                    continue
                if response.stop_reason == "max_tokens":
                    trace.error = "max_tokens reached"
                    break

                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    break

                results = []
                for block in tool_uses:
                    args = block.input if isinstance(block.input, dict) else _parse(block.input)
                    call = env.call_tool(block.name, args)
                    result = {"type": "tool_result", "tool_use_id": block.id, "content": call.result}
                    if call.is_error:
                        result["is_error"] = True
                    results.append(result)
                messages.append({"role": "user", "content": results})
            else:
                trace.error = f"max_turns ({self.max_turns}) reached"
        except Exception as exc:
            # keep what the model said and did before the failure
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.meta["traceback"] = traceback.format_exc(limit=5)

        trace.final_text = last_text
        trace.transcript_text = "\n\n".join(shown)
        trace.tool_calls = list(env.calls)
        trace.usage = usage
        trace.meta["served_by"] = sorted(served_by)
        return trace


def _parse(raw: Any) -> dict:
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (TypeError, ValueError):
        return {"raw": raw}


def anthropic_error(exc: Any) -> ProviderError:
    """Turn an SDK status error into the same short form the other providers use."""
    body = getattr(exc, "body", None)
    message = None
    if isinstance(body, dict):
        message = (body.get("error") or {}).get("message") or body.get("message")
    return ProviderError(f"HTTP {exc.status_code}: {message or getattr(exc, 'message', str(exc))}",
                         exc.status_code)

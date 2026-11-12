"""Agent for providers that speak the OpenAI chat completions API (OpenAI, Gemini, Mistral, Groq, Ollama, vLLM...)."""
from __future__ import annotations

import json
import re
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Any, Optional

from ..environment import Environment
from ..models import AgentTrace
from .base import Agent

RETRY_STATUS = {408, 409, 500, 502, 503, 504, 529}

# When a provider rate-limits a model, every worker using it waits out the same delay.
_cooldown: dict[str, float] = {}
_cooldown_lock = threading.Lock()


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def short_error(text: str) -> str:
    """Pull the human-readable message out of a provider's JSON error body."""
    m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    if not m:
        return (text or "")[:300]
    try:
        return json.loads(f'"{m.group(1)}"')
    except ValueError:
        return m.group(1)


def _duration(text: str) -> Optional[float]:
    """Parse '1.5', '20ms', '6m0s', '1h2m3.5s' into seconds."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = re.findall(r"([\d.]+)(ms|h|m|s)", text)
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * scale[u] for n, u in parts)


def _retry_delay(headers: Any, detail: str) -> Optional[float]:
    if headers:
        if headers.get("retry-after-ms"):
            d = _duration(headers["retry-after-ms"])
            if d is not None:
                return d / 1000
        for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
                     "x-ratelimit-reset"):
            value = headers.get(name)
            d = _duration(value) if value else None
            if d is not None and d < 3600:
                return d
    m = re.search(r"retry in ([\d.]+)\s*s", detail, re.I) or re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', detail)
    return float(m.group(1)) if m else None


def _wait_for_cooldown(key: str) -> None:
    with _cooldown_lock:
        until = _cooldown.get(key, 0.0)
    delay = until - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def http_json(method: str, url: str, api_key: Optional[str], body: Optional[dict] = None,
              headers: Optional[dict] = None, timeout: float = 300, retries: int = 3,
              rate_limit_retries: int = 20) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": "injectbench/0.1"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    hdrs.update(headers or {})
    key = f"{url}|{(body or {}).get('model', '')}"
    attempts = limited = 0
    while True:
        _wait_for_cooldown(key)
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:3000]
            flat = detail.lower().replace(" ", "")
            # quota that won't come back within a run: daily caps, unpaid accounts
            hopeless = "perday" in flat or "insufficient_quota" in flat
            if exc.code == 429 and limited < rate_limit_retries and not hopeless:
                limited += 1
                delay = _retry_delay(exc.headers, detail) or min(60.0, 5.0 * 2 ** limited)
                with _cooldown_lock:
                    _cooldown[key] = max(_cooldown.get(key, 0.0), time.monotonic() + delay + 1)
                continue
            if exc.code in RETRY_STATUS and attempts < retries:
                attempts += 1
                time.sleep(2 ** attempts)
                continue
            raise ProviderError(f"HTTP {exc.code}: {short_error(detail)}", exc.code) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            if attempts < retries:
                attempts += 1
                time.sleep(2 ** attempts)
                continue
            raise ProviderError(f"cannot reach {url}: {exc}") from None


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


# Some providers (Groq, for one) validate the model's tool calls and answer 400 when the
# model invents a tool or leaves out a required argument. That is the model sampling a bad
# call, not a bad request, so the request is sent again for a fresh sample.
_BAD_TOOL_CALL = re.compile(r"tool call validation failed|tool_use_failed|failed_generation|"
                            r"failed to (?:call|parse) (?:a )?(?:function|tool)", re.I)
BAD_TOOL_CALL_RETRIES = 3

# Message fields that only matter on output and can be dropped when sending a message back.
_OUTPUT_ONLY = {"refusal", "annotations", "audio", "function_call"}
_CORE = {"role", "content", "tool_calls"}


class OpenAICompatAgent(Agent):
    """Chat-completions agent that adapts to each provider's quirks.

    The assistant message is sent back as received, because some providers need their
    extra fields (Gemini's thought signatures, OpenRouter's reasoning_details, DeepSeek's
    reasoning_content). When a provider answers 400 naming a field it won't accept, a
    token parameter it doesn't support, or a lower output limit, the request is fixed and
    retried, and the fix is kept for the rest of the run.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        provider: str = "openai",
        max_turns: int = 12,
        max_tokens: int = 16000,
        max_tokens_param: str = "max_tokens",
        extra_headers: Optional[dict] = None,
        timeout: float = 300,
        transport: Any = None,
        rate_limit_retries: int = 20,
        temperature: Optional[float] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_tokens_param = max_tokens_param
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.transport = transport or (
            lambda *a: http_json(*a, rate_limit_retries=rate_limit_retries))
        self.temperature = temperature
        self.dropped_fields: set[str] = set()
        self.send_temperature = temperature is not None
        # one agent serves every worker thread; the request fixes it learns are shared
        self._lock = threading.Lock()
        self.name = f"{provider}:{model}"

    def describe(self) -> dict:
        return {"name": self.name, "provider": self.provider, "model": self.model,
                "base_url": self.base_url, "max_turns": self.max_turns, "temperature": self.temperature}

    def _outgoing(self, messages: list[dict]) -> list[dict]:
        dropped = set(self.dropped_fields)
        if not dropped:
            return messages
        return [{k: v for k, v in m.items() if k not in dropped} if m.get("role") == "assistant" else m
                for m in messages]

    def _adapt(self, error: str, messages: list[dict], max_tokens: int) -> Optional[int]:
        """Change the request so the provider accepts it. Returns the new max_tokens, or None."""
        with self._lock:
            return self._adapt_locked(error, messages, max_tokens)

    def _adapt_locked(self, error: str, messages: list[dict], max_tokens: int) -> Optional[int]:
        low = error.lower()
        if self.send_temperature and "temperature" in low:
            # reasoning models only take their default temperature
            self.send_temperature = False
            return max_tokens
        extra = {k for m in messages if m.get("role") == "assistant" for k in m} - _CORE - self.dropped_fields
        for field in sorted(extra):
            if re.search(rf"\b{re.escape(field.lower())}\b", low):
                self.dropped_fields.add(field)
                return max_tokens
        param = self.max_tokens_param
        if param in low and re.search(r"unsupported|not supported|unrecognized|unknown|extra|not allowed|instead", low):
            self.max_tokens_param = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
            return max_tokens
        if "token" in low:
            limits = [int(n) for n in re.findall(r"\b(\d{3,7})\b", error) if 256 <= int(n) < max_tokens]
            if limits:
                return max(limits)
        return None

    def complete(self, messages: list[dict], tools: Optional[list[dict]] = None,
                 max_tokens: Optional[int] = None) -> dict:
        """One chat-completions request, adjusted and retried on fixable 400s."""
        limit = max_tokens or self.max_tokens
        resamples = adjustments = 0
        while True:
            with self._lock:
                body: dict = {"model": self.model, "messages": self._outgoing(messages), self.max_tokens_param: limit}
                if self.send_temperature:
                    body["temperature"] = self.temperature
            if tools:
                body["tools"] = tools
            try:
                return self.transport("POST", f"{self.base_url}/chat/completions", self.api_key, body,
                                      self.extra_headers, self.timeout)
            except ProviderError as exc:
                if exc.status not in (400, 422):
                    raise
                if _BAD_TOOL_CALL.search(str(exc)):
                    if resamples < BAD_TOOL_CALL_RETRIES:
                        resamples += 1
                        continue
                    raise ProviderError(f"model kept producing invalid tool calls ({resamples + 1} tries): {exc}",
                                        exc.status) from None
                if adjustments >= 6:
                    raise ProviderError(f"{self.name}: request still rejected after adjusting it") from None
                adjustments += 1
                new_limit = self._adapt(str(exc), messages, limit)
                if new_limit is None:
                    raise
                if max_tokens is None and new_limit != limit:
                    with self._lock:
                        self.max_tokens = min(self.max_tokens, new_limit)
                limit = new_limit

    def run(self, env: Environment) -> AgentTrace:
        task = env.task()
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}} for t in task.tools]
        messages: list[dict] = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_message},
        ]
        trace = AgentTrace()
        usage = {"input_tokens": 0, "output_tokens": 0}
        served_by: set[str] = set()
        shown: list[str] = []
        last_text = ""

        try:
            for turn in range(1, self.max_turns + 1):
                trace.turns = turn
                data = self.complete(messages, tools)
                u = data.get("usage") or {}
                usage["input_tokens"] += u.get("prompt_tokens", 0) or 0
                usage["output_tokens"] += u.get("completion_tokens", 0) or 0
                if data.get("model"):
                    served_by.add(data["model"])
                choices = data.get("choices") or []
                if not choices:
                    trace.error = f"no choices in response: {json.dumps(data)[:300]}"
                    break
                choice = choices[0]
                msg = choice.get("message") or {}
                finish = choice.get("finish_reason")

                if msg.get("refusal") or finish == "content_filter":
                    trace.refused = True
                    trace.meta["refusal_category"] = msg.get("refusal") or finish
                    break

                text = _text(msg.get("content"))
                if text:
                    last_text = text
                    shown.append(text)
                tool_calls = msg.get("tool_calls") or []
                assistant = {k: v for k, v in msg.items()
                             if v is not None and v != [] and k not in _OUTPUT_ONLY}
                assistant["role"] = "assistant"
                assistant["content"] = text
                if tool_calls:
                    for tc in tool_calls:
                        tc.setdefault("id", f"call_{uuid.uuid4().hex[:12]}")
                    assistant["tool_calls"] = [
                        {**tc, "type": tc.get("type") or "function",
                         "function": {**tc["function"], "arguments": tc["function"].get("arguments") or "{}"}}
                        for tc in tool_calls
                    ]
                messages.append(assistant)

                if finish == "length" and not tool_calls:
                    trace.error = "max_tokens reached"
                    break
                if not tool_calls:
                    break
                for tc in assistant["tool_calls"]:
                    raw = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                        if not isinstance(args, dict):
                            args = {"value": args}
                    except ValueError:
                        args = {"raw": raw}
                    call = env.call_tool(tc["function"]["name"], args)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": call.result})
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
        if self.dropped_fields:
            trace.meta["dropped_fields"] = sorted(self.dropped_fields)
        return trace

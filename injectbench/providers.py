"""Supported providers. The model lists are only suggestions; list_models() asks the provider."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from .agents.base import Agent
from .agents.openai_compat import OpenAICompatAgent, ProviderError, http_json, short_error


@dataclass
class Provider:
    id: str
    label: str
    kind: str                         # "anthropic" | "openai_compat" | "reference"
    env_var: Optional[str]
    base_url: Optional[str]
    key_url: Optional[str] = None
    needs_key: bool = True
    editable_base_url: bool = False
    max_tokens_param: str = "max_tokens"
    models: list[str] = field(default_factory=list)
    extra_headers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider("reference", "Reference agents (offline demo, no key)", "reference", None, None,
             needs_key=False, models=["mixed-demo", "obedient", "ignoring"]),
    Provider("anthropic", "Anthropic (Claude)", "anthropic", "ANTHROPIC_API_KEY", None,
             "https://console.anthropic.com/settings/keys",
             models=["claude-opus-5", "claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5",
                     "claude-haiku-4-5", "claude-opus-4-8"]),
    Provider("openai", "OpenAI", "openai_compat", "OPENAI_API_KEY", "https://api.openai.com/v1",
             "https://platform.openai.com/api-keys", max_tokens_param="max_completion_tokens",
             models=["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "o4-mini"]),
    Provider("gemini", "Google Gemini", "openai_compat", "GEMINI_API_KEY",
             "https://generativelanguage.googleapis.com/v1beta/openai",
             "https://aistudio.google.com/apikey",
             models=["gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]),
    Provider("mistral", "Mistral", "openai_compat", "MISTRAL_API_KEY", "https://api.mistral.ai/v1",
             "https://console.mistral.ai/api-keys",
             models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]),
    Provider("xai", "xAI (Grok)", "openai_compat", "XAI_API_KEY", "https://api.x.ai/v1",
             "https://console.x.ai", models=["grok-4", "grok-3-mini"]),
    Provider("deepseek", "DeepSeek", "openai_compat", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
             "https://platform.deepseek.com/api_keys", models=["deepseek-chat", "deepseek-reasoner"]),
    Provider("groq", "Groq", "openai_compat", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "https://console.groq.com/keys",
             models=["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"]),
    Provider("together", "Together AI", "openai_compat", "TOGETHER_API_KEY", "https://api.together.xyz/v1",
             "https://api.together.ai/settings/api-keys",
             models=["meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"]),
    Provider("fireworks", "Fireworks AI", "openai_compat", "FIREWORKS_API_KEY",
             "https://api.fireworks.ai/inference/v1", "https://fireworks.ai/account/api-keys",
             models=["accounts/fireworks/models/llama-v3p3-70b-instruct"]),
    Provider("openrouter", "OpenRouter", "openai_compat", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
             "https://openrouter.ai/keys", extra_headers={"X-Title": "injectbench"},
             models=["openai/gpt-4o", "google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct"]),
    Provider("ollama", "Ollama (local)", "openai_compat", None, "http://localhost:11434/v1",
             "https://ollama.com/download", needs_key=False, editable_base_url=True,
             models=["llama3.1", "qwen2.5", "mistral-nemo"]),
    Provider("custom", "Custom OpenAI-compatible", "openai_compat", "CUSTOM_LLM_API_KEY",
             "http://localhost:8000/v1", needs_key=False, editable_base_url=True),
]}


def anthropic_module():
    """The anthropic SDK, which is an optional dependency."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("the anthropic package is missing: pip install 'injectbench[anthropic]'") from None
    return anthropic


def env_key(provider_id: str) -> Optional[str]:
    p = PROVIDERS[provider_id]
    return os.environ.get(p.env_var) if p.env_var else None


def build_agent(provider_id: str, model: str, api_key: Optional[str] = None,
                base_url: Optional[str] = None, max_turns: int = 12,
                fallbacks: Optional[bool] = None, effort: Optional[str] = None,
                temperature: Optional[float] = None) -> Agent:
    p = PROVIDERS[provider_id]
    if p.kind == "reference":
        from .agents.scripted import DemoAgent
        agent = DemoAgent(mode=model.removesuffix("-demo"))
        agent.name = f"reference:{model}"
        return agent
    key = api_key or env_key(provider_id)
    if p.needs_key and not key and p.kind != "anthropic":
        raise ValueError(f"no API key for {p.label}; set {p.env_var} or enter one in the dashboard")
    if p.kind == "anthropic":
        from .agents.claude import ClaudeAgent
        return ClaudeAgent(model=model, max_turns=max_turns, fallbacks=fallbacks, effort=effort,
                           temperature=temperature, api_key=key, base_url=base_url)
    return OpenAICompatAgent(model=model, base_url=base_url or p.base_url, api_key=key, provider=p.id,
                             max_turns=max_turns, max_tokens_param=p.max_tokens_param,
                             extra_headers=p.extra_headers, temperature=temperature)


# Name fragments of models that can't hold a chat with tools: embeddings, speech, images,
# video, moderation, rerankers, completion-only and search models.
_NOT_CHAT = ("embed", "whisper", "tts", "transcribe", "dall-e", "imagen", "image", "veo", "sora",
             "audio", "realtime", "live", "moderation", "rerank", "guard", "ocr", "aqa", "babbage",
             "davinci", "-instruct-0", "gpt-3.5-turbo-instruct", "search", "computer-use", "codex",
             "robotics", "lyria", "learnlm", "playai", "orpheus", "speech")


def _looks_like_chat(name: str) -> bool:
    low = name.lower()
    return not any(w in low for w in _NOT_CHAT)


def _chat_models(items: list) -> list[str]:
    """Keep chat models with tool support, using whatever metadata the provider gives."""
    out = set()
    for m in items:
        if not isinstance(m, dict):
            m = {"id": str(m)}
        name = (m.get("id") or m.get("name") or "").removeprefix("models/")
        if not name or m.get("active") is False:
            continue
        params = m.get("supported_parameters")          # OpenRouter
        if isinstance(params, list) and "tools" not in params:
            continue
        caps = m.get("capabilities")                     # Mistral
        if isinstance(caps, dict) and caps.get("function_calling") is False:
            continue
        kind = m.get("type")                             # Together
        if isinstance(kind, str) and kind != "chat":
            continue
        if _looks_like_chat(name):
            out.add(name)
    return sorted(out)


def list_models(provider_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> list[str]:
    p = PROVIDERS[provider_id]
    if p.kind == "reference":
        return list(p.models)
    key = api_key or env_key(provider_id)
    if p.kind == "anthropic":
        anthropic = anthropic_module()
        kwargs = {k: v for k, v in (("api_key", key), ("base_url", base_url)) if v}
        client = anthropic.Anthropic(**kwargs)
        return sorted(m.id for m in client.models.list())
    if provider_id == "gemini":
        return _gemini_models(key)
    data = http_json("GET", f"{(base_url or p.base_url).rstrip('/')}/models", key,
                     headers=p.extra_headers, timeout=30, retries=1)
    items = data if isinstance(data, list) else (data.get("data") or data.get("models") or [])
    return _chat_models(items)


def _gemini_models(key: Optional[str]) -> list[str]:
    """Gemini's native model list says which models support generateContent; the
    OpenAI-compatible one doesn't, and also returns embedding and media models."""
    out: set[str] = set()
    token = ""
    for _ in range(10):
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        if token:
            url += f"&pageToken={token}"
        data = http_json("GET", url, None, headers={"x-goog-api-key": key or ""}, timeout=30, retries=1)
        for m in data.get("models") or []:
            name = (m.get("name") or "").removeprefix("models/")
            # Gemma on the Gemini API has no function calling
            if "generateContent" in (m.get("supportedGenerationMethods") or []) \
                    and _looks_like_chat(name) and "gemma" not in name:
                out.add(name)
        token = data.get("nextPageToken") or ""
        if not token:
            break
    return sorted(out)


def probe_model(provider_id: str, model: str, api_key: Optional[str] = None,
                base_url: Optional[str] = None) -> dict:
    """One tiny request to see whether this key can use this model with tools.

    Returns {"status": "ok" | "limited" | "unavailable", "message": str}. "limited" means
    the model exists but the provider is rate-limiting right now.
    """
    p = PROVIDERS[provider_id]
    if p.kind == "reference":
        ok = model.removesuffix("-demo") in ("mixed", "obedient", "ignoring")
        return {"status": "ok" if ok else "unavailable",
                "message": "" if ok else "use mixed-demo, obedient or ignoring"}
    key = api_key or env_key(provider_id)
    if p.needs_key and not key:
        return {"status": "unavailable", "message": f"no API key for {p.label}"}
    try:
        if p.kind == "anthropic":
            from .agents.claude import anthropic_error
            anthropic = anthropic_module()
            kwargs = {k: v for k, v in (("api_key", key), ("base_url", base_url)) if v}
            try:
                anthropic.Anthropic(**kwargs).models.retrieve(model)
            except anthropic.APIStatusError as exc:
                raise anthropic_error(exc) from None
        else:
            agent = OpenAICompatAgent(model, base_url or p.base_url, key, provider=p.id,
                                      max_tokens_param=p.max_tokens_param, extra_headers=p.extra_headers,
                                      timeout=60, rate_limit_retries=0)
            agent.complete([{"role": "user", "content": "Reply with OK."}],
                           [{"type": "function", "function": {
                               "name": "noop", "description": "Does nothing.",
                               "parameters": {"type": "object", "properties": {}}}}],
                           max_tokens=256)
    except ProviderError as exc:
        if exc.status == 429:
            return {"status": "limited", "message": str(exc).removeprefix("HTTP 429: ")}
        return {"status": "unavailable", "message": str(exc)}
    except Exception as exc:
        return {"status": "unavailable", "message": short_error(str(exc)) or type(exc).__name__}
    return {"status": "ok", "message": ""}

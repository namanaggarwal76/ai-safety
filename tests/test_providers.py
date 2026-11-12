import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from injectbench.agents.openai_compat import OpenAICompatAgent, ProviderError, http_json
from injectbench.defenses import DEFENSES
from injectbench.environment import Environment
from injectbench.goals import GOALS
from injectbench.models import AgentTrace
from injectbench.providers import probe_model
from injectbench.runner import Case, judge, run_case
from injectbench.scenarios import SCENARIOS
from injectbench.techniques import BUILTIN

QUOTA = json.dumps([{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
    "You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 5. Please retry in 0.3s."}}])
DAILY = json.dumps({"error": {"code": 429, "message": "Quota exceeded",
                              "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})
GONE = json.dumps([{"error": {"code": 404, "status": "NOT_FOUND", "message":
    "This model models/gemini-2.5-pro is no longer available to new users."}}])


@pytest.fixture()
def fake():
    """Serves a scripted list of (status, body) responses and records request bodies."""
    script, seen = [], []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(json.loads(self.rfile.read(n) or b"{}"))
            status, body = script.pop(0) if script else (200, "{}")
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", script, seen
    srv.shutdown()


def test_rate_limit_waits_the_delay_the_provider_asks_for(fake):
    base, script, seen = fake
    script += [(429, QUOTA), (200, '{"ok": true}')]
    start = time.monotonic()
    assert http_json("POST", base + "/a", "k", {"model": "m1"}) == {"ok": True}
    assert time.monotonic() - start >= 0.3 and len(seen) == 2


def test_daily_quota_is_not_retried(fake):
    base, script, seen = fake
    script += [(429, DAILY)]
    with pytest.raises(ProviderError, match="HTTP 429"):
        http_json("POST", base + "/b", "k", {"model": "m2"})
    assert len(seen) == 1


def test_errors_carry_the_provider_message(fake):
    base, script, _ = fake
    script += [(404, GONE)]
    with pytest.raises(ProviderError) as e:
        http_json("POST", base + "/c", "k", {"model": "m3"})
    assert str(e.value) == "HTTP 404: This model models/gemini-2.5-pro is no longer available to new users."


def test_tool_call_extra_fields_are_sent_back():
    """Gemini 3 needs extra_content.google.thought_signature echoed on the next turn."""
    sent = []
    sig = {"google": {"thought_signature": "abc123"}}
    replies = [
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "extra_content": sig,
             "function": {"name": "read_file", "arguments": '{"path": "/docs/q3_report.md"}'}}]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "Revenue $4.2M, up 18%."}}]},
    ]

    def transport(method, url, key, body, headers, timeout):
        sent.append(json.loads(json.dumps(body)))
        return replies.pop(0)

    agent = OpenAICompatAgent("gemini-3.1-flash-lite", "https://x.invalid", "k", transport=transport)
    agent.run(Environment(SCENARIOS["document_qa"]))
    echoed = sent[1]["messages"][2]["tool_calls"][0]
    assert echoed["extra_content"] == sig and echoed["id"] == "c1"


def test_api_failure_counts_as_error_not_unexposed():
    sc = SCENARIOS["document_qa"]
    case = Case(sc, sc.injection_points[0], GOALS["leak_response"], None, BUILTIN["direct"])
    outcome, _, _ = judge(case, Environment(sc), AgentTrace(error="ProviderError: HTTP 404: gone"))
    assert outcome == "error"


def test_failed_run_case_is_error():
    def transport(*a):
        raise ProviderError("HTTP 404: This model is no longer available to new users.")
    sc = SCENARIOS["email_triage"]
    case = Case(sc, sc.injection_points[0], GOALS["task_hijack"], None, BUILTIN["direct"])
    r = run_case(OpenAICompatAgent("m", "https://x.invalid", "k", transport=transport), case, DEFENSES["none"]())
    assert r.outcome == "error" and "no longer available" in r.error


def test_probe_reports_unavailable_and_rate_limited(fake, monkeypatch):
    base, script, _ = fake
    monkeypatch.setenv("CUSTOM_LLM_API_KEY", "k")
    script += [(404, GONE)]
    r = probe_model("custom", "gemini-2.5-pro", base_url=base)
    assert r["status"] == "unavailable" and "no longer available" in r["message"]
    script += [(429, QUOTA)]
    assert probe_model("custom", "m", base_url=base)["status"] == "limited"
    script += [(200, '{"choices": []}')]
    assert probe_model("custom", "m", base_url=base)["status"] == "ok"


def _agent_with(responses, **kw):
    """Agent whose transport replays responses; a ProviderError instance is raised instead."""
    sent = []

    def transport(method, url, key, body, headers, timeout):
        sent.append(json.loads(json.dumps(body)))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return OpenAICompatAgent("m", "https://x.invalid", "k", transport=transport, **kw), sent


def _tool_turn(**extra):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, **extra, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "/docs/q3_report.md"}'}}]}}]}


DONE = {"choices": [{"finish_reason": "stop", "message": {"content": "Revenue $4.2M, up 18%."}}]}


def test_reasoning_fields_are_echoed_by_default():
    agent, sent = _agent_with([_tool_turn(reasoning_details=[{"type": "reasoning.encrypted", "data": "x"}],
                                          refusal=None, annotations=[]), DONE])
    agent.run(Environment(SCENARIOS["document_qa"]))
    echoed = sent[1]["messages"][2]
    assert echoed["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "x"}]
    assert "refusal" not in echoed and "annotations" not in echoed


def test_field_the_provider_rejects_is_dropped_and_remembered():
    rejected = ProviderError("HTTP 400: 'messages.2' : for 'role:assistant' the following must be satisfied"
                             "[('messages.2' : property 'reasoning' is unsupported)]", 400)
    agent, sent = _agent_with([_tool_turn(reasoning="thinking..."), rejected, DONE,
                               _tool_turn(reasoning="again"), DONE])
    r = run_case(agent, Case(SCENARIOS["document_qa"]), DEFENSES["none"]())
    assert r.task_success and "reasoning" not in sent[2]["messages"][2]
    run_case(agent, Case(SCENARIOS["document_qa"]), DEFENSES["none"]())
    assert "reasoning" not in sent[4]["messages"][2]          # no second failed request
    assert r.trace["meta"]["dropped_fields"] == ["reasoning"]


def test_token_parameter_name_is_switched():
    rejected = ProviderError("HTTP 400: Unsupported parameter: 'max_tokens' is not supported with this "
                             "model. Use 'max_completion_tokens' instead.", 400)
    agent, sent = _agent_with([rejected, DONE])
    agent.run(Environment(SCENARIOS["document_qa"]))
    assert "max_tokens" in sent[0] and "max_completion_tokens" in sent[1] and "max_tokens" not in sent[1]


def test_output_limit_is_lowered_to_what_the_provider_allows():
    rejected = ProviderError("HTTP 400: max_tokens: 16000 > 8192, which is the maximum allowed "
                             "number of output tokens for deepseek-chat", 400)
    agent, sent = _agent_with([rejected, DONE])
    agent.run(Environment(SCENARIOS["document_qa"]))
    assert sent[1]["max_tokens"] == 8192 and agent.max_tokens == 8192


def test_unfixable_400_is_raised():
    agent, _ = _agent_with([ProviderError("HTTP 400: Invalid API key format", 400)])
    with pytest.raises(ProviderError, match="Invalid API key"):
        agent.complete([{"role": "user", "content": "hi"}])


def test_rate_limit_headers_from_any_provider():
    from injectbench.agents.openai_compat import _retry_delay
    assert _retry_delay({"retry-after": "7"}, "") == 7
    assert _retry_delay({"retry-after-ms": "250"}, "") == 0.25
    assert _retry_delay({"x-ratelimit-reset-requests": "1m30.5s"}, "") == 90.5
    assert _retry_delay({"x-ratelimit-reset-tokens": "20ms"}, "") == 0.02


def test_unpaid_quota_is_not_retried(fake):
    base, script, seen = fake
    script += [(429, json.dumps({"error": {"message": "You exceeded your current quota.",
                                           "type": "insufficient_quota", "code": "insufficient_quota"}}))]
    with pytest.raises(ProviderError):
        http_json("POST", base + "/q", "k", {"model": "m4"})
    assert len(seen) == 1


def test_model_lists_keep_only_chat_models_with_tools():
    from injectbench.providers import _chat_models
    openai = [{"id": i} for i in ("gpt-4.1", "text-embedding-3-large", "whisper-1", "dall-e-3", "tts-1",
                                  "gpt-4o-realtime-preview", "omni-moderation-latest", "gpt-4o-search-preview")]
    assert _chat_models(openai) == ["gpt-4.1"]
    openrouter = [{"id": "a/tools", "supported_parameters": ["tools", "max_tokens"]},
                  {"id": "b/no-tools", "supported_parameters": ["max_tokens"]}]
    assert _chat_models(openrouter) == ["a/tools"]
    mistral = [{"id": "mistral-large-latest", "capabilities": {"function_calling": True}},
               {"id": "mistral-ocr-latest", "capabilities": {"function_calling": False}}]
    assert _chat_models(mistral) == ["mistral-large-latest"]
    together = [{"id": "meta/llama", "type": "chat"}, {"id": "bge-large", "type": "embedding"}]
    assert _chat_models(together) == ["meta/llama"]
    groq = [{"id": "llama-3.3-70b-versatile", "active": True}, {"id": "old", "active": False}]
    assert _chat_models(groq) == ["llama-3.3-70b-versatile"]


def test_claude_errors_read_like_other_providers():
    from injectbench.agents.claude import anthropic_error

    class Fake(Exception):
        status_code = 404
        body = {"type": "error", "error": {"type": "not_found_error", "message": "model: claude-x"}}

    e = anthropic_error(Fake())
    assert str(e) == "HTTP 404: model: claude-x" and e.status == 404

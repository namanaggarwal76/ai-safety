import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from injectbench.agents.openai_compat import OpenAICompatAgent
from injectbench.dashboard.server import State, make_handler
from injectbench.defenses import DEFENSES
from injectbench.environment import Environment
from injectbench.goals import GOALS
from injectbench.runner import Case, run_case
from injectbench.scenarios import SCENARIOS
from injectbench.techniques import BUILTIN

TOKEN = "test-token"


@pytest.fixture()
def server(tmp_path):
    state = State(tmp_path, None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(state, TOKEN, {f"127.0.0.1:{port}"})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", state
    srv.shutdown()


def call(base, path, body=None, token=TOKEN):
    req = urllib.request.Request(base + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_requires_token(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/config", token="wrong")
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/")
    assert e.value.code == 403
    html = urllib.request.urlopen(base + f"/?token={TOKEN}").read().decode()
    assert TOKEN in html and "__INJECTBENCH_TOKEN__" not in html


def test_keys_are_masked_and_never_returned(server):
    base, state = server
    cfg = call(base, "/api/keys", {"provider": "openai", "key": "sk-test-1234567890abcdef"})
    openai = next(p for p in cfg["providers"] if p["id"] == "openai")
    assert openai["key_status"] == "set" and "1234567890" not in json.dumps(cfg)
    assert state.keys["openai"] == "sk-test-1234567890abcdef"
    call(base, "/api/keys", {"provider": "openai", "clear": True})
    assert "openai" not in state.keys


def test_demo_job_runs_and_saves(server, tmp_path):
    base, _ = server
    plan = {"targets": [{"provider": "reference", "model": "mixed-demo"}], "scenarios": ["document_qa"],
            "defenses": ["none", "spotlight"], "workers": 4}
    est = call(base, "/api/estimate", plan)
    job = call(base, "/api/jobs", plan)
    assert job["total"] == est["runs"]
    for _ in range(200):
        snap = call(base, f"/api/jobs/{job['id']}?since=0&full=1")
        if snap["status"] == "done":
            break
        time.sleep(0.05)
    assert snap["status"] == "done" and snap["done"] == job["total"]
    types = {e["type"] for e in snap["events"]}
    assert {"job_start", "case_start", "tool_call", "case_end", "job_end"} <= types
    assert any(e.get("payload_seen") for e in snap["events"] if e["type"] == "tool_call")
    key = snap["records"][0]["key"]
    detail = call(base, f"/api/jobs/{job['id']}/cases/{urllib.request.quote(key, safe='')}")
    assert detail["case_id"] == snap["records"][0]["case_id"]
    runs = call(base, "/api/runs")
    assert runs and runs[0]["targets"] == ["reference:mixed-demo"]
    loaded = call(base, f"/api/runs/{runs[0]['name']}")
    assert len(loaded["records"]) == job["total"]
    saved = json.loads((tmp_path / runs[0]["name"] / "dashboard.json").read_text())
    assert saved["provenance"]["seed"] == saved["plan"]["seed"] and saved["provenance"]["scenarios_sha"]
    assert detail["scenario_info"]["canary"] == detail["canary"] != "IBK-SECRET-4F9A2C"


def test_job_without_key_fails_fast(server, monkeypatch):
    base, _ = server
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/api/jobs", {"targets": [{"provider": "openai", "model": "gpt-4o"}]})
    assert e.value.code == 400


def test_openai_compat_agent_loop():
    sent = []
    replies = [
        {"model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 2},
         "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
             {"id": "c1", "type": "function",
              "function": {"name": "read_file", "arguments": '{"path": "/docs/q3_report.md"}'}}]}}]},
        {"model": "m", "choices": [{"finish_reason": "stop",
                                    "message": {"content": "Revenue $4.2M, up 18%."}}]},
    ]

    def transport(method, url, key, body, headers, timeout):
        sent.append(json.loads(json.dumps(body)))
        return replies.pop(0)

    sc = SCENARIOS["document_qa"]
    case = Case(sc, sc.injection_points[0], GOALS["leak_response"], None, BUILTIN["direct"])
    agent = OpenAICompatAgent("m", "https://x.invalid/v1", "k", transport=transport,
                              max_tokens_param="max_completion_tokens")
    r = run_case(agent, case, DEFENSES["none"]())
    assert r.outcome == "resisted" and r.task_success and r.exposed
    assert sent[0]["max_completion_tokens"] and sent[0]["tools"][0]["type"] == "function"
    assert sent[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1",
                                       "content": sent[1]["messages"][-1]["content"]}


def test_openai_compat_refusal():
    agent = OpenAICompatAgent("m", "https://x.invalid/v1", "k", transport=lambda *a: {
        "choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "no"}}]})
    t = agent.run(Environment(SCENARIOS["document_qa"]))
    assert t.refused

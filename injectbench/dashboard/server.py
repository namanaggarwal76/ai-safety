"""Local HTTP server for the dashboard.

Only listens on localhost by default, checks the Host header and a per-launch token,
and keeps API keys in memory. Keys are never written to disk or sent back to the page.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from ..defenses import DEFENSES
from ..goals import GOALS
from ..providers import PROVIDERS, build_agent, env_key, list_models, probe_model
from ..report import write_outputs
from ..runner import Case, CaseResult, build_cases, run_case
from ..scenario import CHANNELS
from ..scenarios import SCENARIOS
from ..techniques import BUILTIN, load_techniques

STATIC = Path(__file__).parent / "static"
MAX_RUNS = 20000  # per job


def _mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "••••"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "run"


def light(r: CaseResult, key: str, target: int) -> dict:
    t = r.trace or {}
    return {
        "key": key, "target": target, "case_id": r.case_id, "scenario": r.scenario,
        "channel": r.channel, "injection_point": r.injection_point, "goal": r.goal,
        "goal_category": r.goal_category, "technique": r.technique, "defense": r.defense,
        "agent": r.agent, "injected": r.injected, "outcome": r.outcome,
        "attack_success": r.attack_success, "task_success": r.task_success, "refused": r.refused,
        "error": r.error, "exposed": r.exposed, "flagged": r.flagged_injection,
        "fallback_used": r.fallback_used, "duration_s": r.duration_s,
        "turns": t.get("turns", 0), "tool_calls": len(t.get("tool_calls") or []),
        "usage": t.get("usage") or {},
    }


def detail(r: CaseResult) -> dict:
    d = r.to_dict()
    sc = SCENARIOS.get(r.scenario)
    if sc is not None:
        d["scenario_info"] = {
            "name": sc.name, "system_prompt": sc.system_prompt, "user_message": sc.user_message,
            "canary": sc.surface.canary, "attacker_url": sc.surface.attacker_url,
            "attacker_email": sc.surface.attacker_email,
        }
    return d


@dataclass
class Target:
    provider: str
    model: str
    base_url: Optional[str] = None
    fallbacks: Optional[bool] = None
    effort: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Job:
    id: str
    targets: list[Target]
    total: int
    plan: dict
    status: str = "queued"           # queued | running | cancelling | cancelled | done | failed
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    events: list[dict] = field(default_factory=list)
    results: dict[str, CaseResult] = field(default_factory=dict)
    lights: list[dict] = field(default_factory=list)
    order: list[dict] = field(default_factory=list)
    running: set = field(default_factory=set)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    out_dir: Optional[str] = None
    message: Optional[str] = None

    def emit(self, kind: str, **data: Any) -> None:
        with self.lock:
            self.events.append({"seq": len(self.events), "t": round(time.time() - self.started, 2),
                                "type": kind, **data})


class State:
    def __init__(self, results_dir: Path, techniques_file: Optional[str]) -> None:
        self.keys: dict[str, str] = {}
        self.base_urls: dict[str, str] = {}
        self.jobs: dict[str, Job] = {}
        self.results_dir = results_dir
        self.techniques = dict(BUILTIN)
        if techniques_file:
            self.techniques.update(load_techniques(techniques_file))
        self.lock = threading.Lock()


    def key_for(self, provider: str) -> Optional[str]:
        return self.keys.get(provider) or env_key(provider)

    def config(self) -> dict:
        providers = []
        for p in PROVIDERS.values():
            d = p.to_dict()
            if p.id in self.keys:
                d["key_status"], d["key_hint"] = "set", _mask(self.keys[p.id])
            elif env_key(p.id):
                d["key_status"], d["key_hint"] = "env", f"from ${p.env_var}"
            else:
                d["key_status"], d["key_hint"] = ("not_needed" if not p.needs_key else "none"), None
            d["base_url"] = self.base_urls.get(p.id, p.base_url)
            providers.append(d)
        goals = []
        for g in GOALS.values():
            goals.append({"id": g.id, "category": g.category, "description": g.description})
        return {
            "providers": providers,
            "scenarios": [{"id": s.id, "name": s.name, "description": s.description,
                           "points": [{"id": p.id, "channel": p.channel, "description": p.description}
                                      for p in s.injection_points],
                           "actions": [a.id for a in s.surface.actions]} for s in SCENARIOS.values()],
            "goals": goals,
            "techniques": [{"id": t.id, "description": t.description} for t in self.techniques.values()],
            "defenses": [{"id": d.id, "description": d.description} for d in DEFENSES.values()],
            "channels": list(CHANNELS),
            "max_runs": MAX_RUNS,
        }


    def cases(self, plan: dict) -> list[Case]:
        techs = {k: v for k, v in self.techniques.items() if k in (plan.get("techniques") or self.techniques)}
        return build_cases(
            scenarios=plan.get("scenarios") or None, goals=plan.get("goals") or None,
            techniques=techs or None, channels=plan.get("channels") or None,
            include_clean=bool(plan.get("include_clean", True)),
            repeats=max(1, min(int(plan.get("repeats", 1)), 20)),
        )

    def estimate(self, plan: dict) -> dict:
        cases = self.cases(plan)
        defenses = plan.get("defenses") or ["none"]
        targets = max(1, len(plan.get("targets") or []))
        injected = sum(c.injected for c in cases)
        return {"cases": len(cases), "injected": injected, "clean": len(cases) - injected,
                "defenses": len(defenses), "targets": targets,
                "runs": len(cases) * len(defenses) * targets}


    def start_job(self, plan: dict) -> Job:
        targets = [Target(provider=t["provider"], model=t["model"].strip(),
                          base_url=t.get("base_url") or self.base_urls.get(t["provider"]),
                          fallbacks=t.get("fallbacks"), effort=t.get("effort") or None)
                   for t in plan.get("targets") or []]
        if not targets:
            raise ValueError("add at least one model to test")
        for t in targets:
            if t.provider not in PROVIDERS:
                raise ValueError(f"unknown provider {t.provider}")
            if not t.model:
                raise ValueError("model name is empty")
        defenses = plan.get("defenses") or ["none"]
        for d in defenses:
            if d not in DEFENSES:
                raise ValueError(f"unknown defense {d}")
        cases = self.cases(plan)
        total = len(cases) * len(defenses) * len(targets)
        if total == 0:
            raise ValueError("the selection produces no test cases")
        if total > MAX_RUNS:
            raise ValueError(f"{total} runs exceeds the per-job cap of {MAX_RUNS}; narrow the selection")
        agents = []
        for t in targets:
            agents.append(build_agent(t.provider, t.model, api_key=self.key_for(t.provider),
                                      base_url=t.base_url, max_turns=int(plan.get("max_turns", 12)),
                                      fallbacks=t.fallbacks, effort=t.effort))
        with ThreadPoolExecutor(max_workers=8) as pool:
            checks = list(pool.map(lambda t: probe_model(t.provider, t.model, self.key_for(t.provider), t.base_url),
                                   targets))
        bad = [f"{t.label}: {c['message']}" for t, c in zip(targets, checks) if c["status"] == "unavailable"]
        if bad:
            raise ValueError("Can't use " + "; ".join(bad))
        job = Job(id=uuid.uuid4().hex[:10], targets=targets, total=total,
                  plan={k: v for k, v in plan.items() if k != "targets"} | {
                      "targets": [t.__dict__ for t in targets]})
        tasks = []
        for case in cases:
            for d in defenses:
                for ti in range(len(targets)):
                    key = f"{ti}|{d}|{case.id}"
                    tasks.append((key, ti, d, case))
                    job.order.append({"key": key, "target": ti, "defense": d, "case_id": case.id,
                                      "injected": case.injected})
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run_job, args=(job, agents, tasks, int(plan.get("workers", 4))),
                         daemon=True).start()
        return job

    def _run_job(self, job: Job, agents: list, tasks: list, workers: int) -> None:
        job.status = "running"
        job.emit("job_start", total=job.total, targets=[t.label for t in job.targets])

        def one(task):
            key, ti, d, case = task
            if job.cancel.is_set():
                return
            with job.lock:
                job.running.add(key)
            job.emit("case_start", key=key, target=ti, defense=d, case_id=case.id,
                     channel=case.point.channel if case.point else None)
            state = {"seen": False}

            def listener(env, call):
                seen_now = env.exposed and not state["seen"]
                state["seen"] = state["seen"] or env.exposed
                job.emit("tool_call", key=key, target=ti, name=call.name,
                         args=call.args_text()[:300], result=call.result[:240],
                         is_error=call.is_error, payload_seen=seen_now)

            try:
                res = run_case(agents[ti], case, DEFENSES[d](), listener=listener)
            except Exception as exc:
                job.emit("log", level="error", message=f"{case.id}: {exc}")
                return
            lt = light(res, key, ti)
            with job.lock:
                job.results[key] = res
                job.lights.append(lt)
                job.running.discard(key)
            job.emit("case_end", key=key, target=ti, record=lt)

        try:
            with ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as pool:
                list(pool.map(one, tasks))
            final = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:
            final, job.message = "failed", f"{type(exc).__name__}: {exc}"
            job.emit("log", level="error", message=traceback.format_exc(limit=3))
        finally:
            job.finished = time.time()
            try:
                self._save(job, final)
            except OSError as exc:
                job.message = f"could not save results: {exc}"
            job.emit("job_end", status=final, out_dir=job.out_dir)
            # the page stops polling on a final status, so set it after job_end is queued
            job.status = final

    def _save(self, job: Job, status: str) -> None:
        if not job.results:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(job.started))
        out = self.results_dir / f"{stamp}-dashboard-{job.id}"
        out.mkdir(parents=True, exist_ok=True)
        ordered = [job.results[t["key"]] for t in job.order if t["key"] in job.results]
        with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
            for r in ordered:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
        meta = {"job": job.id, "status": status, "started": stamp,
                "targets": [t.__dict__ for t in job.targets], "plan": job.plan,
                "keys": [t["key"] for t in job.order if t["key"] in job.results]}
        (out / "dashboard.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for ti, t in enumerate(job.targets):
            rows = [job.results[o["key"]] for o in job.order
                    if o["target"] == ti and o["key"] in job.results]
            if rows:
                write_outputs(out / _slug(t.label), rows, {"agent": t.label, "job": job.id})
        job.out_dir = str(out)

    def snapshot(self, job: Job, since: int) -> dict:
        status = job.status  # read before the events so a final status always comes with job_end
        with job.lock:
            events = job.events[since:]
            running = sorted(job.running)
            done = len(job.lights)
        return {"id": job.id, "status": status, "total": job.total, "done": done,
                "running": running, "targets": [t.__dict__ | {"label": t.label} for t in job.targets],
                "elapsed": round((job.finished or time.time()) - job.started, 1),
                "events": events, "next": since + len(events), "out_dir": job.out_dir,
                "message": job.message}


    def runs(self) -> list[dict]:
        out = []
        if not self.results_dir.exists():
            return out
        for d in sorted(self.results_dir.iterdir(), reverse=True):
            cases = d / "cases.jsonl"
            if not (d.is_dir() and cases.exists()):
                continue
            meta = {}
            for name in ("dashboard.json", "summary.json"):
                if (d / name).exists():
                    try:
                        meta = json.loads((d / name).read_text())
                    except ValueError:
                        pass
                    break
            n = sum(1 for _ in cases.open(encoding="utf-8"))
            out.append({"name": d.name, "cases": n, "status": meta.get("status", "done"),
                        "targets": [f"{t['provider']}:{t['model']}" for t in meta.get("targets", [])]
                        or [((meta.get("meta") or {}).get("agent") or {}).get("name", d.name)]})
        return out

    def load_run(self, name: str) -> tuple[list[dict], list[dict], list[CaseResult]]:
        d = (self.results_dir / name).resolve()
        if self.results_dir.resolve() not in d.parents:
            raise ValueError("invalid run name")
        rows = [CaseResult(**json.loads(l)) for l in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        labels: list[str] = []
        lights = []
        for i, r in enumerate(rows):
            if r.agent not in labels:
                labels.append(r.agent)
            lights.append(light(r, str(i), labels.index(r.agent)))
        targets = [{"label": l, "provider": l.split(":", 1)[0], "model": l.split(":", 1)[-1]} for l in labels]
        return targets, lights, rows


def make_handler(state: State, token: str, allowed_hosts: set[str]):
    index_html = (STATIC / "index.html").read_text(encoding="utf-8")
    run_cache: dict[str, list[CaseResult]] = {}

    class Handler(BaseHTTPRequestHandler):
        server_version = "injectbench"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=str).encode(), "application/json")

        def _error(self, msg: str, status: int = 400) -> None:
            self._json({"error": msg}, status)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in allowed_hosts

        def _authed(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Token", ""), token)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if not self._host_ok():
                return self._error("bad host", 403)
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                q = parse_qs(url.query)
                if not secrets.compare_digest((q.get("token") or [""])[0], token):
                    return self._send(403, b"Open the dashboard with the link printed in the terminal "
                                           b"(it includes an access token).", "text/plain; charset=utf-8")
                return self._send(200, index_html.replace("__INJECTBENCH_TOKEN__", token).encode(),
                                  "text/html; charset=utf-8")
            if url.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not self._authed():
                return self._error("unauthorized", 401)
            parts = [unquote(p) for p in url.path.strip("/").split("/")]
            q = parse_qs(url.query)
            try:
                if parts == ["api", "config"]:
                    return self._json(state.config())
                if parts == ["api", "runs"]:
                    return self._json(state.runs())
                if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                    targets, lights, rows = state.load_run(parts[2])
                    run_cache[parts[2]] = rows
                    return self._json({"name": parts[2], "targets": targets, "records": lights})
                if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "cases":
                    rows = run_cache.get(parts[2]) or state.load_run(parts[2])[2]
                    return self._json(detail(rows[int(parts[4])]))
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job = state.jobs.get(parts[2])
                    if not job:
                        return self._error("no such job", 404)
                    snap = state.snapshot(job, int((q.get("since") or ["0"])[0]))
                    if (q.get("full") or ["0"])[0] == "1":
                        with job.lock:
                            snap["records"] = list(job.lights)
                            snap["order"] = list(job.order)
                    return self._json(snap)
                if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "cases":
                    job = state.jobs.get(parts[2])
                    res = job.results.get(parts[4]) if job else None
                    if not res:
                        return self._error("case not finished yet", 404)
                    return self._json(detail(res))
            except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
                return self._error(str(exc), 400)
            return self._error("not found", 404)

        def do_POST(self):
            if not self._host_ok():
                return self._error("bad host", 403)
            if not self._authed():
                return self._error("unauthorized", 401)
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._error("expected application/json", 415)
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
            try:
                body = self._body()
                if parts == ["api", "keys"]:
                    pid = body.get("provider")
                    if pid not in PROVIDERS:
                        return self._error("unknown provider")
                    key = (body.get("key") or "").strip()
                    if key:
                        state.keys[pid] = key
                    elif body.get("clear"):
                        state.keys.pop(pid, None)
                    if "base_url" in body and PROVIDERS[pid].editable_base_url:
                        state.base_urls[pid] = (body.get("base_url") or "").strip() or PROVIDERS[pid].base_url
                    return self._json(state.config())
                if parts == ["api", "models"]:
                    pid = body.get("provider")
                    if pid not in PROVIDERS:
                        return self._error("unknown provider")
                    try:
                        models = list_models(pid, state.key_for(pid), state.base_urls.get(pid))
                    except Exception as exc:
                        return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
                    return self._json({"ok": True, "models": models})
                if parts == ["api", "probe"]:
                    pid = body.get("provider")
                    if pid not in PROVIDERS or not (body.get("model") or "").strip():
                        return self._error("provider and model are required")
                    return self._json(probe_model(pid, body["model"].strip(), state.key_for(pid),
                                                  body.get("base_url") or state.base_urls.get(pid)))
                if parts == ["api", "estimate"]:
                    return self._json(state.estimate(body))
                if parts == ["api", "jobs"]:
                    job = state.start_job(body)
                    return self._json({"id": job.id, "total": job.total})
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                    job = state.jobs.get(parts[2])
                    if not job:
                        return self._error("no such job", 404)
                    job.cancel.set()
                    if job.status == "running":
                        job.status = "cancelling"
                    return self._json({"ok": True})
            except ValueError as exc:
                return self._error(str(exc), 400)
            except Exception as exc:
                return self._error(f"{type(exc).__name__}: {exc}", 500)
            return self._error("not found", 404)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, results_dir: str = "results",
          techniques_file: Optional[str] = None, open_browser: bool = True) -> None:
    state = State(Path(results_dir), techniques_file)
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer((host, port), None)  # type: ignore[arg-type]
    real_port = server.server_address[1]
    hosts = {f"{h}:{real_port}" for h in ("127.0.0.1", "localhost", "[::1]", host)}
    server.RequestHandlerClass = make_handler(state, token, hosts)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{real_port}/?token={token}"
    print(f"injectbench dashboard running at\n\n    {url}\n\nKeys you enter stay in this process's "
          "memory only. Press Ctrl+C to stop.", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: bound to a non-loopback address; anyone with the link can use your keys.", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()

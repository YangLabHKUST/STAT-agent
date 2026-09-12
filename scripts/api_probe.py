#!/usr/bin/env python3
"""Drive a sequence of queries through a running STAT web server and print a compact trace.

    python scripts/api_probe.py --base http://127.0.0.1:8889 --dataset ~/Downloads/bc_demo \
        --model poe/claude-opus-4.8 --provider poe --mode interactive --out probe.json \
        "What can you do for me?" "Describe the data" "Detect spatial niches"

Prompts (clarifications) are answered from a small rule table so the run never blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

REPLY_RULES = [
    (r"species", "human"),
    (r"tissue", "breast cancer"),
    (r"how many|number of (niche|cluster|domain)|expected", "8"),
    (r"reference", "No reference dataset is available; use a marker-based approach instead."),
    (r"gene list|which genes|genes of interest", "Use the top 50 marker genes from the differential expression results already in the session."),
    (r"root", "Myoepi KRT15+"),
    (r"pathway|gene set", "HALLMARK_INFLAMMATORY_RESPONSE"),
]


def answer_for(prompt: dict) -> dict:
    kind = prompt.get("kind")
    if kind == "skill_selection":
        return {"prompt_id": prompt["prompt_id"], "kind": "option", "index": 1}
    questions = [q.get("text", "") for q in prompt.get("questions", [])] or [prompt.get("question") or ""]
    answers = {}
    for q in questions:
        ans = "Proceed with sensible defaults."
        for pat, a in REPLY_RULES:
            if re.search(pat, q, re.I):
                ans = a
                break
        answers[q] = ans
    if kind == "verifier":
        return {"prompt_id": prompt["prompt_id"], "kind": "prerequisites", "answers": answers}
    return {"prompt_id": prompt["prompt_id"], "kind": "answer", "text": next(iter(answers.values()))}


def run_turn(client: httpx.Client, base: str, message: str, mode: str, timeout_s: float) -> dict:
    t0 = time.time()
    r = client.post(f"{base}/api/turns", json={"message": message, "mode": mode})
    if r.status_code != 202:
        return {"message": message, "error": f"{r.status_code} {r.text[:300]}"}
    tid = r.json()["turn_id"]
    replies = []
    state = None
    seen_prompts = set()
    while time.time() - t0 < timeout_s:
        snap = client.get(f"{base}/api/turns/{tid}").json()["turn"]
        state = snap["state"]
        if state in ("done", "error", "cancelled"):
            break
        if state == "awaiting_input":
            evs = client.get(f"{base}/api/turns/{tid}/events", params={"format": "json"}).json()
            evs = evs.get("events", evs) if isinstance(evs, dict) else evs
            prompts = [e for e in evs if e["type"] == "prompt" and e["prompt_id"] not in seen_prompts]
            answered = {e["prompt_id"] for e in evs if e["type"] == "prompt_answered"}
            prompts = [p for p in prompts if p["prompt_id"] not in answered]
            if prompts:
                p = prompts[-1]
                seen_prompts.add(p["prompt_id"])
                dec = answer_for(p)
                rr = client.post(f"{base}/api/turns/{tid}/reply", json=dec)
                replies.append({"kind": p["kind"], "question": (p.get("question") or " | ".join(q.get("text", "") for q in p.get("questions", [])))[:160], "reply": dec, "status": rr.status_code})
        time.sleep(1.5)
    evs = client.get(f"{base}/api/turns/{tid}/events", params={"format": "json"}).json()
    evs = evs.get("events", evs) if isinstance(evs, dict) else evs
    types = {}
    for e in evs:
        types[e["type"]] = types.get(e["type"], 0) + 1
    done = next((e for e in evs if e["type"] in ("turn_done", "turn_error", "turn_cancelled")), {})
    out = {
        "turn_id": tid,
        "message": message,
        "state": state,
        "wall_s": round(time.time() - t0, 1),
        "end": done.get("type"),
        "end_state": done.get("state") or done.get("code"),
        "error": done.get("message") if done.get("type") == "turn_error" else None,
        "replies": replies,
        "plan": [(s["description"][:80], s.get("target_slices")) for e in evs if e["type"] == "plan_updated" for s in e["steps"]][:4],
        "skill": next((e["skill"] for e in evs if e["type"] == "skill_selected"), None),
        "matched": next((e["matched"] for e in evs if e["type"] == "match_complete"), None),
        "verify": next(((e["decided_by"], [(v["prerequisite"][:40], v["status"]) for v in e["verdicts"]]) for e in evs if e["type"] == "verify_complete"), None),
        "unguarded": next((e["reason"][:80] for e in evs if e["type"] == "unguarded_mode"), None),
        "method_card": next((e["card"].get("method_name") for e in evs if e["type"] == "method_card"), None),
        "refusal": next((e["reason"][:200] for e in evs if e["type"] == "refusal"), None),
        "code_blocks": [("ok" if e["success"] else f"FAIL {(e.get('error') or {}).get('type')}: {((e.get('error') or {}).get('message') or '')[:160]}", e["duration_ms"], e.get("attempt")) for e in evs if e["type"] == "code_block_done"],
        "reflections": [e.get("summary", "")[:120] for e in evs if e["type"] == "reflection"],
        "figures": sum(1 for e in evs if e["type"] == "figure"),
        "provenance": [c.get("artifact") for e in evs if e["type"] == "provenance_update" for c in e.get("changes", [])][:8],
        "stale_now": [x for e in evs if e["type"] == "provenance_update" for x in e.get("stale_now", [])][:8],
        "assumptions": [(e["question"][:60], e["chosen"][:40]) for e in evs if e["type"] == "assumption"],
        "usage": done.get("usage"),
        "response": (done.get("response") or "")[:900],
        "event_counts": types,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8889")
    ap.add_argument("--dataset")
    ap.add_argument("--session-name", default="probe")
    ap.add_argument("--model", default="poe/claude-opus-4.8")
    ap.add_argument("--provider", default="poe")
    ap.add_argument("--mode", default="interactive")
    ap.add_argument("--timeout", type=float, default=1500)
    ap.add_argument("--out")
    ap.add_argument("--no-init", action="store_true", help="use the session that is already loaded")
    ap.add_argument("queries", nargs="+")
    a = ap.parse_args()
    client = httpx.Client(timeout=900)
    if not a.no_init:
        r = client.post(f"{a.base}/api/session", json={"dataset_dir": str(Path(a.dataset).expanduser()), "session_name": a.session_name, "provider": a.provider, "model": a.model, "mode": a.mode})
        if r.status_code != 200:
            print("session init failed:", r.status_code, r.text[:300])
            return 2
        s = r.json()["summary"]
        print(f"session {s['name']}: {s['n_slices']} slice(s), {[ (x['n_obs'], x['n_vars'], x['data_level']) for x in s['slices']]}")
    results = []
    for q in a.queries:
        print(f"\n=== {q}")
        res = run_turn(client, a.base, q, a.mode, a.timeout)
        results.append(res)
        for k in ("state", "wall_s", "end", "end_state", "error", "replies", "plan", "skill", "matched", "verify", "unguarded", "method_card", "refusal", "code_blocks", "reflections", "figures", "provenance", "stale_now", "assumptions", "usage"):
            v = res.get(k)
            if v not in (None, [], {}, ""):
                print(f"  {k}: {v}")
        print("  response:", (res.get("response") or "").replace("\n", " ")[:500])
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

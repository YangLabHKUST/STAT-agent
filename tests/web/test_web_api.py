from __future__ import annotations

import time

from tests.web.conftest import init_session, sse_frames


def wait_state(client, turn_id, states, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/turns/{turn_id}").json()["turn"]
        if snap["state"] in states:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"turn {turn_id} never reached {states}: {snap}")


def test_meta_and_session_lifecycle(client, dataset_dir):
    m = client.get("/api/meta").json()
    assert m["protocol"] == 2 and m["success"]
    assert client.get("/api/session/snapshot").status_code == 400
    s = init_session(client, dataset_dir)
    assert s["agent_active"] and s["summary"]["n_slices"] == 1 and "current_slice_id" not in s["summary"]
    snap = client.get("/api/session/snapshot").json()
    assert snap["ledger"]["version"] == 1 and snap["last_seq"] == 0
    legacy = client.get("/api/session/summary").json()
    assert legacy["summary"]["slice_ids"] == [0]


def test_data_endpoints_are_read_only(client, dataset_dir):
    init_session(client, dataset_dir)
    h0 = client.get("/api/debug/consistency").json()["hash"]
    img = client.get("/api/slices/0/image").json()
    assert img["success"] and img["image_width"] == 100 and img["is_scatter"] is False
    cells = client.get("/api/slices/0/cells").json()
    assert cells["n_cells"] == 600 and cells["has_celltype"] is False
    genes = client.get("/api/slices/0/genes").json()
    assert genes["n_genes"] == 40
    expr = client.get("/api/slices/0/expression", params={"gene": genes["genes"][0]}).json()
    assert len(expr["cells"]["expression"]) == 600
    assert client.get("/api/slices/0/expression", params={"gene": "NOPE"}).status_code == 404
    assert client.get("/api/cells/overlay", params={"slice_id": 0}).json()["success"]
    assert client.get("/api/slices/9/cells").status_code == 404
    assert client.get("/api/debug/consistency").json()["hash"] == h0


def test_roi_crud_updates_ledger_and_stream(client, dataset_dir):
    init_session(client, dataset_dir)
    r = client.post("/api/rois", json={"name": "tumor", "slice_id": 0, "type": "bbox", "params": {"min_x": 0, "min_y": 0, "max_x": 50, "max_y": 50}})
    assert r.status_code == 200 and r.json()["n_cells"] > 0 and r.json()["provenance_update"]["summary"]["rois_added"] == ["tumor"]
    assert r.headers["X-STAT-Ledger-Version"] == "2"
    assert client.post("/api/rois", json={"name": "tumor", "slice_id": 0, "type": "bbox", "params": {"min_x": 0, "min_y": 0, "max_x": 5, "max_y": 5}}).status_code == 409
    assert client.patch("/api/rois/tumor", json={"new_name": "core"}).status_code == 200
    rois = client.get("/api/rois").json()["rois"]
    assert [x["name"] for x in rois] == ["core"]
    assert client.delete("/api/rois/core").status_code == 200 and client.get("/api/rois").json()["rois"] == []
    assert client.delete("/api/rois/core").status_code == 404
    events = client.get("/api/turns", params={}).json()
    assert events["success"]


def test_turn_prompt_reply_and_replay(client, dataset_dir):
    init_session(client, dataset_dir)
    r = client.post("/api/turns", json={"message": "annotate cell types", "client_turn_id": "c1"})
    assert r.status_code == 202
    turn_id = r.json()["turn_id"]
    snap = wait_state(client, turn_id, {"awaiting_input"})
    assert snap["pending_prompt"]["kind"] == "verifier"
    # a second turn is refused while this one waits
    assert client.post("/api/turns", json={"message": "again"}).status_code == 409
    # wrong prompt id is refused
    bad = client.post(f"/api/turns/{turn_id}/reply", json={"prompt_id": "nope", "kind": "answer", "text": "x"})
    assert bad.status_code == 409 and bad.json()["error"]["code"] == "prompt_mismatch"
    ok = client.post(f"/api/turns/{turn_id}/reply", json={"prompt_id": snap["pending_prompt_id"], "kind": "prerequisites", "answers": {"q1": "brain"}})
    assert ok.status_code == 202
    done = wait_state(client, turn_id, {"done", "error"})
    assert done["state"] == "done" and "annotate" in done["response"].lower()
    frames = client.get(f"/api/turns/{turn_id}/events").json()["events"]
    types = [f["type"] for f in frames]
    assert types[0] == "turn_started" and "prompt" in types and "prompt_answered" in types and "provenance_update" in types and types[-1] == "turn_done"
    seqs = [f["seq"] for f in frames]
    assert seqs == sorted(seqs)
    cells = client.get("/api/slices/0/cells").json()
    assert cells["has_celltype"] and cells["celltype_colors"]
    snapshot = client.get("/api/session/snapshot").json()
    assert "s0:obs:celltype" in snapshot["ledger"]["artifacts"]


def test_sse_stream_replays_frames(client, dataset_dir):
    init_session(client, dataset_dir)
    client.post("/api/rois", json={"name": "r1", "slice_id": 0, "type": "bbox", "params": {"min_x": 0, "min_y": 0, "max_x": 50, "max_y": 50}})
    assert client.get("/api/turns/t9999/events").status_code == 404
    resp = client.get("/api/events", params={"since": "0", "once": "1"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    frames = sse_frames(resp.text)
    assert frames[0]["event"] == "hello" and frames[0]["data"]["protocol"] == 2
    prov = [f for f in frames if f["event"] == "provenance_update"]
    assert prov and prov[0]["id"] == 1 and prov[0]["data"]["summary"]["rois_added"] == ["r1"]
    resp2 = client.get("/api/events", params={"since": "1", "once": "1"})
    assert [f["event"] for f in sse_frames(resp2.text)] == ["hello"]


def test_cancel_and_reset(client, dataset_dir):
    init_session(client, dataset_dir)
    r = client.post("/api/turns", json={"message": "annotate cell types"})
    turn_id = r.json()["turn_id"]
    wait_state(client, turn_id, {"awaiting_input"})
    c = client.post(f"/api/turns/{turn_id}/cancel")
    assert c.status_code == 202 and c.json()["state"] == "cancelled"
    assert client.post("/api/turns", json={"message": "annotate cell types"}).status_code == 202
    assert client.delete("/api/session").status_code in (200, 409)
    assert client.delete("/api/session", params={"force": "1"}).status_code == 200
    assert client.get("/api/session/snapshot").status_code == 400


def test_legacy_aliases_and_errors(client, dataset_dir):
    init_session(client, dataset_dir)
    assert client.post("/api/select_slice", json={"slice_id": 0}).json()["success"]
    assert client.get("/api/skills/list").json()["success"]
    assert client.get("/api/nope").status_code == 404 and client.get("/api/nope").json()["error"]["code"] == "not_found"
    r = client.post("/api/rois", json={"name": "", "type": "bbox"})
    assert r.status_code == 400
    assert client.get("/api/logs/info").json()["success"]
    assert client.get("/api/export/report.md").status_code == 200


def test_session_save_and_load_roundtrip(client, dataset_dir, tmp_path):
    init_session(client, dataset_dir, mode="auto")
    r = client.post("/api/turns", json={"message": "annotate cell types"})
    assert r.status_code == 202
    wait_state(client, r.json()["turn_id"], {"done"})
    saved = client.post("/api/session/save", json={"dir": str(tmp_path / "saved")}).json()
    assert saved["success"] and (tmp_path / "saved" / "manifest.json").exists()
    r = client.post("/api/session", json={"session_dir": str(tmp_path / "saved"), "model": "scripted/model", "provider": "openai", "mode": "interactive"})
    assert r.status_code == 200, r.text
    cells = client.get("/api/slices/0/cells").json()
    assert cells["has_celltype"] is True
    snap = client.get("/api/session/snapshot").json()
    assert snap["ledger"]["artifacts"]["s0:obs:celltype"]["status"] == "fresh"
    turns = client.get("/api/turns").json()
    assert turns["success"]
    hist = client.get("/api/chat/history").json()
    assert hist["success"] and any("annotate" in (m.get("content") or "").lower() for m in hist["history"])


def test_skill_draft_endpoint(client, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HOME", str(tmp_path / "home"))
    init_session(client, dataset_dir, mode="auto")
    r = client.post("/api/turns", json={"message": "run RNA velocity"})
    wait_state(client, r.json()["turn_id"], {"done", "error"})
    bad = client.post("/api/skills/draft", json={"turn_id": "t9999", "step": 1})
    assert bad.status_code == 400

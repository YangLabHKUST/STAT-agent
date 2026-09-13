from __future__ import annotations

import json
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


def test_bad_dataset_path_keeps_the_live_session(client, dataset_dir):
    init_session(client, dataset_dir)
    r = client.post("/api/session", json={"dataset_dir": "/nope/nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert client.get("/api/session/summary").status_code == 200


def test_new_session_tells_the_old_event_log(client, dataset_dir):
    init_session(client, dataset_dir)
    old = client.app.state.stat.events
    init_session(client, dataset_dir)
    assert client.app.state.stat.events is not old
    assert any(t == "session_reset" for _, t, _ in old.replay(0))


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
    frames = client.get(f"/api/turns/{turn_id}/events").json()["events"]
    assert "turn_cancelled" in [f["type"] for f in frames] and [f for f in frames if f["type"] == "turn_done"][-1]["state"] == "cancelled"
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


def test_skills_list_has_titles_for_the_panel(client, dataset_dir):
    init_session(client, dataset_dir)
    skills = client.get("/api/skills/list").json()["skills"]
    assert skills and all(s.get("title") and s.get("slug") and "enabled" in s for s in skills)


def test_imageless_slice_returns_extent_without_a_jpeg():
    from tests.fixtures.sessions import make_session

    from stat_agent.web.render import image_payload

    s = make_session("S1_cell_gene").get_slice(0)
    s.images.clear()
    payload = image_payload(s)
    assert payload["is_scatter"] is True and payload["image_data"] == ""
    assert payload["image_width"] >= 1 and payload["image_height"] >= 1


def test_roi_list_reports_busy_while_the_agent_holds_the_session(client, dataset_dir):
    init_session(client, dataset_dir)
    stat = client.app.state.stat
    assert stat.session.lock.acquire(timeout=1, holder="turn:test")
    try:
        r = client.get("/api/roi/list")
        assert r.status_code == 409 and "busy" in r.text
    finally:
        stat.session.lock.release()


def test_reply_accepts_documented_shapes(client, dataset_dir):
    """{prompt_id, answers} without a kind, {prompt_id, option_index}, and an empty reply is a 400."""
    init_session(client, dataset_dir)
    turn_id = client.post("/api/turns", json={"message": "annotate cell types"}).json()["turn_id"]
    snap = wait_state(client, turn_id, {"awaiting_input"})
    pid = snap["pending_prompt_id"]
    empty = client.post(f"/api/turns/{turn_id}/reply", json={"prompt_id": pid})
    assert empty.status_code == 400
    ok = client.post(f"/api/turns/{turn_id}/reply", json={"prompt_id": pid, "answers": {"q1": "brain"}})
    assert ok.status_code == 202
    assert wait_state(client, turn_id, {"done", "error"})["state"] == "done"
    frames = client.get(f"/api/turns/{turn_id}/events").json()["events"]
    answered = [f for f in frames if f["type"] == "prompt_answered"]
    assert answered and "brain" in json.dumps(answered[0]["decision"])


def test_snapshot_consistency_and_save_report_busy(client, dataset_dir):
    init_session(client, dataset_dir)
    stat = client.app.state.stat
    assert stat.session.lock.acquire(timeout=1, holder="turn:test")
    try:
        for method, path in (("GET", "/api/session/snapshot"), ("GET", "/api/debug/consistency"), ("POST", "/api/session/save")):
            r = client.request(method, path, json={} if method == "POST" else None)
            assert r.status_code == 409 and r.json()["error"]["code"] == "locked", (path, r.status_code, r.text[:120])
    finally:
        stat.session.lock.release()


def test_events_resync_when_the_client_is_ahead_of_the_log(client, dataset_dir):
    init_session(client, dataset_dir)
    frames = sse_frames(client.get("/api/events", params={"since": "999999", "once": "1"}).text)
    kinds = [f["event"] for f in frames]
    assert "resync" in kinds and [f for f in frames if f["event"] == "resync"][0]["data"]["reason"] == "ahead_of_log"


def test_reply_body_aliases_map_to_decisions():
    """The documented aliases (answer, option_index, answers without kind) map like the viewer's fields; an empty reply is refused."""
    import pytest

    from stat_agent.web.errors import ApiError
    from stat_agent.web.turns import Turn, TurnManager

    turn = Turn(turn_id="t1", generation=1, message="m", mode="interactive", driver="staged")
    turn.pending_prompt = {"prompt_id": "p", "kind": "skill_selection", "questions": [], "options": [{"index": 1, "slug": "a"}, {"index": 2, "slug": "b"}]}
    assert TurnManager._to_decision(turn, {"prompt_id": "p", "option_index": 2}).option_index == 2
    assert TurnManager._to_decision(turn, {"prompt_id": "p", "answer": "2"}).option_index == 2
    turn.pending_prompt = {"prompt_id": "p", "kind": "verifier", "questions": [{"id": "q1", "text": "Which tissue?"}], "options": []}
    d = TurnManager._to_decision(turn, {"prompt_id": "p", "answers": {"q1": "brain"}})
    assert d.kind == "verifier" and d.answers == {"Which tissue?": "brain"}
    assert TurnManager._to_decision(turn, {"prompt_id": "p", "answer": "brain"}).answer == "brain"
    with pytest.raises(ApiError):
        TurnManager._to_decision(turn, {"prompt_id": "p"})


def test_waiting_for_the_session_lock_does_not_block_other_requests(client, dataset_dir):
    """A viewer request that finds the session locked (a block is running) waits for its 409 in a worker
    thread; the event loop keeps serving requests that take no lock (B38)."""
    import threading

    init_session(client, dataset_dir)
    session = client.app.state.stat.session
    held = threading.Event()

    def hold():
        with session.lock.hold("test:block"):
            held.set()
            time.sleep(1.2)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(2)
    codes: list[int] = []
    workers = [threading.Thread(target=lambda: codes.append(client.get("/api/slices/0/cells").status_code)) for _ in range(3)]
    for w in workers:
        w.start()
    time.sleep(0.05)
    t0 = time.time()
    r = client.get("/api/meta")
    dt = time.time() - t0
    for w in workers:
        w.join()
    holder.join()
    assert r.status_code == 200 and dt < 0.15, dt
    assert codes == [409, 409, 409]


def test_celltype_rename_merges_into_an_existing_name_and_refuses_unknown_names(client, tmp_path):
    """Fourth robustness run F11: renaming onto an existing cell type merges the cells into it (the
    old label disappears, the target keeps its colour), a plain rename still relabels, an unknown
    name is 404, an empty name 400, and the same name changes nothing."""
    from tests.fixtures.sessions import make_session, write_dataset

    init_session(client, write_dataset(make_session("S1_cell_gene"), tmp_path / "ds_ct"))
    s = client.app.state.stat.session.get_slice(0)
    before = s.adata.obs["celltype"].value_counts()
    n_t, n_b, n_cells = int(before["T cell"]), int(before["B cell"]), len(s.adata.obs)
    colors_before = dict(s.get_celltype_colors() or {})
    r = client.patch("/api/slices/0/celltypes/T cell", json={"new_name": "B cell"})
    assert r.status_code == 200, r.text
    ev = r.json()["provenance_update"]
    assert ev["origin"] == "gui" and ev["summary"]["celltypes_updated"] == [0] and ev["producer"]["label"] == "celltype_rename 'T cell' -> 'B cell'"
    col = s.adata.obs["celltype"]
    assert "T cell" not in list(col.cat.categories) and int(col.value_counts()["B cell"]) == n_t + n_b and len(col) == n_cells
    colors_after = s.get_celltype_colors() or {}
    if "B cell" in colors_before:
        assert colors_after.get("B cell") == colors_before["B cell"] and "T cell" not in colors_after
    assert client.patch("/api/slices/0/celltypes/Tumor", json={"new_name": "Cancer"}).status_code == 200
    col = s.adata.obs["celltype"]
    assert "Cancer" in list(col.cat.categories) and "Tumor" not in list(col.cat.categories) and int(col.value_counts()["Cancer"]) == int(before["Tumor"])
    assert client.patch("/api/slices/0/celltypes/T cell", json={"new_name": "X"}).status_code == 404
    assert client.patch("/api/slices/0/celltypes/Nope", json={"new_name": "X"}).status_code == 404
    assert client.patch("/api/slices/0/celltypes/B cell", json={"new_name": "  "}).status_code == 400
    assert client.patch("/api/slices/0/celltypes/B cell", json={"new_name": "B cell"}).status_code == 200
    assert list(s.adata.obs["celltype"].cat.categories) == list(col.cat.categories) and len(s.adata.obs) == n_cells

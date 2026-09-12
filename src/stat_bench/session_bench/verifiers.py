"""Verifiers over the live session, ledger, memory and events (no LLM judge)."""

from __future__ import annotations

import re


def column_exists(ctx, slice: int = 0, key: str = "celltype", kind: str = "obs", **_):
    s = ctx.session.get_slice(slice)
    space = {"obs": s.adata.obs, "obsm": s.adata.obsm, "uns": s.adata.uns}[kind]
    keys = list(space.columns) if hasattr(space, "columns") else list(space.keys())
    return key in keys, f"{kind}:{key} {'present' if key in keys else 'absent'} on slice {slice}"


def artefact_inputs_from_roi(ctx, key: str, roi: str, slice: int = 0, **_):
    art = ctx.ledger.get(f"s{slice}:{key}")
    ok = art is not None and f"roi:{roi}" in art.inputs
    return ok, f"inputs of s{slice}:{key} = {art.inputs if art else None}"


def stale_marked(ctx, key: str, slice: int = 0, **_):
    return ctx.ledger.is_stale(f"s{slice}:{key}"), f"stale={ctx.ledger.is_stale(f's{slice}:{key}')}"


def not_stale(ctx, key: str, slice: int = 0, **_):
    return not ctx.ledger.is_stale(f"s{slice}:{key}"), "fresh"


def no_recompute(ctx, key: str, since_turn: int, slice: int = 0, **_):
    """The artefact is still the one produced at (or before) script turn ``since_turn``."""
    art = ctx.ledger.get(f"s{slice}:{key}")
    allowed = {tid for t, tid in ctx.turn_ids.items() if t <= since_turn}
    ok = art is not None and art.produced_by is not None and art.produced_by.turn_id in allowed
    return ok, f"produced_by={art.produced_by.turn_id if art and art.produced_by else None}"


def session_usable_after_failure(ctx, t_fail: int, t_next: int, **_):
    a, b = ctx.turn_status.get(t_fail), ctx.turn_status.get(t_next)
    ok = a in ("blocked", "error", "failed") and b == "done"
    return ok, f"turn {t_fail}={a}, turn {t_next}={b}"


def _tid(ctx, turn: int | None) -> str | None:
    return None if turn is None else ctx.turn_ids.get(turn, f"t{turn:04d}")


def clarification_asked(ctx, kind: str, regex: str = ".", turn: int | None = None, **_):
    evs = [e for e in ctx.events if e.type == "prompt" and e.kind == kind and (turn is None or e.turn_id == _tid(ctx, turn))]
    hit = [e for e in evs if re.search(regex, (e.question or " ".join(q.text for q in e.questions)), re.I)]
    return bool(hit), f"{len(hit)} matching prompt(s)"


def assumption_recorded(ctx, kind: str, turn: int | None = None, **_):
    evs = [e for e in ctx.events if e.type == "assumption" and e.kind == kind and (turn is None or e.turn_id == _tid(ctx, turn))]
    return bool(evs), f"{len(evs)} assumption(s)"


def refusal_names(ctx, regex: str, turn: int | None = None, **_):
    evs = [e for e in ctx.events if e.type == "refusal" and (turn is None or e.turn_id == _tid(ctx, turn))]
    hit = [e for e in evs if re.search(regex, (e.requirement or "") + " " + (e.reason or ""), re.I)]
    return bool(hit), f"{len(hit)} refusal(s) naming /{regex}/"


def answer_mentions(ctx, regex: str, turn: int | None = None, **_):
    t = ctx.memory.turns[-1] if turn is None and ctx.memory.turns else ctx.record(turn) if turn is not None else None
    ok = bool(t and re.search(regex, t.response or "", re.I))
    return ok, f"response {'mentions' if ok else 'omits'} /{regex}/"


def answer_omits(ctx, regex: str, turn: int | None = None, **_):
    ok, msg = answer_mentions(ctx, regex, turn)
    return not ok, msg


def roi_members_equal(ctx, roi: str, key: str, value, slice: int = 0, **_):
    """cells with obs[key]==value are exactly the ROI members."""
    s = ctx.session.get_slice(slice)
    r = ctx.session.get_roi(roi)
    if r is None or key not in s.adata.obs.columns:
        return False, "missing ROI or column"
    members = set(r.adata.obs_names)
    labelled = set(s.adata.obs.index[s.adata.obs[key] == value])
    return members == labelled, f"{len(members)} members vs {len(labelled)} labelled"


def turn_status(ctx, turn: int, status: str, **_):
    st = ctx.turn_status.get(turn)
    return st == status, f"turn {turn} status={st}"


VERIFIERS = {f.__name__: f for f in (column_exists, artefact_inputs_from_roi, stale_marked, not_stale, no_recompute, session_usable_after_failure, clarification_asked, assumption_recorded, refusal_names, answer_mentions, answer_omits, roi_members_equal, turn_status)}

__all__ = ["VERIFIERS"]

"""Analysis report export (Markdown / HTML) from memory, ledger and assumptions."""

from __future__ import annotations

import html
from datetime import datetime


def render_report(agent, session, fmt: str = "md") -> str:
    lines: list[str] = []
    name = getattr(session, "name", "session")
    lines.append(f"# STAT analysis report: {name}")
    lines.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} | model {agent.llm.model_name} | mode {agent.mode}")
    lines.append("")
    lines.append("## Data")
    for s in session.iter_slices():
        lines.append(f"- Slice {s.slice_id}: {s.tissue_name} ({s.modality}, {s.data_level}-level, {s.n_obs:,} obs x {s.n_vars:,} vars)")
    rois = list(session.roi_manager.rois.values())
    if rois:
        lines.append("- ROIs: " + ", ".join(f"{r.name} (slice {r.slice_id}, {r.n_obs} obs)" for r in rois))
    lines.append("")
    lines.append("## Turns")
    for t in agent.memory.turns:
        lines.append(f"### Turn {t.index}: {t.query}")
        lines.append(f"Status: {t.status}")
        for h in t.handoffs:
            lines.append(f"- Step {h.step_number} [{h.skill or 'general'}] ({h.mode}{'' if h.success else ', failed'}): {h.summary}")
            if h.outputs:
                lines.append(f"  - outputs: {', '.join(h.outputs)}")
            if h.key_findings:
                lines.append("  - findings: " + "; ".join(h.key_findings))
        if t.assumptions:
            lines.append("- Assumptions: " + "; ".join(f"{a.question} -> {a.answer}" for a in t.assumptions))
        if t.response:
            lines.append("")
            lines.append(t.response)
        lines.append("")
    lines.append("## Provenance")
    for aid, a in sorted(agent.ledger.live().items()):
        if a.kind in ("X", "var"):
            continue
        src = f"{a.produced_by.skill or 'code'} (turn {a.produced_by.turn_id}, step {a.produced_by.step})" if a.produced_by else a.origin
        lines.append(f"- {aid} v{a.version} [{a.status}] <- {src}")
    md = "\n".join(lines)
    if fmt != "html":
        return md
    body = "".join(f"<p>{html.escape(ln)}</p>\n" if not ln.startswith("#") else f"<h{ln.count('#', 0, 3)}>{html.escape(ln.lstrip('# '))}</h{ln.count('#', 0, 3)}>\n" for ln in lines if ln.strip())
    return f"<!doctype html><html><head><meta charset='utf-8'><title>STAT report</title><style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.5}}</style></head><body>{body}</body></html>"


__all__ = ["render_report"]

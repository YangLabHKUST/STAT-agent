"""Figures, skills, logs, export and LLM-test routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from stat_agent.web.errors import ApiError
from stat_agent.web.routes.common import json, parse_body, state_of
from stat_agent.web.schemas import LLMTest, SkillPatch


async def figure(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    fid = request.path_params["figure_id"]
    png = agent.figures.get(fid)
    if png is None and app.session_dir:
        p = app.session_dir / "figures" / f"{fid}.png"
        if p.exists():
            png = p.read_bytes()
    if png is None:
        raise ApiError("not_found", f"figure {fid} not found", 404)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})


async def list_skills(request: Request):
    app = state_of(request)
    reg = app.agent.registry if app.agent else app.registry
    states = reg.states()
    return json({"success": True, "skills": [{"slug": s, **st} for s, st in states.items()], "states": {s: st["enabled"] for s, st in states.items()}, "broken": [{"path": str(b.path), "error": b.error} for b in reg.broken]})


async def toggle_skill(request: Request):
    app = state_of(request)
    reg = app.agent.registry if app.agent else app.registry
    slug = request.path_params.get("slug")
    body = await parse_body(request, SkillPatch)
    slug = slug or body.slug
    if not slug or slug not in reg:
        raise ApiError("not_found", f"unknown skill {slug}", 404)
    enabled = body.enabled
    if enabled is None:
        enabled = not reg.is_enabled(slug)
    (reg.enable if enabled else reg.disable)(slug)
    return json({"success": True, "slug": slug, "enabled": reg.is_enabled(slug)})


async def skill_detail(request: Request):
    app = state_of(request)
    reg = app.agent.registry if app.agent else app.registry
    slug = request.path_params["slug"]
    if slug not in reg:
        raise ApiError("not_found", f"unknown skill {slug}", 404)
    sk = reg.get(slug)
    return json({"success": True, "slug": slug, "name": sk.name, "title": sk.name, "description": sk.description, "body": sk.body, "content_md": sk.body, "frontmatter": sk.raw_frontmatter, "enabled": reg.is_enabled(slug), "origin": sk.origin, "path": str(sk.path)})


def _turn_entries(d: Path | None) -> list[dict]:
    if not d:
        return []
    out = []
    for i, p in enumerate(sorted(d.glob("turn_*.md")), 1):
        text = p.read_text(encoding="utf-8", errors="replace")
        preview = ""
        if "## User Query" in text:
            block = text.split("## User Query", 1)[1]
            lines = [ln for ln in block.split("```")[1].splitlines() if ln.strip()] if "```" in block else []
            preview = " ".join(lines)[:90]
        out.append({"filename": p.name, "turn_number": i, "size_kb": round(p.stat().st_size / 1024, 1), "preview": preview})
    return out


async def draft_skill(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    body = await request.json()
    try:
        result = await agent.draft_skill(body["turn_id"], int(body.get("step", 1)), extra_notes=body.get("notes", ""))
    except (KeyError, ValueError) as exc:
        raise ApiError("validation", str(exc), 400) from exc
    agent.registry.load()
    return json({"success": True, **{k: v for k, v in result.items() if k != "text"}, "text": result.get("text")})


async def logs_info(request: Request):
    app = state_of(request)
    d = app.session_dir
    entries = _turn_entries(d)
    return json({"success": True, "session_dir": str(d) if d else None, "turns": entries, "turn_logs": [e["filename"] for e in entries], "notebook": str(app.agent.notebook.notebook_file) if app.agent and app.agent.notebook else None, "jupyter_port": app.config.jupyter_port})


async def turn_log(request: Request):
    app = state_of(request)
    name = request.path_params["filename"]
    if not app.session_dir or "/" in name or not name.startswith("turn_"):
        raise ApiError("not_found", "log not found", 404)
    p = app.session_dir / name
    if not p.exists():
        raise ApiError("not_found", "log not found", 404)
    content = p.read_text(encoding="utf-8", errors="replace")
    if request.query_params.get("raw") == "1":
        return PlainTextResponse(content)
    return json({"success": True, "filename": name, "content": content})


async def notebook_content(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    nb = agent.notebook.notebook_file if agent.notebook else None
    if not nb or not Path(nb).exists():
        raise ApiError("not_found", "notebook not found", 404)
    return Response(Path(nb).read_text(encoding="utf-8"), media_type="application/json")


async def notebook_url(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    nb = Path(agent.notebook.notebook_file) if agent.notebook else None
    url = None
    if nb is not None:
        root = Path(app.config.jupyter_root or app.settings.resolved_logs_dir)
        try:
            rel = nb.resolve().relative_to(root.resolve())
        except ValueError:
            rel = Path(nb.name)
        url = f"http://localhost:{app.config.jupyter_port}/lab/tree/{rel.as_posix()}"
    return json({"success": True, "notebook_path": str(nb) if nb else None, "jupyter_port": app.config.jupyter_port, "url": url})


async def export_report(request: Request):
    app = state_of(request)
    agent = app.require_agent()
    from stat_agent.logging.report import render_report

    fmt = request.path_params.get("fmt", "md")
    text = render_report(agent, app.session, fmt=fmt)
    media = "text/html" if fmt == "html" else "text/markdown"
    return Response(text, media_type=media, headers={"Content-Disposition": f'attachment; filename="stat_report.{fmt}"'})


async def test_llm(request: Request):
    body = await parse_body(request, LLMTest)
    from stat_agent.llm import LLMClient, make_spec

    try:
        spec = make_spec(body.model or "openai/gpt-4o", provider=body.provider, api_key=body.api_key, base_url=body.base_url)
        client = LLMClient(spec=spec)
        comp = await asyncio.wait_for(client.complete([("user", "Reply with the single word OK.")], role="test"), timeout=60)
        return json({"success": True, "model": spec.id, "response": comp.text[:200], "usage": comp.usage})
    except Exception as exc:  # noqa: BLE001
        return json({"success": False, "error": f"{type(exc).__name__}: {exc}"}, status=502)


def routes() -> list[Route]:
    return [
        Route("/api/figures/{figure_id}.png", figure, methods=["GET"]),
        Route("/api/skills", list_skills, methods=["GET"]),
        Route("/api/skills/draft", draft_skill, methods=["POST"]),
        Route("/api/skills/{slug}", toggle_skill, methods=["PATCH"]),
        Route("/api/skills/{slug}/detail", skill_detail, methods=["GET"]),
        Route("/api/logs", logs_info, methods=["GET"]),
        Route("/api/logs/turns/{filename}", turn_log, methods=["GET"]),
        Route("/api/logs/notebook", notebook_content, methods=["GET"]),
        Route("/api/notebook/url", notebook_url, methods=["GET"]),
        Route("/api/export/report.{fmt}", export_report, methods=["GET"]),
        Route("/api/llm/test", test_llm, methods=["POST"]),
        # legacy aliases
        Route("/api/skills/list", list_skills, methods=["GET"]),
        Route("/api/skills/toggle", toggle_skill, methods=["POST"]),
        Route("/api/logs/info", logs_info, methods=["GET"]),
        Route("/api/logs/turn/{filename}", turn_log, methods=["GET"]),
        Route("/api/test-llm", test_llm, methods=["POST"]),
    ]

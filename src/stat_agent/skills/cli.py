"""``stat-agent skill ...`` sub-commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import yaml

from stat_agent.skills.lint import lint_file, lint_skill
from stat_agent.skills.registry import SkillRegistry


def _registry(args) -> SkillRegistry:
    paths = [(Path(p), f"path{i}") for i, p in enumerate(args.skills_root or [])]
    if paths:
        reg = SkillRegistry(paths)
        reg.load()
        return reg
    from stat_agent.config import Settings

    return SkillRegistry.from_settings(Settings.from_env())


def cmd_validate(args) -> int:
    reg = _registry(args)
    report = {"skills": {}, "broken": [{"path": str(b.path), "error": b.error} for b in reg.broken]}
    n_err = n_warn = 0
    for slug, skill in sorted(reg.skills.items()):
        if args.slug and slug not in args.slug:
            continue
        issues, facts = lint_skill(skill)
        errs = [i for i in issues if i.level == "error"]
        warns = [i for i in issues if i.level == "warn"]
        n_err += len(errs)
        n_warn += len(warns)
        report["skills"][slug] = {"issues": [i.__dict__ for i in issues], "facts": facts}
        if not args.json:
            status = "ERROR" if errs else ("warn" if warns else "ok")
            print(f"{slug:45s} {status:6s} errors={len(errs)} warnings={len(warns)} produces={len(skill.meta.produces)} consumes={len(skill.meta.consumes)}")
            if args.verbose:
                for i in issues:
                    if i.level != "info" or args.verbose > 1:
                        print(f"    {i}")
    for b in reg.broken:
        n_err += 1
        if not args.json:
            print(f"BROKEN {b.path}: {b.error}")
    if args.json:
        print(json.dumps(report, indent=2))
    elif not args.slug:
        print(f"\n{len(reg.skills)} skills, {len(reg.broken)} broken, {n_err} errors, {n_warn} warnings")
    if n_err:
        return 1
    if args.strict and n_warn:
        return 2
    return 0


def cmd_list(args) -> int:
    reg = _registry(args)
    for slug, st in reg.states().items():
        flag = "on " if st["enabled"] else "off"
        req = reg.get(slug).meta.filter_requirements.describe()
        print(f"{flag} {slug:45s} [{st['origin']}] {req}")
    return 0


def cmd_lint_file(args) -> int:
    issues, _ = lint_file(args.file)
    for i in issues:
        print(i)
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_export(args) -> int:
    """Write Agent-Skills-compatible copies (STAT fields nested under ``metadata``)."""
    reg = _registry(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for slug, skill in sorted(reg.skills.items()):
        raw = dict(skill.raw_frontmatter)
        front = {"name": slug, "description": raw.get("description", "")}
        stat_fields = {k: v for k, v in raw.items() if k not in ("name", "description")}
        front["metadata"] = {"stat": stat_fields}
        target = out / slug
        target.mkdir(exist_ok=True)
        (target / "SKILL.md").write_text("---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n\n" + skill.body + "\n", encoding="utf-8")
    print(f"exported {len(reg.skills)} skills to {out}")
    return 0


def cmd_new(args) -> int:
    from stat_agent.config import Settings

    settings = Settings.from_env()
    target = Path(args.dir) if args.dir else settings.user_skills_dir / args.slug
    if target.exists():
        print(f"{target} already exists", file=sys.stderr)
        return 1
    target.mkdir(parents=True)
    template = f"""---
schema_version: 2
name: {args.slug}
title: {args.title or args.slug.replace('-', ' ').title()}
slug: {args.slug}
description: >-
  One or two sentences: what the method computes and when a user would ask for it.
version: "1.0"

filter_requirements:
  modalities: [gene]
  data_levels: [cell/spot]

prerequisites:
  - "Describe every input the user must provide or that must already exist in the session"

consumes: []
produces:
  - obs:example_label
dependencies:
  python: []
  gpu: false
platforms: []
default_skill: true
---

# {args.title or args.slug}

Describe the method, then give a complete, runnable workflow in ```python fences that
reads from `session.get_slice(slice_id).adata` and writes results back into it.
"""
    (target / "SKILL.md").write_text(template, encoding="utf-8")
    print(f"created {target / 'SKILL.md'}")
    return 0


def add_skill_subparser(subparsers) -> None:
    p = subparsers.add_parser("skill", help="manage analysis skills")
    p.add_argument("--skills-root", action="append", help="skill directory to load instead of the defaults (repeatable)")
    sp = p.add_subparsers(dest="skill_cmd", required=True)

    v = sp.add_parser("validate", help="validate skill contracts and bodies")
    v.add_argument("--strict", action="store_true", help="warnings fail the command (exit 2)")
    v.add_argument("--json", action="store_true")
    v.add_argument("--slug", action="append")
    v.add_argument("-v", "--verbose", action="count", default=0)
    v.set_defaults(func=cmd_validate)

    ls = sp.add_parser("list", help="list skills and their applicability contracts")
    ls.set_defaults(func=cmd_list)

    lf = sp.add_parser("lint-file", help="lint one SKILL.md file")
    lf.add_argument("file")
    lf.set_defaults(func=cmd_lint_file)

    ex = sp.add_parser("export", help="export Agent-Skills-compatible copies")
    ex.add_argument("--out", required=True)
    ex.add_argument("--agent-skills", action="store_true", default=True)
    ex.set_defaults(func=cmd_export)

    nw = sp.add_parser("new", help="scaffold a new skill in the user skills directory")
    nw.add_argument("slug")
    nw.add_argument("--title")
    nw.add_argument("--dir")
    nw.set_defaults(func=cmd_new)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stat-agent skill")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_skill_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

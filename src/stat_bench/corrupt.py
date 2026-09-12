"""Corrupt skill frontmatter at three severities (schema-valid but semantically wrong)."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from stat_agent.skills.schema import split_frontmatter

KINDS = ("data_levels", "modalities", "num_slices", "prereq", "description")


def _mutate(raw: dict[str, Any], kind: str, severity: int, rng: random.Random, sibling_desc: str | None) -> tuple[dict[str, Any], str, str] | None:
    fr = dict(raw.get("filter_requirements") or {})
    before = json.dumps(raw.get("filter_requirements") if kind in ("data_levels", "modalities", "num_slices") else raw.get(kind if kind != "prereq" else "prerequisites"), default=str)
    if kind == "data_levels":
        levels = list(fr.get("data_levels") or [])
        if severity == 1 and levels == ["cell/spot"]:
            fr["data_levels"] = ["cell"]
        elif severity >= 2 and levels:
            fr["data_levels"] = ["cell"] if "spot" in levels[0] else ["spot"]
        elif not levels and severity >= 2:
            fr["data_levels"] = ["spot"]
        else:
            return None
        raw["filter_requirements"] = fr
    elif kind == "modalities":
        mods = list(fr.get("modalities") or [])
        if severity == 1 and mods == ["gene"]:
            fr["modalities"] = ["gene/protein"]
        elif severity >= 2 and mods:
            fr["modalities"] = ["protein"] if severity == 3 else ["gene/protein"]
        else:
            return None
        raw["filter_requirements"] = fr
    elif kind == "num_slices":
        if severity == 1 and "num_slices" in fr:
            fr.pop("num_slices")
        elif severity >= 2:
            fr["num_slices"] = 1
            for f in ("modalities", "data_levels"):
                if len(fr.get(f) or []) > 1:
                    fr[f] = [fr[f][0]]
        else:
            return None
        raw["filter_requirements"] = fr
    elif kind == "prereq":
        pre = list(raw.get("prerequisites") or [])
        if severity == 1 and len(pre) >= 2:
            pre = pre[:-1]
        elif severity == 2:
            pre = []
        elif severity == 3:
            pre = ["A CSV file of pathway scores per cell (adata.obs['pathway_score'])"]
        else:
            return None
        raw["prerequisites"] = pre
    elif kind == "description":
        desc = str(raw.get("description", ""))
        if severity == 1:
            words = desc.split()
            if len(words) > 6:
                del words[len(words) // 2]
            raw["description"] = " ".join(words)
        elif severity == 2 and sibling_desc:
            raw["description"] = sibling_desc
        elif severity == 3:
            raw["description"] = "Deconvolve spot-level data into cell type proportions using a single-cell reference and report per-spot fractions."
        else:
            return None
    after = json.dumps(raw.get("filter_requirements") if kind in ("data_levels", "modalities", "num_slices") else raw.get(kind if kind != "prereq" else "prerequisites"), default=str)
    if before == after:
        return None
    return raw, before, after


def corrupt_registry(src: Path, out: Path, *, severity: int = 1, kinds: tuple[str, ...] = KINDS, n: int = 6, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    skill_dirs = sorted(p for p in src.iterdir() if (p / "SKILL.md").exists())
    for d in skill_dirs:
        target = out / d.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(d, target, ignore=shutil.ignore_patterns("__pycache__", "extdata"))
    descs = {}
    for d in skill_dirs:
        raw, _ = split_frontmatter((d / "SKILL.md").read_text(encoding="utf-8"))
        descs[d.name] = raw.get("description", "")
    manifest = []
    candidates = list(skill_dirs)
    rng.shuffle(candidates)
    for d in candidates:
        if len(manifest) >= n:
            break
        text = (out / d.name / "SKILL.md").read_text(encoding="utf-8")
        raw, body = split_frontmatter(text)
        kind = rng.choice(list(kinds))
        sibling = descs[rng.choice([k for k in descs if k != d.name])]
        res = _mutate(dict(raw), kind, severity, rng, sibling)
        if res is None:
            continue
        new_raw, before, after = res
        new_text = "---\n" + yaml.safe_dump(new_raw, sort_keys=False, allow_unicode=True) + "---\n\n" + body + "\n"
        (out / d.name / "SKILL.md").write_text(new_text, encoding="utf-8")
        manifest.append({"slug": new_raw.get("slug") or new_raw.get("name"), "dir": d.name, "field": kind, "severity": severity, "before": before, "after": after})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


__all__ = ["corrupt_registry", "KINDS"]

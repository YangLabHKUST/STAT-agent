"""Every built-in SKILL.md must load, validate and lint without errors."""

from __future__ import annotations

import pytest

from stat_agent.skills import SkillRegistry, builtin_skills_path
from stat_agent.skills.lint import lint_skill


@pytest.fixture(scope="module")
def registry():
    reg = SkillRegistry([(builtin_skills_path(), "builtin")])
    reg.load()
    return reg


def test_all_builtin_skills_load(registry):
    assert not registry.broken, [f"{b.path}: {b.error}" for b in registry.broken]
    assert len(registry) == 26
    assert len(set(registry.slugs())) == len(registry.slugs())


def test_every_skill_has_valid_contract(registry):
    for slug, skill in registry.skills.items():
        req = skill.meta.filter_requirements
        for item in req.modalities:
            assert all(t in ("gene", "protein") for t in item.split("/")), slug
        for item in req.data_levels:
            assert all(t in ("cell", "spot") for t in item.split("/")), slug
        assert skill.meta.prerequisites, slug
        assert 20 <= len(skill.description) <= 700, slug


def test_lint_reports_no_errors(registry):
    all_errors = []
    for slug, skill in registry.skills.items():
        issues, facts = lint_skill(skill)
        all_errors += [f"{slug}: {i}" for i in issues if i.level == "error"]
        assert facts["n_blocks"] >= 1, f"{slug} has no code blocks"
    assert not all_errors, all_errors


def test_bodies_load_lazily_and_write_known_artefacts(registry):
    writes = set()
    for skill in registry.skills.values():
        _, facts = lint_skill(skill)
        writes |= set(facts["writes"])
    assert "obs:celltype" in writes and "obsm:deconv_weights" in writes


def test_catalog_and_enable_disable(registry):
    catalog = registry.catalog()
    slugs = [s for s, _ in catalog]
    assert len(slugs) == len(registry.enabled())
    assert all(registry.get(s).meta.default_skill for s in slugs)
    some = slugs[0]
    registry.disable(some)
    assert some not in dict(registry.catalog())
    registry.enable(some)
    assert some in dict(registry.catalog())

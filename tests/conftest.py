"""Shared pytest configuration for STAT v2."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("STAT_FAKE_LLM", "replay")


def pytest_collection_modifyitems(config, items):
    live = os.environ.get("STAT_LIVE") == "1"
    skills = os.environ.get("STAT_RUN_SKILLS") == "1"
    for item in items:
        if "live" in item.keywords and not live:
            item.add_marker(pytest.mark.skip(reason="set STAT_LIVE=1 to run live provider tests"))
        if "skills" in item.keywords and not skills:
            item.add_marker(pytest.mark.skip(reason="set STAT_RUN_SKILLS=1 to run real skill tests"))


@pytest.fixture(scope="session")
def fixtures():
    from tests.fixtures import sessions

    return sessions

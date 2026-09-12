from __future__ import annotations

import threading
import time

import pytest
from tests.fixtures.sessions import make_session

from stat_agent.runtime import Runtime, check_code, code_access, diff_snapshots, take_snapshot


@pytest.fixture
def rt():
    return Runtime(make_session("S1_cell_gene"), timeout_s=5, grace_s=1, lock_timeout_s=1)


def test_capture_ordered_output_and_streams(rt):
    lines = []
    r = rt.run("import sys\nprint('a')\nprint('b')\nsys.stderr.write('warn\\n')\nprint('c')", on_output=lambda s, ln: lines.append((s, ln)))
    assert r.success and r.stdout == "a\nb\nc\n" and r.stderr == "warn\n"
    assert [ln for s, ln in lines if s == "stdout"] == ["a", "b", "c"]
    assert ("stderr", "warn") in lines


def test_progress_bar_carriage_returns(rt):
    r = rt.run("import sys\nsys.stdout.write('10%\\r50%\\r100%\\n')\nprint('done')")
    assert r.stdout == "100%\ndone\n"


def test_namespace_persists_only_on_success(rt):
    assert rt.run("x = 41").success
    r = rt.run("y = x + 1\nprint(y)")
    assert r.success and r.stdout == "42\n" and "y" in rt.namespace
    bad = rt.run("z = 1\nraise KeyError('celltype')")
    assert not bad.success and bad.error_type == "KeyError" and "z" not in rt.namespace
    assert "celltype" in bad.error and "Traceback" in bad.traceback and "<stat-run-" in bad.traceback


def test_stdout_tail_available_after_failure(rt):
    r = rt.run("print('before crash')\nraise ValueError('boom')")
    assert "before crash" in r.stdout_tail and r.error_type == "ValueError"


def test_session_is_live_and_diff_detects_changes(rt):
    r = rt.run("s = session.get_slice(0)\ns.adata.obs['new_col'] = 1\ns.adata.uns['note'] = 'x'\nsession.create_roi('r', 0, {'type': 'bbox', 'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10})")
    assert r.success
    assert {"s0:obs:new_col", "s0:uns:note", "roi:r"} <= set(r.diff.added)
    assert "s0:obs:celltype" not in r.diff.changed
    r2 = rt.run("session.get_slice(0).adata.obs['celltype'] = 'T cell'")
    assert "s0:obs:celltype" in r2.diff.changed and r2.access.writes == {"obs:celltype"}


def test_figures_captured_and_closed(rt):
    r = rt.run("import matplotlib.pyplot as plt\nplt.figure(); plt.plot([1,2]); plt.figure(); plt.plot([2,1])\nplt.show()")
    assert r.success and len(r.figures) == 2 and r.figures[0][:8] == b"\x89PNG\r\n\x1a\n"
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "code, blocked",
    [
        ("import os\nos.system('ls')", True),
        ("import subprocess", True),
        ("eval('1+1')", True),
        ("from os import system\nsystem('x')", True),
        ("open('/etc/passwd', 'w')", True),
        ("import pandas as pd\npd.eval('1+1')", False),
        ("df.eval('a+b')", False),
        ("import os\nos.path.exists('x')", False),
        ("open('out.txt')", False),
    ],
)
def test_safety_table(code, blocked):
    assert bool(check_code(code, allowed_write_dirs=["/tmp/stat-work"])) is blocked


def test_safety_refuses_before_running(rt):
    r = rt.run("import os\nos.system('echo hi')")
    assert r.refused and not r.success and r.error_type == "SafetyViolation" and r.violations


def test_timeout_interrupts_pure_python_loop(rt):
    t0 = time.time()
    r = rt.run("import time\nwhile True:\n    time.sleep(0.01)", timeout_s=0.5)
    assert not r.success and r.timed_out and time.time() - t0 < 5
    assert not rt.orphaned
    assert rt.run("print(1)").success  # runtime usable afterwards


def test_code_access_scan():
    acc = code_access("s = session.get_slice(1)\nif 'celltype' in s.adata.obs.columns:\n    s.adata.obs['niche'] = 0\nroi = session.get_roi('tumor')\nref = '/data/ref.h5ad'")
    assert acc.slice_ids == {1} and acc.reads == {"obs:celltype"} and acc.writes == {"obs:niche"}
    assert acc.roi_names == {"tumor"} and acc.paths == ["/data/ref.h5ad"]


def test_lock_blocks_gui_mutation_during_run(rt):
    from stat_agent.core import SessionBusy

    started = threading.Event()
    done = threading.Event()

    def long_run():
        rt.run("import time\nfrom threading import Event\nprint('start')\ntime.sleep(0.6)", on_output=lambda s, ln: started.set())
        done.set()

    t = threading.Thread(target=long_run)
    t.start()
    assert started.wait(3)
    with pytest.raises(SessionBusy):
        with rt.session.lock.hold("gui", timeout=0.05):
            pass
    assert done.wait(5)
    with rt.session.lock.hold("gui", timeout=1):
        pass


def test_snapshot_diff_is_pure():
    s = make_session("S3_two_slice_gene")
    a = take_snapshot(s)
    b = take_snapshot(s)
    assert diff_snapshots(a, b).empty
    s.get_slice(1).adata.obs["celltype"] = "X"
    c = take_snapshot(s)
    d = diff_snapshots(a, c)
    assert d.changed == ["s1:obs:celltype"] and not d.added

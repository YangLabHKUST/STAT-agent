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


def test_code_access_reads_only_count_on_session_slices():
    """Reads become ledger inputs, so only receivers that can be a session slice count: a slice, a
    name bound to one, a slice iterated from the session, a subset or copy of one. Objects built by
    other calls (ad.concat, sc.read_h5ad) and names seeded as derived (a pooled AnnData kept in the
    namespace) are skipped; an unknown name still counts; rebinding a seeded name to a slice restores it."""
    code = (
        "import anndata as ad\n"
        "import scanpy as sc\n"
        "parts = []\n"
        "for sid in [0, 1]:\n"
        "    a = session.get_slice(sid).adata\n"
        "    sub = a[a.obs['celltype'] == 'T'].copy()\n"
        "    sub.obs['tag'] = sid\n"
        "    parts.append(sub)\n"
        "pooled = ad.concat(parts)\n"
        "sc.tl.rank_genes_groups(pooled, 'tag')\n"
        "top = pooled.uns['rank_genes_groups']\n"
        "ref = sc.read_h5ad('ref.h5ad')\n"
        "print(sub.obs['tag'], goblet.uns['rank_genes_groups'], ref.obs['reference_only'], unknown.obs['seen'])\n"
        "if 'k1' in pooled.uns and 'k2' in a.uns:\n"
        "    pass\n"
        "for sl in session.iter_slices():\n"
        "    print(sl.adata.obs['x'])\n"
        "goblet = session.get_slice(1).adata\n"
        "print(goblet.obs['rebound'])\n"
        "session.get_slice(0).adata.uns['top'] = top\n"
        "obs = a.obs\n"
        "pobs = pooled.obs\n"
        "print(obs['aliased'], pobs['skipped'])\n"
        "obs['via_alias'] = 1\n"
        "if 'c3' in obs.columns and 'c4' in pobs.columns and 'k5' in obs:\n"
        "    pass\n"
    )
    acc = code_access(code, derived_names={"goblet"})
    assert acc.reads == {"obs:celltype", "obs:tag", "obs:seen", "uns:k2", "obs:x", "obs:rebound", "obs:aliased", "obs:c3", "obs:k5"}
    assert acc.writes == {"obs:tag", "uns:top", "obs:via_alias"}  # writes stay receiver-agnostic; the snapshot decides
    assert "sc.tl.rank_genes_groups" in acc.calls and acc.paths == ["ref.h5ad"]


def test_reads_on_a_pooled_object_are_not_slice_inputs():
    """The CRC session case: a pooled AnnData kept in the namespace, its own result read back and
    stored on the slices under new keys, then a later step overwriting a slice's key of the same
    name. The pooled results depend on no slice artefact and must not go stale."""
    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef

    session = make_session("S3_two_slice_gene")
    rt = Runtime(session, timeout_s=30)
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))

    def record(res, turn, targets):
        return ledger.record_step(res.snapshot_before, res.snapshot_after, producer=StepRef(turn_id=turn, step=1), reads=res.access.reads, writes=res.access.writes, target_slice_ids=targets)

    r1 = rt.run("for sid in [0, 1]:\n    session.get_slice(sid).adata.uns['rank_genes_groups'] = {'names': ['g1'], 'sid': sid}\n")
    assert r1.success
    record(r1, "t1", [0, 1])
    r2 = rt.run("import anndata as ad\npooled = ad.concat([session.get_slice(0).adata, session.get_slice(1).adata], index_unique='-')\npooled.uns['rank_genes_groups'] = {'names': ['g2']}\n")
    assert r2.success and "pooled" in rt._derived_names()
    r3 = rt.run("res = pooled.uns['rank_genes_groups']\nfor sid in [0, 1]:\n    session.get_slice(sid).adata.uns['pooled_markers'] = res\n")
    assert r3.success and r3.access.reads == set()
    delta = record(r3, "t2", [0, 1])
    assert set(delta.entry.added) == {"s0:uns:pooled_markers", "s1:uns:pooled_markers"}
    assert ledger.get("s0:uns:pooled_markers").inputs == [] and ledger.get("s1:uns:pooled_markers").inputs == []
    r4 = rt.run("session.get_slice(0).adata.uns['rank_genes_groups'] = {'names': ['g3']}\n")
    delta = record(r4, "t3", [0])
    assert delta.entry.changed == ["s0:uns:rank_genes_groups"] and delta.entry.stale_now == []
    assert ledger.stale_ids() == []

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


def test_a_syntax_error_is_an_error_to_reflect_on_not_a_refusal(rt):
    r = rt.run("print('unterminated")
    assert not r.success and not r.refused and r.error_type == "SyntaxError" and "SyntaxError" in r.error


def test_safety_refuses_before_running(rt):
    r = rt.run("import os\nos.system('echo hi')")
    assert r.refused and not r.success and r.error_type == "SafetyViolation" and r.violations


def test_a_refused_block_runs_nothing_and_every_call_is_checked_again(rt):
    """The reflector may rewrite a refused block (the agent's retry), so the check must hold on every call: nothing of a
    refused block executes, a clean rewrite runs, and a rewrite that still violates is refused again."""
    r1 = rt.run("marker_before_refusal = 1\nimport subprocess\nsubprocess.run(['true'])")
    assert r1.refused and "marker_before_refusal" not in rt.namespace and not r1.namespace_added
    r2 = rt.run("marker_after_rewrite = 2")
    assert r2.success and not r2.refused and rt.namespace["marker_after_rewrite"] == 2
    r3 = rt.run("import os\nos.remove('x.tmp')")
    assert r3.refused and r3.error_type == "SafetyViolation"


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


def test_a_region_read_is_recorded_however_the_name_is_spelled():
    named = {
        "roi = session.get_roi('a')": {"a"},
        "roi = session.get_roi(roi_name='a')": {"a"},
        "d = session.get_roi_data('a', slice_id=0)": {"a"},
        "for name in ['a', 'b']:\n    roi = session.get_roi(name)": {"a", "b"},  # the spelling that lost BC-1's dependency
        "names = ('a', 'b')\nfor name in names:\n    roi = session.get_roi(name)": {"a", "b"},
        "A, B = 'a', 'b'\nx, y = session.get_roi(A), session.get_roi(B)": {"a", "b"},
        "names = ['a', 'b']\nrois = {n: session.get_roi(n)\n        for n in names}": {"a", "b"},  # the element is written above its ``for``
        "labels = {'a': 'DCIS', 'b': 'front'}\nfor n in labels:\n    r = session.get_roi(n)": {"a", "b"},
        "r = session.roi_subsets['a']\nq = session.roi_manager.rois.get('b')": {"a", "b"},
        "types = {'a': ['DCIS 1'], 'b': ['Invasive Tumor']}\nfor name, keep in types.items():\n    roi = session.get_roi(name)": {"a", "b"},  # BC-1's DE step
        "names = ['a', 'b']\nprint(names.index('a'))\nfor i, name in enumerate(names):\n    roi = session.get_roi(name)": {"a", "b"},
    }
    for code, names in named.items():
        acc = code_access(code)
        assert acc.roi_names == names and not acc.roi_any, code
    unknown = [
        "for n in session.roi_subsets.keys():\n    r = session.get_roi(n)",  # the generator prompt's own idiom
        "for n, r in session.roi_subsets.items():\n    print(r.n_obs)",
        "def cells(name):\n    return session.get_roi(name).adata\nx = cells('a')",  # a parameter
        "name = 'a'\nif flag:\n    name = 'b'\nr = session.get_roi(name)",  # bound twice
        "names = ['a']\nnames.append('b')\nfor n in names:\n    r = session.get_roi(n)",  # changed in place
        "r = session.get_roi(name)\nname = 'b'",  # bound below its use: the value comes from an earlier block
        "for n in roi_names:\n    r = session.get_roi(n)",  # a list kept from an earlier block
        "r = session.get_roi(f'region_{i}')",
        "rois = session.roi_subsets\nr = rois[k]",
        "for label, name in {'DCIS': 'a'}.items():\n    r = session.get_roi(name)",  # the values of a dict are not followed
    ]
    for code in unknown:
        acc = code_access(code)
        assert acc.roi_any and not acc.roi_names, code
    for code in ("if 'a' in session.roi_subsets:\n    print(len(session.roi_subsets), sorted(session.roi_subsets))", "for i, box in enumerate(boxes):\n    session.create_roi(f'r{i}', slice_id=0, roi_definition=box)", "x = session.get_slice(0).adata.obs['celltype']"):
        acc = code_access(code)
        assert not acc.roi_any and not acc.roi_names, code  # asking for names, or making a region, reads none


def test_a_region_kept_from_an_earlier_block_is_read_by_the_block_that_uses_it(rt):
    rt.session.create_roi("tumor", 0, {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50})
    assert rt.run("roi = session.get_roi('tumor')").access.roi_names == {"tumor"}
    later = rt.run("n = roi.adata.n_obs")  # no get_roi here: the object is in the namespace
    assert later.success and later.access.roi_names == {"tumor"} and not later.access.roi_any
    rebound = rt.run("roi = session.get_slice(0)\nn = roi.adata.n_obs")
    assert rebound.success and not rebound.access.roi_names


def test_a_data_object_carries_the_regions_it_was_computed_from(rt):
    box = {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50}
    for name in ("duct", "front"):
        rt.session.create_roi(name, 0, box)
    built = rt.run("labels = ['Tumor', 'T cell']\nout_dir = 'out'\ncmp = session.get_roi('front').adata[:, :5].copy()\nn_front = cmp.n_obs")
    assert built.success and built.access.roi_names == {"front"}
    nxt = rt.run("import scanpy as sc\nmeans = cmp.X.mean(axis=0)")  # the next block goes on from cmp and never asks for the region
    assert nxt.success and nxt.access.roi_names == {"front"} and not nxt.access.roi_any
    third = rt.run("session.get_slice(0).adata.uns['front_means'] = means")  # and the one after it from means
    assert third.success and third.access.roi_names == {"front"}
    plain = rt.run("print(labels, out_dir, n_front)\nsession.get_slice(0).adata.uns['n'] = 1")
    assert plain.success and not plain.access.roi_names  # labels, paths and numbers hand no region on
    assert rt.run("cmp = session.get_slice(0).adata[:10].copy()").success  # rebuilt from the whole slice
    assert not rt.run("m = cmp.X.sum()").access.roi_names
    taken = code_access("x = x[:, :3]\ny = [v for v in x]\ndef f(cmp):\n    return cmp\nz = cmp").taken
    assert {"x", "cmp"} <= taken and "v" not in taken  # ``x = x[...]`` goes on from the earlier x; a comprehension variable is its own


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


def test_code_access_slice_writes_need_the_slice_itself():
    """Assignments and known library effects count as slice writes only when the receiver is the slice
    object: ``session.get_slice(i).adata``, a name bound to one here or in an earlier block, a slice
    iterated from the session, ``session.slices[i]``, an alias of its container. A subset, a ``.copy()``
    or a pooled object never writes the slice, while the receiver-agnostic list keeps everything."""
    code = (
        "import anndata as ad, scanpy as sc\n"
        "a = session.get_slice(0).adata\n"
        "a.obs['direct'] = 1\n"
        "obs = a.obs\n"
        "obs['via_alias'] = 2\n"
        "sub = a[a.obs['x'] < 50].copy()\n"
        "sub.obs['on_copy'] = 3\n"
        "sub.uns['rank_genes_groups'] = {}\n"
        "view = a[:10]\n"
        "view.obs['on_view'] = 4\n"
        "pobs = sub.obs\n"
        "pobs['on_copy_alias'] = 5\n"
        "pooled = ad.concat([a, session.get_slice(1).adata])\n"
        "sc.tl.leiden(pooled)\n"
        "sc.tl.rank_genes_groups(pooled, 'leiden')\n"
        "sc.pp.neighbors(a)\n"
        "sc.tl.umap(adata=session.get_slice(1).adata)\n"
        "for s in session.iter_slices():\n"
        "    s.adata.obs['looped'] = 6\n"
        "for sid, sl in session.slices.items():\n"
        "    sl.adata.obs['looped_pair'] = 6\n"
        "for c in [session.get_slice(0).adata.copy()]:\n"
        "    c.obs['looped_copy'] = 6\n"
        "session.slices[1].adata.uns['by_index'] = 7\n"
        "kept.obs['from_earlier_block'] = 8\n"
        "goblet.obs['pooled_kept'] = 9\n"
        "unknown.obs['unknown_name'] = 10\n"
        "def store(kept, **kw):\n"
        "    kept.uns['via_param'] = 11\n"
        "    inner = session.get_slice(0).adata\n"
        "    inner.uns['in_body'] = 12\n"
        "store(sub)\n"
        "kept.uns['after_def'] = 13\n"
        "inner.uns['leaked'] = 14\n"
        "session.roi_manager.rois['t'].obs['not_a_slice'] = 15\n"
    )
    acc = code_access(code, derived_names={"goblet"}, slice_names={"kept"})
    assert {"obs:on_copy", "uns:rank_genes_groups", "obs:on_view", "obs:on_copy_alias", "obs:pooled_kept", "obs:unknown_name"} <= acc.writes
    assert acc.slice_writes == {
        "obs:direct", "obs:via_alias", "obsp:connectivities", "obsp:distances", "uns:neighbors", "obsm:X_umap",
        "obs:looped", "obs:looped_pair", "uns:by_index", "obs:from_earlier_block", "uns:in_body", "uns:after_def",
    }
    assert {"obs:looped_copy", "obs:on_copy", "obs:on_view"} & acc.slice_writes == set()


def test_a_write_on_a_copy_does_not_clear_a_stale_mark():
    """Fourth review F1. The ledger clears a stale flag on a key the code re-wrote with identical
    values (a producer re-run, B25). That needs the slice's own value recomputed: the same key
    assigned on a copy, a subset or a pooled object leaves the slice untouched, so the mark, its
    reason and the original producer must survive; a rewrite on the slice itself, also from a loop
    over the session's slices, still clears it."""
    from stat_agent.agent.nodes.execution import _code_writes
    from stat_agent.ledger.provenance import ProvenanceLedger, StepRef

    session = make_session("S1_cell_gene")
    rt = Runtime(session, timeout_s=30)
    ledger = ProvenanceLedger()
    ledger.initialize(take_snapshot(session))
    aid = "s0:uns:rank_genes_groups"

    def record(res, turn):
        assert res.success, res.error
        return ledger.record_step(res.snapshot_before, res.snapshot_after, producer=StepRef(turn_id=turn, step=1), reads=res.access.reads, writes=res.access.slice_writes, target_slice_ids=[0])

    record(rt.run("a = session.get_slice(0).adata\ngroups = sorted(a.obs['celltype'].unique())\na.uns['rank_genes_groups'] = {'names': ['g1', 'g2']}\n"), "t1")
    record(rt.run("a.obs['celltype'] = ['X'] * a.n_obs\n"), "t2")
    assert ledger.get(aid).status == "stale" and ledger.get(aid).produced_by.turn_id == "t1"

    assert "a" in rt._slice_names()  # bound to slice 0's AnnData in an earlier block
    r3 = rt.run("sub = a[a.obs['x'] < 50].copy()\nsub.uns['rank_genes_groups'] = {'names': ['h']}\n")
    assert "uns:rank_genes_groups" in _code_writes(r3) and r3.access.slice_writes == set()
    d3 = record(r3, "t3")
    assert d3.empty and d3.entry.stale_cleared == []
    r4 = rt.run("import anndata as ad\npooled = ad.concat([a, sub], index_unique='-')\npooled.uns['rank_genes_groups'] = {'names': ['p']}\nsub.obs['leiden'] = '0'\n")
    assert r4.success and {"pooled", "sub"} <= rt._derived_names() and r4.access.slice_writes == set()
    d4 = record(r4, "t4")
    assert d4.empty and ledger.stale_ids() == [aid]
    art = ledger.get(aid)
    assert art.status == "stale" and art.produced_by.turn_id == "t1" and "t2" in (art.stale_reason or "")
    assert "STALE" in ledger.state_text([0])

    d5 = record(rt.run("groups = sorted(a.obs['celltype'].unique())\na.uns['rank_genes_groups'] = {'names': ['g1', 'g2']}\n"), "t5")  # the slice itself, identical values
    assert d5.entry.changed == [] and d5.entry.stale_cleared == [aid]
    assert ledger.get(aid).status == "fresh" and ledger.get(aid).produced_by.turn_id == "t5"

    record(rt.run("a.obs['celltype'] = ['Y'] * a.n_obs\n"), "t6")
    assert ledger.stale_ids() == [aid]
    d7 = record(rt.run("for s in session.iter_slices():\n    s.adata.uns['rank_genes_groups'] = {'names': ['g1', 'g2']}\n"), "t7")
    assert d7.entry.stale_cleared == [aid] and ledger.stale_ids() == []


def test_the_namespace_listing_keeps_data_when_a_figure_fills_it(rt):
    assert rt.run(
        "table = pd.DataFrame({'gene': ['A', 'B'], 'score': [1.0, 2.0]})\n"
        "groups = {'TAM': 3, 'Macrophage': 4}\n"
    ).success
    # a figure's locals: many artists and one class, all assigned after the data
    artists = "\n".join(f"art{i} = Rectangle((0, 0), 1, 1)" for i in range(60))
    assert rt.run("from matplotlib.patches import Rectangle\n" + artists).success
    listed = dict(rt.describe_namespace(limit=40))
    assert len(listed) <= 40
    assert "DataFrame" in listed["table"] and "TAM" in listed["groups"]  # data survived
    assert "art59" in listed and "art0" not in listed  # the newest artists are kept, the oldest dropped
    assert "Rectangle" not in listed  # a class is not worth a line


def test_a_short_namespace_is_listed_whole(rt):
    assert rt.run("a = 1\nb = [1, 2]\nc = 'x'").success
    assert [n for n, _ in rt.describe_namespace()] == ["a", "b", "c"]

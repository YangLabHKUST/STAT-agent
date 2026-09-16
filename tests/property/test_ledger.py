"""Ledger invariants, checked on random operation sequences (hypothesis) and on hand-written flows."""

from __future__ import annotations

import os

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from tests.fixtures.sessions import make_session

from stat_agent.events import ListEmitter
from stat_agent.ledger import ProvenanceLedger, SessionMutator, StepRef
from stat_agent.runtime import take_snapshot

settings.register_profile("ci", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("nightly", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

BBOX = {"type": "bbox", "x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50}


def _agent_step(session, ledger, fn, *, reads=(), roi_reads=(), produces_label="step", skill=None, targets=(0,)):
    before = take_snapshot(session)
    fn()
    after = take_snapshot(session)
    return ledger.record_step(before, after, producer=StepRef(turn_id="t", step=1, skill=skill, label=produces_label), reads=reads, roi_reads=roi_reads, target_slice_ids=targets)


def test_load_then_derive_then_edit_marks_downstream_stale():
    s = make_session("S1_cell_gene")
    led = ProvenanceLedger()
    led.initialize(take_snapshot(s))
    assert led.has_live("s0:obs:celltype") and led.version == 1
    # niche detection consumes celltype
    d = _agent_step(s, led, lambda: s.get_slice(0).adata.obs.__setitem__("niche_label", "n1"), reads={"obs:celltype"}, skill="niche-detection")
    assert "s0:obs:niche_label" in d.entry.added and led.get("s0:obs:niche_label").inputs == ["s0:obs:celltype"]
    # user re-annotates through the GUI -> niche labels become stale in the same transaction
    m = SessionMutator(s, led, emitter=ListEmitter())
    res = m.set_column(0, "celltype", "Tumor")
    assert "s0:obs:niche_label" in res.delta.entry.stale_now and led.is_stale("s0:obs:niche_label")
    assert res.event.origin == "gui" and res.event.summary.celltypes_updated == [0]
    # re-running niche detection clears the staleness
    d2 = _agent_step(s, led, lambda: s.get_slice(0).adata.obs.__setitem__("niche_label", "n2"), reads={"obs:celltype"}, skill="niche-detection")
    assert "s0:obs:niche_label" in d2.entry.stale_cleared and not led.is_stale("s0:obs:niche_label")
    assert led.get("s0:obs:niche_label").version == 2


def test_roi_lifecycle_through_mutator_and_v1_summary():
    s = make_session("S1_cell_gene")
    led = ProvenanceLedger()
    led.initialize(take_snapshot(s))
    em = ListEmitter()
    m = SessionMutator(s, led, emitter=em)
    r = m.create_roi("tumor", 0, BBOX)
    assert r.delta.to_v1()["rois_added"] == ["tumor"] and led.live_rois() == {"tumor"}
    _agent_step(s, led, lambda: s.get_slice(0).adata.uns.__setitem__("de_results", {"n": 1}), roi_reads={"tumor"}, skill="differential-expression")
    assert led.get("s0:uns:de_results").inputs == ["roi:tumor"]
    r2 = m.rename_roi("tumor", "tumor2")
    assert r2.event.summary.rois_renamed == [["tumor", "tumor2"]]
    r3 = m.delete_roi("tumor2")
    assert r3.delta.to_v1()["rois_deleted"] == ["tumor2"] and not led.live_rois()
    assert em.events[-1].type == "provenance_update" and em.events[-1].ledger_version == led.version
    assert led.version == len(led.log)


def test_state_text_and_persistence(tmp_path):
    s = make_session("S2_spot_deconv")
    led = ProvenanceLedger()
    led.initialize(take_snapshot(s))
    led.mark_stale("s0:obsm:deconv_weights", "test")
    text = led.state_text([0])
    assert "Slice 0:" in text and "deconv_weights [300" in text and "T cell, B cell, Tumor, Fibroblast] (STALE: test)" in text
    led.save(tmp_path / "ledger.json")
    back = ProvenanceLedger.load(tmp_path / "ledger.json")
    assert back.consistency_hash() == led.consistency_hash() and back.version == led.version


class LedgerMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.session = make_session("S3_two_slice_gene")
        self.ledger = ProvenanceLedger()
        self.ledger.initialize(take_snapshot(self.session))
        self.mutator = SessionMutator(self.session, self.ledger)
        self.stale_history: dict[str, int] = {}
        self.roi_counter = 0

    @rule(slice_id=st.sampled_from([0, 1]))
    def annotate(self, slice_id):
        _agent_step(self.session, self.ledger, lambda: self.session.get_slice(slice_id).adata.obs.__setitem__("celltype", np.random.default_rng(len(self.ledger.log)).choice(["A", "B"], self.session.get_slice(slice_id).n_obs)), skill="annotate", targets=(slice_id,))

    @rule(slice_id=st.sampled_from([0, 1]))
    def derive(self, slice_id):
        if not self.session.get_slice(slice_id).has_celltype():
            return
        _agent_step(self.session, self.ledger, lambda: self.session.get_slice(slice_id).adata.obs.__setitem__("niche_label", "n"), reads={"obs:celltype"}, skill="niche", targets=(slice_id,))

    @rule()
    def add_roi(self):
        self.roi_counter += 1
        self.mutator.create_roi(f"r{self.roi_counter}", 0, BBOX)

    @rule()
    def delete_roi(self):
        names = sorted(self.session.roi_subsets)
        if names:
            self.mutator.delete_roi(names[0])

    @rule(slice_id=st.sampled_from([0, 1]))
    def gui_edit_celltype(self, slice_id):
        if self.session.get_slice(slice_id).has_celltype():
            self.mutator.set_column(slice_id, "celltype", "Z")

    @rule(slice_id=st.sampled_from([0, 1]))
    def delete_niche(self, slice_id):
        adata = self.session.get_slice(slice_id).adata
        if "niche_label" in adata.obs.columns:
            self.mutator.set_column(slice_id, "_tmp", 0)
            with self.mutator.mutation("drop"):
                del adata.obs["niche_label"]
                del adata.obs["_tmp"]

    @invariant()
    def three_views_agree(self):
        assert set(self.session.roi_subsets) == set(self.session.roi_manager.rois) == self.ledger.live_rois()
        for sid in self.session.slice_ids:
            s = self.session.get_slice(sid)
            assert ("celltype" in s.adata.obs.columns) == self.ledger.has_live(f"s{sid}:obs:celltype")
            assert ("niche_label" in s.adata.obs.columns) == self.ledger.has_live(f"s{sid}:obs:niche_label")

    @invariant()
    def log_is_append_only_and_versioned(self):
        assert [e.seq for e in self.ledger.log] == list(range(1, len(self.ledger.log) + 1))
        assert self.ledger.version == len(self.ledger.log)

    @invariant()
    def stale_dependants_are_marked(self):
        # every live dependant of an artefact changed after it was produced is stale
        for aid, art in self.ledger.live().items():
            for inp in art.inputs:
                src = self.ledger.get(inp)
                if src is None:
                    continue
                if src.updated_at > art.updated_at and art.status == "fresh":
                    raise AssertionError(f"{aid} depends on {inp} which changed later but is not stale")


TestLedgerMachine = LedgerMachine.TestCase

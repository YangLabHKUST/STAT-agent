"""Durable sessions: save/load slices, ROIs, ledger and memory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from stat_agent.core import DataSlice, SimpleSession


def save_session(session: SimpleSession, agent=None, target: str | Path | None = None) -> Path:
    target = Path(target) if target else (agent.session_dir if agent and agent.session_dir else Path.cwd()) / "saved_session"
    target.mkdir(parents=True, exist_ok=True)
    (target / "slices").mkdir(exist_ok=True)
    manifest = {"name": session.name, "saved_at": datetime.now().isoformat(), "slices": [], "rois": [], "celltype_colors": session.celltype_colors}
    for s in session.iter_slices():
        fname = f"slice_{s.slice_id}.h5ad"
        s.adata.write_h5ad(target / "slices" / fname)
        images = {}
        for k, img in s.images.items():
            import numpy as np

            np.save(target / "slices" / f"slice_{s.slice_id}_{k}.npy", img)
            images[k] = f"slice_{s.slice_id}_{k}.npy"
        manifest["slices"].append({"slice_id": s.slice_id, "modality": s.modality, "data_level": s.data_level, "file": fname, "images": images, "metadata": {k: v for k, v in s.metadata.items() if isinstance(v, (str, int, float, bool))}})
    for r in session.roi_manager.rois.values():
        d = r.to_dict()
        d["definition"] = r.metadata.get("definition")
        manifest["rois"].append(d)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if agent is not None:
        agent.ledger.save(target / "ledger.json")
        (target / "memory.json").write_text(agent.memory.model_dump_json(indent=1), encoding="utf-8")
    return target


def load_session(source: str | Path) -> tuple[SimpleSession, dict]:
    import anndata as ad
    import numpy as np

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    session = SimpleSession(name=manifest.get("name", source.name))
    for s in manifest["slices"]:
        adata = ad.read_h5ad(source / "slices" / s["file"])
        images = {k: np.load(source / "slices" / f) for k, f in s.get("images", {}).items()}
        session.add_slice(DataSlice(slice_id=int(s["slice_id"]), modality=s["modality"], data_level=s["data_level"], adata=adata, images=images, metadata=dict(s.get("metadata", {}))))
    session.celltype_colors = manifest.get("celltype_colors")
    for r in manifest.get("rois", []):
        definition = r.get("definition")
        if definition and r.get("slice_id") is not None:
            session.create_roi(r["name"], int(r["slice_id"]), definition)
    return session, manifest


__all__ = ["save_session", "load_session"]

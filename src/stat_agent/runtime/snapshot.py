"""Cheap fingerprints of every session artefact, and diffs between two snapshots.

Artefact ids: ``s0:obs:celltype``, ``s1:obsm:deconv_weights``, ``s0:uns:moranI``,
``s0:X``, ``roi:tumor``. Fingerprints use shapes, dtypes and a deterministic sample of
values so a snapshot of a million-cell slice takes milliseconds.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SAMPLE = 512


def _h(*parts: Any) -> str:
    m = hashlib.blake2b(digest_size=12)
    for p in parts:
        m.update(repr(p).encode("utf-8", "replace"))
        m.update(b"\x1f")
    return m.hexdigest()


def _sample_idx(n: int, k: int = SAMPLE) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.linspace(0, n - 1, k).astype(int)


@dataclass
class Fingerprint:
    id: str
    kind: str
    key: str
    slice_id: int | None
    digest: str
    shape: tuple | None = None
    dtype: str | None = None
    n_null: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def fp_series(series) -> tuple[str, str | None, int, dict[str, Any]]:
    n = len(series)
    idx = _sample_idx(n)
    values = series.iloc[idx] if hasattr(series, "iloc") else series[idx]
    try:
        n_null = int(series.isna().sum())
    except Exception:  # noqa: BLE001
        n_null = 0
    extra: dict[str, Any] = {}
    if hasattr(series, "cat"):
        cats = list(series.cat.categories)
        extra["categories"] = cats[:200]
        digest = _h(str(series.dtype), n, n_null, cats, values.astype(str).tolist())
    else:
        digest = _h(str(series.dtype), n, n_null, np.asarray(values).astype(str).tolist())
    return digest, str(series.dtype), n_null, extra


def fp_matrix(obj) -> tuple[str, tuple | None, str | None, dict[str, Any]]:
    extra: dict[str, Any] = {}
    try:
        import scipy.sparse as sp

        if sp.issparse(obj):
            csr = obj.tocsr() if hasattr(obj, "tocsr") else obj
            idx = _sample_idx(csr.shape[0], 64)
            sub = csr[idx]
            digest = _h(csr.shape, str(csr.dtype), int(csr.nnz), sub.indptr.tolist()[:65], np.round(sub.data[:512].astype(float), 4).tolist())
            extra["nnz"] = int(csr.nnz)
            return digest, tuple(csr.shape), str(csr.dtype), extra
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "to_numpy") and hasattr(obj, "columns"):  # DataFrame
        cols = [str(c) for c in obj.columns]
        arr = obj.to_numpy()
        idx = _sample_idx(arr.shape[0] if arr.ndim else 0, 64)
        sample = arr[idx] if arr.ndim else arr
        digest = _h(arr.shape, str(arr.dtype), cols[:100], np.asarray(sample).astype(str).ravel()[:2048].tolist())
        extra["columns"] = cols[:100]
        return digest, tuple(arr.shape), str(arr.dtype), extra
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return _h(arr.item() if arr.size else None), (), str(arr.dtype), extra
    idx = _sample_idx(arr.shape[0], 64)
    sample = arr[idx]
    digest = _h(arr.shape, str(arr.dtype), np.asarray(sample).astype(str).ravel()[:2048].tolist())
    return digest, tuple(arr.shape), str(arr.dtype), extra


def fp_uns(value) -> tuple[str, tuple | None, str | None]:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _h(type(value).__name__, value), None, type(value).__name__
    if isinstance(value, (dict, list, tuple, set)):
        text = repr(value)
        if len(text) <= 65536:
            return _h(type(value).__name__, text), (len(value),), type(value).__name__
        return _h(type(value).__name__, len(value), text[:4096]), (len(value),), type(value).__name__
    if hasattr(value, "shape"):
        digest, shape, dtype, _ = fp_matrix(value)
        return digest, shape, dtype
    return _h(type(value).__name__, repr(value)[:4096]), None, type(value).__name__


@dataclass
class Snapshot:
    artifacts: dict[str, Fingerprint]
    taken_at: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def ids(self) -> set[str]:
        return set(self.artifacts)

    def of_slice(self, slice_id: int) -> dict[str, Fingerprint]:
        return {k: v for k, v in self.artifacts.items() if v.slice_id == slice_id}


def take_snapshot(session, include_X: bool = True) -> Snapshot:
    t0 = time.perf_counter()
    arts: dict[str, Fingerprint] = {}
    for sid, s in session.slices.items():
        adata = s.adata
        for col in adata.obs.columns:
            d, dtype, n_null, extra = fp_series(adata.obs[col])
            aid = f"s{sid}:obs:{col}"
            arts[aid] = Fingerprint(aid, "obs", str(col), sid, d, (adata.n_obs,), dtype, n_null, extra)
        for col in adata.var.columns:
            d, dtype, n_null, extra = fp_series(adata.var[col])
            aid = f"s{sid}:var:{col}"
            arts[aid] = Fingerprint(aid, "var", str(col), sid, d, (adata.n_vars,), dtype, n_null, extra)
        for space, kind in ((adata.obsm, "obsm"), (adata.obsp, "obsp"), (adata.layers, "layers"), (getattr(adata, "varm", {}), "varm")):
            for key in list(space.keys()):
                try:
                    d, shape, dtype, extra = fp_matrix(space[key])
                except Exception as exc:  # noqa: BLE001
                    d, shape, dtype, extra = _h("unhashable", str(exc)), None, None, {}
                aid = f"s{sid}:{kind}:{key}"
                arts[aid] = Fingerprint(aid, kind, str(key), sid, d, shape, dtype, None, extra)
        for key in list(adata.uns.keys()):
            try:
                d, shape, dtype = fp_uns(adata.uns[key])
            except Exception as exc:  # noqa: BLE001
                d, shape, dtype = _h("unhashable", str(exc)), None, None
            aid = f"s{sid}:uns:{key}"
            arts[aid] = Fingerprint(aid, "uns", str(key), sid, d, shape, dtype)
        if include_X and adata.X is not None:
            d, shape, dtype, extra = fp_matrix(adata.X)
            aid = f"s{sid}:X"
            arts[aid] = Fingerprint(aid, "X", "X", sid, d, shape, dtype, None, extra)
    for name, roi in session.roi_manager.rois.items():
        try:
            wkb = roi.geometry.wkb_hex
        except Exception:  # noqa: BLE001
            wkb = repr(roi.geometry)
        aid = f"roi:{name}"
        arts[aid] = Fingerprint(aid, "roi", name, roi.slice_id, _h(wkb, roi.slice_id, roi.n_obs), (roi.n_obs,), roi.type)
    return Snapshot(artifacts=arts, duration_s=time.perf_counter() - t0)


@dataclass
class SnapshotDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return self.added + self.changed

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def by_kind(self, kind: str) -> SnapshotDiff:
        f = lambda ids: [i for i in ids if i.split(":")[1 if i.startswith("s") else 0] == kind]  # noqa: E731
        return SnapshotDiff(added=f(self.added), removed=f(self.removed), changed=f(self.changed))

    def as_dict(self) -> dict[str, list[str]]:
        return {"added": list(self.added), "removed": list(self.removed), "changed": list(self.changed)}


def diff_snapshots(before: Snapshot, after: Snapshot) -> SnapshotDiff:
    b, a = before.artifacts, after.artifacts
    added = sorted(set(a) - set(b))
    removed = sorted(set(b) - set(a))
    changed = sorted(k for k in set(a) & set(b) if a[k].digest != b[k].digest)
    return SnapshotDiff(added=added, removed=removed, changed=changed)


def parse_artifact_id(aid: str) -> tuple[int | None, str, str]:
    """``'s0:obs:celltype' -> (0, 'obs', 'celltype')``; ``'roi:tumor' -> (None, 'roi', 'tumor')``."""
    if aid.startswith("roi:"):
        return None, "roi", aid[4:]
    sid, rest = aid.split(":", 1)
    if rest == "X":
        return int(sid[1:]), "X", "X"
    kind, key = rest.split(":", 1)
    return int(sid[1:]), kind, key


__all__ = ["Fingerprint", "Snapshot", "SnapshotDiff", "take_snapshot", "diff_snapshots", "parse_artifact_id"]

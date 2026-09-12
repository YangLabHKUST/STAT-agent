"""Dataset loading: h5ad files plus images from one directory.

Filename conventions (unchanged from STAT 0.1):

* ``tissue.h5ad``                 -> slice 0, gene
* ``tissue_protein.h5ad``         -> next free id (>= 1), protein
* ``tissue_slice_N.h5ad``         -> slice N, gene
* ``tissue_slice_N_protein.h5ad`` -> slice N, protein
* a stem containing ``spot``      -> spot-level data

Images (``.tif``/``.tiff``/``.npy``) are matched first by the strict patterns
(``he_slice_N.tif``, ``dapi.tif``, ``protein_<name>_slice_N.tif`` ...) and then by
filename similarity for slices still without an image.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from PIL import Image

from stat_agent.core.data_slice import DataSlice

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".tif", ".tiff", ".npy")


@dataclass(frozen=True)
class LoadOptions:
    """Preprocessing applied to every h5ad at load time."""

    min_cells: int = 5  # genes must be detected in at least this many observations
    min_counts: int = 20  # observations must have at least this many counts
    require_raw_counts: bool = True  # raise if no integer count matrix can be found
    filter: bool = True  # apply the two filters above
    spot_threshold: int = 10_000  # fewer observations than this -> spot-level heuristic


DEFAULT_OPTIONS = LoadOptions()


# ---- preprocessing -------------------------------------------------------------------------


def _is_integer_matrix(X) -> bool:
    if sp.issparse(X):
        sample = X.data[: min(1000, len(X.data))] if len(X.data) > 0 else np.array([0])
    else:
        flat = np.asarray(X).ravel()
        sample = flat[: min(1000, len(flat))]
    if sample.size == 0:
        return True
    return bool(np.allclose(sample, np.round(sample)))


def _recover_raw_counts(adata: ad.AnnData, filename: str, options: LoadOptions) -> ad.AnnData:
    if _is_integer_matrix(adata.X):
        return adata
    logger.warning("%s: adata.X does not contain raw counts, searching alternatives", filename)
    if adata.raw is not None and _is_integer_matrix(adata.raw.X):
        logger.info("  using adata.raw.X as adata.X")
        return ad.AnnData(
            X=adata.raw.X,
            obs=adata.obs,
            var=adata.raw.var,
            obsm=adata.obsm,
            obsp=adata.obsp,
            uns=adata.uns,
        )
    if "counts" in adata.layers and _is_integer_matrix(adata.layers["counts"]):
        logger.info("  using adata.layers['counts'] as adata.X")
        adata.X = adata.layers["counts"]
        return adata
    if options.require_raw_counts:
        raise ValueError(
            f"Cannot find raw counts for {filename}: adata.X is non-integer and neither adata.raw "
            "nor adata.layers['counts'] holds integer counts. Provide raw UMI counts."
        )
    logger.warning("  proceeding with non-integer adata.X (require_raw_counts=False)")
    return adata


def preprocess_anndata(adata: ad.AnnData, filename: str, options: LoadOptions = DEFAULT_OPTIONS) -> ad.AnnData:
    """Make names unique, recover raw counts, filter, and validate coordinates."""
    import scanpy as sc

    logger.info("Preprocessing %s", filename)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = _recover_raw_counts(adata, filename, options)

    if options.filter:
        saved_colors = {k: adata.uns[k] for k in list(adata.uns.keys()) if str(k).endswith("_colors")}
        n_vars_before, n_obs_before = adata.n_vars, adata.n_obs
        sc.pp.filter_genes(adata, min_cells=options.min_cells)
        sc.pp.filter_cells(adata, min_counts=options.min_counts)
        for k, v in saved_colors.items():
            adata.uns[k] = v
        if adata.n_vars < n_vars_before:
            logger.info("  filtered genes (min_cells=%d): %d -> %d", options.min_cells, n_vars_before, adata.n_vars)
        if adata.n_obs < n_obs_before:
            logger.info("  filtered obs (min_counts=%d): %d -> %d", options.min_counts, n_obs_before, adata.n_obs)

    missing = [c for c in ("x", "y") if c not in adata.obs.columns]
    if missing and "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])
        adata.obs["x"] = spatial[:, 0]
        adata.obs["y"] = spatial[:, 1]
        logger.info("  created obs x/y from obsm['spatial']")
        missing = [c for c in ("x", "y") if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"AnnData missing required obs columns: {missing}. Provide x, y in adata.obs or obsm['spatial']."
        )
    coord_mask = ~adata.obs["x"].isna() & ~adata.obs["y"].isna()
    n_nan = int((~coord_mask).sum())
    if n_nan:
        adata = adata[coord_mask.to_numpy()].copy()
        logger.info("  removed %d observations with NaN coordinates", n_nan)
    logger.info("  preprocessing complete: %d obs x %d vars", adata.n_obs, adata.n_vars)
    return adata


# ---- single files ----------------------------------------------------------------------------


def load_anndata(path: str | Path, options: LoadOptions = DEFAULT_OPTIONS) -> ad.AnnData:
    path = Path(path)
    logger.info("Loading AnnData from %s", path)
    adata = ad.read_h5ad(path)
    adata = preprocess_anndata(adata, path.name, options)
    if "spot" in path.stem.lower():
        if "deconv_weights" in adata.obsm and "celltype" not in adata.obs.columns:
            weights = adata.obsm["deconv_weights"]
            if hasattr(weights, "idxmax"):
                adata.obs["celltype"] = weights.idxmax(axis=1)
                logger.info("  created virtual celltype column from dominant deconvolution weight")
        adata.uns["has_deconv_weights"] = "deconv_weights" in adata.obsm
    return adata


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        image = np.array(Image.open(path))
    elif suffix == ".npy":
        image = np.load(path)
    else:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    logger.info("Loaded image %s with shape %s", path.name, image.shape)
    return image


# ---- filename parsing -------------------------------------------------------------------------


def parse_slice_id_from_filename(filename: str) -> tuple[int | None, str]:
    """Return ``(slice_id or None, modality)`` from a filename stem."""
    stem = Path(filename).stem
    modality = "protein" if "protein" in stem.lower() else "gene"
    stem_clean = re.sub(r"_protein", "", stem, flags=re.IGNORECASE)
    match = re.search(r"_slice_(\d+)", stem_clean)
    if match:
        return int(match.group(1)), modality
    return None, modality


def detect_data_level(adata: ad.AnnData, filename: str, options: LoadOptions = DEFAULT_OPTIONS) -> tuple[str, str]:
    """Return ``(data_level, reason)`` using filename, ``uns`` spot metadata, then size."""
    stem = Path(filename).stem
    if "spot" in stem.lower():
        return "spot", f'filename contains "spot" ("{stem}")'
    if "spot_diameter" in adata.uns or "spot_shape" in adata.uns:
        return "spot", "adata.uns contains spot_diameter/spot_shape"
    if adata.n_obs < options.spot_threshold:
        return "spot", f"{adata.n_obs} observations < {options.spot_threshold:,} threshold"
    return "cell", f"{adata.n_obs} observations (cell-level)"


# ---- image matching ------------------------------------------------------------------------------


def _find_all_images(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(dataset_dir.glob(f"*{suffix}"))
    return sorted(set(files))


def _detect_image_type(stem: str) -> str:
    s = stem.lower()
    if "he" in s or "h_e" in s:
        return "he"
    if "dapi" in s:
        return "dapi"
    return "primary"


def find_images_for_slice(dataset_dir: Path, slice_id: int, modality: str) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Strict pattern matching; returns ``(images, matched_paths)``."""
    images: dict[str, np.ndarray] = {}
    matched: list[Path] = []
    if modality == "gene":
        patterns = [f"{kind}_slice_{slice_id}{ext}" for kind in ("he", "dapi") for ext in IMAGE_SUFFIXES]
        if slice_id == 0:
            patterns += [f"{kind}{ext}" for kind in ("he", "dapi") for ext in IMAGE_SUFFIXES]
        for pattern in patterns:
            hits = list(dataset_dir.glob(pattern))
            if hits:
                kind = "he" if pattern.startswith("he") else "dapi"
                try:
                    images[kind] = load_image(hits[0])
                    matched.append(hits[0].resolve())
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load %s image %s: %s", kind, hits[0], e)
    elif modality == "protein":
        for img in dataset_dir.glob(f"protein_*_slice_{slice_id}.tif*"):
            name = img.stem.replace("protein_", "").replace(f"_slice_{slice_id}", "")
            try:
                images[name] = load_image(img)
                matched.append(img.resolve())
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load protein image %s: %s", img, e)
        if not images and slice_id == 1:
            for img in dataset_dir.glob("protein_*.tif*"):
                if "_slice_" in img.stem:
                    continue
                try:
                    images[img.stem.replace("protein_", "")] = load_image(img)
                    matched.append(img.resolve())
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load protein image %s: %s", img, e)
    return images, matched


def _assign_images_flexible(dataset_dir: Path, slices: dict[int, DataSlice], stem_map: dict[int, str], used: set) -> None:
    available = [p for p in _find_all_images(dataset_dir) if p.resolve() not in used]
    needing = [(sid, stem_map[sid]) for sid in sorted(slices) if not slices[sid].images]
    if not available or not needing:
        return
    logger.info("Flexible image matching: %d image(s), %d slice(s) without images", len(available), len(needing))
    if len(needing) == 1 and len(available) == 1:
        sid, img_path = needing[0][0], available[0]
        try:
            slices[sid].images[_detect_image_type(img_path.stem)] = load_image(img_path)
            slices[sid].metadata["image_match"] = "flexible (single pair)"
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load image %s: %s", img_path, e)
        return
    pairs = sorted(
        ((SequenceMatcher(None, stem.lower(), p.stem.lower()).ratio(), sid, p) for sid, stem in needing for p in available),
        key=lambda t: -t[0],
    )
    done_slices: set = set()
    done_images: set = set()
    for score, sid, img_path in pairs:
        if sid in done_slices or img_path in done_images or score < 0.3:
            continue
        try:
            slices[sid].images[_detect_image_type(img_path.stem)] = load_image(img_path)
            slices[sid].metadata["image_match"] = f"flexible (similarity: {score:.2f})"
            done_slices.add(sid)
            done_images.add(img_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load image %s: %s", img_path, e)


# ---- directory loader -------------------------------------------------------------------------------


def load_dataset(dataset_dir: str | Path, options: LoadOptions | None = None) -> dict[int, DataSlice]:
    """Load every h5ad in ``dataset_dir`` into ``{slice_id: DataSlice}``."""
    options = options or DEFAULT_OPTIONS
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    h5ad_files = sorted(dataset_dir.glob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found in {dataset_dir}")

    slices: dict[int, DataSlice] = {}
    stem_map: dict[int, str] = {}
    used_images: set = set()
    next_auto_id = 0
    for h5ad_file in h5ad_files:
        stem = h5ad_file.stem
        slice_id, modality = parse_slice_id_from_filename(stem)
        if slice_id is None:
            if modality == "protein":
                slice_id = max(1, next_auto_id)
            else:
                slice_id = 0 if 0 not in slices else next_auto_id
        if slice_id in slices:
            raise ValueError(f"Slice ID conflict: {slice_id} already used ({h5ad_file.name})")
        next_auto_id = max(next_auto_id, slice_id + 1)

        adata = load_anndata(h5ad_file, options)
        data_level, reason = detect_data_level(adata, stem, options)
        images, matched = find_images_for_slice(dataset_dir, slice_id, modality)
        used_images.update(matched)
        slices[slice_id] = DataSlice(
            slice_id=slice_id,
            modality=modality,
            data_level=data_level,
            adata=adata,
            images=images,
            metadata={
                "tissue_name": stem,
                "source_file": str(h5ad_file),
                "data_level_detection": reason,
                "image_match": "strict" if images else "none",
            },
        )
        stem_map[slice_id] = stem
        logger.info("Loaded slice %s: %s, %s (%s), %d obs x %d vars", slice_id, modality, data_level, reason, adata.n_obs, adata.n_vars)

    _assign_images_flexible(dataset_dir, slices, stem_map, used_images)
    return slices


__all__ = [
    "LoadOptions",
    "DEFAULT_OPTIONS",
    "preprocess_anndata",
    "load_anndata",
    "load_image",
    "parse_slice_id_from_filename",
    "detect_data_level",
    "find_images_for_slice",
    "load_dataset",
]

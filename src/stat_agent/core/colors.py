"""Colour helpers shared by DataSlice and SimpleSession.

Colours are generated deterministically from the set of names (seeded by a hash of
the sorted names) so that regenerating colours for the same cell types yields the
same palette; pre-existing colours (scanpy list format or dict format) are kept.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

GOLDEN_RATIO = 0.618033988749895


def hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[float, float, float]:
    """Convert HSL (all in [0, 1]) to RGB floats in [0, 1]."""

    def hue2rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        return lightness, lightness, lightness
    q = lightness * (1 + s) if lightness < 0.5 else lightness + s - lightness * s
    p = 2 * lightness - q
    return hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)


def rgb_string(r: float, g: float, b: float) -> str:
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def normalize_color_to_rgb(color: object) -> str | None:
    """Convert a colour (hex, named, rgb(...)) to the canonical 'rgb(r, g, b)' string."""
    if not isinstance(color, str):
        return None
    color = color.strip()
    if color.startswith("rgb("):
        return color
    if color.startswith("#"):
        hex_str = color[1:]
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str)
        if len(hex_str) == 6:
            try:
                r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
            return f"rgb({r}, {g}, {b})"
        return None
    try:
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(color)
        return rgb_string(r, g, b)
    except (ImportError, ValueError):
        return None


def stable_seed(names: Iterable[str]) -> int:
    digest = hashlib.blake2b(",".join(sorted(names)).encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16)


def generate_distinct_colors(
    names: Iterable[str],
    existing: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Return a colour for every name, keeping ``existing`` colours and generating the rest.

    Generation uses golden-ratio hue spacing with a deterministic RNG so the same set of
    names always maps to the same colours.
    """
    names = sorted({n for n in names if isinstance(n, str)})
    colors: dict[str, str] = {}
    for name in names:
        if existing and name in existing:
            rgb = normalize_color_to_rgb(existing[name])
            if rgb:
                colors[name] = rgb
    missing = [n for n in names if n not in colors]
    if not missing:
        return colors
    rng = random.Random(stable_seed(names) if seed is None else seed)
    hue = rng.random()
    for name in missing:
        hue = (hue + GOLDEN_RATIO) % 1.0
        saturation = 0.6 + rng.random() * 0.2
        lightness = 0.45 + rng.random() * 0.15
        colors[name] = rgb_string(*hsl_to_rgb(hue, saturation, lightness))
    return colors


def extract_existing_colors(adata) -> dict[str, str]:
    """Read ``adata.uns['celltype_colors']`` in either dict or scanpy list format."""
    if "celltype_colors" not in adata.uns:
        return {}
    raw = adata.uns["celltype_colors"]
    if isinstance(raw, dict):
        out = {}
        for ct, color in raw.items():
            rgb = normalize_color_to_rgb(color)
            if rgb:
                out[str(ct)] = rgb
        return out
    if hasattr(raw, "__len__") and not isinstance(raw, str):
        color_list = list(raw)
        col = adata.obs.get("celltype")
        if col is None:
            return {}
        if hasattr(col, "cat"):
            categories = list(col.cat.categories)
        else:
            categories = sorted(ct for ct in col.unique() if isinstance(ct, str))
        out = {}
        for i, cat in enumerate(categories):
            if i < len(color_list):
                rgb = normalize_color_to_rgb(str(color_list[i]))
                if rgb:
                    out[str(cat)] = rgb
        return out
    return {}


__all__ = [
    "GOLDEN_RATIO",
    "hsl_to_rgb",
    "rgb_string",
    "normalize_color_to_rgb",
    "stable_seed",
    "generate_distinct_colors",
    "extract_existing_colors",
]

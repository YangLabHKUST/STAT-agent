"""Colour helpers shared by DataSlice and SimpleSession.

Colours are generated deterministically from the set of names (seeded by a hash of
the sorted names) so that regenerating colours for the same cell types yields the
same palette; pre-existing colours (scanpy list format or dict format) are kept.

Colours are stored as '#rrggbb': generated code plots with the palette it finds in
``adata.uns['celltype_colors']``, and hex is the notation matplotlib, seaborn and the
browser all read (matplotlib rejects CSS 'rgb(r, g, b)', which STAT stored before).
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


def hex_string(r: float, g: float, b: float) -> str:
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def normalize_color(color: object) -> str | None:
    """Convert a colour (hex, named, CSS rgb(...)) to the canonical '#rrggbb' string."""
    if not isinstance(color, str):
        return None
    color = color.strip()
    if color.startswith(("rgb(", "rgba(")) and color.endswith(")"):
        try:
            channels = [float(v) for v in color[color.index("(") + 1 : -1].split(",")[:3]]
        except ValueError:
            return None
        if len(channels) != 3 or not all(0 <= v <= 255 for v in channels):
            return None
        return "#" + "".join(f"{int(v):02x}" for v in channels)
    try:
        from matplotlib.colors import to_hex

        return to_hex(color)
    except (ImportError, ValueError):
        return None


def normalize_palette(colors: dict | None) -> dict[str, str]:
    """``{name: colour}`` with every colour as '#rrggbb'; a colour that cannot be read is dropped (it is
    generated again where it is needed)."""
    out = {}
    for name, color in (colors or {}).items():
        hex_color = normalize_color(color)
        if hex_color:
            out[str(name)] = hex_color
    return out


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
            hex_color = normalize_color(existing[name])
            if hex_color:
                colors[name] = hex_color
    missing = [n for n in names if n not in colors]
    if not missing:
        return colors
    rng = random.Random(stable_seed(names) if seed is None else seed)
    hue = rng.random()
    for name in missing:
        hue = (hue + GOLDEN_RATIO) % 1.0
        saturation = 0.6 + rng.random() * 0.2
        lightness = 0.45 + rng.random() * 0.15
        colors[name] = hex_string(*hsl_to_rgb(hue, saturation, lightness))
    return colors


def extract_existing_colors(adata) -> dict[str, str]:
    """Read ``adata.uns['celltype_colors']`` in either dict or scanpy list format."""
    if "celltype_colors" not in adata.uns:
        return {}
    raw = adata.uns["celltype_colors"]
    if isinstance(raw, dict):
        return normalize_palette(raw)
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
                hex_color = normalize_color(str(color_list[i]))
                if hex_color:
                    out[str(cat)] = hex_color
        return out
    return {}


__all__ = [
    "GOLDEN_RATIO",
    "hsl_to_rgb",
    "hex_string",
    "normalize_color",
    "normalize_palette",
    "stable_seed",
    "generate_distinct_colors",
    "extract_existing_colors",
]

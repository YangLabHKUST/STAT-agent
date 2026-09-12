"""Matplotlib figure capture (PNG bytes), always closing the figures afterwards."""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger(__name__)


def ensure_headless_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg", force=True)
    except Exception:  # noqa: BLE001
        pass


def capture_figures(dpi: int = 100) -> list[bytes]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    out: list[bytes] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
            out.append(buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            logger.warning("figure capture failed: %s", exc)
        finally:
            plt.close(fig)
    return out


def close_all_figures() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:  # noqa: BLE001
        pass


__all__ = ["ensure_headless_backend", "capture_figures", "close_all_figures"]

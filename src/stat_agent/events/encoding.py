"""Safe JSON encoding shared by the event sinks, the web layer and the benchmark runner.

``to_jsonable`` converts numpy scalars/arrays, pandas objects, NaN/inf (-> null), sets,
bytes (-> base64), paths, datetimes, dataclasses, pydantic models and shapely geometries
into plain JSON types; ``dumps`` then serialises with ``allow_nan=False`` so a bad value
can never produce an unparseable frame.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from datetime import date, datetime
from pathlib import PurePath
from typing import Any


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > 64:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, (PurePath,)):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic
        return to_jsonable(obj.model_dump(mode="python"), _depth + 1)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), _depth + 1)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return to_jsonable(obj.item(), _depth + 1)
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist(), _depth + 1)
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd

        if isinstance(obj, (pd.Series, pd.Index)):
            return to_jsonable(obj.tolist(), _depth + 1)
        if isinstance(obj, pd.DataFrame):
            return to_jsonable(obj.to_dict(orient="list"), _depth + 1)
        if obj is pd.NaT:
            return None
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "__geo_interface__"):
        return to_jsonable(obj.__geo_interface__, _depth + 1)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return repr(obj)


def dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("separators", (",", ":"))
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(to_jsonable(obj), allow_nan=False, **kwargs)


__all__ = ["to_jsonable", "dumps"]

"""Static code-access analysis: which artefacts and slices a code block touches."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field

from stat_agent.skills.lint import scan_code


@dataclass
class CodeAccess:
    reads: set[str] = field(default_factory=set)  # 'obs:celltype'
    writes: set[str] = field(default_factory=set)
    roi_names: set[str] = field(default_factory=set)
    slice_ids: set[int] = field(default_factory=set)
    calls: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    dynamic_keys: int = 0
    parse_error: str | None = None
    paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "reads": sorted(self.reads),
            "writes": sorted(self.writes),
            "roi_names": sorted(self.roi_names),
            "slice_ids": sorted(self.slice_ids),
            "imports": sorted(self.imports),
            "dynamic_keys": self.dynamic_keys,
            "paths": list(self.paths),
            "parse_error": self.parse_error,
        }


class _SliceAndPathScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.slice_ids: set[int] = set()
        self.paths: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
        if name == "get_slice" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            self.slice_ids.add(node.args[0].value)
        for kw in node.keywords:
            if kw.arg == "slice_id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                self.slice_ids.add(kw.value.value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value.lower().endswith((".h5ad", ".csv", ".tsv", ".tif", ".tiff", ".npy", ".txt", ".json", ".gmt", ".loom")):
            self.paths.append(node.value)


def code_access(code: str, derived_names: Iterable[str] = ()) -> CodeAccess:
    """Static access scan of one block. ``derived_names`` are namespace variables known to hold an
    AnnData that is not a session slice; reads on them are not slice reads (see ``_BodyScan``)."""
    acc = CodeAccess()
    try:
        scan = scan_code(code, derived_names)
    except SyntaxError as exc:
        acc.parse_error = f"{exc.msg} (line {exc.lineno})"
        return acc
    acc.reads, acc.writes, acc.roi_names = set(scan.reads), set(scan.writes), set(scan.roi_names)
    acc.calls, acc.imports, acc.dynamic_keys = set(scan.calls), set(scan.imports), len(scan.dynamic_keys)
    extra = _SliceAndPathScan()
    extra.visit(ast.parse(code))
    acc.slice_ids, acc.paths = extra.slice_ids, extra.paths
    return acc


__all__ = ["CodeAccess", "code_access"]

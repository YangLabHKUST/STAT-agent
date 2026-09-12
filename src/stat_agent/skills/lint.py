"""Skill linting: schema, filter grammar, body/contract consistency (AST), dependencies.

``lint_skill`` returns a list of :class:`Issue`; ``error`` issues make ``stat-agent skill
validate`` fail, ``warn``/``info`` issues are reported. The same checks back the contract
tests and the corruption generator of the benchmark harness.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from stat_agent.skills.registry import Skill
from stat_agent.skills.schema import ARTIFACT_KINDS, SkillFrontmatter, parse_skill_text

ATTR_KINDS = {"obs", "obsm", "uns", "var", "obsp", "varm", "layers"}

# Keys a body may write without declaring them (bookkeeping, coordinates, colours).
INCIDENTAL_WRITES = {
    "obs:x",
    "obs:y",
    "obs:_slice_id",
    "obs:batch",
    "obs:slice_id",
    "obsm:spatial",
    "uns:celltype_colors",
    "uns:has_deconv_weights",
}

# Library calls that write artefacts without an explicit subscript assignment.
LIBRARY_EFFECTS: Dict[str, List[str]] = {
    "sc.tl.rank_genes_groups": ["uns:rank_genes_groups"],
    "sc.tl.leiden": ["obs:leiden"],
    "sc.tl.louvain": ["obs:louvain"],
    "sc.tl.pca": ["obsm:X_pca", "uns:pca"],
    "sc.tl.umap": ["obsm:X_umap"],
    "sc.pp.neighbors": ["obsp:connectivities", "obsp:distances", "uns:neighbors"],
    "sc.pp.highly_variable_genes": ["var:highly_variable"],
    "sc.tl.score_genes": ["obs:score"],
    "sq.gr.spatial_neighbors": ["obsp:spatial_connectivities", "obsp:spatial_distances", "uns:spatial_neighbors"],
    "sq.gr.nhood_enrichment": ["uns:*_nhood_enrichment"],
    "sq.gr.co_occurrence": ["uns:*_co_occurrence"],
    "sq.gr.ripley": ["uns:*_ripley_*"],
    "sq.gr.centrality_scores": ["uns:*_centrality_scores"],
    "sq.gr.spatial_autocorr": ["uns:moranI", "uns:gearyC"],
    "sce.pp.harmony_integrate": ["obsm:X_pca_harmony"],
    "sce.pp.bbknn": ["obsp:connectivities", "obsp:distances", "uns:neighbors"],
}

STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
IMPORT_TO_DIST = {
    "sc": "scanpy",
    "scanpy": "scanpy",
    "sq": "squidpy",
    "squidpy": "squidpy",
    "np": "numpy",
    "numpy": "numpy",
    "pd": "pandas",
    "pandas": "pandas",
    "scipy": "scipy",
    "anndata": "anndata",
    "ad": "anndata",
    "matplotlib": "matplotlib",
    "plt": "matplotlib",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "shapely": "shapely",
    "geopandas": "geopandas",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "scvi": "scvi-tools",
    "tangram": "tangram-sc",
    "cell2location": "cell2location",
    "liana": "liana",
    "cellphonedb": "cellphonedb",
    "gseapy": "gseapy",
    "harmonypy": "harmonypy",
    "bbknn": "bbknn",
    "scanorama": "scanorama",
    "palantir": "palantir",
    "NaiveDE": "NaiveDE",
    "SpatialDE": "SpatialDE",
    "SpaGCN": "SpaGCN",
    "GraphST": "GraphST",
    "STAGATE_pyG": "STAGATE_pyG",
    "paste": "paste-bio",
    "ot": "POT",
    "ray": "ray",
    "qpsolvers": "qpsolvers",
    "flashdeconv": "flashdeconv",
    "esda": "esda",
    "plotnine": "plotnine",
    "plotly": "plotly",
    "nrrd": "pynrrd",
    "PIL": "pillow",
    "tifffile": "tifffile",
    "yaml": "pyyaml",
    "scipy.sparse": "scipy",
}


@dataclass
class Issue:
    level: str  # error | warn | info
    code: str
    message: str
    slug: Optional[str] = None
    line: Optional[int] = None

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.level}] {self.code}: {self.message}{loc}"


CODE_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)


def extract_code_blocks(body: str) -> List[Tuple[int, str]]:
    """``[(line_offset, code)]`` for every python fence in the body."""
    blocks = []
    for m in CODE_FENCE_RE.finditer(body):
        line = body[: m.start()].count("\n") + 2
        blocks.append((line, m.group(1)))
    return blocks


def _dotted(node: ast.AST) -> Optional[str]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _BodyScan(ast.NodeVisitor):
    """Collect artefact writes/reads, ROI names, calls and imports from one code block."""

    def __init__(self) -> None:
        self.writes: Set[str] = set()
        self.reads: Set[str] = set()
        self.dynamic_keys: List[int] = []
        self.calls: Set[str] = set()
        self.imports: Set[str] = set()
        self.roi_names: Set[str] = set()

    def _artifact(self, node: ast.AST) -> Optional[Tuple[str, Optional[str]]]:
        """Return ``(kind, key)`` for ``x.obs['key']``-like subscripts; key None if dynamic."""
        if not isinstance(node, ast.Subscript):
            return None
        target = node.value
        if not isinstance(target, ast.Attribute) or target.attr not in ATTR_KINDS:
            return None
        kind = target.attr
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            return kind, idx.value
        return kind, None

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._record_write(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_write(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_write(node.target)
        self.generic_visit(node)

    def _record_write(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._record_write(elt)
            return
        art = self._artifact(target)
        if art:
            kind, key = art
            if key is None:
                self.dynamic_keys.append(getattr(target, "lineno", 0))
            else:
                self.writes.add(f"{kind}:{key}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        art = self._artifact(node)
        if art and isinstance(node.ctx, ast.Load):
            kind, key = art
            if key is None:
                self.dynamic_keys.append(node.lineno)
            else:
                self.reads.add(f"{kind}:{key}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # 'celltype' in adata.obs.columns / in adata.obsm / in adata.uns
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) and node.ops:
            if isinstance(node.ops[0], (ast.In, ast.NotIn)) and node.comparators:
                comp = node.comparators[0]
                dotted = _dotted(comp) or ""
                for kind in ATTR_KINDS:
                    if dotted.endswith(f".{kind}") or dotted.endswith(f".{kind}.columns") or dotted.endswith(f".{kind}.keys"):
                        self.reads.add(f"{kind}:{node.left.value}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        if name:
            self.calls.add(name)
            if name.endswith("get_roi") or name.endswith("create_roi"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    self.roi_names.add(node.args[0].value)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self.imports.add(node.module.split(".")[0])


def scan_code(code: str) -> _BodyScan:
    scan = _BodyScan()
    scan.visit(ast.parse(code))
    return scan


def _effects_for(calls: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for call in calls:
        for pattern, effects in LIBRARY_EFFECTS.items():
            if call == pattern or call.endswith("." + pattern.split(".", 1)[1]):
                out.update(effects)
    return out


def _glob_match(declared: str, candidates: Set[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(c, declared) for c in candidates)


def lint_frontmatter(meta: SkillFrontmatter, raw: Dict, slug: str) -> List[Issue]:
    issues: List[Issue] = []
    if len(meta.description) < 40:
        issues.append(Issue("warn", "short_description", f"description has {len(meta.description)} chars (<40)", slug))
    if len(meta.description) > 600:
        issues.append(Issue("warn", "long_description", f"description has {len(meta.description)} chars (>600)", slug))
    for key in meta.extra_keys:
        if key in ("display_title", "applicable_formats", "modality_required", "slice_behavior"):
            issues.append(Issue("info", "legacy_key", f"legacy frontmatter key {key!r} is ignored", slug))
        else:
            issues.append(Issue("warn", "unknown_key", f"unknown frontmatter key {key!r}", slug))
    if not meta.prerequisites:
        issues.append(Issue("info", "no_prerequisites", "no prerequisites declared", slug))
    for p in meta.prerequisites:
        if not p.strip():
            issues.append(Issue("error", "empty_prerequisite", "empty prerequisite entry", slug))
    if meta.schema_version >= 2 and not meta.produces:
        issues.append(Issue("warn", "no_produces", "schema_version 2 skill declares no produces", slug))
    return issues


def lint_body(meta: SkillFrontmatter, body: str, slug: str, vendored: Optional[Set[str]] = None) -> Tuple[List[Issue], Dict]:
    issues: List[Issue] = []
    blocks = extract_code_blocks(body)
    writes: Set[str] = set()
    reads: Set[str] = set()
    calls: Set[str] = set()
    imports: Set[str] = set()
    dynamic: List[int] = []
    parsed = 0
    for line, code in blocks:
        try:
            scan = scan_code(code)
        except SyntaxError as exc:
            issues.append(Issue("warn", "unparseable_block", f"python block does not parse: {exc.msg}", slug, line + (exc.lineno or 1) - 1))
            continue
        parsed += 1
        writes |= scan.writes
        reads |= scan.reads
        calls |= scan.calls
        imports |= scan.imports
        dynamic += [line + ln - 1 for ln in scan.dynamic_keys]
    if blocks and parsed == 0:
        issues.append(Issue("error", "no_parseable_code", "no python block parses", slug))
    if not blocks:
        issues.append(Issue("warn", "no_code_blocks", "body has no python code blocks", slug))
    effects = _effects_for(calls)

    if meta.produces:
        for art in meta.produces:
            if art.kind in ATTR_KINDS and not _glob_match(art.id, writes | effects):
                issues.append(Issue("warn", "produces_not_written", f"declared produces {art.id} is never written by the body (dynamic keys: {len(dynamic)})", slug))
        undeclared = {w for w in writes if w not in INCIDENTAL_WRITES and not any(a.matches(w.split(':', 1)[1]) and a.kind == w.split(':', 1)[0] for a in meta.produces)}
        undeclared = {w for w in undeclared if not w.startswith("uns:") or not w.endswith("_params")}
        for w in sorted(undeclared):
            issues.append(Issue("info", "undeclared_output", f"body writes {w} which is not in produces", slug))
    if meta.consumes:
        prereq_text = " ".join(meta.prerequisites)
        for art in meta.consumes:
            if art.kind in ATTR_KINDS and art.id not in reads and art.key not in prereq_text:
                issues.append(Issue("warn", "consumes_not_read", f"declared consumes {art.id} is neither read by the body nor named in prerequisites", slug))
    for ln in dynamic[:3]:
        issues.append(Issue("info", "dynamic_key", "artefact key is computed at runtime; provenance will rely on the snapshot diff", slug, ln))

    vendored = set(vendored or ())
    declared = {IMPORT_TO_DIST.get(p, p) for p in meta.dependencies.python}
    declared_lower = {d.lower() for d in declared} | {p.lower() for p in meta.dependencies.python}
    unknown = sorted(i for i in imports if i not in STDLIB and i not in vendored and i not in IMPORT_TO_DIST and i not in ("stat_agent",) and i.lower() not in declared_lower)
    if meta.dependencies.python:
        for i in unknown:
            issues.append(Issue("warn", "import_not_declared", f"import {i!r} is not in dependencies.python", slug))
    elif unknown:
        issues.append(Issue("info", "undeclared_imports", f"imports without a dependency declaration: {', '.join(unknown)}", slug))
    facts = {"writes": sorted(writes), "reads": sorted(reads), "effects": sorted(effects), "imports": sorted(imports), "declared": sorted(declared), "n_blocks": len(blocks)}
    return issues, facts


def lint_skill(skill: Skill) -> Tuple[List[Issue], Dict]:
    from stat_agent.skills.loader import vendored_packages

    slug = skill.slug
    issues = lint_frontmatter(skill.meta, skill.raw_frontmatter, slug)
    if skill.path.name != slug and skill.origin == "builtin":
        issues.append(Issue("info", "dir_name_differs", f"directory {skill.path.name!r} differs from slug {slug!r}", slug))
    body_issues, facts = lint_body(skill.meta, skill.body, slug, set(vendored_packages(skill)))
    return issues + body_issues, facts


def lint_file(path) -> Tuple[List[Issue], Dict]:
    """Lint a SKILL.md path that may not be in a registry (draft validation)."""
    from pathlib import Path

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        meta, body, raw = parse_skill_text(text)
    except Exception as exc:  # noqa: BLE001
        return [Issue("error", "schema", str(exc), None)], {}
    skill = Skill(meta=meta, path=p.parent, origin="file", raw_frontmatter=raw, _body=body)
    return lint_skill(skill)


__all__ = ["Issue", "lint_skill", "lint_file", "lint_body", "lint_frontmatter", "scan_code", "extract_code_blocks", "LIBRARY_EFFECTS", "INCIDENTAL_WRITES"]

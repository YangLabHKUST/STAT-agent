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
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


ROI_GETTERS = {"get_roi", "get_roi_data"}  # their first argument (or roi_name=) names the region whose cells the code reaches
ROI_STORES = {"roi_subsets", "rois"}  # session.roi_subsets, session.roi_manager.rois: name -> region, for every region
NAMES_ONLY = {"len", "list", "sorted", "set", "tuple", "print", "str", "repr"}  # of a region store: its names, no region
GROWS = {"append", "extend", "insert", "add", "update", "setdefault", "symmetric_difference_update", "__setitem__", "__iadd__"}  # can put a new string into a list, set or dict


class _StringNames:
    """What the names of one block stand for, where that can only be string literals.

    ``strings(node)`` gives the strings an expression can be (a literal, or a name such as ``name = 'tumor'`` or the
    variable of a loop or comprehension over a list of them, also as ``for name, v in d.items()`` and ``for i, name
    in enumerate(names)``); ``items(node)`` the strings that a list, tuple, set or dict (its keys) holds, written
    out or under a name. A name counts only when the block binds it exactly once, above the place that uses it, and
    never adds to it in place (no second assignment, no ``+=``, no ``names.append(...)``, no ``names[0] = ...``, not
    a parameter, an import or a ``with`` target), so the answer does not depend on the order the code runs in.
    Everything else is None: unknown, never guessed."""

    def __init__(self, tree: Optional[ast.AST] = None) -> None:
        self.first_bound: Dict[str, int] = {}  # every name the block binds itself -> the first line that does (parameters and comprehension variables are local and not listed)
        self.assigned: Set[str] = set()  # the names among them bound as variables (not imports, functions, classes)
        self._one: Dict[str, Tuple[int, Set[str]]] = {}  # name -> (line bound, the strings it can be)
        self._many: Dict[str, Tuple[int, Set[str]]] = {}  # name -> (line bound, the strings it holds)
        if tree is None:
            return
        sites: Dict[str, List[Optional[Tuple[str, ast.AST, int]]]] = {}  # None: a site that is not understood
        understood: Set[int] = set()

        def note(name: str, line: int) -> None:
            self.first_bound[name] = min(self.first_bound.get(name, line), line)

        def bind(target: ast.AST, how: str, source: ast.AST, line: int) -> None:
            if isinstance(target, ast.Name):
                sites.setdefault(target.id, []).append((how, source, line))
                understood.add(id(target))
            elif how == "value" and isinstance(target, (ast.Tuple, ast.List)) and isinstance(source, (ast.Tuple, ast.List)) and len(target.elts) == len(source.elts):
                for t, v in zip(target.elts, source.elts):  # a, b = 'x', 'y'
                    bind(t, how, v, line)

        def bind_item(target: ast.AST, source: ast.AST, line: int) -> None:
            if isinstance(target, ast.Tuple) and len(target.elts) == 2 and isinstance(source, ast.Call) and not source.keywords:
                f = source.func
                if isinstance(f, ast.Attribute) and f.attr == "items" and not source.args:
                    target, source = target.elts[0], f.value  # for name, v in d.items()
                elif isinstance(f, ast.Name) and f.id == "enumerate" and len(source.args) == 1:
                    target, source = target.elts[1], source.args[0]  # for i, name in enumerate(names)
            bind(target, "item", source, line)

        nodes = list(ast.walk(tree))
        for node in nodes:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    bind(t, "value", node.value, node.lineno)
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                bind(node.target, "value", node.value, node.lineno)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bind_item(node.target, node.iter, node.lineno)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in node.generators:  # bound where the comprehension starts: its element may be written above its ``for``
                    bind_item(gen.target, gen.iter, node.lineno)
        in_place: Set[str] = set()
        local = {id(n) for c in nodes if isinstance(c, ast.comprehension) for n in ast.walk(c.target)}
        for node in nodes:
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                if id(node) not in local:
                    note(node.id, node.lineno)
                    self.assigned.add(node.id)
                if id(node) not in understood:
                    sites.setdefault(node.id, []).append(None)  # +=, del, with ... as, an element of an unmatched tuple
            elif isinstance(node, (ast.Subscript, ast.Attribute)) and not isinstance(node.ctx, ast.Load):
                base = node.value
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name):
                    in_place.add(base.id)  # names[0] = ...
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.attr in GROWS:
                in_place.add(node.func.value.id)  # names.append(...)
            else:
                if isinstance(node, ast.arg):
                    sites.setdefault(node.arg, []).append(None)
                    continue
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    other, line = [node.name], node.lineno
                elif isinstance(node, ast.alias):
                    other, line = [(node.asname or node.name).split(".")[0]], getattr(node, "lineno", 0)
                elif isinstance(node, (ast.Global, ast.Nonlocal)):
                    other, line = node.names, node.lineno
                elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
                    other, line = [node.name], node.lineno
                elif isinstance(node, ast.MatchMapping) and node.rest:
                    other, line = [node.rest], node.lineno
                else:
                    continue
                for n in other:
                    sites.setdefault(n, []).append(None)
                    note(n, line)
        single = {n: s[0] for n, s in sites.items() if len(s) == 1 and s[0] is not None and n not in in_place}
        while True:  # a loop variable resolves once the list it runs over has
            known = len(self._one) + len(self._many)
            for name, (how, source, line) in single.items():
                if name in self._one or name in self._many:
                    continue
                if how == "item":
                    if got := self.items(source):
                        self._one[name] = (line, got)
                elif got := self.strings(source):
                    self._one[name] = (line, got)
                elif got := self.items(source):
                    self._many[name] = (line, got)
            if len(self._one) + len(self._many) == known:
                break

    @staticmethod
    def _at(table: Dict[str, Tuple[int, Set[str]]], node: ast.Name) -> Optional[Set[str]]:
        line, got = table.get(node.id, (0, None))
        return got if got and line <= node.lineno else None

    def strings(self, node: Optional[ast.AST]) -> Optional[Set[str]]:
        if isinstance(node, ast.Constant):
            return {node.value} if isinstance(node.value, str) else None
        return self._at(self._one, node) if isinstance(node, ast.Name) else None

    def items(self, node: Optional[ast.AST]) -> Optional[Set[str]]:
        if isinstance(node, ast.Name):
            return self._at(self._many, node)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            parts = [self.strings(e) for e in (node.keys if isinstance(node, ast.Dict) else node.elts)]
            return set().union(*parts) if parts and all(parts) else None
        return None


class _BodyScan(ast.NodeVisitor):
    """Collect artefact writes/reads, ROI names, calls and imports from one code block.

    Reads are what the ledger turns into slice inputs, so a read is recorded only when its
    receiver can be a session slice: ``session.get_slice(i).adata``, a name bound to one, a
    slice iterated from the session, or a subset/copy of one (its values still come from the
    slice). A read on an object built by any other call (``ad.concat(...)``, ``sc.read_h5ad(...)``)
    or on a name listed in ``derived_names`` (the live namespace holds a non-slice AnnData under
    it, such as a pooled object kept from an earlier block) is not a read of any slice and is
    skipped. An unknown name counts (the conservative choice). Two write lists are kept:
    ``writes`` is receiver-agnostic (what the code assigned on any object; the snapshot diff decides
    what changed and the output check uses it), ``slice_writes`` holds only the assignments and the
    known library effects whose receiver is a session slice itself, not a subset or a copy, because
    the ledger clears a stale flag on a key the code re-wrote only when the slice's own value was
    recomputed.

    A region is read when the code obtains the region object: ``get_roi``/``get_roi_data``, or a lookup in the
    region store (``session.roi_subsets[...]``, ``.get(...)``). Its name goes to ``roi_names`` when the scan can
    tell what the argument stands for (``_StringNames``), and a region object kept in the namespace from an earlier
    block counts under its own name (``roi_vars``); ``taken`` lists the names the block uses with the value an
    earlier block left, so that the runtime can add the regions those values were computed from. When the scan cannot tell (a computed name, a parameter, a name
    kept from an earlier block, the store handed on whole), ``roi_any`` is set and the ledger takes every region of
    the step's target slices as an input: a dependency recorded once too often costs a recomputation, one that is
    missed leaves a result looking valid after its region was replaced. Asking the store for its names (``in``,
    ``.keys()``, ``len``, a loop over it) reads no region."""

    def __init__(self, derived_names: Iterable[str] = (), slice_names: Iterable[str] = (), roi_vars: Optional[Dict[str, str]] = None) -> None:
        self.writes: Set[str] = set()
        self.slice_writes: Set[str] = set()  # the writes (plus library effects) whose receiver is a slice itself
        self.reads: Set[str] = set()
        self.dynamic_keys: List[int] = []
        self.calls: Set[str] = set()
        self.imports: Set[str] = set()
        self.roi_names: Set[str] = set()
        self.roi_any = False  # a region was read under a name the scan cannot tell
        self.roi_vars: Dict[str, str] = dict(roi_vars or {})  # namespace variable -> name of the region it holds
        self.taken: Set[str] = set()  # names used with the value an earlier block left in the namespace
        self._locals: List[Set[str]] = []  # parameters and comprehension variables of the scopes being visited
        self.strings = _StringNames()  # set by scan_code from the whole block
        self._store_uses: Set[int] = set()  # region-store mentions already accounted for
        self.slice_names: Set[str] = set(slice_names)  # names bound to a session slice or its AnnData
        self.copy_names: Set[str] = set()  # slice names holding a subset or a copy: reads count, writes stay on the copy
        self.derived_names: Set[str] = set(derived_names)
        self.container_names: Dict[str, Tuple[str, Optional[str]]] = {}  # obs = adata.obs -> {'obs': ('obs', class)}
        self.direct_containers: Set[str] = set()  # container aliases whose owner is a slice itself

    def _artifact(self, node: ast.AST) -> Optional[Tuple[str, Optional[str]]]:
        """Return ``(kind, key)`` for ``x.obs['key']``-like subscripts; key None if dynamic."""
        if not isinstance(node, ast.Subscript):
            return None
        target = node.value
        if isinstance(target, ast.Name) and target.id in self.container_names:
            kind = self.container_names[target.id][0]
        elif not isinstance(target, ast.Attribute) or target.attr not in ATTR_KINDS:
            return None
        else:
            kind = target.attr
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            return kind, idx.value
        return kind, None

    def _classify(self, node: ast.AST) -> Optional[str]:
        """'slice' for a session slice or a subset/copy of one, 'derived' for an object built by any
        other call, None when unknown."""
        if isinstance(node, ast.Name):
            if node.id in self.derived_names:
                return "derived"
            if node.id in self.slice_names:
                return "slice"
            return None
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func) or ""
            if dotted.startswith("session.") or dotted.endswith(".get_slice"):
                return "slice"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":
                return self._classify(node.func.value)
            return "derived"
        if isinstance(node, ast.Attribute):
            inner = self._classify(node.value)
            if node.attr == "adata":
                return "derived" if inner == "derived" else "slice"
            return inner
        if isinstance(node, ast.Subscript):
            return self._classify(node.value)
        return None

    def _receiver_class(self, node: ast.Subscript) -> Optional[str]:
        """Class of the object whose container ``node`` subscripts (``x`` in ``x.obs['k']``)."""
        if isinstance(node.value, ast.Name):
            return self.container_names[node.value.id][1]
        return self._classify(node.value.value)

    def _direct_slice(self, node: Optional[ast.AST]) -> bool:
        """True when ``node`` denotes a session slice itself: a ``session.*`` call such as
        ``session.get_slice(i)``, ``session.slices[i]``, the ``.adata`` of one of those, or a name
        bound to one. A subset, a ``.copy()``, an object built by any other call and an unknown name
        are not: a write on them never reaches the slice."""
        if isinstance(node, ast.Name):
            return node.id in self.slice_names and node.id not in self.copy_names
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":
                return False
            dotted = _dotted(node.func) or ""
            return dotted.startswith("session.") or dotted.endswith(".get_slice")
        if isinstance(node, ast.Attribute):
            return node.attr == "adata" and self._direct_slice(node.value)
        if isinstance(node, ast.Subscript):
            return (_dotted(node.value) or "") == "session.slices"  # session.slices[i]
        return False

    def _writes_slice(self, node: ast.Subscript) -> bool:
        """Whether the assignment target ``x.obs['k']`` (or ``obs['k']`` through an alias) hits a slice."""
        if isinstance(node.value, ast.Name):
            return node.value.id in self.direct_containers
        return self._direct_slice(node.value.value)

    def _bind(self, target: ast.AST, cls: Optional[str], value: Optional[ast.AST] = None) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind(elt, cls)
            return
        if isinstance(target, ast.Name):
            self.slice_names.discard(target.id)
            self.copy_names.discard(target.id)
            self.derived_names.discard(target.id)
            self.container_names.pop(target.id, None)
            self.direct_containers.discard(target.id)
            if isinstance(value, ast.Attribute) and value.attr in ATTR_KINDS:  # obs = adata.obs
                self.container_names[target.id] = (value.attr, self._classify(value.value))
                if self._direct_slice(value.value):
                    self.direct_containers.add(target.id)
            elif cls == "slice":
                self.slice_names.add(target.id)
                if not self._direct_slice(value):  # a subset or a copy: its values come from the slice, a write on it stays there
                    self.copy_names.add(target.id)
            elif cls == "derived":
                self.derived_names.add(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._record_write(t)
        self.generic_visit(node)  # reads in the value refer to the names as bound before this line
        cls = self._classify(node.value)
        for t in node.targets:
            self._bind(t, cls, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_write(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_write(node.target)
        self.generic_visit(node)
        if node.value is not None:
            self._bind(node.target, self._classify(node.value), node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # a parameter shadows an outer slice name inside the body and may be bound to a copy at the
        # call, so a write on it is not a slice write; names bound to slices inside the body stay
        # inside it. Read attribution is untouched (reads on unknown names count anyway).
        saved = (set(self.slice_names), set(self.copy_names), set(self.direct_containers))
        a = node.args
        params = {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if p is not None}
        self.slice_names -= params
        self.copy_names -= params
        self.direct_containers -= params
        self._scoped(node, params)
        self.slice_names, self.copy_names, self.direct_containers = saved

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_For(self, node: ast.For) -> None:
        self._names_only(node.iter)
        self._bind(node.target, self._classify(node.iter))
        if self._direct_slice(node.iter):  # for s in session.iter_slices(); for sid, s in session.slices.items()
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    self.copy_names.discard(n.id)
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
                if self._writes_slice(target):
                    self.slice_writes.add(f"{kind}:{key}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_store(node.value) and isinstance(node.ctx, ast.Load):  # session.roi_subsets['tumor']
            self._store_uses.add(id(node.value))
            self._roi_read(node.slice)
        art = self._artifact(node)
        if art and isinstance(node.ctx, ast.Load):
            kind, key = art
            if key is None:
                self.dynamic_keys.append(node.lineno)
            elif self._receiver_class(node) != "derived":
                self.reads.add(f"{kind}:{key}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._names_only(comp)  # 'tumor' in session.roi_subsets
        # 'celltype' in adata.obs.columns / in adata.obsm / in adata.uns
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) and node.ops:
            if isinstance(node.ops[0], (ast.In, ast.NotIn)) and node.comparators:
                comp = node.comparators[0]
                dotted = _dotted(comp) or ""
                parts = dotted.split(".")
                if parts[0] in self.container_names and parts[1:] in ([], ["columns"], ["keys"]):  # 'k' in obs.columns
                    kind, cls = self.container_names[parts[0]]
                    if cls != "derived":
                        self.reads.add(f"{kind}:{node.left.value}")
                for kind in ATTR_KINDS:
                    if dotted.endswith(f".{kind}") or dotted.endswith(f".{kind}.columns") or dotted.endswith(f".{kind}.keys"):
                        container = comp
                        while isinstance(container, ast.Attribute) and container.attr != kind:
                            container = container.value
                        receiver = container.value if isinstance(container, ast.Attribute) else container
                        if self._classify(receiver) != "derived":
                            self.reads.add(f"{kind}:{node.left.value}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        leaf = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ("roi_name", "name")), None)
        if leaf in ROI_GETTERS:
            self._roi_read(first)
        elif leaf == "create_roi":  # makes a region; reads none
            self.roi_names |= self.strings.strings(first) or set()
        elif leaf == "get_rois_for_slice":
            self.roi_any = True
        elif isinstance(func, ast.Attribute) and self._is_store(func.value) and leaf in ("get", "keys"):
            self._store_uses.add(id(func.value))
            if leaf == "get":  # session.roi_subsets.get('tumor')
                self._roi_read(first)
        elif leaf in NAMES_ONLY and len(node.args) == 1:
            self._names_only(node.args[0])
        name = _dotted(node.func)
        if name:
            self.calls.add(name)
            effects = LIBRARY_EFFECTS.get(name, [])
            if effects:  # sc.tl.leiden(x) writes obs:leiden on the object it is given
                target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "adata"), None)
                if target is not None and self._direct_slice(target):
                    self.slice_writes.update(e for e in effects if "*" not in e)
        self.generic_visit(node)

    @staticmethod
    def _is_store(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and node.attr in ROI_STORES

    def _names_only(self, node: ast.AST) -> None:
        if self._is_store(node):
            self._store_uses.add(id(node))

    def _roi_read(self, arg: Optional[ast.AST]) -> None:
        names = self.strings.strings(arg)
        if names:
            self.roi_names |= names
        else:
            self.roi_any = True

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._names_only(node.iter)
        self.generic_visit(node)

    def _scoped(self, node: ast.AST, names: Set[str]) -> None:
        self._locals.append(names)
        self.generic_visit(node)
        self._locals.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scoped(node, {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)})

    def visit_ListComp(self, node: ast.AST) -> None:
        self._scoped(node, {n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)})

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ROI_STORES and id(node) not in self._store_uses:
            self.roi_any = True  # the store handed on whole: its regions iterated, aliased, passed to a call
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not any(node.id in names for names in self._locals) and node.lineno <= self.strings.first_bound.get(node.id, node.lineno):
            self.taken.add(node.id)  # not bound by this block before this line (``x = x[mask]`` still takes the earlier x)
            if node.id in self.roi_vars:
                self.roi_names.add(self.roi_vars[node.id])  # a region kept from an earlier block

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self.imports.add(node.module.split(".")[0])


def scan_code(code: str, derived_names: Iterable[str] = (), slice_names: Iterable[str] = (), roi_vars: Optional[Dict[str, str]] = None) -> _BodyScan:
    tree = ast.parse(code)
    scan = _BodyScan(derived_names, slice_names, roi_vars)
    scan.strings = _StringNames(tree)
    scan.visit(tree)
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
        issues.append(Issue("info", "no_produces", "declares no produces (mode-dependent outputs or a new slice): the output check is skipped", slug))
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

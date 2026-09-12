"""Static (AST) safety check run before any generated code executes.

Blocks process/shell escapes, dynamic code execution and destructive filesystem calls.
It is deliberately a small allow/deny list: analysis code legitimately uses the network
(gene-set downloads), multiprocessing (Ray workers) and file writes inside the session
working directory.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BLOCKED_CALLS = {
    "os.system",
    "os.popen",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.spawnl",
    "os.spawnv",
    "os.kill",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os._exit",
    "shutil.rmtree",
    "sys.exit",
    "exit",
    "quit",
    "eval",
    "exec",
    "compile",
    "__import__",
    "importlib.import_module",
    "importlib.reload",
    "builtins.eval",
    "builtins.exec",
}
BLOCKED_CALL_PREFIXES = ("subprocess.", "ctypes.", "pty.")
BLOCKED_IMPORTS = {"subprocess", "ctypes", "pty"}
WRITE_MODES = ("w", "a", "x", "+")


@dataclass
class Violation:
    code: str
    message: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.message} (line {self.lineno})"


class SafetyViolation(RuntimeError):
    def __init__(self, violations: Sequence[Violation]):
        self.violations = list(violations)
        super().__init__("; ".join(str(v) for v in self.violations))


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(aliases.get(node.id, node.id))
        return ".".join(reversed(parts))
    return None


class _Checker(ast.NodeVisitor):
    def __init__(self, allowed_write_dirs: Sequence[Path]):
        self.aliases: dict[str, str] = {}
        self.violations: list[Violation] = []
        self.allowed = [Path(p).resolve() for p in allowed_write_dirs]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BLOCKED_IMPORTS:
                self.violations.append(Violation("blocked_import", f"import of '{alias.name}' is not allowed", node.lineno))
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod.split(".")[0] in BLOCKED_IMPORTS:
            self.violations.append(Violation("blocked_import", f"import from '{mod}' is not allowed", node.lineno))
        for alias in node.names:
            full = f"{mod}.{alias.name}" if mod else alias.name
            if full in BLOCKED_CALLS:
                self.violations.append(Violation("blocked_import", f"'{full}' is not allowed", node.lineno))
            self.aliases[alias.asname or alias.name] = full
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func, self.aliases)
        if name:
            if name in BLOCKED_CALLS or name.startswith(BLOCKED_CALL_PREFIXES):
                self.violations.append(Violation("blocked_call", f"call to '{name}' is not allowed", node.lineno))
            elif name in ("open", "io.open", "builtins.open"):
                self._check_open(node)
        self.generic_visit(node)

    def _check_open(self, node: ast.Call) -> None:
        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                mode = kw.value.value
        if not mode or not any(ch in mode for ch in WRITE_MODES):
            return
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            path = Path(node.args[0].value)
            if path.is_absolute():
                resolved = path.resolve()
                if self.allowed and not any(str(resolved).startswith(str(a)) for a in self.allowed):
                    self.violations.append(Violation("write_outside_workdir", f"writing to '{path}' outside the session directory is not allowed", node.lineno))


def check_code(code: str, allowed_write_dirs: Sequence[Path] = ()) -> list[Violation]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [Violation("syntax_error", f"SyntaxError: {exc.msg}", exc.lineno or 0)]
    checker = _Checker(allowed_write_dirs)
    checker.visit(tree)
    return checker.violations


def clean_code(code: str) -> str:
    """Strip a surrounding markdown fence if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


__all__ = ["Violation", "SafetyViolation", "check_code", "clean_code", "BLOCKED_CALLS", "BLOCKED_IMPORTS"]

"""Reliability-oriented validation for generated robot programs.

This is intentionally not presented as an adversarial Python sandbox.  It
reduces accidental access to files, processes, networks, and raw simulator
objects while a spawned process provides the hard reliability boundary.
"""

from __future__ import annotations

import ast
import builtins
import os
from collections.abc import Callable
from typing import Any

ALLOWED_IMPORT_ROOTS = frozenset({"math", "numpy", "scipy"})
FORBIDDEN_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "__import__",
    }
)
FORBIDDEN_NAMES = frozenset({"env", "APIS", "task_completed", "compute_reward"})
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "AUTH_TOKEN",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


class UnsafeProgramError(ValueError):
    pass


class _ProgramValidator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise UnsafeProgramError(f"import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            raise UnsafeProgramError("relative imports are not allowed")
        root = node.module.split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            raise UnsafeProgramError(f"import is not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise UnsafeProgramError(f"private/dunder attribute is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            raise UnsafeProgramError(f"name is not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise UnsafeProgramError(f"call is not allowed: {node.func.id}")
        self.generic_visit(node)


def validate_program(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, filename="generated_robot_program.py", mode="exec")
    except SyntaxError as exc:
        raise UnsafeProgramError(f"invalid Python syntax: {exc}") from exc
    _ProgramValidator().visit(tree)
    return tree


def scrub_sensitive_environment() -> None:
    """Remove credentials inherited by the fresh simulator child process."""
    for name in tuple(os.environ):
        upper_name = name.upper()
        if any(marker in upper_name for marker in SENSITIVE_ENV_MARKERS):
            os.environ.pop(name, None)


def _restricted_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    del globals_, locals_
    if level or name.split(".", 1)[0] not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"generated program import is not allowed: {name}")
    return builtins.__import__(name, fromlist=fromlist, level=0)


def safe_builtins() -> dict[str, Any]:
    """Return the generated program's restricted builtins mapping.

    CPython accepts any mapping for ``exec(..., {"__builtins__": ...})`` at
    first, but extension types such as NumPy scalars later look up
    ``__import__`` through attribute access.  A ``MappingProxyType`` therefore
    fails on otherwise harmless expressions such as
    ``float(array.max())``.  A fresh plain dict keeps the same whitelist while
    satisfying the interpreter and extension-module ABI expectations.
    """
    allowed_names = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "next",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    )
    values: dict[str, Callable[..., Any] | type | object] = {
        name: getattr(builtins, name) for name in allowed_names
    }
    values["__import__"] = _restricted_import
    return values

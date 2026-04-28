#!/usr/bin/env python3
"""Audit values.py assignments and repo references without importing hardware code."""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "values.py"


def _python_files() -> list[Path]:
    ignored_dirs = {".git", "__pycache__", ".venv", "venv", "env"}
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _assigned_values_names() -> tuple[dict[str, list[int]], set[str]]:
    tree = ast.parse(VALUES.read_text(), filename=str(VALUES))
    assignments: dict[str, list[int]] = defaultdict(list)

    def add_target(target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Name) and target.id.isupper():
            assignments[target.id].append(lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                add_target(elt, lineno)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                add_target(target, node.lineno)
    return dict(assignments), set(assignments)


class ValuesReferenceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: set[str] = set()
        self.from_imports: dict[str, int] = {}
        self.attr_refs: dict[str, list[int]] = defaultdict(list)
        self.getattr_refs: dict[str, list[int]] = defaultdict(list)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "values":
                self.aliases.add(alias.asname or "values")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "values":
            for alias in node.names:
                if alias.name == "*":
                    continue
                self.from_imports[alias.name] = node.lineno
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.aliases:
            self.attr_refs[node.attr].append(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.getattr_refs[node.args[1].value].append(node.lineno)
        self.generic_visit(node)


def _collect_references() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    direct: dict[str, list[str]] = defaultdict(list)
    imported: dict[str, list[str]] = defaultdict(list)
    getattr_refs: dict[str, list[str]] = defaultdict(list)
    for path in _python_files():
        if path == VALUES:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"[audit_values] WARNING: could not parse {path.relative_to(ROOT)}: {exc}")
            continue
        visitor = ValuesReferenceVisitor()
        visitor.visit(tree)
        rel = str(path.relative_to(ROOT))
        for name, lines in visitor.from_imports.items():
            imported[name].extend(f"{rel}:{line}" for line in lines)
        for name, lines in visitor.attr_refs.items():
            direct[name].extend(f"{rel}:{line}" for line in lines)
        for name, lines in visitor.getattr_refs.items():
            getattr_refs[name].extend(f"{rel}:{line}" for line in lines)
    return dict(direct), dict(imported), dict(getattr_refs)


def main() -> int:
    assignments, assigned_names = _assigned_values_names()
    direct, imported, getattr_refs = _collect_references()
    referenced_names = set(direct) | set(imported) | set(getattr_refs)
    missing = sorted(name for name in referenced_names if name not in assigned_names)
    duplicates = {name: lines for name, lines in assignments.items() if len(lines) > 1}
    unreferenced = sorted(assigned_names - referenced_names)

    print("[audit_values] values.py duplicate assignments:")
    if duplicates:
        for name in sorted(duplicates):
            print(f"  {name}: lines {duplicates[name]}")
    else:
        print("  none")

    print("\n[audit_values] repo references to missing values.py names:")
    if missing:
        for name in missing:
            refs = direct.get(name, []) + imported.get(name, []) + getattr_refs.get(name, [])
            print(f"  {name}: {', '.join(refs)}")
    else:
        print("  none")

    print("\n[audit_values] assigned values.py names with no in-repo Python reference:")
    if unreferenced:
        for name in unreferenced:
            print(f"  {name}")
    else:
        print("  none")

    print(
        "\n[audit_values] summary: "
        f"{len(assignments)} assigned names, {len(referenced_names)} referenced names, "
        f"{len(duplicates)} duplicate-assigned names, {len(unreferenced)} unreferenced names"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

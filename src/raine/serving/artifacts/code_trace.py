"""Trace and copy project-local Python source needed by a model artifact."""

from __future__ import annotations

import ast
import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence


@dataclass(frozen=True)
class CodeTraceResult:
    modules: tuple[str, ...]
    files: tuple[Path, ...]
    destination: Path | None = None


def _site_package_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for entry in sys.path:
        if not entry or entry == "":
            continue
        path = Path(entry).resolve()
        if path.name == "site-packages" or "site-packages" in path.parts:
            roots.append(path)
    return tuple(roots)


def is_stdlib_module(module_name: str) -> bool:
    top_level = module_name.split(".", 1)[0]
    return top_level in sys.stdlib_module_names


def module_file_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        return None
    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        return origin
    if origin.suffix == ".py":
        return origin
    return None


def is_third_party_module(
    module_name: str,
    *,
    local_roots: Sequence[str] = ("raine",),
) -> bool:
    if is_stdlib_module(module_name):
        return True

    top_level = module_name.split(".", 1)[0]
    if top_level in local_roots:
        return False

    module_path = module_file_path(module_name)
    if module_path is None:
        return True

    resolved = module_path.resolve()
    for site_root in _site_package_roots():
        if resolved.is_relative_to(site_root):
            return True
    return False


def _module_names_from_class(cls: type) -> set[str]:
    modules: set[str] = set()
    for base in cls.__mro__:
        module_name = getattr(base, "__module__", None)
        if module_name and module_name != "__main__":
            modules.add(module_name)
    return modules


def _ancestor_package_modules(module_name: str) -> tuple[str, ...]:
    parts = module_name.split(".")
    if len(parts) <= 1:
        return ()
    return tuple(".".join(parts[:index]) for index in range(1, len(parts)))


def _is_traceable_local_module(
    module_name: str,
    *,
    local_roots: Sequence[str],
    search_paths: Sequence[Path] | None,
) -> bool:
    if is_third_party_module(module_name, local_roots=local_roots):
        return False

    top_level = module_name.split(".", 1)[0]
    if top_level not in local_roots and search_paths:
        module_path = module_file_path(module_name)
        if module_path is None:
            return False
        if not any(
            module_path.resolve().is_relative_to(path.resolve())
            for path in search_paths
        ):
            return False
    return True


def _parse_imports(source_path: Path, module_name: str) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                parts = module_name.split(".")
                package = ".".join(parts[: max(len(parts) - node.level, 0)])
                if node.module:
                    imports.add(f"{package}.{node.module}".lstrip("."))
                continue
            if node.module:
                imports.add(node.module)

    return imports


def collect_local_modules(
    seeds: Sequence[type | str | ModuleType],
    *,
    local_roots: Sequence[str] = ("raine",),
    search_paths: Sequence[Path] | None = None,
) -> tuple[str, ...]:
    """Collect local module names reachable from seed classes/modules via MRO and AST imports.

    Parent package modules (``__init__.py``) for every discovered module are always
    included so copied code remains importable under ``code/`` on ``sys.path``.
    """
    queue: list[str] = []
    seen: set[str] = set()
    discovered: set[str] = set()

    for seed in seeds:
        if isinstance(seed, str):
            queue.append(seed)
        elif isinstance(seed, ModuleType):
            queue.append(seed.__name__)
        else:
            queue.extend(_module_names_from_class(seed))

    if search_paths:
        for path in search_paths:
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    while queue:
        module_name = queue.pop(0)
        if module_name in seen:
            continue
        seen.add(module_name)

        if not _is_traceable_local_module(
            module_name,
            local_roots=local_roots,
            search_paths=search_paths,
        ):
            continue

        discovered.add(module_name)
        for ancestor in _ancestor_package_modules(module_name):
            if ancestor not in seen:
                queue.append(ancestor)

        module_path = module_file_path(module_name)
        if module_path is None:
            continue

        for imported in _parse_imports(module_path, module_name):
            imported_top = imported.split(".", 1)[0]
            if imported_top in local_roots or not is_third_party_module(
                imported, local_roots=local_roots
            ):
                queue.append(imported)

    return tuple(sorted(discovered))


def module_source_files(modules: Sequence[str]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for module_name in modules:
        module_path = module_file_path(module_name)
        if module_path is not None:
            files.add(module_path.resolve())
    return tuple(sorted(files))


def _destination_for_module_file(
    source_file: Path,
    destination_root: Path,
    *,
    anchor_paths: Sequence[Path],
) -> Path:
    resolved = source_file.resolve()
    for anchor in anchor_paths:
        anchor = anchor.resolve()
        if resolved.is_relative_to(anchor):
            relative = resolved.relative_to(anchor)
            return destination_root / relative
    return destination_root / source_file.name


def copy_local_code_paths(
    modules: Sequence[str],
    destination_root: Path,
    *,
    search_paths: Sequence[Path] | None = None,
) -> CodeTraceResult:
    """Copy traced local module files into ``destination_root/code`` preserving structure."""
    files = module_source_files(modules)
    code_root = destination_root / "code"
    anchors = tuple(path.resolve() for path in (search_paths or (destination_root,)))

    copied: list[Path] = []
    for source_file in files:
        target = _destination_for_module_file(source_file, code_root, anchor_paths=anchors)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied.append(target)

    return CodeTraceResult(
        modules=tuple(modules),
        files=tuple(copied),
        destination=code_root,
    )


def materialize_artifact_code(
    destination_root: Path,
    seeds: Sequence[type | str | ModuleType],
    *,
    local_roots: Sequence[str] = ("raine",),
    search_paths: Sequence[Path] | None = None,
) -> CodeTraceResult:
    """Trace and copy all local code required by ``seeds`` into an artifact directory."""
    modules = collect_local_modules(
        seeds,
        local_roots=local_roots,
        search_paths=search_paths,
    )
    return copy_local_code_paths(
        modules,
        destination_root,
        search_paths=search_paths,
    )

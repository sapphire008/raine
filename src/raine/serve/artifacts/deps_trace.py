"""Trace and materialize third-party dependencies for model artifacts.

Uses PEP 621 ``pyproject.toml`` for declarative metadata and PEP 751
``pylock.toml`` (via ``uv export``) for reproducible locked installs.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_VERSION_SPECIFIER_MARKERS = (">=", "<=", "==", "!=", "~=", ">", "<")


def normalize_requires_python(requires_python: str) -> str:
    """Normalize ``requires-python`` to a PEP 440 version specifier.

    Poetry and other tools reject bare versions such as ``3.13.12``; convert
    those to exact pins like ``==3.13.12``. Values that already include a
    specifier operator are returned unchanged.
    """
    normalized = requires_python.strip()
    if not normalized:
        return ">=3.11"
    if any(marker in normalized for marker in _VERSION_SPECIFIER_MARKERS):
        return normalized
    return f"=={normalized}"


def _requirement_key(requirement: str) -> str:
    """Return a normalized package name for deduplication and overrides."""
    token = requirement.strip().split("[", 1)[0]
    for separator in ("===", "==", ">=", "<=", "!=", "~=", ">", "<", " @ "):
        if separator in token:
            token = token.split(separator, 1)[0]
            break
    return token.strip().lower().replace("_", "-")


def _merge_requirements(
    *requirement_groups: Iterable[str],
    overrides: Iterable[str] = (),
) -> tuple[str, ...]:
    """Merge requirement strings, applying ``overrides`` last by package name."""
    merged: dict[str, str] = {}
    for requirements in requirement_groups:
        for requirement in requirements:
            key = _requirement_key(requirement)
            if key and key not in merged:
                merged[key] = requirement.strip()
    for requirement in overrides:
        key = _requirement_key(requirement)
        if key:
            merged[key] = requirement.strip()
    return tuple(merged.values())


@dataclass(frozen=True)
class ArtifactDependencySpec:
    """PEP 621 project metadata written into a model artifact directory."""

    name: str
    version: str = "0.1.0"
    requires_python: str = ">=3.11"
    dependencies: tuple[str, ...] = ()
    description: str = "Raine model artifact runtime environment"


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest directory containing ``pyproject.toml``."""
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found at or above {path}")


def resolve_dependency_project(
    *,
    project_root: Path | None = None,
    pyproject_toml_path: Path | None = None,
    start: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(project_root, pyproject.toml path)`` for dependency export."""
    if pyproject_toml_path is not None:
        resolved_pyproject = Path(pyproject_toml_path).resolve()
        if resolved_pyproject.name != "pyproject.toml":
            raise ValueError(
                f"pyproject_toml_path must point to pyproject.toml, got {resolved_pyproject.name!r}"
            )
        if not resolved_pyproject.is_file():
            raise FileNotFoundError(f"pyproject.toml not found: {resolved_pyproject}")
        return resolved_pyproject.parent, resolved_pyproject

    resolved_root = find_project_root(project_root or start)
    return resolved_root, resolved_root / "pyproject.toml"


def read_project_metadata_at(pyproject_path: Path) -> dict:
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)


def read_project_metadata(
    project_root: Path | None = None,
    *,
    pyproject_toml_path: Path | None = None,
    start: Path | None = None,
) -> dict:
    _, resolved_pyproject = resolve_dependency_project(
        project_root=project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
    )
    return read_project_metadata_at(resolved_pyproject)


def merge_project_dependencies(
    project_root: Path | None = None,
    *,
    pyproject_toml_path: Path | None = None,
    start: Path | None = None,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    extra_dependencies: Sequence[str] = (),
    include_base: bool = True,
) -> ArtifactDependencySpec:
    """Merge base, optional, and dependency-group specs from the source project."""
    _, resolved_pyproject = resolve_dependency_project(
        project_root=project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
    )
    metadata = read_project_metadata_at(resolved_pyproject)
    project = metadata.get("project", {})

    base_dependencies: list[str] = []
    if include_base:
        base_dependencies = list(project.get("dependencies", []))

    optional = project.get("optional-dependencies", {})
    extra_requirements: list[str] = []
    for extra in extras:
        extra_requirements.extend(optional.get(extra, []))

    dependency_groups = metadata.get("dependency-groups", {})
    group_requirements: list[str] = []
    for group in groups:
        group_requirements.extend(dependency_groups.get(group, []))

    merged = _merge_requirements(
        base_dependencies,
        extra_requirements,
        group_requirements,
        overrides=extra_dependencies,
    )

    return ArtifactDependencySpec(
        name=f"{project.get('name', 'raine')}-artifact",
        version=str(project.get("version", "0.1.0")),
        requires_python=normalize_requires_python(
            str(project.get("requires-python", ">=3.11"))
        ),
        dependencies=merged,
        description=f"Runtime dependencies for {project.get('name', 'raine')} model artifacts",
    )


def format_pyproject_toml(spec: ArtifactDependencySpec) -> str:
    """Render a minimal PEP 621 ``pyproject.toml`` for an artifact."""
    lines = [
        "[project]",
        f'name = "{spec.name}"',
        f'version = "{spec.version}"',
        f'description = "{spec.description}"',
        f'requires-python = "{normalize_requires_python(spec.requires_python)}"',
        "dependencies = [",
    ]
    for requirement in spec.dependencies:
        lines.append(f'    "{requirement}",')
    lines.append("]")
    lines.append("")
    lines.append("[build-system]")
    lines.append('requires = ["hatchling"]')
    lines.append('build-backend = "hatchling.build"')
    lines.append("")
    return "\n".join(lines)


def write_artifact_pyproject(output_dir: Path, spec: ArtifactDependencySpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pyproject.toml"
    path.write_text(format_pyproject_toml(spec), encoding="utf-8")
    return path


def export_pylock_toml_from_directory(
    output_dir: Path,
    *,
    no_dev: bool = True,
) -> str:
    """Export a PEP 751 lockfile from an artifact ``pyproject.toml`` using ``uv export``."""
    command = [
        "uv",
        "export",
        "--format",
        "pylock.toml",
        "--directory",
        str(output_dir.resolve()),
        "--no-default-groups",
    ]
    if no_dev:
        command.append("--no-dev")

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def export_pylock_toml(
    project_root: Path | None = None,
    *,
    pyproject_toml_path: Path | None = None,
    start: Path | None = None,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    include_base: bool = True,
    no_dev: bool = True,
) -> str:
    """Export a PEP 751 lockfile from the source project using ``uv export``."""
    resolved_root, _ = resolve_dependency_project(
        project_root=project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
    )
    command = [
        "uv",
        "export",
        "--format",
        "pylock.toml",
        "--directory",
        str(resolved_root),
        "--no-default-groups",
    ]
    if no_dev:
        command.append("--no-dev")
    for extra in extras:
        command.extend(["--extra", extra])
    for group in groups:
        command.extend(["--group", group])

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def write_artifact_pylock(output_dir: Path) -> Path:
    """Write ``pylock.toml`` from the artifact ``pyproject.toml`` in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pylock.toml"
    path.write_text(export_pylock_toml_from_directory(output_dir), encoding="utf-8")
    return path


def materialize_artifact_dependencies(
    output_dir: Path,
    project_root: Path | None = None,
    *,
    pyproject_toml_path: Path | None = None,
    start: Path | None = None,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    extra_dependencies: Sequence[str] = (),
    include_base: bool = True,
    write_lock: bool = True,
) -> ArtifactDependencySpec:
    """Write ``pyproject.toml`` and optionally ``pylock.toml`` into an artifact directory."""
    spec = merge_project_dependencies(
        project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
        extras=extras,
        groups=groups,
        extra_dependencies=extra_dependencies,
        include_base=include_base,
    )
    write_artifact_pyproject(output_dir, spec)
    if write_lock:
        write_artifact_pylock(output_dir)
    return spec


def trace_imported_distributions(
    seed_modules: Sequence[str],
    *,
    exclude: Sequence[str] = ("raine",),
) -> tuple[str, ...]:
    """Return installed distribution names imported by ``seed_modules`` at runtime.

    Useful for validating that declared artifact dependencies cover actual imports.
    This intentionally excludes project-local top-level packages listed in ``exclude``.
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, packages_distributions

    module_to_distributions: dict[str, list[str]] = {}
    for distribution, modules in packages_distributions().items():
        for module in modules:
            module_to_distributions.setdefault(module, []).append(distribution)

    imported: set[str] = set()
    excluded = set(exclude)

    for module_name in seed_modules:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            imported_module = getattr(value, "__module__", None)
            if not imported_module:
                continue
            top_level = imported_module.split(".", 1)[0]
            if top_level in excluded or top_level in sys.stdlib_module_names:
                continue
            for distribution in module_to_distributions.get(top_level, []):
                if distribution not in excluded:
                    imported.add(distribution)

        for distribution in module_to_distributions.get(module.__name__.split(".", 1)[0], []):
            if distribution not in excluded:
                imported.add(distribution)

    resolved: list[str] = []
    for distribution in sorted(imported):
        try:
            from importlib.metadata import version

            resolved.append(f"{distribution}=={version(distribution)}")
        except PackageNotFoundError:
            resolved.append(distribution)
    return tuple(resolved)

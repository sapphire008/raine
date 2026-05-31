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


def read_project_metadata(project_root: Path | None = None) -> dict:
    project_root = find_project_root(project_root)
    with (project_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def merge_project_dependencies(
    project_root: Path | None = None,
    *,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    include_base: bool = True,
) -> ArtifactDependencySpec:
    """Merge base, optional, and dependency-group specs from the source project."""
    project_root = find_project_root(project_root)
    metadata = read_project_metadata(project_root)
    project = metadata.get("project", {})

    merged: list[str] = []
    seen: set[str] = set()

    def _add(requirements: Iterable[str]) -> None:
        for requirement in requirements:
            key = requirement.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(key)

    if include_base:
        _add(project.get("dependencies", []))

    optional = project.get("optional-dependencies", {})
    for extra in extras:
        _add(optional.get(extra, []))

    dependency_groups = metadata.get("dependency-groups", {})
    for group in groups:
        _add(dependency_groups.get(group, []))

    return ArtifactDependencySpec(
        name=f"{project.get('name', 'raine')}-artifact",
        version=str(project.get("version", "0.1.0")),
        requires_python=str(project.get("requires-python", ">=3.11")),
        dependencies=tuple(merged),
        description=f"Runtime dependencies for {project.get('name', 'raine')} model artifacts",
    )


def format_pyproject_toml(spec: ArtifactDependencySpec) -> str:
    """Render a minimal PEP 621 ``pyproject.toml`` for an artifact."""
    lines = [
        "[project]",
        f'name = "{spec.name}"',
        f'version = "{spec.version}"',
        f'description = "{spec.description}"',
        f'requires-python = "{spec.requires_python}"',
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


def export_pylock_toml(
    project_root: Path | None = None,
    *,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    include_base: bool = True,
    no_dev: bool = True,
) -> str:
    """Export a PEP 751 lockfile from the source project using ``uv export``."""
    project_root = find_project_root(project_root)
    command = [
        "uv",
        "export",
        "--format",
        "pylock.toml",
        "--directory",
        str(project_root),
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


def write_artifact_pylock(
    output_dir: Path,
    project_root: Path | None = None,
    *,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    include_base: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pylock.toml"
    path.write_text(
        export_pylock_toml(
            project_root,
            extras=extras,
            groups=groups,
            include_base=include_base,
        ),
        encoding="utf-8",
    )
    return path


def materialize_artifact_dependencies(
    output_dir: Path,
    project_root: Path | None = None,
    *,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
    include_base: bool = True,
    write_lock: bool = True,
) -> ArtifactDependencySpec:
    """Write ``pyproject.toml`` and optionally ``pylock.toml`` into an artifact directory."""
    project_root = find_project_root(project_root)
    spec = merge_project_dependencies(
        project_root,
        extras=extras,
        groups=groups,
        include_base=include_base,
    )
    write_artifact_pyproject(output_dir, spec)
    if write_lock:
        write_artifact_pylock(
            output_dir,
            project_root,
            extras=extras,
            groups=groups,
            include_base=include_base,
        )
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

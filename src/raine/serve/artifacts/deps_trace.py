"""Trace and materialize third-party dependencies for model artifacts.

Uses PEP 621 ``pyproject.toml`` for declarative metadata and PEP 751
``pylock.toml`` (via ``uv export``) for reproducible locked installs.

``uv`` is invoked as an external CLI (``subprocess``). When ``uv`` is not on
``PATH``, lockfile export is skipped with a warning; ``pyproject.toml`` is still
written.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_VERSION_SPECIFIER_MARKERS = (">=", "<=", "==", "!=", "~=", ">", "<")

_UV_LOCKFILE_WARNING = (
    "uv is not available on PATH; skipping pylock.toml export. "
    "The artifact pyproject.toml was written. Install uv "
    "(https://docs.astral.sh/uv/) and re-run export, or run "
    "`uv export --format pylock.toml --directory <bundle_dir>` manually."
)

WHEELS_DIR_NAME = "wheels"


@dataclass(frozen=True)
class ArtifactDependencySpec:
    """PEP 621 project metadata written into a model artifact directory."""

    name: str
    version: str = "0.1.0"
    requires_python: str = ">=3.11"
    dependencies: tuple[str, ...] = ()
    description: str = "Raine model artifact runtime environment"


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


def _package_name_from_requirement(requirement: str) -> str:
    """Return the PEP 508 distribution name from a requirement string."""
    token = requirement.strip().split("[", 1)[0]
    for separator in ("===", "==", ">=", "<=", "!=", "~=", ">", "<", " @ "):
        if separator in token:
            return token.split(separator, 1)[0].strip()
    return token.strip()


def _parse_direct_file_reference(requirement: str) -> tuple[str, str] | None:
    """Return ``(package_name, file_url)`` when ``requirement`` uses ``@ file:...``."""
    if " @ " not in requirement:
        return None
    name, url = requirement.split(" @ ", 1)
    url = url.strip()
    if not url.startswith("file:"):
        return None
    package_name = _package_name_from_requirement(name.strip())
    if not package_name:
        return None
    return package_name, url


def _file_url_to_path(file_url: str) -> str:
    """Convert a PEP 508 ``file:`` URL to a filesystem path string."""
    url = file_url.strip()
    if url.startswith("file:///"):
        return url[7:]
    if url.startswith("file://"):
        remainder = url[7:]
        if remainder.startswith("./") or remainder.startswith("../"):
            return remainder
        return remainder
    if url.startswith("file:"):
        return url[5:]
    raise ValueError(f"Not a file URL: {file_url!r}")


def _resolve_path_against_roots(
    path: Path,
    *,
    project_root: Path,
    output_dir: Path,
) -> Path | None:
    """Resolve ``path`` against ``output_dir`` then ``project_root``."""
    if path.is_absolute():
        candidate = path.resolve()
        return candidate if candidate.is_file() else None

    for root in (output_dir, project_root):
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _resolve_file_reference_path(
    file_url: str,
    *,
    project_root: Path,
    output_dir: Path,
) -> Path:
    """Resolve a ``file:`` URL from a requirement to an existing file path."""
    raw_path = Path(_file_url_to_path(file_url))
    resolved = _resolve_path_against_roots(
        raw_path,
        project_root=project_root,
        output_dir=output_dir,
    )
    if resolved is None:
        raise FileNotFoundError(
            f"Local file dependency not found for {file_url!r} "
            f"(searched under {output_dir} and {project_root})"
        )
    return resolved


def _read_uv_wheel_source_paths(metadata: dict) -> dict[str, str]:
    """Return normalized package name → path for ``[tool.uv.sources]`` wheel paths."""
    sources = metadata.get("tool", {}).get("uv", {}).get("sources", {})
    paths: dict[str, str] = {}
    for name, spec in sources.items():
        if not isinstance(spec, dict):
            continue
        path = spec.get("path")
        if isinstance(path, str) and path.endswith(".whl"):
            paths[_requirement_key(name)] = path
    return paths


def _copy_wheel_to_bundle(
    source_wheel: Path,
    wheels_dir: Path,
    *,
    copied: dict[Path, str],
) -> str:
    """Copy ``source_wheel`` into ``wheels_dir`` and return the bundle-relative filename."""
    resolved_source = source_wheel.resolve()
    if resolved_source in copied:
        return copied[resolved_source]

    wheels_dir.mkdir(parents=True, exist_ok=True)
    destination = wheels_dir / resolved_source.name
    if not destination.exists() or destination.read_bytes() != resolved_source.read_bytes():
        shutil.copy2(resolved_source, destination)
    copied[resolved_source] = destination.name
    return destination.name


def _bundle_wheel_requirement(
    package_name: str,
    source_wheel: Path,
    wheels_dir: Path,
    *,
    copied: dict[Path, str],
) -> str:
    wheel_name = _copy_wheel_to_bundle(source_wheel, wheels_dir, copied=copied)
    return f"{package_name} @ file:./{WHEELS_DIR_NAME}/{wheel_name}"


def bundle_local_wheels(
    spec: ArtifactDependencySpec,
    output_dir: Path,
    *,
    project_root: Path,
    metadata: dict,
) -> ArtifactDependencySpec:
    """Copy local ``.whl`` references into ``output_dir/wheels`` and rewrite deps."""
    wheels_dir = output_dir / WHEELS_DIR_NAME
    copied: dict[Path, str] = {}
    uv_sources = _read_uv_wheel_source_paths(metadata)
    bundled_dependencies: list[str] = []

    for requirement in spec.dependencies:
        direct = _parse_direct_file_reference(requirement)
        if direct is not None:
            package_name, file_url = direct
            source_wheel = _resolve_file_reference_path(
                file_url,
                project_root=project_root,
                output_dir=output_dir,
            )
            if source_wheel.suffix != ".whl":
                bundled_dependencies.append(requirement)
                continue
            if (
                source_wheel.parent == wheels_dir.resolve()
                and file_url.startswith(f"file:./{WHEELS_DIR_NAME}/")
            ):
                bundled_dependencies.append(requirement)
                continue
            bundled_dependencies.append(
                _bundle_wheel_requirement(
                    package_name,
                    source_wheel,
                    wheels_dir,
                    copied=copied,
                )
            )
            continue

        source_path = uv_sources.get(_requirement_key(requirement))
        if source_path is not None:
            source_wheel = _resolve_path_against_roots(
                Path(source_path),
                project_root=project_root,
                output_dir=output_dir,
            )
            if source_wheel is None:
                raise FileNotFoundError(
                    f"[tool.uv.sources] wheel not found for {_requirement_key(requirement)!r}: "
                    f"{source_path!r} (searched under {output_dir} and {project_root})"
                )
            bundled_dependencies.append(
                _bundle_wheel_requirement(
                    _package_name_from_requirement(requirement),
                    source_wheel,
                    wheels_dir,
                    copied=copied,
                )
            )
            continue

        bundled_dependencies.append(requirement)

    return ArtifactDependencySpec(
        name=spec.name,
        version=spec.version,
        requires_python=spec.requires_python,
        dependencies=tuple(bundled_dependencies),
        description=spec.description,
    )


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


def write_artifact_pylock(output_dir: Path) -> Path | None:
    """Write ``pylock.toml`` from the artifact ``pyproject.toml`` in ``output_dir``.

    Returns the lockfile path, or ``None`` when ``uv`` is not available on ``PATH``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pylock.toml"
    try:
        path.write_text(export_pylock_toml_from_directory(output_dir), encoding="utf-8")
    except FileNotFoundError:
        warnings.warn(_UV_LOCKFILE_WARNING, stacklevel=2)
        return None
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
    """Write ``pyproject.toml`` and optionally ``pylock.toml`` into an artifact directory.

    When ``write_lock`` is true and ``uv`` is not on ``PATH``, emits a warning and
    skips ``pylock.toml`` instead of failing export.

    Local wheel references (PEP 508 ``@ file:...`` or ``[tool.uv.sources]`` paths
    ending in ``.whl``) are copied into ``wheels/`` and rewritten as portable
    ``file:./wheels/<name>.whl`` requirements in the artifact ``pyproject.toml``.
    """
    resolved_root, resolved_pyproject = resolve_dependency_project(
        project_root=project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
    )
    metadata = read_project_metadata_at(resolved_pyproject)
    spec = merge_project_dependencies(
        project_root,
        pyproject_toml_path=pyproject_toml_path,
        start=start,
        extras=extras,
        groups=groups,
        extra_dependencies=extra_dependencies,
        include_base=include_base,
    )
    spec = bundle_local_wheels(
        spec,
        output_dir,
        project_root=resolved_root,
        metadata=metadata,
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

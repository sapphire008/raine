from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence


def build_search_paths(
    source_dir: Path | None,
    project_root: Path | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if source_dir is not None:
        paths.append(source_dir.resolve())
    if project_root is not None:
        src_root = project_root / "src"
        if src_root.is_dir():
            paths.append(src_root.resolve())
    cwd = Path.cwd().resolve()
    if cwd not in paths:
        paths.append(cwd)
    return tuple(paths)


def local_roots_for_model_class(model_class: str) -> tuple[str, ...]:
    model_module = model_class.rsplit(".", 1)[0]
    return tuple(dict.fromkeys(["raine", model_module.split(".", 1)[0]]))


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` when saving a model artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_extra_artifacts(
    artifacts: Sequence[str | Path],
    output_dir: Path,
) -> list[Path]:
    copied: list[Path] = []
    for artifact in artifacts:
        source = Path(artifact)
        if not source.exists():
            raise FileNotFoundError(f"Extra artifact not found: {source}")
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            _copy_file(source, destination)
        copied.append(destination)
    return copied

"""Artifact bundle index and runtime context for model packages."""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

ARTIFACTS_DIR_NAME = "artifacts"
CODE_DIR_NAME = "code"
ARTIFACTS_INDEX_NAME = "artifacts.json"
SCHEMA_VERSION = "1"
_ARTIFACT_KEY_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\-]*$")


@dataclass
class ArtifactBundle:
    """Persisted artifact index for a model bundle."""

    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> ArtifactBundle:
        if not data:
            return cls()
        valid = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in valid}
        artifacts = kwargs.pop("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("artifacts must be a mapping of logical names to relative paths")
        return cls(artifacts={str(key): str(value) for key, value in artifacts.items()}, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelContext:
    """Runtime view of a saved model bundle, similar to MLflow's PythonModelContext."""

    model_dir: Path
    code_dir: Path
    artifacts_dir: Path
    bundle: ArtifactBundle
    artifacts: dict[str, Path]

    @classmethod
    def from_uri(
        cls,
        model_uri: str | Path,
        *,
        configure_path: bool = True,
    ) -> ModelContext:
        model_dir = Path(model_uri).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model bundle directory not found: {model_dir}")

        bundle = read_artifacts_index(model_dir)
        code_dir = model_dir / CODE_DIR_NAME
        artifacts_dir = model_dir / ARTIFACTS_DIR_NAME
        resolved_artifacts = _resolve_artifact_paths(model_dir, bundle.artifacts)

        if configure_path and code_dir.is_dir():
            configure_code_path(code_dir)

        return cls(
            model_dir=model_dir,
            code_dir=code_dir,
            artifacts_dir=artifacts_dir,
            bundle=bundle,
            artifacts=resolved_artifacts,
        )

    def artifact(self, name: str) -> Path:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"Artifact {name!r} not found in bundle {self.model_dir}") from exc

    @property
    def metadata(self) -> dict[str, Any]:
        return self.bundle.metadata


def configure_code_path(code_dir: Path) -> None:
    resolved = str(code_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _validate_artifact_key(key: str) -> None:
    if not _ARTIFACT_KEY_PATTERN.match(key):
        raise ValueError(
            f"Invalid artifact key {key!r}; use letters, numbers, underscores, or hyphens"
        )


def _relative_artifact_path(key: str, source: Path) -> str:
    if source.is_dir():
        return f"{ARTIFACTS_DIR_NAME}/{key}"
    suffix = source.suffix
    return f"{ARTIFACTS_DIR_NAME}/{key}{suffix}"


def materialize_bundle_artifacts(
    bundle_root: Path,
    artifacts: Mapping[str, str | Path],
) -> dict[str, str]:
    """Copy user artifacts into ``bundle_root/artifacts`` and return the relative index."""
    if not artifacts:
        return {}

    bundle_root.mkdir(parents=True, exist_ok=True)
    artifacts_root = bundle_root / ARTIFACTS_DIR_NAME
    artifacts_root.mkdir(parents=True, exist_ok=True)

    index: dict[str, str] = {}
    destinations: dict[str, str] = {}

    for key, raw_source in artifacts.items():
        _validate_artifact_key(key)
        source = Path(raw_source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Artifact source not found for {key!r}: {source}")

        relative_path = _relative_artifact_path(key, source)
        if relative_path in destinations and destinations[relative_path] != key:
            raise ValueError(
                f"Artifacts {destinations[relative_path]!r} and {key!r} "
                f"would both copy to {relative_path}"
            )

        destination = bundle_root / relative_path
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        index[key] = relative_path
        destinations[relative_path] = key

    return index


def write_artifacts_index(bundle_root: Path, bundle: ArtifactBundle) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)
    index_path = bundle_root / ARTIFACTS_INDEX_NAME
    index_path.write_text(json.dumps(bundle.to_dict(), indent=2) + "\n", encoding="utf-8")
    return index_path


def read_artifacts_index(bundle_root: Path) -> ArtifactBundle:
    index_path = bundle_root / ARTIFACTS_INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Artifact index not found: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return ArtifactBundle.from_dict(data)


def _resolve_artifact_paths(
    bundle_root: Path,
    artifacts: Mapping[str, str],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, relative_path in artifacts.items():
        path = (bundle_root / relative_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact {key!r} listed in {ARTIFACTS_INDEX_NAME} "
                f"was not found at {path}"
            )
        resolved[key] = path
    return resolved

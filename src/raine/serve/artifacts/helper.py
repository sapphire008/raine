"""Helpers for local testing with staged model bundles."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from raine.serve.artifacts.code_trace import link_staged_code_dir
from raine.serve.artifacts.context import (
    CODE_DIR_NAME,
    ArtifactBundle,
    link_bundle_artifacts,
    write_artifacts_index,
)


def stage_model_bundle_at(
    bundle_root: str | Path,
    *,
    artifacts: Mapping[str, str | Path],
    source_dir: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    code_renames: Mapping[str, str] | None = None,
) -> Path:
    """Create a bundle layout at ``bundle_root`` using symlinks instead of copies.

    The resulting directory matches an exported artifact bundle closely enough for
    ``ModelContext.from_uri`` and handler ``setup()`` code paths, but skips
    dependency locking and file copies. Intended for local tests and dev servers.
    """
    resolved_root = Path(bundle_root).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    artifact_index = link_bundle_artifacts(resolved_root, artifacts)
    bundle = ArtifactBundle(artifacts=artifact_index, metadata=dict(metadata or {}))
    write_artifacts_index(resolved_root, bundle)

    if source_dir is not None:
        resolved_source = Path(source_dir).resolve()
        if not resolved_source.is_dir():
            raise NotADirectoryError(f"source_dir must be a directory: {resolved_source}")

        link_staged_code_dir(
            resolved_root / CODE_DIR_NAME,
            resolved_source,
            code_renames=code_renames,
        )

    return resolved_root


@contextmanager
def staged_model_bundle(
    *,
    artifacts: Mapping[str, str | Path],
    source_dir: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    code_renames: Mapping[str, str] | None = None,
    bundle_root: str | Path | None = None,
) -> Iterator[Path]:
    """Yield a staged model bundle directory for local functional tests.

    When ``bundle_root`` is omitted, the bundle is created in a temporary
    directory and removed when the context exits. When ``bundle_root`` is
    provided, the caller owns cleanup.
    """
    if bundle_root is not None:
        yield stage_model_bundle_at(
            bundle_root,
            artifacts=artifacts,
            source_dir=source_dir,
            metadata=metadata,
            code_renames=code_renames,
        )
        return

    with tempfile.TemporaryDirectory(prefix="raine-model-bundle-") as tmp:
        yield stage_model_bundle_at(
            tmp,
            artifacts=artifacts,
            source_dir=source_dir,
            metadata=metadata,
            code_renames=code_renames,
        )

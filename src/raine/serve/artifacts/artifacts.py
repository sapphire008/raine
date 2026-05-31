from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from litserve.loops.base import LitLoop
from litserve.mcp import MCP
from litserve.specs.base import LitSpec

from raine.serve.artifacts.code_trace import materialize_artifact_code
from raine.serve.artifacts.context import (
    ArtifactBundle,
    ModelContext,
    materialize_bundle_artifacts,
    write_artifacts_index,
)
from raine.serve.artifacts.deps_trace import materialize_artifact_dependencies
from raine.serve.artifacts.utils import (
    build_search_paths,
    handler_module_dir,
    local_roots_from_seeds,
)


@dataclass
class LitAPIConfig:
    """Validated passthrough config for ls.LitAPI constructor kwargs."""

    max_batch_size: int = 1
    batch_timeout: float = 0.0
    api_path: str = "/predict"
    stream: bool = True
    loop: str | LitLoop | None = "auto"
    spec: LitSpec | None = None
    mcp: MCP | None = None
    enable_async: bool = True

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be greater than 0")
        if self.batch_timeout < 0:
            raise ValueError("batch_timeout must be greater than or equal to 0")
        if not self.api_path.startswith("/"):
            raise ValueError("api_path must start with '/'")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> LitAPIConfig:
        if not data:
            return cls()
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid})

    def lit_api_kwargs(self) -> dict[str, Any]:
        return asdict(self)


class RaineModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_model_class(model_class: str):
        """Load model main class object."""
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def load_model(cls, model_uri: str | Path, *, configure_path: bool = True) -> ModelContext:
        """Load a saved model bundle and return its runtime context."""
        return ModelContext.from_uri(model_uri, configure_path=configure_path)

    def save_model(
        self,
        output_dir: str | Path,
        artifacts: Mapping[str, str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
        source_dir: str | Path | None = None,
        code_seeds: Sequence[type | str | ModuleType] | None = None,
        dependency_extras: Sequence[str] = ("serve", "torch"),
        dependency_groups: Sequence[str] = (),
        project_root: str | Path | None = None,
        pyproject_toml_path: str | Path | None = None,
        code_renames: Mapping[str, str] | None = None,
    ) -> Path:
        """Package this handler into a deployable artifact directory.

        Writes:
        - ``code/`` — project-local Python modules traced from this handler class
          and optional ``code_seeds``
        - ``artifacts/`` — copied model assets keyed by ``artifacts``
        - ``artifacts.json`` — logical artifact name to bundle-relative path index
        - ``pyproject.toml`` and ``pylock.toml`` — PEP 621 / PEP 751 runtime deps

        Args:
            output_dir: Destination directory for the artifact bundle.
            artifacts: Logical artifact names mapped to source files or directories.
                Sources may live anywhere; copies are normalized under ``artifacts/``.
            metadata: Optional user-defined metadata persisted in ``artifacts.json``.
            source_dir: Optional directory for code tracing. Defaults to the
                directory containing this handler class's module (e.g. the folder
                with ``inference.py``).
            code_seeds: Extra modules or classes to include in code tracing.
                Local package roots are inferred from this handler class and
                any ``code_seeds`` modules.
            dependency_extras: Names of ``[project.optional-dependencies]`` extras
                from the source ``pyproject.toml`` to merge into the artifact
                environment (PEP 621 extras, installed via ``uv sync --extra``).
            dependency_groups: Names of ``[dependency-groups]`` from the source
                ``pyproject.toml`` to merge into the artifact environment (uv/pip
                dependency groups, installed via ``uv sync --group``).
            project_root: Project whose ``pyproject.toml`` defines runtime deps.
                Defaults to the nearest ``pyproject.toml`` above the handler
                module directory. Ignored when ``pyproject_toml_path`` is set.
            pyproject_toml_path: Explicit path to a ``pyproject.toml`` for
                dependency export. When set, takes precedence over
                ``project_root`` search.
            code_renames: Optional mapping of bundle-relative paths within
                ``code/`` to rename after tracing, e.g.
                ``{"inference_en.py": "inference.py"}``.

        Returns:
            Resolved path to ``output_dir``.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        handler_module_path = handler_module_dir(type(self))
        resolved_project_root = (
            Path(project_root).resolve() if project_root is not None else None
        )
        resolved_pyproject_toml = (
            Path(pyproject_toml_path).resolve() if pyproject_toml_path is not None else None
        )
        resolved_source_dir = (
            Path(source_dir).resolve()
            if source_dir is not None
            else handler_module_path
        )
        search_paths = build_search_paths(resolved_source_dir)

        resolved_metadata = dict(metadata or {})
        seeds: list[type | str | ModuleType] = [type(self)]
        if code_seeds:
            seeds.extend(code_seeds)
        local_roots = local_roots_from_seeds(seeds)

        materialize_artifact_code(
            output_dir,
            seeds=seeds,
            local_roots=local_roots,
            search_paths=search_paths,
            code_renames=code_renames,
        )
        materialize_artifact_dependencies(
            output_dir,
            resolved_project_root,
            pyproject_toml_path=resolved_pyproject_toml,
            start=handler_module_path,
            extras=dependency_extras,
            groups=dependency_groups,
            include_base=True,
        )

        artifact_index = materialize_bundle_artifacts(output_dir, artifacts)
        bundle = ArtifactBundle(artifacts=artifact_index, metadata=resolved_metadata)
        write_artifacts_index(output_dir, bundle)
        return output_dir

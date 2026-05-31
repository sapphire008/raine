from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, TypeVar

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



T = TypeVar("T", bound="RaineModel")


class RaineModel:
    def __init__(self, *args, **kwargs):
        self._context: ModelContext | None = None
        super().__init__(*args, **kwargs)

    @property
    def context(self) -> ModelContext:
        """Runtime bundle context, set by :meth:`from_bundle`."""
        if self._context is None:
            raise RuntimeError(
                "Model bundle not loaded; construct the handler with "
                f"{type(self).__name__}.from_bundle(model_uri, ...)"
            )
        return self._context

    @staticmethod
    def load_model_class(model_class: str):
        """Load model main class object."""
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def from_bundle(
        cls: type[T],
        model_uri: str | Path,
        *args: Any,
        configure_path: bool = True,
        **kwargs: Any,
    ) -> T:
        """Load a saved bundle and return a handler with ``self.context`` bound."""
        instance = cls(*args, **kwargs)
        instance._context = ModelContext.from_uri(
            model_uri,
            configure_path=configure_path,
        )
        return instance

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
        extra_dependencies: Sequence[str] = (),
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
        - ``pyproject.toml`` — merged PEP 621 runtime deps (always written)
        - ``pylock.toml`` — PEP 751 lockfile when ``uv`` is on ``PATH`` (optional)

        If ``uv`` is missing, export completes without ``pylock.toml`` and emits a
        warning. ``uv`` does not need to be installed in the same environment as
        ``raine`` (see README).

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
            extra_dependencies: Additional PEP 508 requirements merged last into
                the artifact environment. When a package name matches an entry
                from base/extras/groups, the ``extra_dependencies`` value wins.
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
            extra_dependencies=extra_dependencies,
            include_base=True,
        )

        artifact_index = materialize_bundle_artifacts(output_dir, artifacts)
        bundle = ArtifactBundle(artifacts=artifact_index, metadata=resolved_metadata)
        write_artifacts_index(output_dir, bundle)
        return output_dir

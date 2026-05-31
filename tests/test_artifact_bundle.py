from __future__ import annotations

import json
from pathlib import Path

import pytest

from raine.serve.artifacts.artifacts import RaineModel
from raine.serve.artifacts.utils import handler_module_dir, local_roots_from_seeds
from raine.serve.artifacts.context import (
    ARTIFACTS_INDEX_NAME,
    ArtifactBundle,
    ModelContext,
    materialize_bundle_artifacts,
    read_artifacts_index,
    write_artifacts_index,
)


class DummyHandler(RaineModel):
    pass


def test_handler_module_dir_uses_handler_file_directory() -> None:
    assert handler_module_dir(DummyHandler) == Path(__file__).resolve().parent


def test_local_roots_from_seeds_uses_handler_module() -> None:
    assert local_roots_from_seeds([DummyHandler]) == ("tests",)
    assert local_roots_from_seeds([DummyHandler, "models.arch"]) == ("tests", "models")


def test_materialize_bundle_artifacts_copies_files_and_dirs(tmp_path: Path) -> None:
    weights = tmp_path / "sources" / "best.pt"
    config = tmp_path / "other" / "hparams.json"
    vocab_dir = tmp_path / "shared" / "vocab"
    weights.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    vocab_dir.mkdir(parents=True)
    weights.write_text("weights", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    (vocab_dir / "tokens.txt").write_text("a b c", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    index = materialize_bundle_artifacts(
        bundle_root,
        {
            "weights": weights,
            "config": config,
            "vocab": vocab_dir,
        },
    )

    assert index == {
        "weights": "artifacts/weights.pt",
        "config": "artifacts/config.json",
        "vocab": "artifacts/vocab",
    }
    assert (bundle_root / "artifacts/weights.pt").read_text(encoding="utf-8") == "weights"
    assert (bundle_root / "artifacts/config.json").is_file()
    assert (bundle_root / "artifacts/vocab/tokens.txt").is_file()


def test_write_and_read_artifacts_index(tmp_path: Path) -> None:
    bundle = ArtifactBundle(
        artifacts={"weights": "artifacts/weights.pt"},
        metadata={"loader": "pytorch"},
    )
    write_artifacts_index(tmp_path, bundle)

    loaded = read_artifacts_index(tmp_path)
    assert loaded.artifacts == {"weights": "artifacts/weights.pt"}
    assert loaded.metadata == {"loader": "pytorch"}


def test_model_context_from_uri_resolves_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "artifacts").mkdir()
    (bundle_root / "artifacts" / "weights.pt").write_text("weights", encoding="utf-8")
    (bundle_root / "code").mkdir()

    write_artifacts_index(
        bundle_root,
        ArtifactBundle(
            artifacts={"weights": "artifacts/weights.pt"},
            metadata={"model_class": "models.Example"},
        ),
    )

    inserted_paths: list[Path] = []

    def fake_configure_code_path(code_dir: Path) -> None:
        inserted_paths.append(code_dir)

    monkeypatch.setattr(
        "raine.serve.artifacts.context.configure_code_path",
        fake_configure_code_path,
    )

    ctx = ModelContext.from_uri(bundle_root)

    assert ctx.artifacts["weights"] == (bundle_root / "artifacts/weights.pt").resolve()
    assert ctx.metadata["model_class"] == "models.Example"
    assert inserted_paths == [bundle_root / "code"]


def test_save_model_writes_bundle_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights = tmp_path / "weights.pt"
    config = tmp_path / "config.json"
    weights.write_text("weights", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "model-bundle"

    captured: dict[str, object] = {}

    def fake_materialize_code(destination_root, seeds, *, local_roots, search_paths, code_renames=None):
        captured["seeds"] = list(seeds)
        captured["local_roots"] = local_roots
        code_root = destination_root / "code"
        code_root.mkdir(parents=True, exist_ok=True)
        (code_root / "marker.txt").write_text("code", encoding="utf-8")
        return type("Result", (), {"destination": code_root})()

    def fake_materialize_deps(
        destination_root,
        project_root,
        *,
        pyproject_toml_path,
        start,
        extras,
        groups,
        extra_dependencies=(),
        include_base,
        write_lock=True,
    ):
        captured["dependency_extras"] = extras
        captured["dependency_groups"] = groups
        captured["extra_dependencies"] = extra_dependencies
        captured["pyproject_toml_path"] = pyproject_toml_path
        captured["start"] = start
        return None

    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_code",
        fake_materialize_code,
    )
    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_dependencies",
        fake_materialize_deps,
    )

    handler = DummyHandler()
    handler.save_model(
        output_dir,
        artifacts={"weights": weights, "config": config},
        metadata={"loader": "pytorch"},
    )

    assert captured["local_roots"] == ("tests",)
    assert captured["seeds"] == [DummyHandler]
    assert captured["dependency_extras"] == ("serve", "torch")
    assert captured["dependency_groups"] == ()

    index = json.loads((output_dir / ARTIFACTS_INDEX_NAME).read_text(encoding="utf-8"))
    assert index["artifacts"] == {
        "weights": "artifacts/weights.pt",
        "config": "artifacts/config.json",
    }
    assert index["metadata"]["loader"] == "pytorch"
    assert (output_dir / "artifacts/weights.pt").is_file()
    assert (output_dir / "code/marker.txt").is_file()

    handler = DummyHandler.from_bundle(output_dir, configure_path=False)
    assert handler.context.artifacts["weights"].name == "weights.pt"
    assert handler.context.artifacts["config"].name == "config.json"


def test_save_model_passes_dependency_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "model-bundle"
    weights = tmp_path / "weights.pt"
    weights.write_text("weights", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_materialize_code(*args, **kwargs):
        return type("Result", (), {"destination": output_dir / "code"})()

    def fake_materialize_deps(
        destination_root,
        project_root,
        *,
        pyproject_toml_path,
        start,
        extras,
        groups,
        extra_dependencies=(),
        include_base,
        write_lock=True,
    ):
        captured["dependency_extras"] = extras
        captured["dependency_groups"] = groups
        captured["extra_dependencies"] = extra_dependencies
        captured["pyproject_toml_path"] = pyproject_toml_path
        captured["start"] = start

    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_code",
        fake_materialize_code,
    )
    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_dependencies",
        fake_materialize_deps,
    )

    DummyHandler().save_model(
        output_dir,
        artifacts={"weights": weights},
        dependency_extras=(),
        dependency_groups=("dev",),
    )

    assert captured["dependency_extras"] == ()
    assert captured["dependency_groups"] == ("dev",)


def test_merge_project_dependencies_extras_vs_groups() -> None:
    from raine.serve.artifacts.deps_trace import merge_project_dependencies

    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"

    from_extras = merge_project_dependencies(
        pyproject_toml_path=pyproject_path,
        extras=("serve", "torch"),
        include_base=False,
    )
    assert "litserve==0.2.17" in from_extras.dependencies
    assert "torch==2.8.0" in from_extras.dependencies

    from_groups = merge_project_dependencies(
        pyproject_toml_path=pyproject_path,
        groups=("dev",),
        include_base=False,
    )
    assert any("pytest" in requirement for requirement in from_groups.dependencies)


def test_merge_project_dependencies_extra_dependencies_override() -> None:
    from raine.serve.artifacts.deps_trace import merge_project_dependencies

    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"

    spec = merge_project_dependencies(
        pyproject_toml_path=pyproject_path,
        extras=("serve",),
        include_base=False,
        extra_dependencies=("litserve==0.1.0", "raine"),
    )
    assert spec.dependencies == ("litserve==0.1.0", "raine")


def test_normalize_requires_python_bare_version() -> None:
    from raine.serve.artifacts.deps_trace import (
        ArtifactDependencySpec,
        format_pyproject_toml,
        normalize_requires_python,
    )

    assert normalize_requires_python("3.13.12") == "==3.13.12"
    assert normalize_requires_python(">=3.11,<3.14") == ">=3.11,<3.14"
    assert normalize_requires_python("==3.13.12") == "==3.13.12"

    rendered = format_pyproject_toml(
        ArtifactDependencySpec(
            name="demo-artifact",
            requires_python="3.13.12",
            dependencies=("raine",),
        )
    )
    assert 'requires-python = "==3.13.12"' in rendered


def test_save_model_passes_extra_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "model-bundle"
    weights = tmp_path / "weights.pt"
    weights.write_text("weights", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_materialize_code(*args, **kwargs):
        return type("Result", (), {"destination": output_dir / "code"})()

    def fake_materialize_deps(
        destination_root,
        project_root,
        *,
        pyproject_toml_path,
        start,
        extras,
        groups,
        extra_dependencies=(),
        include_base,
        write_lock=True,
    ):
        captured["extra_dependencies"] = extra_dependencies

    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_code",
        fake_materialize_code,
    )
    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_dependencies",
        fake_materialize_deps,
    )

    DummyHandler().save_model(
        output_dir,
        artifacts={"weights": weights},
        dependency_extras=(),
        extra_dependencies=("raine", "litserve==0.2.17"),
    )

    assert captured["extra_dependencies"] == ("raine", "litserve==0.2.17")


def test_resolve_dependency_project_prefers_explicit_pyproject() -> None:
    from raine.serve.artifacts.deps_trace import resolve_dependency_project

    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"

    resolved_root, resolved_path = resolve_dependency_project(
        pyproject_toml_path=pyproject_path,
    )
    assert resolved_root == project_root
    assert resolved_path == pyproject_path


def test_save_model_passes_pyproject_toml_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    output_dir = tmp_path / "model-bundle"
    weights = tmp_path / "weights.pt"
    weights.write_text("weights", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_materialize_code(*args, **kwargs):
        return type("Result", (), {"destination": output_dir / "code"})()

    def fake_materialize_deps(
        destination_root,
        project_root,
        *,
        pyproject_toml_path,
        start,
        extras,
        groups,
        extra_dependencies=(),
        include_base,
        write_lock=True,
    ):
        captured["pyproject_toml_path"] = pyproject_toml_path

    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_code",
        fake_materialize_code,
    )
    monkeypatch.setattr(
        "raine.serve.artifacts.artifacts.materialize_artifact_dependencies",
        fake_materialize_deps,
    )

    DummyHandler().save_model(
        output_dir,
        artifacts={"weights": weights},
        pyproject_toml_path=pyproject_path,
    )

    assert captured["pyproject_toml_path"] == pyproject_path


def test_from_bundle_sets_context(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "artifacts").mkdir()
    (bundle_root / "artifacts" / "weights.pt").write_text("weights", encoding="utf-8")
    write_artifacts_index(
        bundle_root,
        ArtifactBundle(artifacts={"weights": "artifacts/weights.pt"}),
    )

    handler = DummyHandler.from_bundle(bundle_root, configure_path=False)
    assert handler.context.artifacts["weights"].read_text(encoding="utf-8") == "weights"


def test_context_raises_before_from_bundle() -> None:
    handler = DummyHandler()
    with pytest.raises(RuntimeError, match="from_bundle"):
        _ = handler.context


def test_model_context_missing_artifact_raises(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    write_artifacts_index(bundle_root, ArtifactBundle(artifacts={}))

    ctx = ModelContext.from_uri(bundle_root, configure_path=False)
    with pytest.raises(KeyError, match="weights"):
        _ = ctx.artifacts["weights"]

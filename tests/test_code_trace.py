from __future__ import annotations

from pathlib import Path

from raine.serving.artifacts.code_trace import (
    collect_local_modules,
    materialize_artifact_code,
    module_source_files,
)


def test_collect_local_modules_includes_ancestor_packages() -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    modules = collect_local_modules(
        ["raine.serving.artifacts.artifacts"],
        search_paths=search_paths,
    )

    assert "raine.serving.artifacts.artifacts" in modules
    assert "raine.serving.artifacts" in modules
    assert "raine.serving" in modules
    assert "raine" in modules


def test_module_source_files_includes_package_inits() -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    modules = collect_local_modules(
        ["raine.serving.artifacts.artifacts"],
        search_paths=search_paths,
    )
    files = module_source_files(modules)

    assert project_root / "src/raine/__init__.py" in files
    assert project_root / "src/raine/serving/__init__.py" in files
    assert project_root / "src/raine/serving/artifacts/__init__.py" in files
    assert project_root / "src/raine/serving/artifacts/artifacts.py" in files


def test_materialize_artifact_code_preserves_package_layout(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    result = materialize_artifact_code(
        tmp_path,
        ["raine.serving.artifacts.artifacts"],
        search_paths=search_paths,
    )

    code_root = tmp_path / "code"
    assert (code_root / "raine/__init__.py").is_file()
    assert (code_root / "raine/serving/__init__.py").is_file()
    assert (code_root / "raine/serving/artifacts/__init__.py").is_file()
    assert (code_root / "raine/serving/artifacts/artifacts.py").is_file()
    assert result.destination == code_root

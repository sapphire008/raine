from __future__ import annotations

from pathlib import Path

import pytest

from raine.serve.artifacts.code_trace import (
    collect_local_modules,
    copy_local_code_paths,
    materialize_artifact_code,
    module_source_files,
)


def test_collect_local_modules_includes_ancestor_packages() -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    modules = collect_local_modules(
        ["raine.serve.artifacts.artifacts"],
        search_paths=search_paths,
    )

    assert "raine.serve.artifacts.artifacts" in modules
    assert "raine.serve.artifacts" in modules
    assert "raine.serve" in modules
    assert "raine" in modules


def test_module_source_files_includes_package_inits() -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    modules = collect_local_modules(
        ["raine.serve.artifacts.artifacts"],
        search_paths=search_paths,
    )
    files = module_source_files(modules)

    assert project_root / "src/raine/__init__.py" in files
    assert project_root / "src/raine/serve/__init__.py" in files
    assert project_root / "src/raine/serve/artifacts/__init__.py" in files
    assert project_root / "src/raine/serve/artifacts/artifacts.py" in files


def test_materialize_artifact_code_preserves_package_layout(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_paths = (project_root / "src",)

    result = materialize_artifact_code(
        tmp_path,
        ["raine.serve.artifacts.artifacts"],
        search_paths=search_paths,
    )

    code_root = tmp_path / "code"
    assert (code_root / "raine/__init__.py").is_file()
    assert (code_root / "raine/serve/__init__.py").is_file()
    assert (code_root / "raine/serve/artifacts/__init__.py").is_file()
    assert (code_root / "raine/serve/artifacts/artifacts.py").is_file()
    assert result.destination == code_root


def test_copy_local_code_paths_renames_bundle_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "inference_en.py").write_text("HANDLER = 1\n", encoding="utf-8")
    (source_dir / "models.py").write_text("MODEL = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(source_dir))

    result = copy_local_code_paths(
        ["inference_en", "models"],
        tmp_path / "bundle",
        search_paths=(source_dir,),
        code_renames={"inference_en.py": "inference.py"},
    )

    code_root = tmp_path / "bundle" / "code"
    assert (code_root / "inference.py").read_text(encoding="utf-8") == "HANDLER = 1\n"
    assert not (code_root / "inference_en.py").exists()
    assert (code_root / "models.py").is_file()
    assert result.files == (code_root / "inference.py", code_root / "models.py")

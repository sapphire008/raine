from __future__ import annotations

from pathlib import Path

import pytest

from raine.serving.artifacts.context import ModelContext, read_artifacts_index
from raine.serving.artifacts.helper import stage_model_bundle_at, staged_model_bundle


def test_stage_model_bundle_at_symlinks_artifacts_and_code(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    code_root = source_root / "project"
    weights = source_root / "weights.pt"
    config = source_root / "config.json"
    code_root.mkdir(parents=True)
    weights.write_text("weights", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    (code_root / "handler.py").write_text("print('ok')\n", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    stage_model_bundle_at(
        bundle_root,
        artifacts={"weights": weights, "config": config},
        source_dir=code_root,
        metadata={"mode": "test"},
    )

    weights_link = bundle_root / "artifacts/weights.pt"
    config_link = bundle_root / "artifacts/config.json"
    code_link = bundle_root / "code"

    assert weights_link.is_symlink()
    assert config_link.is_symlink()
    assert code_link.is_symlink()
    assert weights_link.resolve() == weights.resolve()
    assert config_link.resolve() == config.resolve()
    assert code_link.resolve() == code_root.resolve()

    bundle = read_artifacts_index(bundle_root)
    assert bundle.artifacts == {
        "weights": "artifacts/weights.pt",
        "config": "artifacts/config.json",
    }
    assert bundle.metadata == {"mode": "test"}

    ctx = ModelContext.from_uri(bundle_root, configure_path=False)
    assert ctx.artifact("weights").resolve() == weights.resolve()


def test_staged_model_bundle_cleans_up_temp_dir() -> None:
    seen: Path | None = None

    with staged_model_bundle(
        artifacts={},
        source_dir=None,
    ) as bundle_dir:
        seen = bundle_dir
        assert bundle_dir.is_dir()
        assert (bundle_dir / "artifacts.json").is_file()

    assert seen is not None
    assert not seen.exists()


def test_staged_model_bundle_with_explicit_root(tmp_path: Path) -> None:
    bundle_root = tmp_path / "explicit-bundle"
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    with staged_model_bundle(
        artifacts={"config": config},
        bundle_root=bundle_root,
    ) as bundle_dir:
        assert bundle_dir == bundle_root.resolve()
        assert (bundle_root / "artifacts.json").is_file()

    assert bundle_root.exists()

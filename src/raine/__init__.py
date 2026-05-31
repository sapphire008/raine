"""Raine MLOps toolkit."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_pyproject_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        metadata = tomllib.load(handle)
    return str(metadata["project"]["version"])


def _resolve_version() -> str:
    try:
        return version("raine")
    except PackageNotFoundError:
        return _read_pyproject_version()


__version__ = _resolve_version()

__all__ = ["__version__"]

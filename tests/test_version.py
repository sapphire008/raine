import tomllib
from importlib.metadata import version
from pathlib import Path

import raine


def test_version_attribute() -> None:
    assert raine.__version__


def test_version_matches_package_metadata() -> None:
    assert raine.__version__ == version("raine")


def test_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]
    assert raine.__version__ == expected

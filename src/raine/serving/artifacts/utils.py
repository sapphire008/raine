from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Sequence


def build_search_paths(
    source_dir: Path | None,
    project_root: Path | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if source_dir is not None:
        paths.append(source_dir.resolve())
    if project_root is not None:
        src_root = project_root / "src"
        if src_root.is_dir():
            paths.append(src_root.resolve())
    cwd = Path.cwd().resolve()
    if cwd not in paths:
        paths.append(cwd)
    return tuple(paths)


def _module_name_from_seed(seed: type | str | ModuleType) -> str | None:
    if isinstance(seed, str):
        return seed
    if isinstance(seed, ModuleType):
        return seed.__name__
    module_name = getattr(seed, "__module__", None)
    if module_name and module_name != "__main__":
        return module_name
    return None


def local_roots_from_seeds(seeds: Sequence[type | str | ModuleType]) -> tuple[str, ...]:
    """Derive top-level package roots from handler classes and optional code seeds."""
    roots: list[str] = []
    for seed in seeds:
        module_name = _module_name_from_seed(seed)
        if module_name is None:
            continue
        top_level = module_name.split(".", 1)[0]
        roots.append(top_level)
    return tuple(dict.fromkeys(roots))

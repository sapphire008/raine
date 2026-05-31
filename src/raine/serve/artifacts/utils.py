from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Sequence


def build_search_paths(source_dir: Path) -> tuple[Path, ...]:
    """Return code-tracing anchor paths rooted at the handler project directory."""
    return (source_dir.resolve(),)


def handler_module_dir(handler: type) -> Path:
    """Return the directory containing the handler class's module file."""
    module_name = getattr(handler, "__module__", None)
    if module_name and module_name != "__main__":
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin not in (None, "built-in", "frozen"):
            return Path(spec.origin).resolve().parent
    return Path(inspect.getfile(handler)).resolve().parent


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

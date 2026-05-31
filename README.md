# Raine

Raine is an MLOps toolkit for the full model lifecycle: data preparation and feature engineering, training pipeline orchestration (Kubeflow), and model serving. The goal is reusable building blocks you can compose across local development and cloud deployment.

**Status:** only **`raine.serve.artifacts`** is implemented and documented today — packaging a handler, traced local code, model assets, and runtime dependencies into a self-contained bundle. Other areas of the repo are work in progress.

## Install

```bash
pip install raine
```

Model packaging does not require LitServe. To run a LitServe-based handler, install the optional `serve` extra for convenience:

```bash
pip install "raine[serve]"
```

For PyTorch handlers, also install the `torch` extra. Other extras (`gcp`, `train`, `data`) cover optional tooling elsewhere in the repo and are not required for model packaging.

### Exporting bundles (`save_model`)

`save_model` writes a PEP 751 `pylock.toml` by shelling out to the **`uv` CLI** when it is available. Put the `uv` executable on your **`PATH`** when you export (for example via [uv's standalone installer](https://docs.astral.sh/uv/getting-started/installation/)). It does **not** need to be installed in the same Python environment as `raine` — only the command must be reachable.

If `uv` is missing, export still succeeds: raine writes `pyproject.toml` and emits a **warning** that `pylock.toml` was skipped. Install [`uv`](https://docs.astral.sh/uv/) and re-run export, or run `uv export --format pylock.toml --directory <bundle_dir>` manually against the bundle directory.

Local proprietary wheels referenced via PEP 508 ``@ file:...`` (or ``[tool.uv.sources]`` paths ending in ``.whl``) are copied into ``wheels/`` and rewritten as portable ``file:./wheels/<name>.whl`` entries in the artifact manifest.

You do not need `uv` at all if you use another installer. The artifact `pyproject.toml` is standard PEP 621 — Poetry, pip, pdm, etc. can install from it. A Poetry consumer can run `poetry lock` / `poetry install` in the bundle directory and ignore `pylock.toml` entirely.

## Model bundles

`save_model` writes a directory like:

```
my_model/
├── code/              # traced local Python (handler + imports)
├── artifacts/         # weights, configs, and other assets
├── artifacts.json     # logical name → bundle path index
├── pyproject.toml     # merged runtime dependencies
├── wheels/            # local .whl deps (copied when referenced)
└── pylock.toml        # locked deps (when uv is on PATH; optional)
```

At serving time, load the handler with `from_bundle`. In `setup()`, access bundle assets via `self.context.artifacts`, similar to MLflow's `PythonModelContext`.

## Quick start

`RaineModel` is a mixin — it does not depend on any serving framework. The example below uses LitServe (`pip install "raine[serve]"`); mix it with your own base class instead if you prefer.

```python
import litserve as ls
from raine.serve.artifacts import RaineModel

class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self):
        super().__init__(max_batch_size=1)

    def setup(self, device):
        weights = self.context.artifacts["weights"]
        config = self.context.artifacts["config"]
        ...

api = MyInferenceAPI.from_bundle("/path/to/my_model", max_batch_size=1)
ls.LitServer(api).run(port=8080)
```

Export from a dedicated script:

```python
handler = MyInferenceAPI()
handler.save_model(
    output_dir="./my_model",
    artifacts={
        "config": "/path/to/config.json",
        "weights": "/path/to/weights",
    },
)
```

### Common options

| Parameter | Purpose |
|---|---|
| `source_dir` | Root for code tracing (defaults to the handler module directory) |
| `code_seeds` | Extra modules/classes to include in `code/` |
| `code_renames` | Rename files in the bundle, e.g. `{"inference_en.py": "inference.py"}` |
| `dependency_extras` | PEP 621 optional deps from your `pyproject.toml` |
| `dependency_groups` | uv/poetry dependency groups to merge |
| `extra_dependencies` | Extra PEP 508 reqs merged last; `@ file:...` wheels copied into `wheels/` |
| `include_base` | When `false`, skip source `[project].dependencies`; merge only extras/groups/overrides |
| `pyproject_toml_path` | Explicit deps manifest (overrides upward search) |

### Local testing without a full export

Use `staged_handler` to symlink a bundle layout and yield a loaded handler:

```python
from raine.serve.artifacts import staged_handler

with staged_handler(
    MyInferenceAPI,
    artifacts={"weights": weights_path, "config": config_path},
    source_dir=handler_dir,
    code_renames={"inference_en.py": "inference.py"},
    max_batch_size=1,
) as api:
    ls.LitServer(api).run(port=8080)
```

## Example

See [`examples/artifact_packaging/`](examples/artifact_packaging/) for a minimal export script, handler, and sample assets.

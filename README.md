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

## Model bundles

`save_model` writes a directory like:

```
my_model/
├── code/              # traced local Python (handler + imports)
├── artifacts/         # weights, configs, and other assets
├── artifacts.json     # logical name → bundle path index
├── pyproject.toml     # merged runtime dependencies
└── pylock.toml        # locked dependency export (uv)
```

At serving time, load the bundle in `setup()` via `ModelContext` and resolve assets by logical name (`ctx.artifact("weights")`), similar to MLflow's `PythonModelContext`.

## Quick start

`RaineModel` is a mixin — it does not depend on any serving framework. The example below uses LitServe (`pip install "raine[serve]"`); mix it with your own base class instead if you prefer.

```python
import litserve as ls
from raine.serve.artifacts import RaineModel

class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self, model_dir: str | None = None):
        super().__init__(max_batch_size=1)
        self.model_dir = model_dir

    def setup(self, device):
        ctx = self.load_model(self.model_dir)
        weights = ctx.artifact("weights")
        config = ctx.artifact("config")
        ...
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
| `pyproject_toml_path` | Explicit deps manifest (overrides upward search) |

### Local testing without a full export

Use `staged_model_bundle` to symlink artifacts and code into a bundle layout for dev servers and tests:

```python
from raine.serve.artifacts import staged_model_bundle

with staged_model_bundle(
    artifacts={"weights": weights_path, "config": config_path},
    source_dir=handler_dir,
    code_renames={"inference_en.py": "inference.py"},
) as bundle_dir:
    api = MyInferenceAPI(model_dir=str(bundle_dir))
    ls.LitServer(api).run(port=8080)
```

## Example

See [`examples/artifact_packaging/`](examples/artifact_packaging/) for a minimal export script, handler, and sample assets.

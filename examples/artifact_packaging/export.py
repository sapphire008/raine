"""Export a deployable model bundle for the artifact_packaging example."""
from pathlib import Path

from inference import MyInferenceAPI

EXAMPLE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_ROOT / "exported" / "my_model"

from pdb import set_trace

def main() -> Path:
    handler = MyInferenceAPI()
    return handler.save_model(
        output_dir=OUTPUT_DIR,
        artifacts={
            "config": EXAMPLE_ROOT / "trained_model/configs.json",
            "weights": EXAMPLE_ROOT / "trained_model/weights",
        },
        code_renames={
            "inference.py": "api.py",
        },
    )


def load(bundle_dir: str | Path) -> MyInferenceAPI:
    """Load the exported bundle as a LitServe-ready handler (MLflow-style)."""
    handler = MyInferenceAPI.from_bundle(bundle_dir, max_batch_size=1)
    set_trace()
    return handler


if __name__ == "__main__":
    bundle_dir = main()
    print(f"Wrote model bundle to {bundle_dir}")
    api = load(bundle_dir)
    print(f"Loaded handler: {type(api).__name__}, context={api.context.model_dir}")

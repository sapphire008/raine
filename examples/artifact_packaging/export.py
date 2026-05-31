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
            "inference.py": "api.py",  # rename code
        }
    )
    
def load(bundle_dir: str | Path):
    handler = MyInferenceAPI.load_model(bundle_dir)


if __name__ == "__main__":
    bundle_dir = main()
    print(f"Wrote model bundle to {bundle_dir}")
    load(bundle_dir)

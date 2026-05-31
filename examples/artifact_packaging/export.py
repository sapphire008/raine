"""Script to export a package of inference"""
from pathlib import Path
from inference import MyInferenceAPI

EXAMPLE_ROOT = Path(__file__).resolve().parent


handler = MyInferenceAPI()
handler.save_model(
    output_dir="./my_model",
    artifacts={
        "configs": "./trained_model/configs.json",
        "weights": "./trained_model/weights",
    },
    source_dir=EXAMPLE_ROOT / "my_model",
    project_root=EXAMPLE_ROOT, # example pyproject, not raine repo
)

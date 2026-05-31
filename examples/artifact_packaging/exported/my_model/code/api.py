import litserve as ls
from pathlib import Path

from raine.serve.artifacts import RaineModel
from modules.module import ModelModule


class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self, model_dir: str | Path | None = None):
        super().__init__(max_batch_size=1)
        self.model_dir = model_dir

    def setup(self, device):
        _ = ModelModule()
        ctx = self.load_model_context(self.model_dir)
        weights_path = ctx.artifact("weights")
        config_path = ctx.artifact("config")
        print(f"device:{device}, weights:{weights_path}, configs:{config_path}")
        



#%%
# from pathlib import Path

# SRC = Path(__file__).resolve().parent
# RINOA_ROOT = SRC.parents[1]  # rinoa-ai-model

# handler.save_model(
#     output_dir="./model-bundle",
#     artifacts={
#         "config": SRC / "egs/visinger2_en/checkpoints/config.json",
#         "checkpoints": SRC / "egs/visinger2_en/checkpoints",
#         # optional: "example_inputs": ...,
#     },
#     metadata={
#         "artifact_type": "eager",
#         "language": "en",
#     },
#     source_dir=SRC,
#     project_root=RINOA_ROOT,
#     code_seeds=["models", "utils.inference_base"],  # optional extra safety
#     dependency_extras=("gcp",),  # [project.optional-dependencies]
#     dependency_groups=("torch", "inspect", "visinger2"),  # [dependency-groups]
#     code_renames={"inference_en.py": "inference.py"},
# )

# Local functional tests without a full export:
# from raine.serve.artifacts import staged_model_bundle
#
# EXAMPLE_ROOT = Path(__file__).resolve().parent
# with staged_model_bundle(
#     artifacts={...},
#     source_dir=SRC,
#     code_renames={"inference_en.py": "inference.py"},
# ) as bundle_dir:
#     api = MyInferenceAPI(model_dir=str(bundle_dir))
#     ls.LitServer(api).run(port=8080)

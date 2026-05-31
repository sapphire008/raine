import litserve as ls

from raine.serving.artifacts import RaineModel



class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self):
        super().__init__(max_batch_size=1)

    def setup(self, device):
        ctx = self.load_model(self.model_dir)
        weights_path = ctx.artifact("weights")
        config_path = ctx.artifact("config")
        _ = device, weights_path, config_path
        



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
# )

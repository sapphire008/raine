import litserve as ls

from raine.serve.artifacts import RaineModel
from modules.module import ModelModule


class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self, **lit_api_kwargs):
        lit_api_kwargs.setdefault("max_batch_size", 1)
        super().__init__(**lit_api_kwargs)

    def setup(self, device):
        self.model = ModelModule()
        weights_path = self.context.artifacts["weights"]
        config_path = self.context.artifacts["config"]
        print(f"device:{device}, weights:{weights_path}, configs:{config_path}")

if __name__ == '__main__':
    # Deploy:
    api = MyInferenceAPI.from_bundle("./model_artifact/my_model", max_batch_size=1)
    ls.LitServer(api).run(port=8080, generate_client_file=False)

# Local functional tests without a full export:
# from pathlib import Path
# from raine.serve.artifacts import staged_handler
#
# EXAMPLE_ROOT = Path(__file__).resolve().parent
# with staged_handler(
#     MyInferenceAPI,
#     artifacts={
#         "config": EXAMPLE_ROOT / "trained_model/configs.json",
#         "weights": EXAMPLE_ROOT / "trained_model/weights",
#     },
#     source_dir=EXAMPLE_ROOT,
#     code_renames={"inference.py": "api.py"},
#     max_batch_size=1,
# ) as api:
#     ls.LitServer(api).run(port=8080)

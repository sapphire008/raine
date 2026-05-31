import litserve as ls
from src.raine.serving.artifacts.inference import LitAPIConfig, RaineModel, ArtifactManifest


class MyInferenceAPI(RaineModel, ls.LitAPI):
    def __init__(self, lit_api: LitAPIConfig |None= None):
        cfg = lit_api or LitAPIConfig()
        super().__init__(self, **cfg.lit_api_kwargs())
        
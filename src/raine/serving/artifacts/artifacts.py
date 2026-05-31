from __future__ import annotations

import sys
import os
import importlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import asyncio
from typing import Dict, List, Literal, Sequence, Tuple, Any, Iterable
from itertools import zip_longest
import json
import unicodedata
import re
import numpy as np
import pandas as pd
import litserve as ls
from litserve.loops.base import LitLoop
from litserve.mcp import MCP
from litserve.specs.base import LitSpec

from raine.serving.artifacts.code_trace import materialize_artifact_code
from raine.serving.artifacts.deps_trace import find_project_root, materialize_artifact_dependencies
from raine.serving.artifacts.utils import (
    build_search_paths,
    copy_extra_artifacts,
    local_roots_for_model_class,
    write_manifest,
)


@dataclass
class LitAPIConfig:
    """Validated passthrough config for ls.LitAPI constructor kwargs."""

    max_batch_size: int = 1
    batch_timeout: float = 0.0
    api_path: str = "/predict"
    stream: bool = True
    loop: str | LitLoop | None = "auto"
    spec: LitSpec | None = None
    mcp: MCP | None = None
    enable_async: bool = True

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be greater than 0")
        if self.batch_timeout < 0:
            raise ValueError("batch_timeout must be greater than or equal to 0")
        if not self.api_path.startswith("/"):
            raise ValueError("api_path must start with '/'")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> LitAPIConfig:
        if not data:
            return cls()
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid})

    def lit_api_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactManifest:
    model_name: str | None = None
    model_version: str | None = None
    model_class: str = "models.SynthesizerTrn"
    artifact_type: Literal["script", "eager"] = "script"
    weights: str = "scripted_model.pt"
    config: str = "config.json"
    example_inputs: str = "example_inputs.pt"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> ArtifactManifest:
        if not data:
            return cls()
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RaineModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def load_model_class(model_class: str):
        """Load model main class object"""
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    @classmethod
    def load_model(cls, model_uri: str | Path):
        """Loading model artifact back"""
        pass
    
    def save_model(
        self,
        output_dir: str | Path,
        manifest: ArtifactManifest,
        *,
        source_dir: str | Path | None = None,
        extra_artifacts: Sequence[str | Path] | None = None,
        dependency_extras: Sequence[str] = ("serve", "torch"),
        project_root: str | Path | None = None,
    ) -> Path:
        """Package this handler into a deployable artifact directory.

        Writes:
        - ``code/`` — project-local Python modules traced from this handler class
          and ``manifest.model_class``
        - ``pyproject.toml`` and ``pylock.toml`` — PEP 621 / PEP 751 runtime deps
        - ``manifest.json`` — serialized from ``manifest``
        - any paths listed in ``extra_artifacts`` (e.g. weights, configs)

        Args:
            output_dir: Destination directory for the artifact bundle.
            manifest: Metadata describing the saved model. Written to
                ``manifest.json``; not copied from an existing file.
            source_dir: Optional directory added to code-tracing search paths,
                typically where custom model code lives.
            extra_artifacts: Files or directories to copy into ``output_dir``.
                Use this for model weights, configs, and other non-code assets
                referenced by ``manifest``.
            dependency_extras: Optional dependency groups from the source
                ``pyproject.toml`` to include in the artifact environment.
            project_root: Root of the raine project. Defaults to the nearest
                directory containing ``pyproject.toml``.

        Returns:
            Resolved path to ``output_dir``.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else find_project_root()
        )
        resolved_source_dir = Path(source_dir).resolve() if source_dir is not None else None
        search_paths = build_search_paths(resolved_source_dir, resolved_project_root)

        model_module = manifest.model_class.rsplit(".", 1)[0]
        materialize_artifact_code(
            output_dir,
            seeds=[type(self), model_module],
            local_roots=local_roots_for_model_class(manifest.model_class),
            search_paths=search_paths,
        )
        materialize_artifact_dependencies(
            output_dir,
            resolved_project_root,
            extras=dependency_extras,
            include_base=True,
        )

        if extra_artifacts:
            copy_extra_artifacts(extra_artifacts, output_dir)

        write_manifest(output_dir, manifest.to_dict())
        return output_dir



# %% TorchServe Model Handler
# class InferenceHandlerBase(ls.LitAPI):
#     def __init__(
#         self,
#         model_dir: str,
#         output_dir: str = None,
#         write_single_audio: bool = False,
#         audio_format: Literal[".wav", ".mp3", ".wav+mp3", ".wav+wav"] = ".mp3",
#         return_durations: bool = False,
#         gt_f0_weight: float = 0.85,
#         stream_batch_size: int = 3,
#         max_concurrent_predict: int = 1,
#         lit_api: LitAPIConfig | None = None,
#     ):
#         assert os.path.exists(model_dir), "Model directory is required"

#         cfg = lit_api or LitAPIConfig()
#         super().__init__(**cfg.lit_api_kwargs())
#         self.model_dir = model_dir
#         self.output_dir = output_dir
#         self.write_single_audio = write_single_audio
#         self.audio_format = audio_format
#         self.return_durations = return_durations
#         self.gt_f0_weight = gt_f0_weight
#         self.stream_batch_size = stream_batch_size

#         self.audio_format = ".wav+mp3"
#         self._context = None
#         self.initialized = False
#         self.model = None
#         self.hps = HParams()
#         self.file_handler = None
#         # Limit how many predict can run simultaneously in the same app
#         self.predict_semaphore = asyncio.Semaphore(max_concurrent_predict)

#     @staticmethod
#     def _load_model_class(model_class: str):
#         """'models.SynthesizerTrn' -> class object"""
#         module_name, class_name = model_class.rsplit(".", 1)
#         module = importlib.import_module(module_name)
#         return getattr(module, class_name)
    
#     def setup(self, device):
#         """
#         Initialize model. setup is called once at startup.
#         """
#         with open(os.path.join(self.model_dir, "manifest.json"), "r") as fid:
#             self.manifest = ArtifactManifest.from_dict(json.load(fid))
#         # Set attributes based on env variables
#         self.output_dir = os.environ.get("AUDIO_OUTPUT_DIR") or self.output_dir
#         if self.output_dir is not None and self.output_dir.startswith("gs://"):  # google storage
#             import gcsfs

#             self.file_handler = gcsfs.GCSFileSystem()

#         # Set inference devices
#         self.device = device or "cpu"

#         # Read model config file
#         hparams_path = os.path.join(self.model_dir, self.manifest.config)
#         self.hps = hps = get_hparams_from_file(hparams_path)

#         # Load either eager model or scripted model
#         model_weight_path = os.path.join(self.model_dir, self.manifest.weights)
#         if self.manifest.artifact_type == "eager":
#             # defining and loading the custom model
#             if self.model_dir not in sys.path:
#                 sys.path.insert(0, os.path.abspath(self.model_dir))
#             self.model = self._load_model_class(self.manifest.model_class)(hps)
#             # Load the weights
#             _ = load_checkpoint(model_weight_path, self.model, optimizer=None)
#             self.model.to(self.device)
#             self.model.eval()  # set to eval model
#             self.model = torch.compile(self.model) # compile for faster inference
#         elif self.manifest.artifact_type == "script":  # torchscript
#             # Load the compiled script model (has weights already)
#             self.model = torch.jit.load(model_weight_path, map_location=self.device)
#             self.model.eval()  # set to eval model
#         else:
#             raise ValueError(f"Unrecognized artifact_type: {self.manifest.artifact_type}")

        
#         # Warm up the model
#         # example_input_file = os.path.join(self.model_dir, self.manifest.example_inputs)
#         # if os.path.isfile(example_input_file):
#         #     example_inputs = torch.load(example_input_file, weights_only=True)
#         #     # Move tensors to correct device
#         #     example_inputs = {
#         #         k: v.to(self.device) if isinstance(v, torch.Tensor) else v
#         #         for k, v in example_inputs.items()
#         #     }
#         #     with torch.no_grad():
#         #         self.model.infer(**example_inputs)
#         #     print("Done warming up model")

#         self.initialized = True


#     @staticmethod
#     def pad_sequence(x: List[np.ndarray], pad_value: str | int | float = ""):
#         return np.array(list(zip_longest(*x, fillvalue=pad_value))).T

#     @staticmethod
#     def slugify(value):
#         """
#         Use unidecode to conver to ASCII. Convert spaces or repeated
#         dashes to single dashes. Remove characters that aren't alphanumerics,
#         underscores, or hyphens. Convert to lowercase. Also strip leading and
#         trailing whitespace, dashes, and underscores.
#         """
#         value = str(value)
#         value = unidecode(value)
#         value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
#         value = re.sub(r"[^\w\s-]", "", value.lower())
#         return re.sub(r"[-\s]+", "-", value).strip("-_")

#     def create_features(
#         self, data: List[Dict[str, Any]], params: Dict[str, Any] = {}
#     ) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
#         raise NotImplementedError("Method to be implemented")

#     async def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
#         """Music score to ready-to-use features dictionary."""
#         features, metadata = self.create_features(request["instances"], request["parameters"])
#         parameters = request["parameters"] | {"nframes": int(metadata["NFrames"].iloc[0])}
#         return {
#             "features": features,
#             "metadata": metadata,
#             "parameters": parameters,
#         }

#     def encode_response_data(self, output: Dict[str, Any]) -> Dict[str, Any]:
#         # Append base_dir
#         if output.get("parameters", {}).get("session_id"):
#             base_dir = os.path.join(self.output_dir, output["parameters"]["session_id"])
#         else:
#             base_dir = self.output_dir

#         # File list to write
#         audio_format = output.get("parameters", {}).get("audio_format", self.audio_format)
#         wav_file_name = None
#         if audio_format in (".wav+mp3", ".wav+wav"):  # adding single wav file name
#             wav_file_name = output["outputs"]["FileName"][0]
#             # stripping the 0000- part from file name
#             wav_file_name = wav_file_name.split("-", 1)[-1]
#             if audio_format == ".wav+mp3":
#                 wav_file_name = wav_file_name.replace(".mp3", ".wav")
#             wav_file_name = os.path.join(base_dir, wav_file_name)
#         file_names = [os.path.join(base_dir, fn) for fn in output["outputs"]["FileName"]]
#         # write individual files
#         return {
#             "single_wav_file": wav_file_name,
#             "file_names": file_names,
#             "base_dir": base_dir,
#             "audio_format": audio_format,
#             "nframes": output.get("parameters", {}).get("nframes"),
#             "return_durations": output.get("parameters", {}).get("return_durations", self.return_durations),
#         }

#     async def _chain_async(self, first_items, async_iter):
#         """Helper to chain first items with async iterator"""
#         for item in first_items:
#             yield item
#         async for item in async_iter:
#             yield item

#     async def encode_response(self, output_stream: Iterable[Dict[str, Any]]):
#         """Takes the model output and create audio files.
#         Dict contains
#             "outputs": A dictionary of mini-batched arrays of output series
#             "parameters": dict of params passed from the request
#         Returns a Dictionary of
#             "audio_files": List of audio files
#             "phoneme_duration_pred": Predicted phoneme durations
#             "num_samples": Num of samples in the audio
#             "base_dir": Directory that contains the audio files
#             "sample_rate": Audio sample rate
#         """
#         # this is not an async iterable by litserve, but just an iterable
#         first_item = await output_stream.__anext__()
#         # output_stream = iter(output_stream)
#         # first_item = next(output_stream)
#         # if first_item is None:
#         #     return
#         # Obtain request-time configs from inputs
#         config = self.encode_response_data(first_item)
#         if config["audio_format"] in (".wav+mp3", ".wav+wav"):
#             async for output in write_combined_wav_separate_chunks(
#                 self._chain_async([first_item], output_stream),
#                 wav_output_file=config["single_wav_file"],
#                 sample_rate=self.hps.data.sample_rate,
#                 # nframes=config["nframes"],
#                 file_handler=self.file_handler,
#                 chunk_ext=".mp3" if config["audio_format"] == ".wav+mp3" else ".wav"
#             ):
#                 output_payload = {
#                     "audio_files": output["outputs"]["FileName"],
#                     "audio_format": config["audio_format"],
#                     "num_samples": output["outputs"]["NumSamples"],
#                     "base_dir": config["base_dir"],
#                     "sample_rate": self.hps.data.sample_rate,
#                 }
#                 if config["return_durations"]:  # flatten
#                     output_payload["phoneme_duration_pred"] = output["outputs"]["PredPhoneDuration"]
#                     output_payload["f0_pred"] = output["outputs"]["PredF0"]
#                 yield output_payload
#         else: # only individual files
#             async for output in self._chain_async([first_item], output_stream):
#                 data = self.encode_response_data(output)
#                 if config["audio_format"] == ".wav":
#                     num_samples = await write_wav_async(
#                         output["outputs"]["Audio"],
#                         data["file_names"],
#                         self.hps.data.sample_rate,
#                         file_handler=self.file_handler,
#                     )
#                 elif config["audio_format"] == ".mp3":
#                     num_samples = await write_mp3_async(
#                         output["outputs"]["Audio"],
#                         data["file_names"],
#                         self.hps.data.sample_rate,
#                         file_handler=self.file_handler,
#                     )
#                 output_payload = {
#                     "audio_files": output["outputs"]["FileName"],
#                     "audio_format": config["audio_format"],
#                     "num_samples": num_samples,
#                     "base_dir": config["base_dir"],
#                     "sample_rate": self.hps.data.sample_rate,
#                 }
#                 if config["return_durations"]:
#                     output_payload["phoneme_duration_pred"] = output["outputs"]["PredPhoneDuration"]
#                 yield output_payload

# # %%

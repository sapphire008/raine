"""Package a trained model into a model archive .mar file"""
import os
from typing import List, Optional
import subprocess
import re


def is_valid_model_name(model_name):
    """
    Check if a model name is valid.

    A valid model name must begin with a letter of the alphabet
    and can only contains letters, digits, underscores _, dashes -
    and periods ..
    """
    # Define the regex pattern
    pattern = r"^[A-Za-z0-9][A-Za-z0-9_\-.]*$"

    # Use re.match to check if the entire string matches the pattern
    if re.match(pattern, model_name):
        return True
    else:
        return False


def export_to_model_archive(
    model_name: str,
    model_version: str,
    model_file: str = "model.py",
    serialized_file: str = "pytorch_model.pth",
    handler_file: str = None,
    extra_files: List[str] = [],
    config_file: Optional[str] = "torchserve_config.yaml",
    export_path: Optional[str] = None,
    requirement_txt_path: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Save a PyTorch model into .mar model archive file for serving.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_version : str
        Version of the model
    model_file : str, optional
        Path to python file containing model architecture.
        This parameter is mandatory for eager mode models.
        The model architecture file must contain only one
        class definition extended from torch.nn.Module
    serialized_file : str, optional
        Path to .pt or .pth file containing state_dict in
        case of eager mode or an executable ScriptModule
        in case of TorchScript.
        By default, pytorch_model.pth
    handler_file : str, optional
        TorchServe's default handler name  or handler python
        file path to handle custom TorchServe inference logic.
        - Default handler names include:
            - image_classifier
            - object_detector
            - text_classifier
            - image_segmenter
        - To specify custom handler file, use
            "./handler.py"
        - To specify a specific handler function, use
            "./handler:my_inference_func" (no .py in the file name)
    extra_files : List[Text], optional
        Comma separated path to extra dependency files.
        For example [ "config.json", "spiece.model",
        "tokenizer.json", "setup_config.json"]. This
        could also be a good place to include other modules
        imported by model_file.
    config_file: str, optional
        Path to a model config yaml file.
        Default to None.
    export_path : str, optional
        Folder path to export the model to,
        by default None, which saves a .mar file
        to the current directory
    requirement_txt_path : str, optional
        requirement.txt file path, by default None
    overwrite: bool, optional
        Whether or not to overwrite existing .mar file.
        Default to False
    """
    # Check if model name is valid
    assert is_valid_model_name(model_name), (
        "Model name is not valid. "
        "A valid model name must begin with a letter of the alphabet"
        " and can only contains letters, digits, underscores _, dashes - "
        "and periods ."
    )
    # from pdb import set_trace; set_trace()

    # argument is required to be a comma separated string
    extra_files = ",".join(extra_files)
    # Build command
    cmd = [
        "torch-model-archiver",
        f'--model-name={model_name}',
        f'--version={model_version}',
        f'--model-file={model_file}',
        f'--serialized-file={serialized_file}',
        f'--handler={handler_file}',
        f'--extra-files={extra_files}',
        f'--runtime=python',
    ]
    if config_file:
        cmd.append(f'--config-file={config_file}')
    if export_path:
        cmd.append(
            f'--export-path={export_path}',
        )
    # If requirement.txt is added
    if requirement_txt_path:
        cmd.append(
            f'requirements-file={requirement_txt_path}',
        )
    # Overwrite
    if overwrite:
        cmd.append("--force")
    # Run, create the archive file
    subprocess.call(cmd)

    # Return the path of the .mar file
    if export_path:
        return os.path.join(export_path, f"{model_name}.mar")
    else:
        return f"./{model_name}.mar"

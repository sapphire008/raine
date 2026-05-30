import os
from typing import List, Any, Dict, Set
import yaml
import tomllib
from dynaconf import Dynaconf

project_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def load_settings(
    settings_files: str | List[str] = f"{project_dir}/settings.yaml",
    # secret_file: str = f"{project_dir}/.secrets.toml",
    pipeline_env: str = "default",
) -> Dynaconf:
    """Loading Dynaconf settings file"""
    pipeline_env = os.environ.get("ENV_FOR_DYNACONF", pipeline_env)
    # Allow single setting file override
    settings_files = os.environ.get("DYNACONF_SETTINGS_FILE", settings_files)
    if isinstance(settings_files, str):
        settings_files = [settings_files]  # use

    env_set: Set[str] = set()
    for file in settings_files:
        # Check if settings file exists
        if not os.path.isfile(file):
            raise (FileNotFoundError(f"{file} is not found."))
        # Check if environment exists
        # environment variable can overwrite this default
        if file.endswith((".yaml", ".yml")):
            with open(file, "r") as fid:
                tmp_set = yaml.safe_load(fid) or {}
        elif file.endswith(".toml"):
            with open(file, "rb") as fid:
                tmp_set = tomllib.load(fid)
        else:
            tmp_set: Dict[str, Any] = {}

        # Aggregate the env
        env_set = env_set.union(tmp_set.keys())

    # Forcing the user to select a correct environment
    if pipeline_env not in env_set:
        raise (Exception(f"'{pipeline_env}' is not a valid environment. Use one of {list(env_set)}"))

    # Load the settings with Dynaconf
    settings = Dynaconf(
        # Not mounting secret here because it will be printed Kubeflow/Vertex
        # Pipeline interface
        settings_files=settings_files,  # secret_file
        environments=True,
        # Alternatively, skip this variable, and assign
        # ENV_FOR_DYNACONF variable as a environment variable, and dynaconf will
        # automatically load the correct environment
        # https://dynaconf.readthedocs.io/en/docs_223/guides/configuration.html
        env=pipeline_env,
    )
    settings.set("LOADED_DYNACONF_ENVIRONMENT", pipeline_env)
    return settings


# settings = load_settings()
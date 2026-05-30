import os
import re
import yaml
import tempfile
import shutil
import importlib
from typing import Union, Dict, Any, Optional, Literal, List
from kfp import dsl
from kfp.dsl.base_component import BaseComponent
from kfp import compiler
from google.cloud import aiplatform
from pdb import set_trace

class VertexPipelineRunner:
    def __init__(self, gcp_project_id: str, run_region: str):
        """
        GCP Vertex Pipeline Runner for Kubeflow Pipelines

        Parameters
        ----------
        gcp_project_id : str
            GCP Project ID
        run_region : str
            Run region of the pipeline, e.g. us-east1
        """
        aiplatform.init(project=gcp_project_id, location=run_region)

    def sanitize_pipeline_name(self, pipeline_name):
        # Convert to lowercase
        sanitized = pipeline_name.lower()

        # Replace any character that's not a-z, 0-9, or - with a hyphen
        sanitized = re.sub(r"[^a-z0-9-]", "-", sanitized)

        # Remove any leading hyphens
        sanitized = sanitized.lstrip("-")

        # Ensure it starts with a letter or number if it's empty after previous operations
        if not sanitized:
            sanitized = "pipeline"
        elif not sanitized[0].isalnum():
            sanitized = "p" + sanitized

        # Truncate to 128 characters
        sanitized = sanitized[:128]

        return sanitized
    
    def apply_image_uri_to_pipeline(self, pipeline_package_path: str, image_uri: str):
        """Replace all the component images with custom docker image"""
        with open(pipeline_package_path, "r") as fid:
            pipeline_yaml = yaml.safe_load(fid)
            for component in pipeline_yaml["deploymentSpec"]["executors"].values():
                component["container"]["image"] = image_uri
                
        with open(pipeline_package_path, "w") as fid:
            yaml.dump(pipeline_yaml, fid)

    def compile_pipeline_package(
        self,
        pipeline: Union[str, BaseComponent],
        pipeline_name: str,
        pipeline_path: Optional[str] = None,
        image_uri: Optional[str] = None,
    ):
        """Compile pipeline_func into a yaml file"""
        # Compile the pipeline_func
        if isinstance(pipeline, str):  # import the pipeline from module
            if (
                pipeline.endswith(".yaml") or pipeline.endswith(".yml")
            ) and os.path.isfile(pipeline):
                self.pipeline = pipeline  # existing file
                return
            else:  # assuming path to the pipeline function
                module_path, function_name = pipeline.rsplit(".", 1)
                # Import the module
                module = importlib.import_module(module_path)
                # Get the function from the module
                pipeline: BaseComponent = getattr(module, function_name)

        # Compile BaseComponent
        temp_dir = None
        if pipeline_path is None:
            temp_dir = tempfile.mkdtemp()
            pipeline_path = os.path.join(
                temp_dir, f"{pipeline_name}.yaml"
            )
        else:
            pipeline_path = os.path.join(
                pipeline_path, f"{pipeline_name}.yaml"
            )
        self.pipeline = pipeline_path
                    
        # Compile the pipeline func into a .yaml file
        compiler.Compiler().compile(
            pipeline_func=pipeline,
            package_path=pipeline_path,
            pipeline_name=pipeline_name,
        )
        
        if image_uri is not None:
            self.apply_image_uri_to_pipeline(self.pipeline, image_uri)

        return temp_dir

    def create_run(
        self,
        pipeline: Union[str, BaseComponent],
        pipeline_name: str,
        pipeline_root: str,
        pipeline_parameters: Dict[str, Any] = {},
        pipeline_path: Optional[str] = None,
        enable_caching: Optional[bool] = False,
        image_uri: Optional[str] = None,
        labels: Optional[List[str]] = None,
        failure_policy: Optional[Literal["fast", "slow"]] = None,
    ):
        """
        Create a run and submit to GCP Vertex AI

        Parameters
        ----------
        pipeline: one of the following 3:
            - @dsl.pipeline decorated pipeline function
            - Import path to the @dsl.pipeline decoration pipeline function, separated by "."
            - .yaml file compiled from the pipeline function
        pipeline_name : str
            Displayed name of the pipeline.
        pipeline_root : str
            Remote storage path to store pipeline output artifacts
        pipeline_parameters : Dict[str, Any], optional
            Input parameters of the pipeline, by default {}
        pipeline_path : Optional[str], optional
            A directory to store the compiled pipeline.yaml file.
            Only used when `pipeline` argument is a BaseComponent
            @dsl.pipeline decorated function, or the import path of
            the fnuction. By default, None, which will save the
            compiled .yaml file to a temporary directory.
        enable_caching : bool, optional
            Whether or not to enable run caching, by default False
        image_uri : str, optional
            Docker image path to use to run the pipeline, by default None
        labels: List[str], optional
            List of labels for the Vertex Pipeline run
        failure_policy: str, optional
            - To configure the pipeline to fail after one task fails, use 'fast'.
            - To configure the pipeline to continue scheduling tasks after one task 
                fails, use 'slow' (default if not set).
        """
        # Clean the display name
        pipeline_name = self.sanitize_pipeline_name(pipeline_name)
        temp_dir = self.compile_pipeline_package(
            pipeline, pipeline_name, pipeline_path, image_uri
        )

        # Create the job
        job = aiplatform.PipelineJob(
            display_name=pipeline_name,
            template_path=self.pipeline,
            # job_id="kfp2-vertex-run",
            pipeline_root=pipeline_root,
            parameter_values=pipeline_parameters,
            enable_caching=enable_caching,
            # encryption_spec_key_name = CMEK,
            labels = labels,
            # credentials = CREDENTIALS,
            failure_policy = failure_policy,
        )

        # Submit the job
        job.submit()
        
        # Clean up
        if temp_dir:
            shutil.rmtree(temp_dir)

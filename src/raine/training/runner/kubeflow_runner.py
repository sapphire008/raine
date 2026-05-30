import os
from typing import Union, Callable, List, Dict, Any
import importlib
from kfp import compiler, Client
from kfp.dsl.base_component import BaseComponent
from kfp.client.client import RunPipelineResult


class KubeflowPipelineRunner:
    """
    Create a Kubeflow pipeline and run it
    * kubeflow_host: remote host of the Kubeflow pipeline
    """

    def __init__(
        self,
        kubeflow_host: str,
    ):
        # Yaml file path
        self.client = Client(host=kubeflow_host)

    def create_run(
        self,
        pipeline: Union[str, BaseComponent],
        pipeline_root: str,
        pipeline_parameters: Dict[str, Any] = {},
        experiment_name: str = None,
        run_name: str = None,
        namespace: str = None,
        service_account: str = None,
        enable_caching: bool = False,
    ) -> RunPipelineResult:
        """Create a Kubeflow run
        * pipeline: one of the following 3:
            * @dsl.pipeline decorated pipeline function
            * Import path to the @dsl.pipeline decoration pipeline function, separated by "."
            * .yaml file compiled from the pipeline function
        * pipeline_root: Root path of the pipeline outputs.
        * pipeline_parameters: Arguments to the pipeline function provided as a dict.
        * experiment_name: Name of the experiment to add the run to.
        * run_name: Name of the run to be shown in the UI.
        * namespace: Kubernetes namespace to use. Used for multi-user deployments.
            For single-user deployments, this should be left as None.
        * service_account: Specifies which Kubernetes service
            account to use for this run.
         * enable_caching: Whether or not to enable caching for the
            run. If not set, defaults to the compile time settings, which is True
            for all tasks by default, while users may specify different caching options
            for individual tasks. If set, the setting applies to all tasks in the pipeline
            (overrides the compile time settings).
        """
        # Parse pipeline_func
        if isinstance(pipeline, str):  # import the pipeline from module
            if (
                pipeline.endswith(".yaml") or pipeline.endswith(".yml")
            ) and os.path.isfile(pipeline):
                self.pipeline = pipeline
                _pipeline_type = "pipeline_package"
            else:  # assuming path to the pipeline function
                module_path, function_name = pipeline.rsplit(".", 1)
                # Import the module
                module = importlib.import_module(module_path)
                # Get the function from the module
                self.pipeline = getattr(module, function_name)
                _pipeline_type = "pipeline_func"
        else:
            self.pipeline = pipeline
            _pipeline_type = "pipeline_func"

        # Creating run
        kwargs = {
            "arguments": pipeline_parameters,
            "run_name": run_name,
            "experiment_name": experiment_name,
            "namespace": namespace,
            "pipeline_root": pipeline_root,
            "enable_caching": enable_caching,
            "service_account": service_account,
            # "experiment_id": experiment_id,
        }
        if _pipeline_type == "pipeline_func":
            self.run: RunPipelineResult = (
                self.client.create_run_from_pipeline_func(
                    pipeline_func=self.pipeline, **kwargs
                )
            )
        else:  # pipeline_package
            self.run: RunPipelineResult = (
                self.client.create_run_from_pipeline_package(
                    pipeline_file=self.pipeline, **kwargs
                )
            )

        return self.run

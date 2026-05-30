import tempfile
from typing import Literal
from kfp import dsl, local


class LocalPipelineRunner:
    """Local pipeline runner"""

    def __init__(
        self,
        pipeline_root: str = None,
        runner: Literal["subprocess", "docker"] = "subprocess",
    ):
        # Initialize the pipeline runner
        self.pipeline_root = pipeline_root or tempfile.mkdtemp()
        if runner == "subprocess":
            self.runner = local.SubprocessRunner(use_venv=False)
        elif runner == "docker":
            self.runner = local.DockerRunner()
        local.init(
            runner=self.runner,
            pipeline_root=self.pipeline_root,
            raise_on_error=True,
        )

    def create_run(
        self,
        pipeline: dsl.graph_component.GraphComponent,
        payload: dict = None,
    ):
        payload = payload or {}
        # Run by directlycalling the pipeline function
        result = pipeline(**payload)
        # Return the result
        return result

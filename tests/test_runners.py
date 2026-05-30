import pytest
import json
from kfp import dsl
from kfp.dsl import Output, Artifact
from ..runner.local_runner import LocalPipelineRunner
from ..runner.kubeflow_runner import KubeflowPipelineRunner
from ..runner.vertex_runner import VertexPipelineRunner
from pdb import set_trace


# %% Local Subprocess Runner
@dsl.component
def say_hello(name: str) -> str:
    hello_text = f"Hello, {name}!"
    print(hello_text)
    return hello_text


# Create the pipeline
@dsl.pipeline
def hello_pipeline(recipient: str) -> str:
    hello_task = say_hello(name=recipient)
    return hello_task.output


@pytest.mark.skip(reason="skip")
def test_local_subprocess_runner():
    # Initialize the runner
    runner = LocalPipelineRunner(runner="subprocess")

    # Define parameters
    payload = {
        "recipient": "World",
    }

    # Run the pipeline
    result = runner.create_run(hello_pipeline, payload)

    # Check results
    assert result.output == "Hello, World!"


# %% Local Docker Runner
@dsl.component
def add(a: int, b: int, output_artifact: Output[Artifact]):
    import json  # importing needed components inside

    result = json.dumps(a + b)
    with open(output_artifact.path, "w") as f:
        print("Writing to path:", output_artifact.path)
        f.write(result)

    output_artifact.metadata["operation"] = "addition"


@pytest.mark.skip(reason="skip")
def test_local_docker_run():
    # Initialize the runner
    runner = LocalPipelineRunner(runner="docker")

    # Define parameters
    payload = {
        "a": 1,
        "b": 2,
    }

    # Run the pipeline
    task = runner.create_run(add, payload)

    with open(task.outputs["output_artifact"].path) as f:
        contents = f.read()
    assert json.loads(contents) == 3
    assert (
        task.outputs["output_artifact"].metadata["operation"]
        == "addition"
    )


# %% Kubeflow runner
@pytest.mark.skip(reason="skip")
def test_kubeflow_run():
    runner = KubeflowPipelineRunner("http://localhost:8080/")
    runner.create_run(
        hello_pipeline,
        pipeline_root="./",
        pipeline_parameters={"recipient": "Kubeflow"},
        experiment_name="test_kubeflow_pipeline_run",
    )

# %% GCP Vertex runner
# @pytest.mark.skip(reason="Requires remote job submission, skip")
def test_vertex_run():
    runner = VertexPipelineRunner(
        gcp_project_id="nfa-core-prod", run_region="us-east1"
    )
    runner.create_run(
        hello_pipeline,
        pipeline_name="test-vertex-pipeline-run",
        pipeline_parameters={"recipient": "Vertex"},
        pipeline_root="gs://ml-pipeline-runs",
        pipeline_path="./",
        # image_uri="gcr.io/ml-pipelines/test-ml-pipeline"
    )

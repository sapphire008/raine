from typing import List
from kfp import dsl
from kfp.dsl import Output, Artifact
from linke.evaluation.beam.evaluator import EvalConfig, MetricSpec, run_evaluation_pipeline


@dsl.component
def evaluator(
    eval_config: EvalConfig,
    metric_result_output: str,
    blessing_result_output: str,
    beam_pipeline_args: List[str]
):
    """Beam evaluator Kubeflow component."""
    # Run the pipeline
    run_evaluation_pipeline(
        eval_config=eval_config,
        metric_result=metric_result_output,
        blessing_result=blessing_result_output,
        beam_pipeline_args=beam_pipeline_args,
    )
    
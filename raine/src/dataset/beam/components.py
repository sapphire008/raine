"""Living as an independent file so that it can import the helper functions from another file"""

from typing import List, Dict
from kfp import dsl
from kfp.dsl import Output, Artifact


@dsl.component
def beam_data_processing_component(
    processing_fn: str,
    setup_fn: str,
    input_data: Dict,
    output_data: Dict,
    # input_artifact: Optional[Input[Artifact]] = None,
    output_artifact: Output[Artifact],
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
    use_output_artifact: bool = True,
):
    """Beam data processing Kubeflow component."""
    # Importing all the helper functions
    import os
    from linke.dataset.beam.data_processor import (
        run_data_processing_pipeline,
        BaseData,
    )

    # Getting the data class
    input_dataclass = BaseData.get_class(input_data["__class__"])
    input_data_obj = input_dataclass.from_dict(input_data)
    output_dataclass = BaseData.get_class(output_data["__class__"])
    output_data_obj = output_dataclass.from_dict(output_data)

    # Setting outputdata to use output artifact path
    if output_dataclass.has_field("file") and use_output_artifact:
        if output_data_obj.file is not None:
            # Reusing the filename
            output_data_obj.file = os.path.join(
                output_artifact.path,
                os.path.basename(output_data_obj.file),
            )
        else:
            output_data_obj.file = os.path.join(
                output_artifact.path, "output"
            )
    elif output_dataclass.has_field("file") and not use_output_artifact:
        assert (
            output_data_obj.file is not None,
            "Need to specify the output file name when not "
            "use_output_artifact = False",
        )

    # Call the data processor
    run_data_processing_pipeline(
        input_data=input_data_obj,
        output_data=output_data_obj,
        processing_fn=processing_fn,
        setup_fn=setup_fn,
        beam_pipeline_args=beam_pipeline_args,
    )


@dsl.component
def example_gen():
    pass
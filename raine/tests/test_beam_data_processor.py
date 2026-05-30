"""Integration test for I/O of various format.
Make sure to test from root directory"""

import os
import tempfile
import shutil
import gzip
import pytest
import numpy as np
import pandas as pd
from apache_beam.io.tfrecordio import _TFRecordUtil
from apache_beam.coders import BytesCoder
from linke.runner.local_runner import LocalPipelineRunner
# fmt: off
from linke.dataset.beam.data_processor import (
    run_data_processing_pipeline,
    CsvInputData, CsvOutputData,
    BigQueryInputData, BigQuerySchemaField, BigQueryOutputData,
    TFRecordFeatureSchema, TFRecordInputData, TFRecordOutputData,
    ParquetSchemaField, ParquetInputData, ParquetOutputData,
)
# fmt: on
from linke.dataset.beam.components import (
    beam_data_processing_component,
)
from linke.dataset.beam.utils import (
    TFRecordIOUtils,
)
from pdb import set_trace


def test_csv_reader_writer():
    input_file = "linke/tests/data/input.csv"
    processing_fn = (
        "linke.tests.conftest.csv_processing_fn"
    )
    setup_fn = "linke.tests.conftest.csv_setup_fn"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output")
        run_data_processing_pipeline(
            input_data=CsvInputData(file=input_file, batch_size=2),
            output_data=CsvOutputData(
                file=output_file,
                num_shards=3,
                headers=["A", "B"],
            ),
            processing_fn=processing_fn,
            setup_fn=setup_fn,
        )
        # Check results
        df = []
        for ii, file in enumerate(os.listdir(temp_dir)):
            if not file.endswith(".csv"):
                continue
            df.append(pd.read_csv(os.path.join(temp_dir, file)))
        df = pd.concat(df)
        assert ii == 2, "Expecting 3 shards"
        assert (
            df.columns[0] == "A" and df.columns[1] == "B"
        ), f"Expecting only 2 columns, but got {df.columns}"
        df_input = pd.read_csv(input_file)
        assert (
            df_input.shape[0] == df.shape[0]
        ), "Expecting same number of outputs as inputs"


def test_beam_data_processing_single_component():
    """No input/output artifacts. Just static files linked"""
    input_file = "linke/tests/data/input.csv"
    # Initialize the runner
    runner = LocalPipelineRunner(runner="subprocess")
    with tempfile.TemporaryDirectory() as temp_dir:
        # output_file = os.path.join(temp_dir, "output")
        # make payload
        payload = {
            "processing_fn": "linke.tests.conftest.csv_processing_fn",
            "setup_fn": "linke.tests.conftest.csv_setup_fn",
            "input_data": CsvInputData(
                file=input_file, batch_size=2
            ).as_dict(),
            "output_data": CsvOutputData(
                # file=output_file,
                num_shards=3,
                headers=["A", "B"],
            ).as_dict(),
        }
        # Run the pipeline
        task = runner.create_run(
            beam_data_processing_component, payload
        )

        # Check that there are 3 files
        files = os.listdir(task.output.uri)
        assert len(files) == 3, "Expected 3 files"
        shutil.rmtree(task.output.uri)  # clean up


def test_bigquery_output_data():
    # Two modes of specifying BigQuery Output
    output_data1 = BigQueryOutputData(
        "project_id.dataset.table",
        schema=[
            BigQuerySchemaField(
                name="id",
                type="STRING",
                mode="REQUIRED",
                description="video id",
            ),
            BigQuerySchemaField(
                name="title", type="STRING", mode="NULLABLE"
            ),
        ],
    )
    output_data2 = BigQueryOutputData(
        "project_id.dataset.table",
        schema=[
            {
                "name": "id",
                "type": "STRING",
                "mode": "REQUIRED",
                "description": "video id",
            },
            {
                "name": "title",
                "type": "STRING",
                "mode": "NULLABLE",
                "description": "",
            },
        ],
    )

    assert (
        output_data1.schema["fields"][0]
        == output_data2.schema["fields"][0]
    )


@pytest.mark.skip(reason="Skip for now during development")
def test_bigquery_reader_writer():
    """To Run this test, set up is needed on a Google Cloud project.
    * Create the table using the sql from data/input_bq.sql
    * Create a storage bucket for temp data writes
    * Run `gcloud auth application-default login` if running locally
    """
    query = """
        SELECT id, title, tags
        FROM `rinoa-core-prod.public.videos`
        WHERE age_rating = "mpa:pg-13"
    """
    # Make sure this schema is in the same order as the output from the processing_fn
    schema = [
        BigQuerySchemaField(name="id", type="STRING", mode="REQUIRED"),
        BigQuerySchemaField(
            name="transformed_title", type="STRING", mode="NULLABLE"
        ),
        BigQuerySchemaField(
            name="num_tags", type="INT64", mode="NULLABLE"
        ),
    ]

    run_data_processing_pipeline(
        input_data=BigQueryInputData(sql=query, batch_size=3),
        output_data=BigQueryOutputData(
            output_table="rinoa-core-prod.temp_dataset.test_beam_writer",
            schema=schema,
        ),
        processing_fn="linke.tests.conftest.bq_processing_fn",
        setup_fn="linke.tests.conftest.csv_setup_fn",
        beam_pipeline_args=[
            "--runner=DirectRunner",
            "--temp_location=gs://rinoa-core-prod-ml-pipelines/bigquery",
            "--project=rinoa-core-prod",
        ],
    )


def test_tfrecord_reader_writer():
    input_file = "linke/tests/data/input.tfrecord"
    processing_fn = (
        "linke.tests.conftest.tfrecord_processing_fn"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output")
        run_data_processing_pipeline(
            input_data=TFRecordInputData(
                file=input_file,
                format="feature",
                schema=[
                    TFRecordFeatureSchema(
                        name="A", type="int", fixed_length=False
                    ),
                    TFRecordFeatureSchema(name="B", type="byte"),
                    TFRecordFeatureSchema(name="C", type="float"),
                ],
                batch_size=2,
            ),
            output_data=TFRecordOutputData(
                file=output_file,
                schema=[
                    TFRecordFeatureSchema(name="A", type="byte"),
                    TFRecordFeatureSchema(name="B", type="float"),
                    TFRecordFeatureSchema(name="C", type="int"),
                ],
            ),
            processing_fn=processing_fn,
            setup_fn=None,
        )
        # Check the output file
        _coder = BytesCoder()
        counter = 0
        with gzip.open(output_file, "rb") as fid:
            while True:
                raw_record = _TFRecordUtil.read_record(fid)
                counter += 1
                if raw_record is None:
                    break
                record = _coder.decode(raw_record)
                result = TFRecordIOUtils.deserialize_tf_example(
                    record, {"A": "byte", "B": "float", "C": "int"}
                )
                assert isinstance(
                    result["A"][0], bytes
                ), "Expected A to be bytes"
                assert isinstance(
                    result["B"][0], np.float32
                ), "Expected B to be floats"
                assert isinstance(
                    result["C"][0], np.int64
                ), "Expected C to be ints"
        assert counter == 34, "Expecting 34 records"


def test_parquet_reader_writer():
    input_file = "linke/tests/data/input.parquet"
    processing_fn = (
        "linke.tests.conftest.parquet_processing_fn"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output")
        # output_file = "./data/output.parquet"
        run_data_processing_pipeline(
            input_data=ParquetInputData(
                file=input_file,
                format="dict",
                batch_size=2,
            ),
            output_data=ParquetOutputData(
                file=output_file,
                schema=[
                    ParquetSchemaField(
                        name="E", type="string", nullable=False
                    ),
                    ParquetSchemaField(name="F", type="int"),
                    ParquetSchemaField(name="G", type="float"),
                    ParquetSchemaField(name="H", type="array(string)"),
                ],
            ),
            processing_fn=processing_fn,
            setup_fn=None,
        )
        # Check the output file
        df_output = pd.read_parquet(output_file)
        assert set(["E", "F", "G", "H"]) == set(df_output.columns)
        assert df_output["E"].dtype == "object"
        assert df_output["F"].dtype == "int"
        assert df_output["G"].dtype == "float32"
        assert df_output["H"].dtype == "object"
        assert isinstance(df_output["H"][0], np.ndarray)
        assert all(df_output["H"].apply(lambda x: len(x)) > 0)

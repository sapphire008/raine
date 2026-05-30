import tempfile
import apache_beam as beam
from linke.dataset.beam.data_processor import CsvInputData, CsvOutputData
from linke.dataset.beam.example_gen import run_example_gen_data_split_pipeline, StatisticsType, DataSplitConfigs


class TestExampleGenDataSplitPipeline:
    def setup_method(self, method=None):
        pass
        # Create two data splits from csv file
    
    def teardown_method(self, method=None):
        pass
        # remove the temporary file
    
    def test_example_gen_e2e(self):
        split_configs = DataSplitConfigs(
            statistics=[{}],
            input_data={"train": CsvInputData(batch_size=2, file="linke/tests/data/examples_train.csv"), "eval": CsvInputData(batch_size=2, file="")}
        )
        run_example_gen_data_split_pipeline(
            split_configs=split_configs, statistics_result=output_path)
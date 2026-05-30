import os
import json
import pytest
import tempfile
import copy
import numpy as np
import pandas as pd
from linke.dataset.beam.data_processor import (
    CsvInputData,
)
from linke.evaluation.beam.evaluator import (
    run_evaluation_pipeline,
    determine_blessing,
    EvalConfig,
    DataSpec,
    SliceConfig,
    ModelSpec,
    MetricSpec,
    MetricThreshold,
)

from pdb import set_trace
import warnings

warnings.filterwarnings("ignore")


def label_transform_fn(labels, config={}):
    alphabet = "abcdefghijk"
    transformed_labels = []
    for val in labels:
        label = [alphabet[ii] for ii in range(val)]
        np.random.RandomState(val + 42).shuffle(label)
        transformed_labels.append(label)
    return transformed_labels


def inference_fn(inputs, config={}):
    alphabet = "abcdefghijk"
    predictions = []
    # Faking predictions
    for index in inputs["A"]:
        prediction = list(alphabet)
        np.random.RandomState(index + 327).shuffle(prediction)
        predictions.append(prediction)
    predictions = np.array(predictions)
    return predictions


def setup_fn():
    return {}


class TestEvaluator:
    def setup_method(self, method=None):
        self.data_spec = DataSpec(
            input_data=CsvInputData(
                # Not really doing anything
                file="linke/tests/data/input.csv",
                batch_size=2,
            ),
            label_key="E",
            slices=None,
        )
        self.model_spec = ModelSpec(
            name="model_1",
            inference_fn=inference_fn,
            setup_fn=setup_fn,
            label_transform_fn=label_transform_fn,
        )
        # Create metric from module_path
        self.metric_hit_ratio = MetricSpec(
            name="hit_ratio",
            metric="linke.evaluation.beam.metrics.HitRatioTopK",
            config={"top_k": [1, 4, 5]},
        )
        self.metric_ndcg = MetricSpec(
            name="ndcg",
            metric="linke.evaluation.beam.metrics.NDCGTopK",
            config={"top_k": [1, 4, 5]},
        )
        self.metric_unqiue_count = MetricSpec(
            name="unique_count",
            metric="linke.evaluation.beam.metrics.UniqueCountTopK",
            config={"top_k": [1, 4, 5]},
        )

    # @pytest.mark.skip(reason="")
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline from end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # temp_dir = "./"
            metric_result = os.path.join(temp_dir, "metric_result.json")
            blessing_result = os.path.join(temp_dir, "blessing_result.json")
            run_evaluation_pipeline(
                eval_config=EvalConfig(
                    model=self.model_spec,
                    metrics=[self.metric_hit_ratio, self.metric_ndcg],
                    data=self.data_spec,
                ),
                metric_result=metric_result,
                blessing_result=blessing_result,
                beam_pipeline_args=["--runner=DirectRunner"],
            )
            # Check results
            with open(metric_result, "r") as fid:
                result = json.load(fid)
                assert "ndcg" in result
                assert "hit_ratio" in result
                assert all(
                    [
                        str(k) in result["ndcg"]
                        for k in self.metric_ndcg.metric.combiner.top_k
                    ]
                )
                assert all(
                    [
                        str(k) in result["hit_ratio"]
                        for k in self.metric_ndcg.metric.combiner.top_k
                    ]
                )
            with open(blessing_result, "r") as fid:
                blessing = json.load(fid)
                assert blessing.get("is_blessed") == True
                assert isinstance(blessing.get("explanations"), str)

    # @pytest.mark.skip(reason="")
    def test_evaluation_pipeline_keyed_combiner(self):
        """Test evaluation pipeline from end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metric_result = os.path.join(temp_dir, "metric_result.json")
            blessing_result = os.path.join(temp_dir, "blessing_result.json")
            run_evaluation_pipeline(
                eval_config=EvalConfig(
                    model=self.model_spec,
                    metrics=[self.metric_unqiue_count],
                    data=self.data_spec,
                ),
                metric_result=metric_result,
                blessing_result=blessing_result,
                beam_pipeline_args=["--runner=DirectRunner"],
            )
            # Check results
            with open(metric_result, "r") as fid:
                result = json.load(fid)
                assert "unique_count" in result

                assert all(
                    [
                        str(k) in result["unique_count"]
                        for k in self.metric_unqiue_count.metric.combiner
                    ]
                )
                assert "label" in result["unique_count"]  # label

    # @pytest.mark.skip(reason="")
    def test_sliced_evaluation_pipeline(self):
        data_path = "linke/tests/data/input.csv"
        data_spec = DataSpec(
            input_data=CsvInputData(
                # Not really doing anything
                file=data_path,
                batch_size=2,
            ),
            label_key="E",
            slices=[
                SliceConfig(feature_keys=["B"]),
                SliceConfig(feature_keys=["C", "D"]),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            metric_result = os.path.join(temp_dir, "metric_result.json")
            blessing_result = os.path.join(temp_dir, "blessing_result.json")
            run_evaluation_pipeline(
                eval_config=EvalConfig(
                    model=self.model_spec,
                    metrics=[self.metric_hit_ratio, self.metric_ndcg],
                    data=data_spec,
                ),
                metric_result=metric_result,
                blessing_result=blessing_result,
                beam_pipeline_args=["--runner=DirectRunner"],
            )
            # Create expected results for hit_ratio
            df = pd.read_csv(data_path)
            predictions = inference_fn(df)
            transformed_labels = label_transform_fn(df["E"])
            for k in [1, 4, 5]:
                hit_ratios = [
                    len(set(pred).intersection(set(lab))) > 0
                    for pred, lab in zip(predictions[:, :k], transformed_labels)
                ]
                df[f"hit_ratio_{k}"] = np.array(hit_ratios).astype(float)

            # Check results
            with open(metric_result, "r") as fid:
                result = json.load(fid)
                assert "ndcg" in result
                assert "hit_ratio" in result
                hit_ratio_dict = result["hit_ratio"]
                # Check results
                for k in [1, 4, 5]:
                    # Check global
                    hit_ratio_global = df[f"hit_ratio_{k}"].mean()
                    assert np.allclose(hit_ratio_global, hit_ratio_dict[""][""][str(k)])
                # Check the first partition
                hit_ratio_B = df.groupby(by="B").mean()
                for k in [1, 4, 5]:
                    for key, val in hit_ratio_dict["B"].items():
                        expected = hit_ratio_B.loc[int(key), f"hit_ratio_{k}"]
                        obtained = val[str(k)]
                        assert np.allclose(expected, obtained)

                hit_ratio_CD = df.groupby(by=["C", "D"]).mean()
                for k in [1, 4, 5]:
                    for key, val in hit_ratio_dict["(C, D)"].items():
                        index = key.replace("(", "").replace(")", "").split(",")
                        index = tuple([int(ii.strip()) for ii in index])
                        expected = hit_ratio_CD.loc[index, f"hit_ratio_{k}"]
                        obtained = val[str(k)]
                        assert np.allclose(expected, obtained)

    def test_sliced_evaluation_pipeline_keyed_combiner(self):
        data_path = "linke/tests/data/input.csv"
        data_spec = DataSpec(
            input_data=CsvInputData(
                # Not really doing anything
                file=data_path,
                batch_size=2,
            ),
            label_key="E",
            slices=[
                SliceConfig(feature_keys=["B"]),
                SliceConfig(feature_keys=["C", "D"]),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            metric_result = os.path.join(temp_dir, "metric_result.json")
            blessing_result = os.path.join(temp_dir, "blessing_result.json")
            run_evaluation_pipeline(
                eval_config=EvalConfig(
                    model=self.model_spec,
                    metrics=[self.metric_unqiue_count],
                    data=data_spec,
                ),
                metric_result=metric_result,
                blessing_result=blessing_result,
                beam_pipeline_args=["--runner=DirectRunner"],
            )
            with open(metric_result, "r") as fid:
                result = json.load(fid)
                assert "unique_count" in result
                assert all(
                    [
                        str(k) in result["unique_count"]["(C, D)"]["(2, 4)"]
                        for k in self.metric_unqiue_count.metric.combiner
                    ]
                )

    def test_determine_blessing(self):
        # Mocking a metric output
        metric_results = {
            "hit_ratio": {
                "": {
                    "": {1: 0.28, 4: 0.54, 5: 0.71},
                },
                "B": {
                    "0": {1: 0.31, 4: 0.62, 5: 0.75},
                    "1": {1: 0.24, 4: 0.37, 5: 0.64},
                    "3": {1: 0.11, 4: 0.27, 5: 0.53},
                },
            }
        }
        metric_spec = copy.deepcopy(self.metric_hit_ratio)
        metric_spec.thresholds = [
            # Only lower bound: pass
            MetricThreshold(["B", "0", 1], lower=0.10),
            # needs to be in-between: pass
            MetricThreshold(["B", "0", 4], lower=0.27, upper=0.75),
            # needs to be in-between: fail
            MetricThreshold(["B", "0", 5], lower=0.27, upper=0.65),
        ]

        blessing_results = determine_blessing(metric_results, [metric_spec])
        assert blessing_results["is_blessed"] == False
        field = blessing_results["explanations"]["hit_ratio"]["B"]["0"]
        expectation = {1: "(Passed)", 4: "(Passed)", 5: "(Failed)"}
        for k, v in field.items():
            assert v.startswith(expectation[k])

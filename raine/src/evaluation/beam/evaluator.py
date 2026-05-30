import json
import importlib
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    Type,
    Iterable,
    Literal,
    ClassVar,
)
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from linke.utils.beam_routines import CombineToJson
from linke.dataset.beam.data_processor import (
    BaseInputData,
    BatchReader,
    DataProcessingDoFn,
)
from linke.evaluation.beam.metrics import (
    BaseMetric,
    DEFAULT_FEATURE_KEY,
    DEFAULT_LABEL_KEY,
    DEFAULT_PREDICTION_KEY,
)


from pdb import set_trace

# %% TFMA-like Model Eval without Protobuf
# and use models in addition to Tensorflow


@dataclass
class ModelSpec:
    name: str
    inference_fn: Union[
        str, Callable[[List[Dict], Dict], Union[Dict, List[Dict]]]
    ]
    setup_fn: Union[str, Callable[[], Dict]] = None
    label_transform_fn: Union[str, Callable[[Any, Dict], Any]] = None
    prediction_transform_fn: Union[str, Callable[[Any, Dict], Any]] = (
        None
    )


@dataclass
class MetricThreshold:
    # Direction Enum
    HIGHER_IS_BETTER: ClassVar[str] = "higher_is_better"
    LOWER_IS_BETTER: ClassVar[str] = "lower_is_better"

    # Inner key of the metric, if the metric
    # returns a dictionary rather than a single value
    # e.g. slice = "pg" and top_k = 5 metric
    # can be specified as ["pg", 5]
    metric_keys: List[Any] = None
    # Metric needs to be >= this value
    lower: Union[float, int] = None
    # Metric needs to be <= this value
    upper: Union[float, int] = None
    # Minimum examples needed for a valid metric
    min_examples: int = None
    # Direction, whether higher or lower is better
    direction: Literal["higher_is_better", "lower_is_better"] = (
        "higher_is_better"
    )


@dataclass
class MetricSpec:
    name: str
    # metric instance or str path to the metric class
    metric: Union[str, BaseMetric, None] = None
    # keyword arguments passed to the metric
    # when constructing from a module path
    config: Dict = None
    # Determines if the evaluator should give blessing
    thresholds: Optional[List[MetricThreshold]] = None

    def __post_init__(self):
        if not isinstance(self.metric, str):
            return  # already created
        # Use module_path to create metric
        module_path, class_name = self.metric.rsplit(".", 1)
        module = importlib.import_module(module_path)
        metric_cls = getattr(module, class_name)
        self.metric = metric_cls(**self.config)


@dataclass
class SliceConfig:
    """Slice data input and perform
    group-wise metric calculation."""

    feature_keys: Union[List[str], None] = None


@dataclass
class DataSpec:
    input_data: BaseInputData
    label_key: str
    # By defualt, use full dataset
    slices: Optional[List[SliceConfig]] = None


class EvalConfig:
    def __init__(
        self,
        model: ModelSpec,
        metrics: List[MetricSpec],
        data: DataSpec,
        baseline_model: Optional[ModelSpec] = None,
    ):
        """
        Evaluation Configuration.

        Args:
            model (ModelSpec): model configuration
            metrics (List[MetricSpec]): List of metrics to compute
            data (DataSpec): evaluation data configuration
            baseline_model (Optional[ModelSpec], optional): baseline
                model to compare to. This is useful to calculate
                metrics that compares baseline models, such as
                the difference in the set of predictions. Defaults to
                None.
        """
        self.model = model
        self.metrics = metrics
        self.data = data
        self.baseline_model = baseline_model

    @classmethod
    def from_json(cls):
        pass

    def to_json(self):
        pass


# %% Model inference
class ModelInferenceDoFn(DataProcessingDoFn):
    def __init__(
        self,
        label_key: str,
        inference_fn: Union[
            str, Callable[[List[Dict], Dict], Union[Dict, List[Dict]]]
        ],
        setup_fn: Union[str, Callable[[], Dict]] = None,
        label_transform_fn: Union[
            str, Callable[[Any, Dict], Any]
        ] = None,
        prediction_transform_fn: Union[
            str, Callable[[Any, Dict], Any]
        ] = None,
        config: Dict = {},
        output_label_key: str = DEFAULT_LABEL_KEY,
        output_prediction_key: str = DEFAULT_PREDICTION_KEY,
        output_feature_key: str = DEFAULT_FEATURE_KEY,
        batched_output: bool = True,
    ):
        self.label_key = label_key
        self.label_transform_fn = (
            self.import_function(label_transform_fn)
            if isinstance(label_transform_fn, str)
            else label_transform_fn
        )
        self.prediction_transform_fn = (
            self.import_function(prediction_transform_fn)
            if isinstance(prediction_transform_fn, str)
            else prediction_transform_fn
        )
        self.output_label_key = output_label_key
        self.output_prediction_key = output_prediction_key
        self.output_feature_key = output_feature_key
        self.batched_output = batched_output
        super(ModelInferenceDoFn, self).__init__(
            processing_fn=inference_fn, setup_fn=setup_fn, config=config
        )

    def process(self, element):
        # Force convert to dict
        if not isinstance(element, dict):
            element = self.list2dict(element)
        # Extract labels
        labels = element.pop(self.label_key)
        if self.label_transform_fn:
            labels = self.label_transform_fn(labels, self.config)
        # Make predictions
        predictions = self.processing_fn(element, self.config)
        if self.prediction_transform_fn:
            predictions = self.prediction_transform_fn(
                predictions, self.config
            )
        # Make output to be used by metric calculation
        if self.batched_output:
            output = {
                self.output_feature_key: element,
                self.output_prediction_key: predictions,
                self.output_label_key: labels,
            }
            yield output
        else:  # unbatched output, batch size 1
            for ii in range(len(predictions)):
                output = {
                    self.output_feature_key: {
                        k: v[ii : ii + 1] for k, v in element.items()
                    },
                    self.output_prediction_key: predictions[
                        ii : ii + 1
                    ],
                    self.output_label_key: labels[ii : ii + 1],
                }
                # yield one element at a time
                yield output


# %% Metric writer


class EvaluateMetric(beam.PTransform):
    """Evaluate a single metric."""

    def __init__(
        self,
        metric: MetricSpec,
        group_keys: Optional[List[List[str]]] = None,
    ):
        self.metric = metric
        self.group_keys = group_keys

    @staticmethod
    def _key_comb_global(pcoll, metric_name, key, combine_fn):
        return (
            pcoll
            | f"Filter {metric_name} result at {key}"
            >> beam.Filter(lambda x: key in x)
            | f"Extract {metric_name} result at {key}"
            >> beam.Map(lambda x: x[key])
            | f"Combine {metric_name} at {key}"
            >> beam.CombineGlobally(combine_fn)
            | f"Label combined {metric_name} at {key}"
            >> beam.Map(lambda x: {key: x})
        )

    def _handle_key_specific_combiner(self, pcoll):
        """Handle case where combiner is a dict."""
        pcoll_combine_list = []

        # use dedicated combiner for each of the results in the output
        for key, combine_fn in self.metric.metric.combiner.items():
            pcoll_combine_list.append(
                self._key_comb_global(
                    pcoll, self.metric.name, key, combine_fn
                )
            )
        pcoll_combine = (
            tuple(pcoll_combine_list)
            | "Flatten key-specific combiners" >> beam.Flatten()
            | f"Combine {self.metric.name} all groups"
            >> beam.CombineGlobally(CombineToJson("dict"))
        )
        return pcoll_combine

    @staticmethod
    def _clean_key(key: Union[List, Tuple, str]):
        if isinstance(key, str):
            return key.replace("'", "").replace('"', "")
        key = tuple(key) if len(key) > 1 else key[0]
        return str(key).replace("'", "").replace('"', "")

    @staticmethod
    def _key_comb_per_group(pcoll, metric_name, group, key, combine_fn):
        return (
            pcoll
            | f"Filter {metric_name} result at {group} {key}"
            >> beam.Filter(lambda x: key in x[1])
            | f"Extract {metric_name} result at {group} {key}"
            >> beam.Map(lambda x: (x[0], x[1][key]))
            | f"Combine {metric_name} at {group} {key}"
            >> beam.CombinePerKey(combine_fn)
            | f"Label combined {metric_name} at {group} {key}"
            >> beam.Map(lambda x: (x[0], {key: x[1]}))
        )

    def _eval_metric_group(self, pcoll, group_name: str, index: int):
        """Evaluate the metric on specific feature group by slices."""
        if group_name == "":  # global key
            _extract_key = lambda _: ""
        else:
            _extract_key = lambda x: self._clean_key(x[index])
        pcoll = (
            pcoll
            | f"Extract key {self.metric.name} {group_name}"
            >> beam.Map(lambda x: (_extract_key(x[0]), x[1]))
        )
        # Combine
        if isinstance(self.metric.metric.combiner, dict):
            pcoll_combine_list = []
            for key, combine_fn in self.metric.metric.combiner.items():
                pcoll_combine_list.append(
                    self._key_comb_per_group(
                        pcoll,
                        self.metric.name,
                        group_name,
                        key,
                        combine_fn,
                    )
                )
            pcoll = (
                tuple(pcoll_combine_list)
                | f"Flatten key-specific combiners ({group_name})"
                >> beam.Flatten()
                | f"Combine {self.metric.name} {group_name} all groups"
                >> beam.CombinePerKey(CombineToJson("dict"))
            )
        else:
            pcoll = (
                pcoll
                | f"Combine slice {self.metric.name} {group_name}"
                >> beam.CombinePerKey(self.metric.metric.combiner)
            )
        return (
            pcoll
            | f"Label {self.metric.name} {group_name} values"
            >> beam.Map(lambda x: {x[0]: x[1]})
            | f"Merge {self.metric.name} {group_name} slice groups"
            >> beam.CombineGlobally(CombineToJson("dict"))
            | f"Label {self.metric.name} {group_name} group"
            >> beam.Map(lambda x: {group_name: x})
        )

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        """Evaluate a single metric"""
        if not self.group_keys:  # combine globally
            for preprocessor in self.metric.metric.preprocessors:
                pcoll = (
                    pcoll
                    | f"Compute {self.metric.name}"
                    >> beam.ParDo(preprocessor)
                )
            if isinstance(self.metric.metric.combiner, dict):
                pcoll_combine = self._handle_key_specific_combiner(
                    pcoll
                )
            else:  # single combiner
                pcoll_combine = (
                    pcoll
                    | f"Combine {self.metric.name}"
                    >> beam.CombineGlobally(self.metric.metric.combiner)
                )
        else:  # compute metrics by slice
            for preprocessor in self.metric.metric.preprocessors:
                pcoll = (
                    pcoll
                    | f"Compute {self.metric.name}"
                    >> beam.ParDo(
                        preprocessor.with_group_keys(self.group_keys)
                        if hasattr(preprocessor, "with_group_keys")
                        else preprocessor
                    )
                )

            metric_groups = []
            for ii, keys in enumerate(self.group_keys):
                group_name = self._clean_key(keys)
                metric_groups.append(
                    self._eval_metric_group(pcoll, group_name, ii)
                )

            # Use of of the groups to compute the global result
            metric_groups.append(self._eval_metric_group(pcoll, "", -1))

            # Merge all groups into a single dictionary
            pcoll_combine = (
                tuple(metric_groups)
                | f"Flatten keyed {self.metric.name} all groups"
                >> beam.Flatten()
                | f"Combine keyed {self.metric.name} all groups"
                >> beam.CombineGlobally(CombineToJson("dict"))
            )

        # Wrap the current metric with a name key
        pcoll_labeled = (
            pcoll_combine
            | f"Label {self.metric.name}"
            >> beam.Map(lambda x: {self.metric.name: x})
        )

        return pcoll_labeled


class MetricWriter(beam.PTransform):
    def __init__(self, output_file):
        self.output_file = output_file

    def expand(self, pcolls: Iterable[beam.PCollection]):
        return (
            pcolls
            | beam.Flatten()
            | beam.CombineGlobally(CombineToJson())
            | beam.io.WriteToText(
                self.output_file,
                num_shards=1,
                shard_name_template="",
            )
        )


# %% Evaluation pipeline
def _validate_metric_names(metrics: List[MetricSpec]):
    metric_keys = []
    for metric in metrics:
        metric_keys.append(metric.name)
    assert (
        len(metric_keys) == len(set(metric_keys)),
        "Duplicated metric names detected. "
        "Metric names need to be unique within a single job.",
    )


def set_nested_value(my_dict, path, value):
    current = my_dict
    for i, key in enumerate(path):
        if i == len(path) - 1:
            # Set the value at the last key
            current[key] = value
        else:
            # Ensure the next level dictionary exists
            if key not in current:
                current[key] = {}
            # Move to the next level
            current = current[key]


def determine_blessing(
    metric_results: Dict, metrics_specs: List[MetricSpec]
) -> Dict:
    # by default, if no threshold is specified,
    # the metric is passed
    is_blessed = True
    explanations = {}
    for metric in metrics_specs:
        if not metric.thresholds:
            continue
        result = metric_results[metric.name]
        # Iterate over list of thresholds
        for thresh in metric.thresholds:
            # extracting nested field
            value = result
            for key in thresh.metric_keys or []:
                value = value[key]
            if thresh.lower:
                is_above = value >= thresh.lower
                is_blessed = is_blessed and is_above
            else:
                is_above = None
            if thresh.upper:
                is_below = value <= thresh.upper
                is_blessed = is_blessed and is_below
            else:
                is_below = None

            # Add explanation
            if is_above is not None and is_below is not None:
                if is_above and is_below:
                    explain = f"(Passed) {thresh.lower:.3f} <= {value:.3f} <= {thresh.upper:.3f}"
                elif not is_above:
                    explain = f"(Failed) {value:.3f} < {thresh.lower:.3f} = lower bound"
                elif not is_below:
                    explain = f"(Failed) {value:.3f} > {thresh.upper:.3f} = upper bound"
            elif is_above is not None:
                if is_above:
                    explain = f"(Passed) {value:.3f} >= {thresh.lower:.3f} = lower bound"
                else:
                    explain = f"(Failed) {value:.3f} < {thresh.lower:.3f} = lower bound"
            elif is_below is not None:
                if is_below:
                    explain = f"(Passed) {value:.3f} <= {thresh.upper:.3f} = upper bound"
                else:
                    explain = f"(Failed) {value:.3f} > {thresh.upper:.3f} = upper bound"

            # Set the value of the explanation
            set_nested_value(
                explanations,
                [metric.name] + (thresh.metric_keys or []),
                explain,
            )
    blessing = {
        "is_blessed": is_blessed,
        "explanations": (
            explanations if explanations else "No metric thresholds"
        ),
    }
    return blessing


def run_evaluation_pipeline(
    eval_config: EvalConfig,
    metric_result: str,
    blessing_result: str = None,
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
):
    """Create and run the evaluation pipeline."""
    # Validate
    _validate_metric_names(eval_config.metrics)

    # Combine keys
    # (whether or not compute metrics globally or per slice)
    if eval_config.data.slices is None:
        group_keys = None
    else:
        group_keys = [
            s.feature_keys
            for s in eval_config.data.slices
            if s and s.feature_keys
        ]

    # Create beam pipeline
    options = PipelineOptions(flags=beam_pipeline_args)
    with beam.Pipeline(options=options) as pipeline:
        # Read from data source
        pcoll = pipeline | BatchReader(
            input_data=eval_config.data.input_data
        )
        # Make model inference
        pcoll_pred = pcoll | beam.ParDo(
            ModelInferenceDoFn(
                label_key="E",
                inference_fn=eval_config.model.inference_fn,
                setup_fn=eval_config.model.setup_fn,
                config={"label_key": eval_config.data.label_key},
                label_transform_fn=eval_config.model.label_transform_fn,
                prediction_transform_fn=eval_config.model.prediction_transform_fn,
                batched_output=group_keys is None,
            )
        )

        # Run each metric
        combined_metrics = []
        for metric in eval_config.metrics:
            pcoll_combined = (
                pcoll_pred
                | f"Evaluate {metric.name}"
                >> EvaluateMetric(metric, group_keys)
            )
            combined_metrics.append(pcoll_combined)

        # Combine metrics
        metric_json = (
            # Necessary need to cast to tuple!
            tuple(combined_metrics)
            | "Flatten all mertics" >> beam.Flatten()
            | "Combine all metrics"
            >> beam.CombineGlobally(CombineToJson())
        )

        # Write results
        metric_json | "Write metrics" >> beam.io.WriteToText(
            metric_result,
            num_shards=1,
            shard_name_template="",
        )

        # Compute and write blessing
        (
            metric_json
            | "Determine blessing"
            >> beam.Map(
                lambda x: determine_blessing(
                    json.loads(x), eval_config.metrics
                )
            )
            | "Blessing to string" >> beam.Map(lambda x: json.dumps(x))
            | "Write blessing"
            >> beam.io.WriteToText(
                blessing_result,
                num_shards=1,
                shard_name_template="",
            )
        )

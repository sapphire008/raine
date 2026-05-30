import os
from typing import Dict, Union, Tuple, List, Iterable, Any, Literal, Optional
from enum import Enum
import datetime
from collections import Counter
from dataclasses import dataclass
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.stats import ApproximateQuantilesCombineFn

from linke.utils.beam_routines import CombineToJson

from linke.dataset.beam.data_processor import (
    BaseInputData,
    BaseOutputData,
    BatchReader,
    BatchWriter,
)

class StatisticsType(str, Enum):
    """Feature statistics types"""

    unique = "unique"
    frequency = "frequency"
    mean = "mean"
    variance = "variance"
    standard_deviation = "standard_deviation"
    minimum = "minimum"
    maximum = "maximum"
    min_max_range = "min_max_range"
    # For the approximated algorithms, use beam's ApproximateQuantile
    approximate_median = "approximate_median"
    approximate_quartiles = "approximate_quartiles"
    approximate_percentiles = "approximate_percentiles"
    
    @classmethod
    def get_approximated_quantile_combiner(
        cls, stats_type: str, epsilon: float = 1e-2
    ):
        if stats_type == cls.approximate_median:
            return ApproximateQuantilesCombineFn.create(
                num_quantiles=3,
                epsilon=epsilon,
            )
        elif stats_type == cls.approximate_quartiles:
            return ApproximateQuantilesCombineFn.create(
                num_quantiles=5,
                epsilon=epsilon,
            )
        elif stats_type == cls.approximate_percentiles:
            return ApproximateQuantilesCombineFn.create(
                num_quantiles=101,
                epsilon=epsilon,
            )


@dataclass
class StatisticsConfig:
    """Statistics Configurations.
    handle_sequence: for sequence features,
        there are two ways to handle the data.
        "expand": (default) in this case, the sequence will 
            be expanded so that it will contribute to
            multiple instances of the data point.
        "reduce": first reduce within the sequence 
            and then reduce across the dataset. 
            In this case, each sequence is treated as 
            1 single data point, and will contribute
            to a single counter in the population
            statistics. Note that "reduce" is not valid
            for varianec, standard_deviation and 
            approximate_* metrics. Only "expand" will
            be used.
        
        For certain statistics like `unique` or `frequency`
        on categorical features, or `minimum`, `maximum`,
        `min_max_range`, the two ways will yield the same result.
        
    """
    feature: str
    type: StatisticsType
    handle_sequence: Literal["expand", "reduce"] = "expand"


@dataclass
class SplitConfigs:
    """
    Base Split Configs class
    statistics: List[StatisticsConfig]
        Specifies how to accumulate statistics
        for each feature. In this set-up, all
        data splits will compute the same set
        of statistics. Alternatively, specify
        as {split_name: [...]} for split-specific
        statistics.
    """

    statistics: Union[
        Dict[str, List[StatisticsConfig]],
        List[StatisticsConfig],
    ] = []
    output_data: Union[BaseOutputData, Dict[str, BaseOutputData]]


@dataclass
class DataSplitConfigs(SplitConfigs):
    """
    Create a data split using each split-specific data config.
    This can refence either different directories or queries.
    Parameters:
    ----
    input_data: dict of {split_name: InputData]}
        Input data for each split
    output_data: dict of {split_name: OutputData}
        Output data for each split. If the output are file data,
            subfolders will be created with the split name to store
            the transformed data.
    """

    input_data: Dict[str, BaseInputData]


@dataclass
class DataSplitOnFeatureConfigs(SplitConfigs):
    """Split the data ona single feature.
    The value of the feature will be the split name"""

    input_data: BaseInputData
    output_data: BaseOutputData
    partition_feature: str


@dataclass
class DataSplitOnRangeConfigs(SplitConfigs):
    """Split the data on a numerical feature, given
    each split brackets a range of the data.
    * partitioning: Dict of feature name, and a tuple
        that marks the start (inclusive) and end (exclusive)
        of the range. Valid numerical feature types are
        datetime.datetime, int, or float.
    """

    input_data: BaseInputData
    output_data: BaseOutputData
    numerical_feature: str
    partitioning: Dict[
        str,
        Union[
            Tuple[datetime.datetime, datetime.datetime],
            Tuple[int, int],
            Tuple[float, float],
        ],
    ]


@dataclass
class RandomDataSplitByFeaturesConfigs(SplitConfigs):
    """
    Randomly split based on feature definition.
    if output is a file, then subfolders are created
    to store each data split.
    Parameters:
    -----
    * input_data: Input data configs
    * output_data: Output data configs
    * partition_features: list of features whose value
        combinations will be hashed.
    * hash_buckets: Proportion of each split.
        For example, {"train": 8, "eval": 1, "test": 1}
        would result a random split of 80% data in "train",
        and 10% in "eval" and "test", respectively.
    """

    input_data: BaseInputData
    output_data: BaseOutputData
    partition_features: List[str]
    hash_buckets: Dict[str, int]


class StatisticsGenDoFn(beam.DoFn):
    def __init__(
        self,
        statistics: List[StatisticsConfig],
        approximated_metrics: bool = False,
    ):
        self.statistics = statistics
        self.approximated_metrics = approximated_metrics
        
    @staticmethod
    def batch_compute_stats(stats_type: str, x_batched: Iterable):
        # TODO: handle sequence features
        if stats_type == StatisticsType.unique:
            return set(x_batched)
        elif stats_type == StatisticsType.frequency:
            return Counter(x_batched)
        elif stats_type == StatisticsType.mean:
            x_batched = np.array(x_batched, dtype=float)
            # total, count
            return np.nansum(x_batched), np.sum(~np.isnan(x_batched))
        elif stats_type in (
            StatisticsType.variance,
            StatisticsType.standard_deviation,
        ):
            # sum(x), sum(x^2), count
            x_batched = np.array(x_batched, dtype=float)
            return (
                np.nansum(x_batched),
                np.nansum(x_batched**2),
                np.sum(~np.isnan(x_batched)),
            )
        elif stats_type == StatisticsType.minimum:
            x_batched = np.array(x_batched, dtype=float)
            return np.nanmin(x_batched)
        elif stats_type == StatisticsType.maximum:
            x_batched = np.array(x_batched, dtype=float)
            return np.nanmax(x_batched)
        elif stats_type in (StatisticsType.min_max_range):
            x_batched = np.array(x_batched, dtype=float)
            return np.nanmin(x_batched), max(x_batched)

    def process(self, elements):
        if isinstance(elements, dict):
            elements = [elements]
        # Iterate over each feature
        for s in self.statistics:
            if self.approximated_metrics:
                for element in elements:
                    yield (s.feature, s.type), element[s.feature]
            else:
                output = self.batch_compute_stats(
                    s.type, [ele[s.feature] for ele in elements]
                )
                if output is not None:  # custom combiner can handle
                    yield output
                else:  # catch fall-through
                    for element in elements:
                        yield (s.feature, s.type), element[s.feature]


class StatisticsGenCombiner(beam.CombineFn):
    def __init__(self, statistics: List[StatisticsConfig]):
        # filter out approximated statistics
        self.statistics, _ = self.separate_stats(statistics)
    
    @staticmethod
    def separate_stats(
        statistics: List[StatisticsConfig]
    ) -> Tuple[List[StatisticsConfig], List[StatisticsConfig]]:
        # ignore approximate statistics
        _special_stats = [
            StatisticsType.approximate_median,
            StatisticsType.approximate_percentiles,
            StatisticsType.approximate_quartiles,
        ]
        orig_stats = []
        filtered_stats = []
        for feature_stats in statistics:
            feature, stat = list(feature_stats.items())[0]
            if stat not in _special_stats:
                orig_stats.append({feature: stat})
            else:
                filtered_stats.append({feature: stat})
        return orig_stats, filtered_stats
    
    @staticmethod
    def get_accumulator(stats_type: str):
        if stats_type == StatisticsType.unique:
            return set()
        elif stats_type == StatisticsType.frequency:
            return Counter()
        elif stats_type == StatisticsType.mean:
            # total, count
            return 0.0, 0
        elif stats_type in (
            StatisticsType.variance,
            StatisticsType.standard_deviation,
        ):
            # sum(x), sum(x^2), count
            return 0.0, 0.0, 0
        elif stats_type in (
            StatisticsType.minimum,
            StatisticsType.maximum,
        ):
            return 0.0
        elif stats_type == StatisticsType.min_max_range:
            return 0.0, 0.0

    @staticmethod
    def reduce_accumulator(stats_type: str, x, y):
        if stats_type == StatisticsType.unique:
            return x + y
        elif stats_type == StatisticsType.frequency:
            return x + y
        elif stats_type == StatisticsType.mean:
            # total, count
            return x[0] + y[0], x[1] + y[1]
        elif stats_type in (
            StatisticsType.variance,
            StatisticsType.standard_deviation,
        ):
            # sum(x), sum(x^2), count
            return x[0] + y[0], x[1] + y[1], x[2] + y[2]
        elif stats_type == StatisticsType.minimum:
            return min(x, y)
        elif stats_type == StatisticsType.maximum:
            return max(x, y)
        elif stats_type == StatisticsType.min_max_range:
            return min(x[0], y[0]), max(x[1], y[1])

    @staticmethod
    def compute_statistics(stats_type: str, accumulator: Any):
        if stats_type in (
            StatisticsType.unique,
            StatisticsType.frequency,
            StatisticsType.minimum,
            StatisticsType.maximum,
            StatisticsType.min_max_range,
        ):
            return accumulator
        elif stats_type == StatisticsType.mean:
            # total, count
            return accumulator[0] / accumulator[1]
        elif stats_type == StatisticsType.variance:
            # sum(x^2)/n - (sum(x)/n)^2
            return (accumulator[1] / accumulator[2]) - (
                accumulator[0] / accumulator[2]
            ) ** 2
        elif StatisticsType.standard_deviation:
            variance = (accumulator[1] / accumulator[2]) - (
                accumulator[0] / accumulator[2]
            ) ** 2
            # sum(x), sum(x^2), count
            return np.sqrt(variance)

    def create_accumulator(self) -> Dict[Tuple[str, str], Any]:
        accumulators = {}
        for s in self.statistics:
            key = (s.feature, str(s.type))
            value = self.get_accumulator(s.type)
            accumulators[key] = value
        return accumulators

    def add_input(
        self,
        accumulator: Dict[Tuple[str, str], Any],
        state: Dict[Tuple[str, str], Any],
    ):
        result = {}
        for s in self.statistics:
            key = (s.feature, str(s.type))
            value1 = accumulator[key]
            value2 = state[key]
            result[key] = self.reduce_accumulator(
                s.type, value1, value2
            )
        return result

    def merge_accumulators(
        self, accumulators: Iterable[Dict[Tuple[str, str], Any]]
    ):
        accumulators = iter(accumulators)
        result = next(accumulators)
        for accumulator in accumulator:
            for feature_stat in self.statistics:
                feature, stat = list(feature_stat.items())[0]
                key = (feature, str(stat))
                value1 = accumulator[key]
                value2 = result[key]
                result[key] = self.reduce_accumulator(
                    stat, value1, value2
                )
        return result

    def extract_output(
        self, accumulator: Dict[Tuple[str, str], Any]
    ) -> List[Tuple[Tuple[str, str], Any]]:
        result = []
        for feature_stat in self.statistics:
            feature, stat = list(feature_stat.items())[0]
            key = (feature, str(stat))
            value = self.compute_statistics(
                stat, accumulator[key]
            )
            result.append(key, value)
        return result


class StatisticsGenWriter(beam.PTransform):
    def __init__(self, statistics_output: str):
        self.statistics_output = statistics_output

    def expand(self, pcoll):
        return (
            pcoll
            | "Flatten key-specific combiners" >> beam.Flatten()
            | "Map approx statistics to dict"
            >> beam.Map(lambda x: (x[0][0], {x[0][1]: x[1]}))
            | f"Combine approximate all groups"
            >> beam.CombinePerKey(CombineToJson("dict"))
            | f"Format all statistics"
            >> beam.Map(lambda x: {x[0]: x[1]})
            | f"Combine all statistics"
            >> beam.CombineGlobally(CombineToJson("string"))
            | f"Write statistics results"
            >> beam.io.WriteToText(
                self.statistics_output,
                num_shards=1,
                shard_name_template="",
            )
        )


def run_example_gen_data_split_pipeline(
    split_configs: DataSplitConfigs,
    statistics_result: str = None,
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
):
    options = PipelineOptions(flags=beam_pipeline_args)
    with beam.Pipeline(options=options) as pipeline:
        # Run a pipeline for each split
        for split_name, input_data in split_configs.input_data.items():
            # Input
            pcoll = pipeline | f"Read {split_name}" >> BatchReader(
                input_data
            )

            # Output
            output_data = split_configs.output_data[split_name]
            output_data.is_batched = input_data.batch_size is not None
            pcoll | f"Write {split_name}" >> BatchWriter(output_data)

            if not split_configs.statistics:
                return

            # Check if statistics result path exist
            if statistics_result is None:
                assert hasattr(
                    output_data, "file"
                ), "statistics_result argument is required for non-file output."
                statistics_output = os.path.join(
                    getattr(output_data, "file"), split_name
                )
            else:
                statistics_output = os.path.join(
                    statistics_result, split_name
                )

            # Gathering statistics, separating into
            # approximated and non-approximated metrics
            concrete_statistics, approximated_statistics = (
                StatisticsGenCombiner.separate_stats(
                    split_configs.statistics
                )
            )
            pcoll_stats = (
                pcoll
                | f"Compute Statistics {split_name}"
                >> beam.ParDo(
                    StatisticsGenDoFn(
                        statistics=concrete_statistics,
                        approximated_metrics=False,
                    )
                )
                | f"Combine Statistics {split_name}"
                >> beam.CombineGlobally(
                    StatisticsGenCombiner(
                        statistics=split_configs.statistics
                    )
                )
            )

            # Gathering approximate stats
            stat_features = {}
            for approx_stat in approximated_statistics:
                feature, stat = list(approx_stat.items())[0]
                if stat not in stat_features:
                    stat_features[stat] = set()
                stat_features[stat].add(feature)

            # Making list of list of grouped stats
            pcoll_combined = [
                pcoll_stats
                | f"Flatten regular stats {split_name}"
                >> beam.FlatMap(lambda x: x)
            ]
            for stat, features in stat_features.items():
                grouped_stats = [{feat: stat} for feat in features]
                _approx = (
                    pcoll
                    | f"Compute {stat} {split_name}"
                    >> beam.ParDo(
                        StatisticsGenDoFn(statistics=grouped_stats)
                    )
                    | f"Combine {stat} {split_name}"
                    >> beam.CombinePerKey(
                        StatisticsType.get_approximated_quantile_combiner(
                            stats_type=stat
                        )
                    )
                )
                pcoll_combined.append(_approx)

            # Merge the all the metrics into a dictionary
            tuple(
                pcoll_combined
            ) | f"Write statistics results for {split_name}" >> StatisticsGenWriter(
                statistics_output
            )

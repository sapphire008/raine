from typing import Dict, Literal
import functools
import numpy as np

from linke.dataset.beam import example_pb2


# %% Reading and Writing TFRecords
class TFRecordIOUtils:
    TYPENAME_MAPPING = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list",
    }

    # From: https://github.com/vahidk/tfrecord/blob/main/tfrecord/reader.py
    @classmethod
    def process_feature(
        cls,
        feature: example_pb2.Feature,
        typename: str,
        typename_mapping: dict,
        key: str,
    ):
        # NOTE: We assume that each key in the example has only one field
        # (either "bytes_list", "float_list", or "int64_list")!
        field = feature.ListFields()[0]
        inferred_typename, value = field[0].name, field[1].value

        if typename is not None:
            tf_typename = typename_mapping[typename]
            if tf_typename != inferred_typename:
                reversed_mapping = {
                    v: k for k, v in typename_mapping.items()
                }
                raise TypeError(
                    f"Incompatible type '{typename}' for `{key}` "
                    f"(should be '{reversed_mapping[inferred_typename]}')."
                )

        if inferred_typename == "bytes_list":
            value = np.array(value, dtype=bytes)
        elif inferred_typename == "float_list":
            value = np.array(value, dtype=np.float32)
        elif inferred_typename == "int64_list":
            value = np.array(value, dtype=np.int64)
        return value

    @classmethod
    def extract_feature_dict(
        cls, features, description, typename_mapping
    ):
        if isinstance(features, example_pb2.FeatureLists):
            features = features.feature_list

            def get_value(typename, typename_mapping, key):
                feature = features[key].feature
                fn = functools.partial(
                    cls.process_feature,
                    typename=typename,
                    typename_mapping=typename_mapping,
                    key=key,
                )
                return list(map(fn, feature))

        elif isinstance(features, example_pb2.Features):
            features = features.feature

            def get_value(typename, typename_mapping, key):
                return cls.process_feature(
                    features[key], typename, typename_mapping, key
                )

        else:
            raise TypeError(
                f"Incompatible type: features should be either of type "
                f"example_pb2.Features or example_pb2.FeatureLists and "
                f"not {type(features)}"
            )

        all_keys = list(features.keys())

        if description is None or len(description) == 0:
            description = dict.fromkeys(all_keys, None)
        elif isinstance(description, list):
            description = dict.fromkeys(description, None)

        processed_features = {}
        for key, typename in description.items():
            if key not in all_keys:
                raise KeyError(
                    f"Key {key} doesn't exist (select from {all_keys})!"
                )

            processed_features[key] = get_value(
                typename, typename_mapping, key
            )

        return processed_features

    # From: https://github.com/vahidk/tfrecord/blob/main/tfrecord/writer.py
    @classmethod
    def deserialize_tf_example(
        cls,
        inputs: bytes,
        schema: Dict[str, Literal["byte", "int", "float"]],
    ) -> Dict:
        """Parse a byte string loaded from tfrecord into a dictionary of features."""
        example = example_pb2.Example()
        example.ParseFromString(inputs)  # protobuf with data
        return cls.extract_feature_dict(
            example.features, schema, cls.TYPENAME_MAPPING
        )

    @classmethod
    def serialize_tf_example(
        cls,
        inputs: Dict,
        schema: Dict[str, Literal["byte", "int", "float"]],
    ) -> bytes:
        """Serialize a single record dict into a tf_example byte string."""
        feature_map = {
            "byte": lambda f: example_pb2.Feature(
                bytes_list=example_pb2.BytesList(value=f)
            ),
            "float": lambda f: example_pb2.Feature(
                float_list=example_pb2.FloatList(value=f)
            ),
            "int": lambda f: example_pb2.Feature(
                int64_list=example_pb2.Int64List(value=f)
            ),
        }

        def serialize(value, dtype):
            if not isinstance(value, (list, tuple, np.ndarray)):
                if isinstance(value, str):
                    value = value.encode()
                value = [value]
            return feature_map[dtype](value)

        features = {
            key: serialize(value, schema[key])
            for key, value in inputs.items()
        }
        example_proto = example_pb2.Example(
            features=example_pb2.Features(feature=features)
        )
        return example_proto.SerializeToString()

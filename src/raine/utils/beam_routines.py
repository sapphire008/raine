"""Set of beam routines shared by multiple components."""
from typing import Literal
import json
import apache_beam as beam

class CombineToJson(beam.CombineFn):
    def __init__(self, output_format: Literal["string", "dict"] = "string"):
        self.output_format = output_format

    def create_accumulator(self):
        return {}  # start with an empty list

    def add_input(self, accumulator, input):
        accumulator.update(input)  # add input to the list
        return accumulator

    def merge_accumulators(self, accumulators):
        result = {}
        for accumulator in accumulators:
            result.update(accumulator)  # combine lists
        return result

    def extract_output(self, accumulator):
        # convert list to JSON string
        if self.output_format == "string":
            return json.dumps(accumulator, indent=4)
        else:  # dict
            return accumulator
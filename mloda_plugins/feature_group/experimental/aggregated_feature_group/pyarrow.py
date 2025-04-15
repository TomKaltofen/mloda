"""
PyArrow implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class PyArrowAggregatedFeatureGroup(AggregatedFeatureGroup):
    """
    PyArrow implementation of aggregated feature group.

    Supports multiple aggregation types in a single class.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with PyArrow."""
        return {PyarrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform aggregations using PyArrow.

        Processes all requested features, determining the aggregation type
        and source feature from each feature name.

        Adds the aggregated results to the input Table.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            aggregation_type = cls.get_aggregation_type(feature_name)
            source_feature = cls.mloda_source_feature(feature_name)

            # Check if source feature exists in the table
            if source_feature not in data.schema.names:
                raise ValueError(f"Source feature '{source_feature}' not found in data")

            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            # Apply the appropriate aggregation function
            result = cls._perform_aggregation(data, aggregation_type, source_feature)

            # Create an array with the aggregated result repeated for each row
            repeat_count = data.num_rows
            repeated_result = pa.array([result] * repeat_count)

            # Add the new column to the table
            data = data.append_column(feature_name, repeated_result)

        # Return the modified Table
        return data

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Perform the aggregation using PyArrow compute functions.

        Args:
            data: The PyArrow Table
            aggregation_type: The type of aggregation to perform
            source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        # Get the column to aggregate
        column = data.column(mloda_source_feature)

        if aggregation_type == "sum":
            return pc.sum(column).as_py()
        elif aggregation_type == "min":
            return pc.min(column).as_py()
        elif aggregation_type == "max":
            return pc.max(column).as_py()
        elif aggregation_type in ["avg", "mean"]:
            return pc.mean(column).as_py()
        elif aggregation_type == "count":
            return pc.count(column).as_py()
        elif aggregation_type == "std":
            return pc.stddev(column).as_py()
        elif aggregation_type == "var":
            return pc.variance(column).as_py()
        elif aggregation_type == "median":
            # PyArrow doesn't have a direct median function
            # We can approximate it using quantile with q=0.5
            # quantile returns an array, so we need to extract the first value
            result = pc.quantile(column, q=0.5)
            return result[0].as_py()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

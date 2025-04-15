"""
Integration tests for time window feature groups.
"""

from typing import List
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup

from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.test_time_window_utils import (
    PandasTimeWindowTestDataCreator,
    PyArrowTimeWindowTestDataCreator,
    validate_time_window_features,
)


# List of time window features to test
TIME_WINDOW_FEATURES: List[Feature | str] = [
    "avg_3_day_window__temperature",  # 3-day average temperature
    "max_5_day_window__humidity",  # 5-day maximum humidity
    "min_2_day_window__pressure",  # 2-day minimum pressure
    "sum_4_day_window__wind_speed",  # 4-day sum of wind speed
]


class TestTimeWindowPandasIntegration:
    """Integration tests for the time window feature group using Pandas."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PandasTimeWindowTestDataCreator, PandasTimeWindowFeatureGroup}
        )

        # Run the API with multiple time window features
        result = mlodaAPI.run_all(
            [
                "temperature",  # Source data
                "avg_3_day_window__temperature",  # 3-day average temperature
                "max_5_day_window__humidity",  # 5-day maximum humidity
                "min_2_day_window__pressure",  # 2-day minimum pressure
                "sum_4_day_window__wind_speed",  # 4-day sum of wind speed
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for time window features

        # Find the DataFrame with the time window features
        window_df = None
        for df in result:
            if "avg_3_day_window__temperature" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)


class TestTimeWindowPyArrowIntegration:
    """Integration tests for the time window feature group using PyArrow."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PyArrowTimeWindowTestDataCreator, PyArrowTimeWindowFeatureGroup}
        )

        # Run the API with multiple time window features
        result = mlodaAPI.run_all(
            [
                "temperature",  # Source data
                "avg_3_day_window__temperature",  # 3-day average temperature
                "max_5_day_window__humidity",  # 5-day maximum humidity
                "min_2_day_window__pressure",  # 2-day minimum pressure
                "sum_4_day_window__wind_speed",  # 4-day sum of wind speed
            ],
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for time window features

        # Find the Table with the time window features
        window_table = None
        for table in result:
            if "avg_3_day_window__temperature" in table.schema.names:
                window_table = table
                break

        assert window_table is not None, "Table with time window features not found"

        # Convert PyArrow Table to Pandas DataFrame for validation
        window_df = window_table.to_pandas()

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)

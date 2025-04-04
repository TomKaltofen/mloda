import pandas as pd
import pytest
from typing import Any, Optional

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import BaseAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "sales": [100, 200, 300, 400, 500],
            "quantity": [10, 20, 30, 40, 50],
            "price": [10.0, 9.5, 9.0, 8.5, 8.0],
            "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
            "customer_rating": [4, 5, 3, 4, 5],
        }
    )


@pytest.fixture
def feature_set_sum() -> FeatureSet:
    """Create a feature set with a sum aggregation feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("sum_aggr_sales"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple aggregation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("sum_aggr_sales"))
    feature_set.add(Feature("avg_aggr_price"))
    feature_set.add(Feature("min_aggr_discount"))
    feature_set.add(Feature("max_aggr_customer_rating"))
    return feature_set


class TestBaseAggregatedFeatureGroup:
    """Tests for the BaseAggregatedFeatureGroup class."""

    def test_get_aggregation_type(self) -> None:
        """Test extraction of aggregation type from feature name."""
        assert BaseAggregatedFeatureGroup.get_aggregation_type("sum_aggr_sales") == "sum"
        assert BaseAggregatedFeatureGroup.get_aggregation_type("min_aggr_quantity") == "min"
        assert BaseAggregatedFeatureGroup.get_aggregation_type("max_aggr_price") == "max"
        assert BaseAggregatedFeatureGroup.get_aggregation_type("avg_aggr_discount") == "avg"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_aggregation_type("invalid_feature_name")

        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_aggregation_type("_aggr_sales")

        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_aggregation_type("sum_aggr_")

    def test_get_source_feature(self) -> None:
        """Test extraction of source feature from feature name."""
        assert BaseAggregatedFeatureGroup.get_source_feature("sum_aggr_sales") == "sales"
        assert BaseAggregatedFeatureGroup.get_source_feature("min_aggr_quantity") == "quantity"
        assert BaseAggregatedFeatureGroup.get_source_feature("max_aggr_price") == "price"
        assert BaseAggregatedFeatureGroup.get_source_feature("avg_aggr_discount") == "discount"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_source_feature("invalid_feature_name")

        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_source_feature("sum_aggr_")

        with pytest.raises(ValueError):
            BaseAggregatedFeatureGroup.get_source_feature("_aggr_sales")

    def test_supports_aggregation_type(self) -> None:
        """Test _supports_aggregation_type method."""
        # Test with supported aggregation types
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("sum")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("min")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("max")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("avg")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("mean")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("count")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("std")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("var")
        assert BaseAggregatedFeatureGroup._supports_aggregation_type("median")

        # Test with unsupported aggregation type
        assert not BaseAggregatedFeatureGroup._supports_aggregation_type("unsupported")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria("sum_aggr_sales", options)
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria("min_aggr_quantity", options)
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria("max_aggr_price", options)
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria("avg_aggr_discount", options)

        # Test with FeatureName objects
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria(FeatureName("sum_aggr_sales"), options)
        assert BaseAggregatedFeatureGroup.match_feature_group_criteria(FeatureName("min_aggr_quantity"), options)

        # Test with invalid feature names
        assert not BaseAggregatedFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not BaseAggregatedFeatureGroup.match_feature_group_criteria("sum_invalid_sales", options)
        assert not BaseAggregatedFeatureGroup.match_feature_group_criteria("invalid_aggr_sales", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = BaseAggregatedFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("sum_aggr_sales"))
        assert input_features == {Feature("sales")}

        input_features = feature_group.input_features(options, FeatureName("min_aggr_quantity"))
        assert input_features == {Feature("quantity")}

        input_features = feature_group.input_features(options, FeatureName("max_aggr_price"))
        assert input_features == {Feature("price")}

        input_features = feature_group.input_features(options, FeatureName("avg_aggr_discount"))
        assert input_features == {Feature("discount")}


class TestPandasAggregatedFeatureGroup:
    """Tests for the PandasAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasAggregatedFeatureGroup.compute_framework_rule() == {PandasDataframe}

    def test_perform_aggregation_sum(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with sum aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "sum", "sales")
        assert result == 1500  # Sum of [100, 200, 300, 400, 500]

    def test_perform_aggregation_min(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with min aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "min", "sales")
        assert result == 100  # Min of [100, 200, 300, 400, 500]

    def test_perform_aggregation_max(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with max aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "max", "sales")
        assert result == 500  # Max of [100, 200, 300, 400, 500]

    def test_perform_aggregation_avg(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with avg aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "avg", "sales")
        assert result == 300  # Avg of [100, 200, 300, 400, 500]

    def test_perform_aggregation_mean(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with mean aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "mean", "sales")
        assert result == 300  # Mean of [100, 200, 300, 400, 500]

    def test_perform_aggregation_count(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with count aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "count", "sales")
        assert result == 5  # Count of [100, 200, 300, 400, 500]

    def test_perform_aggregation_std(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with std aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "std", "sales")
        assert abs(result - 158.11) < 0.1  # Std of [100, 200, 300, 400, 500]

    def test_perform_aggregation_var(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "var", "sales")
        assert abs(result - 25000) < 0.1  # Var of [100, 200, 300, 400, 500]

    def test_perform_aggregation_median(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with median aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "median", "sales")
        assert result == 300  # Median of [100, 200, 300, 400, 500]

    def test_perform_aggregation_invalid(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with invalid aggregation type."""
        with pytest.raises(ValueError):
            PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "invalid", "sales")

    def test_calculate_feature_single(self, sample_dataframe: pd.DataFrame, feature_set_sum: FeatureSet) -> None:
        """Test calculate_feature method with a single aggregation."""
        result = PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set_sum)

        # Check that the result contains the original data plus the aggregated feature
        assert "sum_aggr_sales" in result.columns
        assert result["sum_aggr_sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # Check that the original data is preserved
        assert "sales" in result.columns
        assert "quantity" in result.columns
        assert "price" in result.columns
        assert "discount" in result.columns
        assert "customer_rating" in result.columns

    def test_calculate_feature_multiple(self, sample_dataframe: pd.DataFrame, feature_set_multiple: FeatureSet) -> None:
        """Test calculate_feature method with multiple aggregations."""
        result = PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set_multiple)

        # Check that the result contains all aggregated features
        assert "sum_aggr_sales" in result.columns
        assert result["sum_aggr_sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr_price" in result.columns
        assert result["avg_aggr_price"].iloc[0] == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr_discount" in result.columns
        assert result["min_aggr_discount"].iloc[0] == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr_customer_rating" in result.columns
        assert result["max_aggr_customer_rating"].iloc[0] == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in result.columns
        assert "quantity" in result.columns
        assert "price" in result.columns
        assert "discount" in result.columns
        assert "customer_rating" in result.columns

    def test_calculate_feature_missing_source(self, sample_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr_missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = BaseAggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            BaseAggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("min_aggr_sales"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            BaseAggregatedFeatureGroup.AGGREGATION_TYPES = original_types


class TestAggPandasIntegration:
    """Integration tests for the aggregated feature group using DataCreator."""

    def test_aggregation_with_data_creator(self) -> None:
        """Test aggregation features with mlodaAPI using DataCreator."""

        # Create a feature group that uses DataCreator to provide test data
        class TestDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator({"sales", "quantity", "price", "discount", "customer_rating"})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Return the test data
                return {
                    "sales": [100, 200, 300, 400, 500],
                    "quantity": [10, 20, 30, 40, 50],
                    "price": [10.0, 9.5, 9.0, 8.5, 8.0],
                    "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
                    "customer_rating": [4, 5, 3, 4, 5],
                }

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestDataCreator, PandasAggregatedFeatureGroup})

        # Run the API with multiple aggregation features
        result = mlodaAPI.run_all(
            [
                "sales",  # Source data
                "sum_aggr_sales",  # Sum of sales
                "avg_aggr_price",  # Average price
                "min_aggr_discount",  # Minimum discount
                "max_aggr_customer_rating",  # Maximum customer rating
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for aggregated features

        # Find the DataFrame with the aggregated features
        agg_df = None
        for df in result:
            if "sum_aggr_sales" in df.columns:
                agg_df = df
                break

        assert agg_df is not None, "DataFrame with aggregated features not found"

        # Verify the aggregated features
        assert "sum_aggr_sales" in agg_df.columns
        assert agg_df["sum_aggr_sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr_price" in agg_df.columns
        assert agg_df["avg_aggr_price"].iloc[0] == 9.0  # Average of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr_discount" in agg_df.columns
        assert agg_df["min_aggr_discount"].iloc[0] == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr_customer_rating" in agg_df.columns
        assert agg_df["max_aggr_customer_rating"].iloc[0] == 5  # Max of [4, 5, 3, 4, 5]

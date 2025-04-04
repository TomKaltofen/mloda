import pyarrow as pa
import pytest
from typing import Any, Optional

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import BaseAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup


@pytest.fixture
def sample_table() -> pa.Table:
    """Create a sample PyArrow Table for testing."""
    return pa.table(
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


class TestPyArrowAggregatedFeatureGroup:
    """Tests for the PyArrowAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PyArrowAggregatedFeatureGroup.compute_framework_rule() == {PyarrowTable}

    def test_perform_aggregation_sum(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with sum aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "sum", "sales")
        assert result == 1500  # Sum of [100, 200, 300, 400, 500]

    def test_perform_aggregation_min(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with min aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "min", "sales")
        assert result == 100  # Min of [100, 200, 300, 400, 500]

    def test_perform_aggregation_max(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with max aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "max", "sales")
        assert result == 500  # Max of [100, 200, 300, 400, 500]

    def test_perform_aggregation_avg(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with avg aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "avg", "sales")
        assert result == 300  # Avg of [100, 200, 300, 400, 500]

    def test_perform_aggregation_mean(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with mean aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "mean", "sales")
        assert result == 300  # Mean of [100, 200, 300, 400, 500]

    def test_perform_aggregation_count(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with count aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "count", "sales")
        assert result == 5  # Count of [100, 200, 300, 400, 500]

    def test_perform_aggregation_std(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with std aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "std", "sales")
        # PyArrow uses a different formula for standard deviation than Pandas
        # PyArrow uses the population standard deviation (n), while Pandas uses the sample standard deviation (n-1)
        assert abs(result - 141.42) < 0.1  # Std of [100, 200, 300, 400, 500] with population formula

    def test_perform_aggregation_var(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "var", "sales")
        # PyArrow uses a different formula for variance than Pandas
        # PyArrow uses the population variance (n), while Pandas uses the sample variance (n-1)
        assert abs(result - 20000) < 0.1  # Var of [100, 200, 300, 400, 500] with population formula

    def test_perform_aggregation_median(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with median aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "median", "sales")
        assert result == 300  # Median of [100, 200, 300, 400, 500]

    def test_perform_aggregation_invalid(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with invalid aggregation type."""
        with pytest.raises(ValueError):
            PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "invalid", "sales")

    def test_calculate_feature_single(self, sample_table: pa.Table, feature_set_sum: FeatureSet) -> None:
        """Test calculate_feature method with a single aggregation."""
        result = PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set_sum)

        # Check that the result contains the original data plus the aggregated feature
        assert "sum_aggr_sales" in result.schema.names
        assert result.column("sum_aggr_sales")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        # Check that the original data is preserved
        assert "sales" in result.schema.names
        assert "quantity" in result.schema.names
        assert "price" in result.schema.names
        assert "discount" in result.schema.names
        assert "customer_rating" in result.schema.names

    def test_calculate_feature_multiple(self, sample_table: pa.Table, feature_set_multiple: FeatureSet) -> None:
        """Test calculate_feature method with multiple aggregations."""
        result = PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set_multiple)

        # Check that the result contains all aggregated features
        assert "sum_aggr_sales" in result.schema.names
        assert result.column("sum_aggr_sales")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr_price" in result.schema.names
        assert result.column("avg_aggr_price")[0].as_py() == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr_discount" in result.schema.names
        assert result.column("min_aggr_discount")[0].as_py() == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr_customer_rating" in result.schema.names
        assert result.column("max_aggr_customer_rating")[0].as_py() == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in result.schema.names
        assert "quantity" in result.schema.names
        assert "price" in result.schema.names
        assert "discount" in result.schema.names
        assert "customer_rating" in result.schema.names

    def test_calculate_feature_missing_source(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr_missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = BaseAggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            BaseAggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("min_aggr_sales"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            BaseAggregatedFeatureGroup.AGGREGATION_TYPES = original_types


class TestAggPyArrowIntegration:
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
                # Return the test data as a dictionary (will be converted to PyArrow Table)
                return {
                    "sales": [100, 200, 300, 400, 500],
                    "quantity": [10, 20, 30, 40, 50],
                    "price": [10.0, 9.5, 9.0, 8.5, 8.0],
                    "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
                    "customer_rating": [4, 5, 3, 4, 5],
                }

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestDataCreator, PyArrowAggregatedFeatureGroup})

        # Run the API with multiple aggregation features
        result = mlodaAPI.run_all(
            [
                "sales",  # Source data
                "sum_aggr_sales",  # Sum of sales
                "avg_aggr_price",  # Average price
                "min_aggr_discount",  # Minimum discount
                "max_aggr_customer_rating",  # Maximum customer rating
            ],
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for aggregated features

        # Find the Table with the aggregated features
        agg_table = None
        for table in result:
            if "sum_aggr_sales" in table.schema.names:
                agg_table = table
                break

        assert agg_table is not None, "Table with aggregated features not found"

        # Verify the aggregated features
        assert "sum_aggr_sales" in agg_table.schema.names
        assert agg_table.column("sum_aggr_sales")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr_price" in agg_table.schema.names
        assert agg_table.column("avg_aggr_price")[0].as_py() == 9.0  # Average of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr_discount" in agg_table.schema.names
        assert agg_table.column("min_aggr_discount")[0].as_py() == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr_customer_rating" in agg_table.schema.names
        assert agg_table.column("max_aggr_customer_rating")[0].as_py() == 5  # Max of [4, 5, 3, 4, 5]

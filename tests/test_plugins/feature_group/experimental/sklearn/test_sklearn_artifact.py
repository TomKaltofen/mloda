"""
Tests for SklearnArtifact.
"""

import pytest
from unittest.mock import Mock, patch
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options


class TestSklearnArtifact:
    """Test cases for SklearnArtifact."""

    def test_serialize_deserialize_artifact(self) -> None:
        """Test serialization and deserialization of sklearn artifacts."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        # Create a simple sklearn transformer
        scaler = StandardScaler()
        # Fit with some dummy data
        import numpy as np

        dummy_data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(dummy_data)

        # Create artifact
        artifact = {
            "fitted_transformer": scaler,
            "feature_names": ["feature1", "feature2"],
            "training_timestamp": "2023-01-01T00:00:00",
        }

        # Serialize
        serialized = SklearnArtifact._serialize_artifact(artifact)
        assert isinstance(serialized, str)

        # Deserialize
        deserialized = SklearnArtifact._deserialize_artifact(serialized)

        # Verify contents
        assert "fitted_transformer" in deserialized
        assert "feature_names" in deserialized
        assert "training_timestamp" in deserialized

        assert deserialized["feature_names"] == ["feature1", "feature2"]
        assert deserialized["training_timestamp"] == "2023-01-01T00:00:00"

        # Verify the transformer works
        result = deserialized["fitted_transformer"].transform(dummy_data)
        assert result.shape == (3, 2)

    def test_serialize_artifact_missing_joblib(self) -> None:
        """Test serialization when joblib is not available."""
        with patch.dict("sys.modules", {"joblib": None}):
            with pytest.raises(ImportError, match="joblib is required"):
                SklearnArtifact._serialize_artifact({"fitted_transformer": Mock()})

    def test_deserialize_artifact_missing_joblib(self) -> None:
        """Test deserialization when joblib is not available."""
        with patch.dict("sys.modules", {"joblib": None}):
            with pytest.raises(ImportError, match="joblib is required"):
                SklearnArtifact._deserialize_artifact('{"fitted_transformer": "dummy"}')

    def test_custom_saver(self) -> None:
        """Test custom_saver method."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            import tempfile
            import os
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        features = Mock(spec=FeatureSet)
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "test_custom_saver_feature"
        features.options = Mock()
        features.options.data = {}

        artifact = {"fitted_transformer": StandardScaler(), "feature_names": ["test_custom_saver_feature"]}

        try:
            result = SklearnArtifact.custom_saver(features, artifact)
            assert isinstance(result, str)
            # Verify file was created
            assert os.path.exists(result)
        finally:
            # Clean up
            try:
                file_path = SklearnArtifact._get_artifact_file_path(features)
                if file_path.exists():
                    file_path.unlink()
            except Exception:  # nosec
                pass

    def test_custom_loader_no_options(self) -> None:
        """Test custom_loader when no options are available."""
        features = Mock(spec=FeatureSet)
        features.options = None
        features.name_of_one_feature = None

        result = SklearnArtifact.custom_loader(features)
        assert result is None

    def test_custom_loader_no_artifact(self) -> None:
        """Test custom_loader when no artifact is stored."""
        features = Mock(spec=FeatureSet)
        features.options = Mock()
        features.options.data = {}
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "test_no_artifact_feature_unique"

        result = SklearnArtifact.custom_loader(features)
        assert result is None

    def test_custom_loader_with_artifact(self) -> None:
        """Test custom_loader with stored artifact."""
        # Skip test if sklearn/joblib not available
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            import os
        except ImportError:
            pytest.skip("scikit-learn or joblib not available")

        # Create and serialize an artifact
        scaler = StandardScaler()
        import numpy as np

        dummy_data = np.array([[1, 2], [3, 4]])
        scaler.fit(dummy_data)

        artifact = {"fitted_transformer": scaler, "feature_names": ["feature1", "feature2"]}

        # Mock features with unique name
        features = Mock(spec=FeatureSet)
        features.options = Mock()
        features.options.data = {}
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "test_with_artifact_feature_unique"

        try:
            # First save the artifact
            saved_path = SklearnArtifact.custom_saver(features, artifact)
            assert os.path.exists(saved_path)  # type: ignore

            # Then load it
            result = SklearnArtifact.custom_loader(features)

            assert result is not None
            assert "fitted_transformer" in result
            assert "feature_names" in result
            assert result["feature_names"] == ["feature1", "feature2"]
        finally:
            # Clean up
            try:
                file_path = SklearnArtifact._get_artifact_file_path(features)
                if file_path.exists():
                    file_path.unlink()
            except Exception:  # nosec
                pass

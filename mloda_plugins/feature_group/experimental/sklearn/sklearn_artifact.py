"""
Artifact for storing fitted scikit-learn transformers and estimators.
"""

import json
import base64
import os
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class SklearnArtifact(BaseArtifact):
    """
    Artifact for storing fitted scikit-learn transformers and estimators.

    This artifact stores fitted scikit-learn objects using joblib serialization,
    allowing for efficient persistence and reuse of trained models and transformers.

    The artifact contains:
    - fitted_transformer: The fitted scikit-learn object
    - feature_names: Names of input features used during fitting
    - training_metadata: Information about the training process
    """

    @classmethod
    def _serialize_artifact(cls, artifact: Dict[str, Any]) -> str:
        """
        Serialize the artifact to a JSON string.

        Args:
            artifact: The artifact to serialize

        Returns:
            A JSON string representation of the artifact
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        # Create a copy of the artifact
        serializable_artifact = {}

        # Serialize each component of the artifact
        for key, value in artifact.items():
            if key == "fitted_transformer":
                # Use joblib to serialize the fitted transformer
                import io

                buffer = io.BytesIO()
                joblib.dump(value, buffer)
                serializable_artifact[key] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif key == "feature_names":
                # Convert list to JSON
                serializable_artifact[key] = json.dumps(value)
            elif key == "training_timestamp":
                # Convert timestamp to string
                serializable_artifact[key] = str(value)
            else:
                # Keep other values as is
                serializable_artifact[key] = value

        # Convert the entire artifact to a JSON string
        return json.dumps(serializable_artifact)

    @classmethod
    def _deserialize_artifact(cls, serialized_artifact: str) -> Dict[str, Any]:
        """
        Deserialize the artifact from a JSON string.

        Args:
            serialized_artifact: The JSON string to deserialize

        Returns:
            The deserialized artifact
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        # Parse the JSON string
        serializable_artifact = json.loads(serialized_artifact)

        # Create a new artifact
        artifact = {}

        # Deserialize each component of the artifact
        for key, value in serializable_artifact.items():
            if key == "fitted_transformer":
                # Use joblib to deserialize the fitted transformer
                import io

                buffer = io.BytesIO(base64.b64decode(value))
                artifact[key] = joblib.load(buffer)
            elif key == "feature_names":
                # Parse JSON list
                artifact[key] = json.loads(value)
            elif key == "training_timestamp":
                # Keep timestamp as string for now
                artifact[key] = value
            else:
                # Keep other values as is
                artifact[key] = value

        return artifact

    @classmethod
    def _get_artifact_file_path(cls, features: FeatureSet) -> Path:
        """
        Generate a file path for storing the artifact.

        Args:
            features: The feature set

        Returns:
            Path object for the artifact file
        """
        if features.name_of_one_feature is None:
            raise ValueError("Feature name is required for artifact storage")

        # Get storage path from options or use default temp directory
        storage_path = None
        if features.options:
            storage_path = features.options.data.get("artifact_storage_path")

        if storage_path is None:
            storage_path = tempfile.gettempdir()

        # Create a unique filename based on feature name and configuration
        feature_name = features.name_of_one_feature.name

        # Create a hash of the feature configuration for uniqueness
        # Exclude artifact-related keys to ensure consistent hashing
        config_data = {}
        if features.options:
            config_data = {
                k: v
                for k, v in features.options.data.items()
                if not k.startswith(feature_name) and k != "artifact_storage_path"
            }

        # Convert non-serializable objects for hashing
        serializable_config = {}
        for k, v in config_data.items():
            if isinstance(v, frozenset):
                # Convert frozenset to sorted list for consistent hashing
                serializable_config[k] = sorted(list(v))
            else:
                serializable_config[k] = v

        config_str = json.dumps(serializable_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]

        filename = f"sklearn_artifact_{feature_name}_{config_hash}.joblib"

        # Ensure the directory exists
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        return storage_dir / filename

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Save the sklearn artifact to file.

        Args:
            features: The feature set
            artifact: The sklearn artifact to save (dict containing fitted_transformer, etc.)

        Returns:
            The file path where the artifact was saved
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        file_path = cls._get_artifact_file_path(features)
        print(f"DEBUG: SklearnArtifact.custom_saver called - saving to {file_path}")

        # Save the artifact directly using joblib (more efficient than JSON serialization)
        joblib.dump(artifact, file_path)

        print(f"DEBUG: SklearnArtifact.custom_saver completed - artifact saved to {file_path}")

        return str(file_path)

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        Load the sklearn artifact from file.

        Args:
            features: The feature set

        Returns:
            The loaded artifact (dict containing fitted_transformer, etc.)
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for SklearnArtifact. Install with: pip install joblib")

        if features.name_of_one_feature is None:
            print("DEBUG: SklearnArtifact.custom_loader - no feature name, returning None")
            return None

        file_path = cls._get_artifact_file_path(features)

        print(f"DEBUG: SklearnArtifact.custom_loader called - looking for {file_path}")

        # Check if the artifact file exists
        if not file_path.exists():
            print(f"DEBUG: SklearnArtifact.custom_loader - file does not exist: {file_path}")
            return None

        try:
            # Load the artifact using joblib
            artifact = joblib.load(file_path)
            print(f"DEBUG: SklearnArtifact.custom_loader - successfully loaded artifact from {file_path}")
            return artifact
        except Exception as e:
            # If loading fails, return None to trigger re-creation
            print(f"Warning: Failed to load artifact from {file_path}: {e}")
            return None

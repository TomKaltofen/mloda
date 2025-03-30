from abc import ABC
from typing import Any, Optional, final

from mloda_core.abstract_plugins.components.feature_set import FeatureSet


class BaseArtifact(ABC):
    """
    Abstract base class for handling artifacts within a FeatureSet.

    Artifacts are data generated by a feature group and can be used by other feature groups.
    This is necessary for scenarios such as embeddings, where the output of one feature group
    serves as an input for another, enabling complex data transformations and feature engineering
    workflows.

    This class provides mechanisms to load and save artifacts with validation and customization points.
    It ensures that artifacts are correctly managed, validated, and can be customized as per the
    specific requirements of different feature groups.
    """

    @final
    @classmethod
    def load(cls, features: FeatureSet) -> Optional[Any]:
        """
        Loads an artifact from the given config of the feature set, when the custom_loader is not overwritten.
        If the custom_loader is overwritten, this method will call the custom_loader and return the result.

        This method is crucial for data science processes where the reuse of previously computed data
        (artifacts) is necessary. For example, in machine learning pipelines,
        precomputed embeddings can be loaded and reused across different stages of the pipeline.
        """

        if features.artifact_to_load is None:
            return None

        cls._validate(features)

        loaded_artifact = cls.custom_loader(features)

        if loaded_artifact is None:
            raise ValueError("No artifact to load although it was requested.")

        return loaded_artifact

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        In the default case, it loads an artifact from the given config of the features.

        However, you can overwrite this method to load the artifact by any means necessary.
        """
        return features.options.get(features.name_of_one_feature.name)  # type: ignore

    @final
    @classmethod
    def save(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        The default implementation is to return the artifact, as then the framework will handle it.

        In case that the data is larger or cannot be pickled, you can overwrite the custom_saver function to save the artifact by any means necessary.
        In that case, the return value would not be the artifact, but any metadata to identify this artifact.

        Returns:
            Optional[Any]: The artifact or metadata identifying the artifact, depending on implementation.
                           Default behavior is to return the artifact, as then the framework will handle it.

        """
        if features.artifact_to_save is None:
            return None

        cls._validate(features)

        artifact = cls.custom_saver(features, artifact)

        if artifact is None:
            raise ValueError("No artifact to save although it was requested.")

        return artifact

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Subclasses can override this method to implement custom saving logic.

        The default implementation is to return the artifact, as then the framework will handle it.
        """
        return artifact

    @staticmethod
    def _validate(features: FeatureSet) -> None:
        """
        Validates that the FeatureSet has the necessary attributes set.
        """
        if features.options is None:
            raise ValueError("No options set. This should only be called after adding a feature.")

        if features.name_of_one_feature is None:
            raise ValueError("Feature name missing in feature set.")

    @classmethod
    @final
    def get_class_name(cls) -> str:
        return cls.__name__

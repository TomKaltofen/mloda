import pytest
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


def test_feature_equals() -> None:
    feature1 = Feature(name="Feature1", options={"option1": 1})
    feature2 = Feature(name="Feature1", options={"option1": 1})
    feature3 = Feature(name="Feature2", options={"option2": 2})
    assert feature1 == feature2
    assert feature1 != feature3


def test_feature_set_compute_framework() -> None:
    feature = Feature(name="Feature1")

    # Test if frameworks are not found
    with pytest.raises(ValueError):
        feature.set_compute_framework(None, "ComputeFrameworkNotExists")
    with pytest.raises(ValueError):
        feature.set_compute_framework("ComputeFrameworkNotExists", None)

    # Test when neither compute_framework nor compute_framework_options are set
    assert feature.set_compute_framework(None, None) is None

    # Test valid cases
    valid_fw_subclases = get_all_subclasses(ComputeFrameWork)
    result1 = feature.set_compute_framework(next(iter(valid_fw_subclases)).get_class_name(), None)
    result2 = feature.set_compute_framework(None, next(iter(valid_fw_subclases)).get_class_name())
    assert result1 == result2 is not None


def test_feature_set_domain() -> None:
    feature = Feature(name="Feature1")

    # Test when domain is set
    result = feature.set_domain("example_domain", None)
    assert result.name == "example_domain"  # type: ignore

    # Test when domain_options is set
    result = feature.set_domain(None, "example_domain_options")
    assert result.name == "example_domain_options"  # type: ignore

    # Test when neither domain nor domain_options are set
    assert feature.set_domain(None, None) is None

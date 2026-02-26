"""Tests to verify mloda dependencies and demo plugin can be imported."""


def test_mloda_provider_imports() -> None:
    """Verify mloda.provider module imports work."""
    from mloda.provider import FeatureGroup, ComputeFramework

    assert FeatureGroup is not None
    assert ComputeFramework is not None


def test_mloda_core_imports() -> None:
    """Verify mloda.core module imports work."""
    from mloda.core.abstract_plugins.function_extender import Extender

    assert Extender is not None


def test_mloda_testing_imports() -> None:
    """Verify mloda.testing module imports work."""
    from mloda.testing import FeatureGroupTestBase

    assert FeatureGroupTestBase is not None


def test_employee_data_features_import() -> None:
    """Verify the demo FeatureGroup can be imported and extends FeatureGroup."""
    from mloda.provider import FeatureGroup

    from ptc_mloda_demo.feature_groups.sample_data.sample_data_features import EmployeeDataFeatures

    assert issubclass(EmployeeDataFeatures, FeatureGroup)
    assert len(EmployeeDataFeatures.feature_names_supported()) == 5

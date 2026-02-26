"""Tests for EmployeeDataFeatures — 3-level testing per guide 10-testing-guide.md."""

from typing import Union, cast

import pandas as pd
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginLoader, mloda

from ptc_mloda_demo.feature_groups.sample_data.sample_data_features import EMPLOYEE_FEATURES, EmployeeDataFeatures


# ---------------------------------------------------------------------------
# Level 1: Unit tests — class structure, matching logic
# ---------------------------------------------------------------------------


def test_extends_feature_group() -> None:
    assert issubclass(EmployeeDataFeatures, FeatureGroup)


def test_input_data_returns_data_creator() -> None:
    inp = EmployeeDataFeatures.input_data()
    assert isinstance(inp, DataCreator)
    assert isinstance(inp, BaseInputData)


def test_feature_names_supported() -> None:
    names = EmployeeDataFeatures.feature_names_supported()
    assert names == EMPLOYEE_FEATURES


def test_match_feature_group_criteria() -> None:
    opts = Options()
    for name in EMPLOYEE_FEATURES:
        assert EmployeeDataFeatures.match_feature_group_criteria(name, opts)
    assert not EmployeeDataFeatures.match_feature_group_criteria("nonexistent_feature", opts)


# ---------------------------------------------------------------------------
# Level 2: Framework test — calculate_feature returns expected DataFrame
# ---------------------------------------------------------------------------


def test_calculate_feature_returns_dataframe() -> None:
    result = EmployeeDataFeatures.calculate_feature(None, cast(FeatureSet, None))
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == EMPLOYEE_FEATURES
    assert len(result) == 10


# ---------------------------------------------------------------------------
# Level 3: Integration test — mloda.run_all end-to-end
# ---------------------------------------------------------------------------


def test_run_all_integration() -> None:
    PluginLoader.all()
    features: list[Union[Feature, str]] = ["employee_id", "department", "salary"]
    results = mloda.run_all(features, compute_frameworks=["PandasDataFrame"])
    assert len(results) == 1
    df = results[0]
    assert len(df) == 10
    assert {"employee_id", "department", "salary"}.issubset(set(df.columns))

"""Tests for ObservabilityExtender (TDD: red then green)."""

import logging
from typing import Any, Union

import pandas as pd
from mloda.steward import Extender, ExtenderHook
from mloda.user import Feature, PluginLoader
from mloda.user import mloda as mlodaAPI

from ptc_mloda_demo.extenders.observability.observability_extender import ObservabilityExtender

import ptc_mloda_demo.feature_groups.sample_data.sample_data_features  # noqa: F401


# ---------------------------------------------------------------------------
# Level 1: Class structure
# ---------------------------------------------------------------------------


class TestObservabilityExtenderStructure:
    """Verify the extender has the correct class structure."""

    def test_is_subclass_of_extender(self) -> None:
        assert issubclass(ObservabilityExtender, Extender)

    def test_wraps_calculate_feature(self) -> None:
        ext = ObservabilityExtender()
        assert ext.wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def test_is_callable(self) -> None:
        ext = ObservabilityExtender()
        assert callable(ext)


# ---------------------------------------------------------------------------
# Level 2: Wrapping behavior
# ---------------------------------------------------------------------------


class TestObservabilityExtenderBehavior:
    """Verify wrapping: passthrough, timing, logging, shape."""

    def test_passthrough_returns_function_result(self) -> None:
        ext = ObservabilityExtender()
        result = ext(lambda x, y: x + y, 1, 2)
        assert result == 3

    def test_passthrough_dataframe(self) -> None:
        ext = ObservabilityExtender()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ext(lambda: df)
        assert result is df

    def test_logs_execution_time(self, caplog: Any) -> None:
        ext = ObservabilityExtender()
        with caplog.at_level(logging.INFO):
            ext(lambda: "ok")
        time_messages = [r for r in caplog.records if "elapsed" in r.message.lower()]
        assert len(time_messages) >= 1

    def test_logs_result_shape_for_dataframe(self, caplog: Any) -> None:
        ext = ObservabilityExtender()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with caplog.at_level(logging.INFO):
            ext(lambda: df)
        shape_messages = [r for r in caplog.records if "2, 2" in r.message or "(2, 2)" in r.message]
        assert len(shape_messages) >= 1

    def test_does_not_log_shape_for_non_dataframe(self, caplog: Any) -> None:
        ext = ObservabilityExtender()
        with caplog.at_level(logging.INFO):
            ext(lambda: 42)
        shape_messages = [r for r in caplog.records if "shape" in r.message.lower()]
        assert len(shape_messages) == 0


# ---------------------------------------------------------------------------
# Level 3: Integration with mloda.run_all()
# ---------------------------------------------------------------------------


class TestObservabilityExtenderIntegration:
    """Integration: extender works inside mloda.run_all()."""

    def test_run_all_with_observability_extender(self, caplog: Any) -> None:
        PluginLoader.all()
        features: list[Union[Feature, str]] = [Feature.not_typed("employee_id"), Feature.not_typed("salary")]
        with caplog.at_level(logging.INFO):
            results = mlodaAPI.run_all(
                features,
                compute_frameworks=["PandasDataFrame"],
                function_extender={ObservabilityExtender()},
            )
        assert len(results) == 1
        assert "employee_id" in results[0].columns
        assert "salary" in results[0].columns
        elapsed_logs = [r for r in caplog.records if "elapsed" in r.message.lower()]
        assert len(elapsed_logs) >= 1

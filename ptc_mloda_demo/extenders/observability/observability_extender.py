"""Observability extender: logs execution time and result shape for each feature calculation."""

import logging
import time
from typing import Any, Set

import pandas as pd
from mloda.steward import Extender, ExtenderHook

logger = logging.getLogger(__name__)


class ObservabilityExtender(Extender):
    """Wraps calculate_feature to log execution time and result shape.

    Designed for PTC (Programmatic Tool Calling) where the orchestration loop
    is internal to Claude, so observability must be injected into mloda.run_all()
    via function_extender rather than the loop itself.
    """

    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        logger.info("calculate_feature elapsed=%.4fs", elapsed)

        if isinstance(result, pd.DataFrame):
            logger.info("result shape=(%d, %d)", result.shape[0], result.shape[1])

        return result

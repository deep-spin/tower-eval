# -*- coding: utf-8 -*-
from tower_eval.metrics.base.result_handler import MetricResult


class PerplexityResult(MetricResult):
    """
    Perplexity result handler
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

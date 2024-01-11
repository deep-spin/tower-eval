# -*- coding: utf-8 -*-
from tower_eval.metrics.result_handler import MetricResult


class AccuracyResult(MetricResult):
    """
    Accuracy Result Handler.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

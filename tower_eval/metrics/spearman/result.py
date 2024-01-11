# -*- coding: utf-8 -*-
from tower_eval.metrics.result_handler import MetricResult


class SpearmanResult(MetricResult):
    """
    Spearman Correlation Result Handler.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

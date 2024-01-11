# -*- coding: utf-8 -*-
from tower_eval.metrics.result_handler import MetricResult


class ErrorSpanPrecisionResult(MetricResult):
    """
    Error Span Detection Precision Result Handler.
    """

    def __init__(
        self,
        result: list,
    ) -> None:
        super().__init__(result)

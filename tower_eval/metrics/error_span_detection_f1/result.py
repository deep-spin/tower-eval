# -*- coding: utf-8 -*-
from tower_eval.metrics.result_handler import MetricResult


class ErrorSpanF1Result(MetricResult):
    """
    Error Span Detection F1 Result Handler.
    """

    def __init__(
        self,
        result: list,
    ) -> None:
        super().__init__(result)

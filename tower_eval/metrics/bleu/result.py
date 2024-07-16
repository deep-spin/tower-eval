# -*- coding: utf-8 -*-
from tower_eval.metrics.base.result_handler import MetricResult


class BLEUResult(MetricResult):
    """
    BLEU Result Handler.
    TODO: Add the extra information (such as the brevity penalty, scores of different n-grams) to the output.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

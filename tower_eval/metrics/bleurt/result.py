# -*- coding: utf-8 -*-
from tower_eval.metrics.base.result_handler import MetricResult


class BLEURTResult(MetricResult):
    """
    BLEURT Result Handler.
    TODO: Add the segment-level to the output as additional params.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

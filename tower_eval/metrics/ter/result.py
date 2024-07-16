# -*- coding: utf-8 -*-
from tower_eval.metrics.base.result_handler import MetricResult


class TERResult(MetricResult):
    """
    TER Result Handler.
    TODO: Add the extra information (such as the casing, version of the metric, etc) to the output.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

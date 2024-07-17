# -*- coding: utf-8 -*-
from tower_eval.metrics.base.error_span_detection import ErrorSpanDetectionMetric


class ErrorSpanDetectionPrecision(ErrorSpanDetectionMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(key="precision", **kwargs)

    @staticmethod
    def metric_name():
        return "error-span-detection-precision"

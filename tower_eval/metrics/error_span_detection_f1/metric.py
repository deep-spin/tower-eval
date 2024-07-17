# -*- coding: utf-8 -*-
from tower_eval.metrics.base.error_span_detection import ErrorSpanDetectionMetric


class ErrorSpanDetectionF1(ErrorSpanDetectionMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(key="f1", **kwargs)

    @staticmethod
    def metric_name():
        return "error-span-detection-f1"

# -*- coding: utf-8 -*-
from tower_eval.metrics.error_span_detection import (
    ErrorSpanDetectionMetric,
    ErrorSpanDetectionResult,
)


class ErrorSpanDetectionPrecision(ErrorSpanDetectionMetric):
    def __init__(self, severity_mismatch_penalty: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.severity_mismatch_penalty = kwargs.get(
            "severity_mismatch_penalty", severity_mismatch_penalty
        )

    def run(self):
        hypothesis_list, reference_list = self._handle_inputs(
            self.hypothesis_path, self.gold_data_path
        )
        result = ErrorSpanDetectionResult(
            self.evaluate(
                hypothesis_list,
                reference_list,
            )["precision"]
        )
        result.print_result(self.metric_name())

        return result.format_result(self.metric_name())

    @staticmethod
    def metric_name():
        return "error-span-detection-precision"

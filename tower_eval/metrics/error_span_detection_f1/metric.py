# -*- coding: utf-8 -*-
from tower_eval.metrics.error_span_detection import (
    ErrorSpanDetectionMetric,
    ErrorSpanDetectionResult,
)


class ErrorSpanDetectionF1(ErrorSpanDetectionMetric):
    def __init__(self, severity_mismatch_penalty: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.severity_mismatch_penalty = kwargs.get(
            "severity_mismatch_penalty", severity_mismatch_penalty
        )
        self.hyp_type = kwargs.get("hyp_type", "jsonl")

    def run(self, hypothesis_path, gold_data_path) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        reference_list = gold_data["ref"]
        result = ErrorSpanDetectionResult(
            self.evaluate(
                hypotheses,
                reference_list,
            )["f1"]
        )
        result.print_result(self.metric_name())

        return result.format_result(self.metric_name())

    @staticmethod
    def metric_name():
        return "error-span-detection-f1"

# -*- coding: utf-8 -*-
from typing import Dict, List, Union

from tower_eval.metrics.base.result_handler import MetricResult


class F1SequenceResult(MetricResult):
    """
    F1Sequence Result Handler.
    """

    def __init__(
        self,
        result: float,
        tags_f1: Dict[str, float],
        valid_tags: List[str],
    ) -> None:
        super().__init__(result)
        self.tags_f1 = tags_f1
        self.valid_tags = valid_tags

    def print_result(self, metric_name: str, round_to_decimals: int = 4) -> None:
        """Function used to display a particular Metric result.
        :param metric_name: Metric name.
        :param round_to_decimals: decimals that we want to present.
        """
        print(f"{metric_name}: {round(self.result, round_to_decimals)}")
        for tag in self.valid_tags:
            print(f"{tag}: {round(self.tags_f1[tag], round_to_decimals)}")

    def format_result(
        self, metric_name: str, round_to_decimals: int = 4
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Function used to format a particular Metric result.
        :param metric_name: Metric name.
        :param round_to_decimals: decimals that we want to present.
        """
        out = {}
        out[f"{metric_name}"] = round(self.result, round_to_decimals)
        out["valid_tags"] = self.valid_tags
        out[f"{metric_name}_by_tag"] = {
            tag: round(self.tags_f1[tag], round_to_decimals) for tag in self.valid_tags
        }
        return out

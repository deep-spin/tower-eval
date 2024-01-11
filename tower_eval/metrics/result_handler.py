# -*- coding: utf-8 -*-
from abc import ABC


class MetricResult(ABC):
    """
    Abstract class defining a shared interface for all Metric Result Handlers.

    :param result: float value to be displayed.
    """

    def __init__(self, result: float) -> None:
        self.result = result

    def print_result(self, metric_name: str, round_to_decimals: int = 4) -> None:
        """Function used to display a particular Metric result.
        :param metric_name: Metric name.
        :param round_to_decimals: decimals that we want to present.
        """
        if type(self.result) == dict:
            print(
                f'{metric_name}: {round(self.result["system_score"], round_to_decimals)}'
            )
        else:
            print(f"{metric_name}: {round(self.result, round_to_decimals)}")

    def format_result(self, metric_name: str, round_to_decimals: int = 4) -> dict:
        """Function used to format a particular Metric result.
        :param metric_name: Metric name.
        :param round_to_decimals: decimals that we want to present.
        """
        if type(self.result) == dict:
            out = {
                f"{metric_name}": round(self.result["system_score"], round_to_decimals),
                f"{metric_name}_segments": self.result["segments_scores"],
            }
        else:
            out = {f"{metric_name}": round(self.result, round_to_decimals)}
        return out

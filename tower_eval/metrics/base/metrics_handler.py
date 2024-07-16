# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path

from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.utils import list_to_dict, load_jsonl_file, read_lines


class Metric(ABC):
    """Abstract class defining a shared interface for all the Metrics"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, hypothesis_path: Path = None, references_path: Path = None, **kwargs):
        """
        The runner function that performs all the necessary checks on the files,
        applies the needed preprocessing (like lowercasing, tokenization, etc),
        and calls the evaluate method of the class to get the scores.
        """
        pass

    @abstractmethod
    def evaluate(self, hypotheses: list, references: list, **kwargs) -> MetricResult:
        """
        Evaluate function receives the hypotheses and the reference files and returns a MetricResult object.

        :param hypotheses: the path to the hypothese file.
        :param references: the path to the reference file.
        :return: MetricResult object.
        """
        pass

    @abstractmethod
    def process_result(self, result) -> MetricResult:
        """
        Process the result to be ready and complient with the MetricResult format.

        :param result: the raw result produced by the metric
        :return: MetricResult object.
        """
        pass

    @staticmethod
    @abstractmethod
    def metric_name() -> None:
        """Metric name used to address the metric via cli."""
        pass

    @staticmethod
    def _handle_inputs(
        hypotheses: Path,
        data_path: Path,
    ) -> tuple:
        """
        Function to handle input files.
        All inputs will be returned as list of strings.

        :param hypotheses: either the handler to the file storing the hypotheses.
        :param references: either the handler to the file storing the refereneces.
        :param sources: either the handler to the file storing the source sentences.

        :return:
            -  Tuple with hypotheses, references and kwargs
            -  If sources not None, Tuple with hypotheses, references, sources and kwargs
        """
        hypotheses = read_lines(hypotheses, unescape_newline=True)
        # gold data keys depend on the task; e.g., for MT, it will include "ref", for APE "pe"
        gold_data = load_jsonl_file(data_path)
        assert len(hypotheses) == len(
            gold_data
        ), f"The number of hypotheses ({len(hypotheses)}) and rows in the gold data ({len(gold_data)}) should be the same."
        gold_data = list_to_dict(gold_data)

        return hypotheses, gold_data

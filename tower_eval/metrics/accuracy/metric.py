# -*- coding: utf-8 -*-
from typing import List

from loguru import logger
from sklearn.metrics import accuracy_score

from tower_eval.metrics.accuracy.result import AccuracyResult
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.utils import read_lines, text_to_label


class ACCURACY(Metric):
    def __init__(
        self,
        source_type: str,
        source_labels: List[str] = None,
        reference_col: str = "answer",
        **kwargs,
    ) -> None:
        """Initializes an instance of the Accuracy metric.

        Args:
            source_type (str): The type of source data. Either "categorical" or "text".
            source_labels (List[str]): A list of labels for the source data. Required if source_type is "text".
            **kwargs: Additional keyword arguments. Must include "hypothesis" and "references".

        Raises:
            AssertionError: If multiple references are provided.

        Returns:
            None
        """
        super().__init__(**kwargs)
        assert (
            len(self.references) == 1
        ), "ERROR: multiple references are not supported for Accuracy."
        assert (
            source_labels is not None if source_type == "text" else True
        ), "ERROR: source_labels must be provided if source_type is text."

        # This is because multiple references are not supported for Accuracy.
        self.reference_col = reference_col
        self.source_type = source_type
        self.source_labels = source_labels

    def run(self, hypothesis_path, gold_data_path) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        reference_lines = gold_data["ref"]
        gold_labels = []
        predicted_labels = []
        is_random_count = 0
        # if hypothesis is already numbered
        for ref_line, hyp_line in zip(reference_lines, hypotheses):
            # reference is always assumed to be in categorical format; i.e., [0,1,2,3,...]
            gold_labels.append(text_to_label(ref_line, "categorical"))
            label, is_random = text_to_label(
                hyp_line,
                self.source_type,
                self.source_labels,
                return_is_random=True,
            )
            is_random_count += 1 if is_random else 0
            predicted_labels.append(label)
        # warn user that some labels were randomly assigned
        if is_random_count > 0:
            pct_random = (is_random_count / len(gold_labels)) * 100
            logger.opt(colors=True).warning(
                f"<red>{is_random_count} ({pct_random:.2f}% of total) labels</red> did not correspond to any label in source_labels, so a random label was a assigned."
            )

        result = self.evaluate(
            gold_labels=gold_labels, predicted_labels=predicted_labels
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(self, gold_labels, predicted_labels) -> AccuracyResult:
        """
        Evaluate function receives the gold labels as well as the predicted ones and returns the Accuracy score of the predictions.
        The accuracy is calculate by calling the corresponding function in Scikit Learn library

        Args:
            gold_labels: The gold labels.
            predicted_labels: The predicted labels.

        Returns:
            AccuracyResult: The accuracy score.
        """
        score = accuracy_score(y_true=gold_labels, y_pred=predicted_labels)
        result = AccuracyResult(score)
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        """Returns the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return "accuracy"

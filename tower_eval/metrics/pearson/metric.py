# -*- coding: utf-8 -*-
from scipy.stats import pearsonr

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.pearson.result import PearsonResult


class PEARSON(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self, hypothesis_path, gold_data_path) -> dict:
        predicted_scores, gold_data = self._handle_inputs(
            hypothesis_path, gold_data_path
        )
        gold_scores = gold_data["score"]

        result = self.evaluate(
            gold_scores=gold_scores, predicted_scores=predicted_scores
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(self, gold_scores, predicted_scores) -> PearsonResult:
        """
        Evaluate function receives the gold scores as well as the predicted ones and returns the Pearson Correlation Coefficient score of the predictions and the gold scores.
        The Pearson Correlation Coefficient is calculate by calling the corresponding function in Scikit Learn library
        """
        pearson_corr_coef = pearsonr(gold_scores, predicted_scores)
        statistic, pvalue = pearson_corr_coef.statistic, pearson_corr_coef.pvalue
        result = PearsonResult(statistic)
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "pearson"

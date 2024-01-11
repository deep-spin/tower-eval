# -*- coding: utf-8 -*-
from scipy.stats import spearmanr
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.metrics.spearman.result import SpearmanResult
from tqdm import tqdm


class SPEARMAN(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self) -> dict:
        predicted_scores, gold_data = self._handle_inputs(
            self.hypothesis_path, self.gold_data_path
        )
        gold_scores = gold_data["score"]

        result = self.evaluate(
            gold_scores=gold_scores, predicted_scores=predicted_scores
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(self, gold_scores, predicted_scores) -> SpearmanResult:
        """
        Evaluate function receives the gold scores as well as the predicted ones and returns the Pearson Correlation Coefficient score of the predictions and the gold scores.
        The Pearson Correlation Coefficient is calculate by calling the corresponding function in Scikit Learn library
        """
        spearman_corr_coef = spearmanr(gold_scores, predicted_scores)
        statistic, pvalue = spearman_corr_coef.statistic, spearman_corr_coef.pvalue
        result = SpearmanResult(statistic)
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "spearman"

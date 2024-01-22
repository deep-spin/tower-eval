# -*- coding: utf-8 -*-
import torch
from evaluate import load
from tower_eval.metrics.bleurt.result import BLEURTResult
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tqdm import tqdm


class BLEURT(Metric):
    def __init__(self, batch_size: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.model = load("bleurt", "BLEURT-20", module_type="metric")

    def run(self) -> dict:
        hypotheses, gold_data = self._handle_inputs(
            self.hypothesis_path, self.gold_data_path
        )
        references = gold_data["ref"]
        result = self.evaluate(hypotheses, references)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(self, hypotheses: list, references: list) -> BLEURTResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        segments_scores = []
        for i in tqdm(range(0, len(references), self.batch_size)):
            with torch.no_grad():
                batch_references = references[i : i + self.batch_size]
                batch_hypotheses = hypotheses[i : i + self.batch_size]
                segments_scores.extend(
                    self.model.compute(
                        predictions=batch_hypotheses, references=batch_references
                    )["scores"]
                )
        system_score = sum(segments_scores) / len(segments_scores)

        result = BLEURTResult(
            {
                "system_score": system_score,
                "segments_scores": segments_scores,
            }
        )
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "bleurt"

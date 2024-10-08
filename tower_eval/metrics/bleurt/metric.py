# -*- coding: utf-8 -*-
import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from tqdm import tqdm

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.bleurt.result import BLEURTResult


class BLEURT(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20")
        self.model = BleurtForSequenceClassification.from_pretrained(
            "lucadiliello/BLEURT-20"
        )
        self.tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")
        self.model.eval()
        self.model = self.model.to("cuda")

    def run(
        self, hypothesis_path, gold_data_path, batch_size: int = 16, **kwargs
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]
        result = self.evaluate(hypotheses, references, batch_size)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self, hypotheses: list, references: list, batch_size: int
    ) -> BLEURTResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        segments_scores = []
        for i in tqdm(range(0, len(references), batch_size)):
            with torch.no_grad():
                batch_references = references[i : i + batch_size]
                batch_hypotheses = hypotheses[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_references,
                    batch_hypotheses,
                    padding="longest",
                    return_tensors="pt",
                    truncation=True,
                ).to("cuda")
                segments_scores.extend(self.model(**inputs).logits.flatten().tolist())
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

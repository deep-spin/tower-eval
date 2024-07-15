# -*- coding: utf-8 -*-
from comet import download_model, load_from_checkpoint

from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.metrics.xcomet_xxl.result import XCOMETXXLResult


class XCOMETXXL(Metric):
    def __init__(
        self,
        lowercase: bool = False,
        batch_size: int = 16,
        gpus: int = 1,
        comet_model="Unbabel/XCOMET-XXL",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.source = kwargs.get("source")
        self.lowercase = kwargs.get("lowercase", lowercase)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.gpus = kwargs.get("gpus", gpus)
        self.model = kwargs.get("comet_model", comet_model)
        model_path = download_model(self.model)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def run(self, hypothesis_path, gold_data_path, **kwargs) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references, sources = gold_data["ref"], gold_data["src"]
        result = self.evaluate(hypotheses, references, sources)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self, hypotheses: list, references: list, sources: list
    ) -> XCOMETXXLResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        samples = {"src": sources, "mt": hypotheses, "ref": references}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]

        outputs = self.model.predict(
            samples=samples,
            batch_size=self.batch_size,
            gpus=self.gpus,
            accelerator="auto",
        )
        system_score, segments_scores = outputs.system_score, outputs.scores

        comet_result = XCOMETXXLResult(
            {
                "system_score": system_score,
                "segments_scores": segments_scores,
            }
        )
        return comet_result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "xcomet_xxl"

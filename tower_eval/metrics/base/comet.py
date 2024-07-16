# -*- coding: utf-8 -*-
from comet import download_model, load_from_checkpoint

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult


class BaseCOMETResult(MetricResult):
    """
    COMET Result Handler.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)


class BaseCOMET(Metric):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)
        model_path = download_model(model)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def load_gold_data(self, gold_data):
        pass

    def make_samples(
        self, sources: list[str], hypotheses: list[str], references: list[str] = None
    ):
        pass

    def run(
        self, hypothesis_path, gold_data_path, gpus: int = 1, batch_size: int = 16
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references, sources = self.load_gold_data(gold_data)
        result = self.evaluate(hypotheses, references, sources, gpus, batch_size)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self, hypotheses: list, references: list, sources: list, gpus, batch_size
    ) -> BaseCOMETResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        samples = self.make_samples(sources, hypotheses, references)

        outputs = self.model.predict(
            samples=samples,
            batch_size=batch_size,
            gpus=gpus,
            accelerator="auto",
        )
        system_score, segments_scores = outputs.system_score, outputs.scores

        comet_result = BaseCOMETResult(
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
        pass


class RefCOMET(BaseCOMET):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def load_gold_data(self, gold_data):
        references, sources = gold_data["ref"], gold_data["src"]
        return references, sources

    def make_samples(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ):
        samples = {"src": sources, "mt": hypotheses, "ref": references}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
        return samples


class QECOMET(BaseCOMET):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)

    def load_gold_data(self, gold_data):
        references, sources = None, gold_data["src"]
        return references, sources

    def make_samples(
        self, sources: list[str], hypotheses: list[str], references: list[str] = None
    ):
        samples = {"src": sources, "mt": hypotheses}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
        return samples

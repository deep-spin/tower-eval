# -*- coding: utf-8 -*-
from sacrebleu.metrics import CHRF as SacreCHRF

from tower_eval.metrics.chrf.result import CHRFResult
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.utils import get_sacrebleu_segment_scores


class CHRF(Metric):
    def __init__(self, lowercase: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lowercase = kwargs.get("lowercase", lowercase)

    def run(self, hypothesis_path, gold_data_path) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]
        result = self.evaluate(hypotheses, references, lowercase=self.lowercase)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
        lowercase: bool = False,
    ) -> CHRFResult:
        """
        Evaluate function receives the hypotheses and the references and returns a CHRFResult object.
        The chrF score is calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        chrf = SacreCHRF(lowercase=lowercase)
        if type(references[0]) == str:
            segment_references = [[r] for r in references]
            references = [references]
        score = chrf.corpus_score(hypotheses, references)
        segment_scores = get_sacrebleu_segment_scores(
            hypotheses, segment_references, method=chrf
        )
        result = CHRFResult(score.score)
        result = CHRFResult(
            {
                "system_score": score.score,
                "segments_scores": segment_scores,
            }
        )
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "chrf"

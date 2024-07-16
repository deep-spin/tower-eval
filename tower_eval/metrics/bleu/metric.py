# -*- coding: utf-8 -*-
from sacrebleu.metrics import BLEU as SacreBLEU

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.bleu.result import BLEUResult
from tower_eval.utils import get_sacrebleu_segment_scores


class BLEU(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(
        self,
        hypothesis_path,
        gold_data_path,
        lowercase: bool = False,
        tokenizer: str = None,
        **kwargs
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]
        result = self.evaluate(
            hypotheses, references, lowercase=lowercase, tokenize=tokenizer
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypothesis: list,
        references: list,
        lowercase: bool = False,
        tokenize: str = None,
    ) -> BLEUResult:
        """
        Evaluate function receives the hypotheses and the references and returns a BLEUResult object.
        The BLEU score is calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        sacrebleu = SacreBLEU(lowercase=lowercase, tokenize=tokenize)
        if type(references[0]) == str:
            segment_references = [[r] for r in references]
            references = [references]
        score = sacrebleu.corpus_score(hypothesis, references)
        sacrebleu = SacreBLEU(
            lowercase=lowercase, tokenize=tokenize, effective_order=True
        )
        segment_scores = get_sacrebleu_segment_scores(
            hypothesis, segment_references, method=sacrebleu
        )
        result = BLEUResult(score.score)
        result = BLEUResult(
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
        return "bleu"

# -*- coding: utf-8 -*-
from sacrebleu.metrics import TER as SacreTER

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.ter.result import TERResult


class TER(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(
        self,
        hypothesis_path,
        gold_data_path,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
        **kwargs
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]
        result = self.evaluate(
            hypotheses, references, normalized, no_punct, asian_support, case_sensitive
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
    ) -> TERResult:
        """
        Evaluate function receives the hypotheses and the references and returns a TERResult object.
        The TER score is calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        ter = SacreTER(
            normalized=normalized,
            no_punct=no_punct,
            asian_support=asian_support,
            case_sensitive=case_sensitive,
        )
        if type(references[0]) == str:
            references = [references]
        score = ter.corpus_score(hypotheses, references)
        result = TERResult(score.score)
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "ter"

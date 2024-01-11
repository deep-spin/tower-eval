# -*- coding: utf-8 -*-
from sacrebleu.metrics import TER as SacreTER
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.metrics.ter.result import TERResult


class TER(Metric):
    def __init__(
        self,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.normalized = kwargs.get("normalized", normalized)
        self.no_punct = kwargs.get("no_punct", no_punct)
        self.asian_support = kwargs.get("asian_support", asian_support)
        self.case_sensitive = kwargs.get("case_sensitive", case_sensitive)

    def run(self) -> dict:
        hypotheses, gold_data = self._handle_inputs(
            self.hypothesis_path, self.gold_data_path
        )
        references = gold_data["ref"]
        result = self.evaluate(hypotheses, references)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
    ) -> TERResult:
        """
        Evaluate function receives the hypotheses and the references and returns a TERResult object.
        The TER score is calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        ter = SacreTER(
            normalized=self.normalized,
            no_punct=self.no_punct,
            asian_support=self.asian_support,
            case_sensitive=self.case_sensitive,
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

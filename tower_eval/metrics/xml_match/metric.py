# -*- coding: utf-8 -*-
from loguru import logger
from lxml import etree

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.xml_match.result import XML_MatchResult
from tower_eval.utils import match_xml


class XML_MATCH(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(
        self, hypothesis_path, gold_data_path, lowercase: bool = False, **kwargs
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]

        result = self.evaluate(hypotheses, references)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
    ) -> XML_MatchResult:
        """
        Evaluate function receives the hypotheses and the references and returns a XML_MatchResult object.

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        assert type(references[0]) == str, logger.error(
            "Mutli-reference is not supported for XML_CHRF"
        )

        """
        Based on the information provided in the original papers, 
        the XML-Match is the percentage of outputs that have exactly the same XML structures as their references.
        """
        segment_scores = [0] * len(hypotheses)

        for id, (hyp, ref) in enumerate(zip(hypotheses, references)):
            try:
                hyp = etree.fromstring(f"<root>{hyp}</root>")
                ref = etree.fromstring(f"<root>{ref}</root>")
                if match_xml(hyp, ref):
                    segment_scores[id] = 1
            except:
                pass

        score = sum(segment_scores) / len(segment_scores)
        result = XML_MatchResult(
            {
                "system_score": score,
                "segments_scores": segment_scores,
            }
        )
        return result

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "xml_match"

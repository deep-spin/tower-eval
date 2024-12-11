# -*- coding: utf-8 -*-
from loguru import logger
from sacrebleu.metrics import CHRF as SacreCHRF

from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.metrics.xml_chrf.result import XML_CHRFResult
from tower_eval.utils import get_sacrebleu_segment_scores, split_pairs_by_xml_tags


class XML_CHRF(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(
        self, hypothesis_path, gold_data_path, lowercase: bool = False, **kwargs
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        references = gold_data["ref"]

        result = self.evaluate(hypotheses, references, lowercase=lowercase)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
        lowercase: bool = False,
    ) -> XML_CHRFResult:
        """
        Evaluate function receives the hypotheses and the references and returns a XML_CHRFResult object.
        The chrF scores of the segments are calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        chrf = SacreCHRF(lowercase=lowercase)
        assert type(references[0]) == str, logger.error("Mutli-reference is not supported for XML_CHRF")

        """
        Based on the information provided in the original papers, xml-chrf (and xml-bleu) are calcualted as follows:
        We first use etree to extract the XML structure of the output and reference.
        The XML-Match is the percentage of outputs that have exactly the same XML structures as their references.
        If the XML structures of an output and its reference match, then the translation and reference are split by the XML tags
        and we evaluate the chrF score by comparing each split segment.
        If the structures do not match, the chrF score is counted as zero to penalize the irrelevant outputs.
        """
        segment_scores = []
        count_non_matching_xml = 0
        hypothesis_segmented, references_segmented = [], []
        for hyp, ref in zip(hypotheses, references):
            hs, rs = split_pairs_by_xml_tags(hyp, ref)
            if hs and rs:
                hypothesis_segmented.append(hs)
                references_segmented.append(rs)
            else:
                count_non_matching_xml += 1

        hypothesis_segmented = [item for sublist in hypothesis_segmented for item in sublist]
        references_segmented = [[item] for sublist in references_segmented for item in sublist]
        
        segment_scores = get_sacrebleu_segment_scores(
            hypothesis_segmented, references_segmented, method=chrf
        )

        # These extra 0 scores are added to penalise the irrelevant outputs.
        # This is based on the information provided in the original paper:
        # How Effective is Synthetic Data and Instruction Fine-tuning for Translation with Markup using LLMs?
        # https://aclanthology.org/2024.amta-research.8
        # Appendix B: Details of Evaluation Metrics
        segment_scores.extend([0.0] * count_non_matching_xml)

        score = sum(segment_scores) / len(segment_scores)
        result = XML_CHRFResult(score)
        result = XML_CHRFResult(
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
        return "xml_chrf"

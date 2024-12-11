# -*- coding: utf-8 -*-
from loguru import logger

from tower_eval.metrics.chrf.metric import CHRF
from tower_eval.metrics.chrf.result import CHRFResult
from tower_eval.utils import prepare_xml_markup_pairs


class XML_CHRF(CHRF):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(
        self,
        hypotheses: list,
        references: list,
        lowercase: bool = False,
    ) -> CHRFResult:
        """
        Evaluate function receives the hypotheses and the references and returns a XML_CHRFResult object.
        The chrF scores of the segments are calculate by calling sacreBLEU

        :param hypotheses: path to the hypotheses file.
        :param references: path to the references file.
        """
        chrf = CHRF(lowercase=lowercase)
        assert type(references[0]) == str, logger.error(
            "Mutli-reference is not supported for XML_CHRF"
        )

        """
        Based on the information provided in the original papers, xml-chrf (and xml-bleu) are calcualted as follows:
        We first use etree to extract the XML structure of the output and reference.
        The XML-Match is the percentage of outputs that have exactly the same XML structures as their references.
        If the XML structures of an output and its reference match, then the translation and reference are split by the XML tags
        and we evaluate the chrF score by comparing each split segment.
        If the structures do not match, the chrF score is counted as zero to penalize the irrelevant outputs.
        """

        hypothesis_segmented, references_segmented, non_matching_indices = (
            prepare_xml_markup_pairs(hypotheses, references)
        )
        results = chrf.evaluate(hypothesis_segmented, references_segmented)
        segment_scores = results.result["segments_scores"]
        # Now, add 0 scores for the instance that their markup structure doesn't match the one of their corresponding references.
        # This is to penalise the irrelevant outputs and is based on the information provided in the original paper:
        # How Effective is Synthetic Data and Instruction Fine-tuning for Translation with Markup using LLMs?
        # https://aclanthology.org/2024.amta-research.8
        # Appendix B: Details of Evaluation Metrics
        for index in reversed(non_matching_indices):
            segment_scores.insert(index, 0.0)

        score = sum(segment_scores) / len(segment_scores)

        result = CHRFResult(
            {
                "system_score": score,
                "segments_scores": segment_scores,
            }
        )
        return result

    @staticmethod
    def metric_name():
        return "xml_chrf"

# -*- coding: utf-8 -*-
import re
import string
import sys
from collections import Counter

from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult


class TYDIQA(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert (
            len(kwargs["references"]) == 1
        ), "ERROR: multiple references are not supported for F1."
        self.source = kwargs.get("source")
        self.hypothesis = kwargs.get("hypothesis")
        self.reference = kwargs["references"][
            0
        ]  # This is because multiple references are not supported for F1.

    def evaluate(self, hypotheses, references) -> dict:
        """
        Evaluate function receives the hypotheses and the references and returns F1 and ExactMatch scores.

        :param hypotheses: List of the answers
        :param references: List dictionaries (context, questions and answers)
        """
        ###################################################################################################################################################
        ## NOTE: the code is directly copied from https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py#L54                          ##
        ## This is suggested in the main tydi-QA repository: https://github.com/google-research-datasets/tydiqa/tree/master#gold-passage-task-evaluation ##
        ###################################################################################################################################################
        f1 = exact_match = total = 0
        for article in references:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    total += 1
                    if qa["id"] not in hypotheses:
                        message = (
                            "Unanswered question " + qa["id"] + " will receive score 0."
                        )
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                    prediction = hypotheses[qa["id"]]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths
                    )
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths
                    )
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {"f1": f1, "exact_match": exact_match}

    def predictions2json(self, ref_json, hyp_list):
        predictions = {}
        qid = 0
        for article in ref_json:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    id = qa["id"]
                    predictions[id] = hyp_list[qid]
                    qid += 1
        return predictions

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "tydi"


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class TYDIQAResult(MetricResult):
    """
    TYDIQA Result Handler.
    """

    def __init__(
        self,
        result: float,
    ) -> None:
        super().__init__(result)

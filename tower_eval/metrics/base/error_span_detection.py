# -*- coding: utf-8 -*-
from pathlib import Path

from tower_eval.error_span_utils import det_to_annotation, tag_to_annotation
from tower_eval.metrics.base.metrics_handler import Metric
from tower_eval.metrics.base.result_handler import MetricResult
from tower_eval.utils import load_jsonl_file, read_lines


class ErrorSpanDetectionMetric(Metric):
    def __init__(self, key, **kwargs) -> None:
        super().__init__(**kwargs)
        self.key = key

    def run(
        self,
        hypothesis_path,
        gold_data_path,
        severity_mismatch_penalty: float = 0.5,
        hyp_type: str = "jsonl",
    ) -> dict:
        hypotheses, gold_data = self._handle_inputs(
            hypothesis_path, gold_data_path, hyp_type=hyp_type
        )
        reference_list = gold_data["ref"]
        result = ErrorSpanDetectionResult(
            self.evaluate(
                hypotheses,
                reference_list,
                severity_mismatch_penalty,
            )[self.key]
        )
        result.print_result(self.metric_name())

        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypotheses: list,
        references: list,
        severity_mismatch_penalty: float = 0.5,
    ) -> dict:
        """
        Computes the Error Span Detection metric.
        """
        system_preds = self.load_annotations_from_list(hypotheses)
        gold_labels = self.load_annotations_from_list(references)

        tp = 0
        tn = 0
        fp = 0
        total_sys = 0
        total_gold = 0
        for segid in gold_labels:
            for (
                character_gold_major,
                character_sys_major,
                character_gold_minor,
                character_sys_minor,
            ) in zip(
                gold_labels[segid]["major"],
                system_preds[segid]["major"],
                gold_labels[segid]["minor"],
                system_preds[segid]["minor"],
            ):
                if character_gold_major != 0 or character_gold_minor != 0:
                    total_gold += 1
                if character_sys_major != 0 or character_sys_minor != 0:
                    total_sys += 1
                if character_gold_major == 0 and character_gold_minor == 0:
                    if character_sys_major == 0 and character_sys_minor == 0:
                        tn += 1
                    else:
                        # fp+=(character_sys_major + character_sys_minor)
                        fp += 1
                else:
                    if character_gold_major > 0 and character_gold_minor == 0:
                        if character_sys_major > 0:
                            tp += 1
                        elif character_sys_minor > 0:
                            tp += severity_mismatch_penalty
                    elif character_gold_minor > 0 and character_gold_major == 0:
                        if character_sys_minor > 0:
                            tp += 1
                        elif character_sys_major > 0:
                            tp += severity_mismatch_penalty
                    elif character_gold_minor > 0 and character_gold_major > 0:
                        if character_sys_minor > 0 or character_sys_major > 0:
                            tp += 1

        precision = tp / (total_sys)
        recall = tp / (total_gold)
        f1 = 2 * precision * recall / (precision + recall)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def load_annotations_from_list(
        self,
        segments: list,
    ) -> dict:
        """Converts list of annotation dictionaries into format that is amenable for evaluation fnctn.

        The file should contain lines with the following format:
        {"mt": <mt>, "annotations": [{"start": <start>, "end": <end>, "severity":

        Args:
            annotations: list of dictionaries with mt and respective annotations.
        Returns:
            a dictionary mapping document id's to a list of annotations.
        """
        out_dict = {}
        for i, line in enumerate(segments):
            mt = line["mt"]

            seg_id = i
            out_dict[seg_id] = {}
            out_dict[seg_id]["major"] = [0] * (len(mt) + 1)
            out_dict[seg_id]["minor"] = [0] * (len(mt) + 1)
            for annotation in line["annotations"]:
                s = int(annotation["start"])
                e = int(annotation["end"])
                t = annotation["severity"]
                if e > len(mt):
                    e = len(mt)

                if s != -1 and e != -1:
                    if s == e:
                        out_dict[seg_id][t][s] += 1
                    else:
                        i = s
                        while i < e:
                            out_dict[seg_id][t][i] += 1
                            i += 1
        return out_dict

    def _handle_inputs(
        self,
        hypotheses: Path,
        references: Path,
        hyp_type: str,
    ) -> tuple:
        """
        Function to handle input files.
        All inputs will be returned as list of strings.

        :param hypotheses: either the handler to the file storing the hypotheses.
        :param references: either the handler to the file storing the refereneces.
        :param sources: either the handler to the file storing the source sentences.

        :return:
            -  Tuple with hypotheses, references and kwargs
        """
        hypotheses_list = []
        references_list = load_jsonl_file(references)
        mts = [ref["mt"] for ref in references_list]
        # if handling existing jsonl file
        if hyp_type == "jsonl":
            hypotheses_list_no_mt = load_jsonl_file(hypotheses)
            for i, mt in enumerate(mts):
                hypotheses_list_no_mt[i].update({"mt": mt})
                hypotheses_list.append(hypotheses_list_no_mt[i])
        # if handling model generations
        elif hyp_type in ["tag", "det"]:
            generations = read_lines(hypotheses, unescape_newline=True)
            hypotheses_list = self.parse_generations(generations, mts, hyp_type)

        assert len(hypotheses) == len(
            references_list
        ), f"The number of hypotheses {len(hypotheses)} and rows in the gold data {len(references_list)} should be the same."

        return hypotheses_list, references_list

    def process_result(self):
        pass

    @classmethod
    def parse_generations(
        self,
        generations: list[str],
        mts: list[str],
        hyp_type: str,
    ) -> list[dict[str, list[dict[str, str | int]]]]:
        """Takes a list of model generations (strings) and converts them into a list of records, each containing
        information like an error span raw test set.
        """
        if hyp_type == "tag":
            _parsing_func = tag_to_annotation
        elif hyp_type == "det":
            _parsing_func = det_to_annotation

        return [
            {"mt": mt, "annotations": _parsing_func(g, mt)}
            for g, mt in zip(generations, mts)
        ]


class ErrorSpanDetectionResult(MetricResult):
    """
    Error Span Detection Recall Result Handler.
    """

    def __init__(
        self,
        result: list,
    ) -> None:
        super().__init__(result)

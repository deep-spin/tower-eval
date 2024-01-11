# -*- coding: utf-8 -*-
import json
import re
from typing import List, Literal

from loguru import logger
from tower_eval.metrics.f1_sequence import conlleval
from tower_eval.metrics.f1_sequence.result import F1SequenceResult
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.utils import (
    PATTERN_CLOSE_TAG,
    PATTERN_OPEN_TAG,
    list_to_dict,
    load_jsonl_file,
    read_lines,
    tokenize_spacy,
)
from tqdm import tqdm


class F1SEQUENCE(Metric):
    def __init__(
        self,
        language: str,
        hypothesis_format: str = "xml",
        reference_format: str = "tsv",
        tokenize_hypothesis: bool = True,
        default_noent_tag: str = "O",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Terminate if the length of hypothesis doesn't match the length of the reference?
        # Or simply trim/pad the hypothesis to make it the same length as the reference
        self.language = kwargs.get("language", language)
        self.tokenize_hypothesis = kwargs.get(
            "tokenize_hypothesis", tokenize_hypothesis
        )
        self.hypothesis_format = kwargs.get("hypothesis_format", hypothesis_format)
        self.reference_format = kwargs.get("reference_format", reference_format)
        self.default_noent_tag = kwargs.get("default_noent_tag", default_noent_tag)
        # Having self.valid_ner_tags set to None means all tags produced by the model are acceptable.
        self.valid_ner_tags = kwargs.get("valid_ner_tags")

    def run(self) -> dict:
        hypothesis = self._load_samples(
            self.hypothesis_path,
            format=self.hypothesis_format,
            tokenize=self.tokenize_hypothesis,
        )
        hypothesis = self.filter_tags(
            hypothesis, self.valid_ner_tags, self.default_noent_tag
        )
        reference = self._load_samples(
            self.gold_data_path, format=self.reference_format, tokenize=False
        )
        reference = self.filter_tags(
            reference, self.valid_ner_tags, self.default_noent_tag
        )

        result = self.evaluate(hypothesis=hypothesis, reference=reference)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self, hypothesis: List[List[str]], reference: List[List[str]]
    ) -> F1SequenceResult:
        # evaluate by tag
        f1s_by_tag = {}
        for tag in self.valid_ner_tags:
            filtered_hypothesis = self.filter_tags(
                hypothesis, tag, self.default_noent_tag
            )
            filtered_reference = self.filter_tags(
                reference, tag, self.default_noent_tag
            )
            true_seqs, pred_seqs = align_hyp_ref(
                gold_labels=filtered_reference,
                predicted_labels=filtered_hypothesis,
                noent_tag=self.default_noent_tag,
            )
            _, _, f1s_by_tag[tag] = conlleval.evaluate(
                true_seqs=true_seqs, pred_seqs=pred_seqs
            )
            logger.info(f"{self.metric_name()}_{tag}: {f1s_by_tag[tag]}")

        # evaluate global f1
        true_seqs, pred_seqs = align_hyp_ref(
            gold_labels=reference,
            predicted_labels=hypothesis,
            noent_tag=self.default_noent_tag,
        )

        precision, recall, global_f1 = conlleval.evaluate(
            true_seqs=true_seqs, pred_seqs=pred_seqs, verbose=False
        )
        result = F1SequenceResult(
            result=global_f1, tags_f1=f1s_by_tag, valid_tags=self.valid_ner_tags
        )
        return result

    def process_result(self, result) -> MetricResult:
        pass

    def _load_samples(
        self,
        filename: str,
        format: Literal["text", "tsv", "xml"] = "text",
        separator="|",
        tokenize: bool = False,
    ) -> List[List[str]]:
        """ "
        It reads the labeled file, and returns only the tags.
        For now it only supports the following two formats:
        - xml: in which the NER tags are added to the named entities, like: This is <PER>John</PER>.
        - tsv: in which the firs two columns store the words and their corresponding tags.
               empty lines define the sentence boundaries
        - text: in which each line contains one labeled string in the form of token1|label1 token2|label2 ...
        """

        labels = []
        if format == "xml":
            if tokenize:
                input_lines = read_lines(filename)
                input_lines = tokenize_spacy(
                    lines=input_lines, language=self.language, keep_xml=True
                )
            else:
                with open(filename, "r", encoding="utf8") as ifh:
                    input_lines = ifh.readlines()
                    input_lines = [line.strip() for line in input_lines]
            labels = [self.xml2iob(hyp) for hyp in input_lines]
        elif format == "tsv":
            separator = "\t"
            with open(filename, "r", encoding="utf8") as infh:
                current_string = []
                for token in infh:
                    token = token.strip()
                    if token:
                        token = token.split(separator)
                        current_string.append(token[2])
                    else:
                        labels.append(current_string)
                        current_string = []
        elif "text" in format:
            with open(filename, "r", encoding="utf8") as infh:
                for line in infh:
                    if format == "text":
                        tokens = line.split()
                        # It assumes to have the tokens in the form of (word|tag).
                        # And we are only interested in the tags.
                        # So, we iterate over all the tokens, split them by the separator and only keep their tags
                        tokens = [token.split(separator)[1] for token in tokens]
                        labels.append(tokens)
                    elif format == "text-tuple-list":
                        # assumes format is [("This", "O"), ("is", "O"), ("a", "O"), ("Named", "B-<entity>"), ("Entity", "I-<entity>"), (".", "O")]
                        labels.append(self.list_of_tuples_to_tokens(line))
        elif format == "jsonl":
            gold_data = load_jsonl_file(filename)
            gold_data = list_to_dict(gold_data)
            labels = [self.list_of_tuples_to_tokens(line) for line in gold_data["answer"]]
        return labels

    def list_of_tuples_to_tokens(self, line):
        # can't just use ast.literal_eval() because the strings may be like "06" which raises an error of trailing zeroes.
        pattern = r'\("(.*?)", "(.*?)"\)'
        list_of_tuples = [tuple(match) for match in re.findall(pattern, line)]
        tokens = [token[1] for token in list_of_tuples]
        return tokens

    def xml2iob(self, xml_string):
        tokens = xml_string.split()
        annotations = []
        # tag is set to "O" (self.noent_tag) by default
        tag = self.default_noent_tag
        for token in tokens:
            matching_open_tag = re.search(PATTERN_OPEN_TAG, token)
            matching_close_tag = re.search(PATTERN_CLOSE_TAG, token)

            if matching_open_tag:
                tag = re.sub(r"^.*<([\w]+)>?.*$", r"\1", token)
                annotations.append(f"B-{tag}")

            if matching_close_tag:
                # You are processing the last token of a multi-token tag.
                # Else, it was a single-token tag and you don't need to do anything.
                # That is taken care of already, in the very first check ("if matching_open_tag")
                if not matching_open_tag:
                    annotations.append(f"I-{tag}")
                tag = self.default_noent_tag
            if not matching_open_tag and not matching_close_tag:
                # You are processing one of the middle tokens of a multi-token tag.
                annotations.append(f"I-{tag}")
        return annotations

    @staticmethod
    def filter_tags(
        annotations: List[List[str]], valid_tags: List[str], default_noent_tag
    ):
        """
        Remove all tags that are not in the list of valid tags.
        """
        filtered_annotations = []
        for annotation in annotations:
            filtered_annotation = []
            for tag in annotation:
                parsed_tag = tag[2:]  # ignore B- or I- prefix
                if parsed_tag in valid_tags:
                    filtered_annotation.append(tag)
                else:
                    filtered_annotation.append(default_noent_tag)
            filtered_annotations.append(filtered_annotation)
        return filtered_annotations

    @staticmethod
    def metric_name():
        return "f1sequence"


def align_hyp_ref(gold_labels, predicted_labels, noent_tag="O"):
    true_seqs, pred_seqs = [], []
    for g, p in zip(gold_labels, predicted_labels):
        # pad or trim to size of target
        if len(p) > len(g):
            p = p[: len(g)]
        elif len(p) < len(g):
            p.extend([noent_tag] * (len(g) - len(p)))
        true_seqs.extend(g)
        true_seqs.append(noent_tag)

        pred_seqs.extend(p)
        pred_seqs.append(noent_tag)
    return true_seqs, pred_seqs

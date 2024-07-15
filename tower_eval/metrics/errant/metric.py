# -*- coding: utf-8 -*-
import re
import subprocess
import tempfile

from tower_eval.metrics.errant.result import ERRANTResult
from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.utils import read_lines, tokenize_spacy


class ERRANT(Metric):
    def __init__(
        self, tokenize_source: bool = False, tokenize_hypothesis: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.tokenize_source = kwargs.get("tokenize_source", tokenize_source)
        self.tokenize_hypothesis = kwargs.get(
            "tokenize_hypothesis", tokenize_hypothesis
        )

    def run(self, hypothesis_path, gold_data_path, **kwargs) -> dict:
        gold_data_path_m2 = kwargs["references_m2"]
        language = kwargs["lp"]["src_lang"]
        hypothesis_m2 = self.preprocess(hypothesis_path, gold_data_path, language=language)
        result = self.evaluate(hypothesis_m2, gold_data_path_m2)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self,
        hypothesis_m2, 
        references
    ) -> ERRANTResult:
        """
        Evaluate function receives the source, hypothesis as well as the reference and returns an ERRANTResult object.
        """
        errant_score = subprocess.run(
            ["errant_compare", "-hyp", hypothesis_m2, "-ref", references],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
        )
        pattern = r"\d+\.\d+|\d+"
        _, _, score_values, _ = errant_score.stdout.decode("utf-8").strip().split("\n")
        matches = re.findall(pattern, score_values)
        # Assign the extracted values to the respective fields
        fields = ["TP", "FP", "FN", "Prec", "Rec", "F0.5"]
        values = [int(matches[i]) if i < 3 else float(matches[i]) for i in range(6)]
        # Create the output dictionary
        output_dict = {fields[i]: values[i] for i in range(6)}
        result = ERRANTResult(output_dict["F0.5"])
        return result

    def preprocess(self, hypothesis_path, gold_data_path, language):
        hyp_lines, gold_data = self._handle_inputs(hypothesis_path, gold_data_path)
        src_lines = gold_data["src"]

        if self.tokenize_source:
            src_tokenized = tokenize_spacy(src_lines, language)
        else:
            # assumes gold data already has tokenized sources
            src_tokenized = gold_data["tok_src"]
        if self.tokenize_hypothesis:
            hyp_tokenized = tokenize_spacy(hyp_lines, language)
        else:
            hyp_tokenized = hyp_lines

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as sfh_out:
            for line in src_tokenized:
                sfh_out.write(line + "\n")
            self.source = sfh_out.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as hfh_out:
            for line in hyp_tokenized:
                hfh_out.write(line + "\n")
            self.hypothesis = hfh_out.name

        # Create the m2 version of the hypothesis file to be used by the evaluator.
        # TODO: replace this part by a script that calls native python code of errant lib
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as hyp_m2:
            subprocess.run(
                [
                    "errant_parallel",
                    "-orig",
                    self.source,
                    "-cor",
                    self.hypothesis,
                    "-out",
                    hyp_m2.name,
                ],
                check=True,
            )
        return hyp_m2.name

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "errant"

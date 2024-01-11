# -*- coding: utf-8 -*-
from pathlib import Path

from tower_eval.metrics.tydiqa import TYDIQA, TYDIQAResult
from tower_eval.utils import load_json_file, read_lines


class TYDIQAExactMatch(TYDIQA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self) -> dict:
        hypotheses, references = self._handle_inputs(
            self.hypothesis_path,
            self.gold_data_path,
        )
        result = TYDIQAResult(
            self.evaluate(hypotheses=hypotheses, references=references)["exact_match"]
        )
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def _handle_inputs(
        self,
        hypotheses: Path,
        data_path: Path,
    ) -> tuple:
        """
        Function to handle input files.
        """
        hypotheses = read_lines(hypotheses, unescape_newline=True)
        references = load_json_file(data_path)
        references = references["data"]
        hypotheses_json = self.predictions2json(references, hypotheses)

        return hypotheses_json, references

    @staticmethod
    def metric_name():
        return "tydiqa-exact-match"

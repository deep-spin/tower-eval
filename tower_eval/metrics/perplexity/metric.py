# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import vllm
from loguru import logger

from tower_eval.metrics.metrics_handler import Metric
from tower_eval.metrics.perplexity.result import PerplexityResult
from tower_eval.metrics.result_handler import MetricResult
from tower_eval.utils import (
    handle_subprocess,
    list_to_dict,
    load_jsonl_file,
    tokenize_text,
)


class Perplexity(Metric):
    def __init__(self, model: str, max_model_context: int, **kwargs) -> None:
        """
        Calculates perplexity over some corpus. Truncates the ending of each instance to fit the max_model_context.
        """
        super().__init__(**kwargs)
        self.model_id = model
        self.max_model_context = max_model_context
        self.vllm_args = kwargs.get(
            "vllm_args",
            {
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1,
                "trust_remote_code": True,
            },
        )

    @staticmethod
    def _handle_inputs(
        data_path: Path,
    ) -> tuple:
        """ """
        gold_data = load_jsonl_file(data_path)
        gold_data = list_to_dict(gold_data)

        return gold_data

    def run(self, hypothesis_path, gold_data_path, **kwargs) -> dict:
        result = self.evaluate(gold_data_path, self.model_id, self.max_model_context)
        result.print_result(self.metric_name())
        return result.format_result(self.metric_name())

    def evaluate(
        self, gold_data_path: Path, model_id: str, max_model_context: int
    ) -> PerplexityResult:
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        current_dir = os.getcwd()
        subprocess_args = [
            f"python",
            f"{current_dir}/tower_eval/metrics/perplexity/vllm_subprocess.py",
            "--gold_data_path",
            f"{str(gold_data_path)}",
            "--model_id",
            f"{model_id}",
            "--max_model_context",
            f"{str(max_model_context)}",
        ]
        output = handle_subprocess(subprocess_args, check_output=True)
        perplexities, mean_perplexity = self.parse_subprocess_output(output)
        assert perplexities is not None and mean_perplexity is not None, (
            f"Error when parsing the output of the perplexity subprocess, aborting."
            f"Output: {output}"
        )
        result = PerplexityResult(
            {
                "system_score": mean_perplexity,
                "segments_scores": perplexities,
            }
        )
        return result

    @staticmethod
    def parse_subprocess_output(output: str):
        # Correcting the regular expression to accurately capture the perplexities and mean perplexity
        regex = r"PERPLEXITIES: \[([0-9.,\s]+)\]\n PERPLEXITY: ([0-9.]+)"

        match = re.search(regex, output)

        if match:
            perplexities = list(map(float, match.group(1).split(", ")))
            mean_perplexity = float(match.group(2))
        else:
            perplexities, mean_perplexity = None, None

        return perplexities, mean_perplexity

    @staticmethod
    def truncate_prompts(prompts: List[str], max_model_context: int):
        return [prompt[:max_model_context] for prompt in prompts]

    @staticmethod
    def get_perplexity_from_vllm_output(vllm_output):
        perplexities = []
        for output in vllm_output:
            l_probs = []
            # ignore very first token, for which there is no logprob
            for l_prob_dict in output.prompt_logprobs[1:]:
                l_probs.append(list(l_prob_dict.values())[0].logprob)
            perplexities.append(np.exp(-(sum(l_probs) / len(l_probs))))
        mean_perplexity = np.mean(perplexities).astype(float)

        return perplexities, mean_perplexity

    def process_result(self, result) -> MetricResult:
        pass

    @staticmethod
    def metric_name():
        return "perplexity"

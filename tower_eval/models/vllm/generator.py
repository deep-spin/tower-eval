import gc
import time
from typing import List

import torch
from loguru import logger
from tower_eval.models.inference_handler import Generator
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


class VLLM(Generator):
    """VLLM Generate Wrapper."""

    def __init__(
        self,
        max_tokens: int = 1024,
        stop_sequences: list = ["\n", "\\n", "</s>"],
        do_sample: bool = False,
        seed: int = 42,
        n_gpus: int = 1,
        run_async: bool = True,
        batch_size: int = 16,
        quantization: str = None,  # "awq", "gptq" or "squeezellm"
        trust_remote_code: bool = True,
        vllm_sampling_params: dict = {},  # see vllm SamplingParams and pass desired kwargs in yaml
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens  # actually max new tokens
        if len(stop_sequences) == 0:
            self.stop_sequences = None
        else:
            self.stop_sequences = stop_sequences
        self.do_sample = do_sample
        self.temperature = vllm_sampling_params.get(
            "temperature", 1.0 if do_sample else 0
        )
        self.seed = seed
        self.model_dir = kwargs.get("model_dir")
        self.run_async = run_async
        self.batch_size = batch_size
        self.quantization = quantization
        self.n_gpus = n_gpus
        self.trust_remote_code = trust_remote_code
        self.sampling_params = SamplingParams(
            stop=self.stop_sequences,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **vllm_sampling_params
        )
        self.model = LLM(
            model=self.model_dir,
            quantization=self.quantization,
            seed=self.seed,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.n_gpus,
            gpu_memory_utilization=0.7,
        )

    def _generate(self, input_line: str) -> str:
        self._batch_generate([input_line])[0]

    def _batch_generate(self, input_lines: List[str]) -> List[str]:
        model_output = self.model.generate(
            input_lines, self.sampling_params, use_tqdm=True
        )
        generations = [output.outputs[0].text for output in model_output]
        return generations

    @staticmethod
    def model_name():
        return "vllm"

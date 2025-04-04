from typing import List

from vllm import LLM, SamplingParams

from tower_eval.models.inference_handler import Generator


class VLLM(Generator):
    """VLLM Generate Wrapper."""

    def __init__(
        self,
        max_tokens: int = 1024,
        stop_sequences: list = ["\n", "\\n", "</s>"],
        seed: int = 42,
        n_gpus: int = 1,
        run_async: bool = True,
        batch_size: int = 16,
        quantization: str = None,  # "awq", "gptq" or "squeezellm"
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.0,  # greedy by default
        vllm_sampling_params: dict = {},  # see vllm SamplingParams and pass desired kwargs
        vllm_engine_args: dict = {},  # see vllm LLM and pass desired kwargs
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens  # actually max new tokens
        if len(stop_sequences) == 0:
            self.stop_sequences = None
        else:
            self.stop_sequences = stop_sequences
        self.temperature = kwargs.get("temperature", temperature)
        self.seed = seed
        self.model_dir = kwargs.get("model_dir")
        self.run_async = run_async
        self.batch_size = batch_size
        self.quantization = quantization
        self.n_gpus = n_gpus
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sampling_params = SamplingParams(
            stop=self.stop_sequences,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **vllm_sampling_params,
        )
        self.model = LLM(
            model=self.model_dir,
            quantization=self.quantization,
            seed=self.seed,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.n_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            **vllm_engine_args,
        )

    def _generate(self, input_line: str) -> str:
        pass

    def _batch_generate(self, input_lines: List[str]) -> List[str]:
        model_output = self.model.generate(
            input_lines, self.sampling_params, use_tqdm=True
        )
        generations = [output.outputs[0].text for output in model_output]
        return generations

    def apply_chat_template(self, input_line: str) -> str:
        tokenizer = self.model.get_tokenizer()
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": input_line})
        input_line = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=(
                None
                if self.model_dir
                not in [
                    "openGPT-X/Teuken-7B-instruct-research-v0.4",
                    "openGPT-X/Teuken-7B-instruct-commercial-v0.4",
                ]
                else "EN"
            ),
        )
        return input_line

    @staticmethod
    def model_name():
        return "vllm"

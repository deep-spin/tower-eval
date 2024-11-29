import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StopStringCriteria,
)

from tower_eval.models.inference_handler import Generator


class HF(Generator):
    """HF Generate Wrapper."""

    def __init__(
        self,
        max_tokens: int = 1024,
        stop_sequences: list = ["\n", "\\n", "</s>"],
        seed: int = 42,
        run_async: bool = True,
        batch_size: int = 16,
        trust_remote_code: bool = True,
        temperature: float = 0.0,  # greedy by default
        strip_output: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.model_dir = kwargs.get("model_dir")
        self.run_async = run_async
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.strip_output = strip_output
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=self.trust_remote_code
        )
        if stop_sequences:
            self.stopping_criteria = StoppingCriteriaList(
                [StopStringCriteria(self.tokenizer, stop_sequences)]
            )
        else:
            self.stopping_criteria = None

        # Set up device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
        )

        torch.manual_seed(self.seed)

    def _generate(self, input_line: str) -> str:
        """Generate text for a single input."""

        # Tokenize input
        inputs = self.tokenizer(
            input_line, return_token_type_ids=False, return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            stopping_criteria=self.stopping_criteria
        )
        # decode only the generated part of the text
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] : -1]  # exclude stopping tokens
        )

        if self.strip_output:
            generated_text = generated_text.strip()

        return generated_text

    def apply_chat_template(self, input_line: str) -> str:
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": input_line})
        input_line = self.tokenizer.apply_chat_template(
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
        return "hf"

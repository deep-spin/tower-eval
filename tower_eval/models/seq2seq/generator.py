from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tower_eval.models.inference_handler import Generator
from tower_eval.models.seq2seq import NLLB_LANGUAGE_CODES


class Seq2Seq(Generator):
    """Seq2Seq Models Generation code.

    Args:
        model (str, required): The model name or path to the model.
        batch_size (int, optional): The batch size for the model. Defaults to 16.
        model_family (str, optional): The model family. Defaults to "nllb".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
        do_sample (bool, optional): Whether to use sampling or not. Defaults to False.
        gpu (int, optional): The GPU device to use. Defaults to 0.
    """

    def __init__(
        self,
        model: str = None,
        batch_size: int = 16,
        model_family: str = None,
        max_tokens: int = 1024,
        do_sample: bool = False,
        hf_generate_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.max_tokens = kwargs.get("max_tokens", max_tokens)
        self.do_sample = kwargs.get("do_sample", do_sample)
        self.model_id = kwargs.get("model", model)
        self.model_family = kwargs.get("model_family", model_family)
        self.max_tokens = kwargs.get("max_tokens", max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, device_map="auto"
        )
        self.hf_generate_kwargs = hf_generate_kwargs

    def _generate(self) -> str:
        pass

    def _batch_generate(self, input_lines: List[str]) -> List[str]:
        """It calls the model to generate the sequences.

        Args:
            input_lines (List[str]): The input lines for the model


        Returns:
            str: Returns the generated sequences.
        """
        # NLLB requires that source language be passed to the tokenizer and the target language be passed to the model
        if self.model_family == "nllb":
            self.tokenizer.src_lang = self.source_language
        inputs = self.tokenizer(input_lines, return_tensors="pt", padding=True).to(
            "cuda"
        )
        if self.model_family == "nllb":
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[
                    NLLB_LANGUAGE_CODES[self.target_language]
                ],
                do_sample=self.do_sample,
                max_new_tokens=self.max_tokens,
                **self.hf_generate_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                max_new_tokens=self.max_tokens,
                **self.hf_generate_kwargs,
            )
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        sequences = [seq.strip() for seq in sequences]

        return sequences

    @staticmethod
    def model_name():
        return "seq2seq"

import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

from loguru import logger
from tqdm import tqdm

from tower_eval.utils import (
    get_num_processed_lines,
    load_json_file,
    log_response,
    read_lines,
    save_to_json,
    write_lines,
)


class Generator(ABC):
    """Abstract class defining a shared interface for all the generators (OpenAI models as well as our internal LLMs)"""

    def __init__(self, **kwargs) -> None:
        self.batch_size = kwargs.get("batch_size", 1)
        self.strip = kwargs.get("strip", True)
        self.use_chat_template = kwargs.get("use_chat_template", False)
        self.system_prompt = kwargs.get("system_prompt", None)

    def generate(self, prompt: str, **kwargs):
        """
        The function that given the prompt sends the inference requtest to the model and gets the output.
        """
        pass

    @abstractmethod
    def _generate(self, prompt: str):
        """ """
        pass

    def _batch_generate(self, batch: List[str]):
        """ """
        return [self._generate(b) for b in batch]

    @staticmethod
    @abstractmethod
    def model_name() -> None:
        """Model name to be called for inference."""
        pass

    def assess_progress(
        self,
        input_lines: List[str],
        output_file: str,
        metadata: dict,
        metadata_file: str,
    ) -> Tuple[List[str], List[str], dict, int, int]:
        """ """
        total_lines = len(input_lines)
        if os.path.exists(output_file):
            processed_lines = read_lines(output_file, unescape_newline=True)
        else:
            processed_lines = []
        # We assume that if the metadata file exists it already contains the information of the generation times.
        # If it doesn't exist, then the metadata will be the config of the task and we will add the generation_time field to it
        if os.path.exists(metadata_file):
            metadata = load_json_file(metadata_file)
        else:
            metadata["generation_time"] = []

        num_processed_lines = get_num_processed_lines(output_file)
        assert (
            num_processed_lines <= total_lines
        ), f"MORE PROCESSED LINES ({num_processed_lines}) THAN INPUT LINES ({total_lines})!"
        # Skip the lines already processed
        input_lines = input_lines[num_processed_lines:]

        return input_lines, processed_lines, metadata, num_processed_lines, total_lines

    def apply_chat_template(self, input_line: str) -> str:
        return input_line

    def preprocess_lines(self, input_lines: List[str]) -> List[str]:
        """ """
        if self.strip:
            input_lines = [input_line.strip() for input_line in input_lines]
        else:
            input_lines = [input_line for input_line in input_lines]
        if self.use_chat_template:
            if self.model_name() in ["hf", "vllm"]:
                logger.warning("Applying chat template to loaded instructions.")
            else:
                raise NotImplementedError(
                    "Applying chat template on the fly is only supported by hf or vllm models; please set the use_chat_template flag to False."
                )
            input_lines = [
                self.apply_chat_template(input_line) for input_line in input_lines
            ]
        logger.info(f"Example processed line: {input_lines[-1]}")
        return input_lines

    def generate_to_file(
        self,
        input_lines: List[str],
        processed_lines: List[str],
        num_processed_lines: int,
        total_lines: int,
        output_file: str,
        metadata: dict,
        metadata_file: str,
    ):
        input_lines = self.preprocess_lines(input_lines)
        inference_batch_size = self.batch_size
        # for vllm, handle the case where input lines is finished
        if self.batch_size == -1:
            inference_batch_size = max(len(input_lines), 1)
        with tqdm(initial=num_processed_lines, total=total_lines) as pbar:
            for batch_id in range(0, len(input_lines), inference_batch_size):
                batch = input_lines[batch_id : batch_id + inference_batch_size]
                start_time = time.time()
                responses = self._batch_generate(batch)
                end_time = time.time()
                metadata["generation_time"].append(end_time - start_time)

                for response_id, response in enumerate(responses):
                    processed_lines.append(response.strip())
                    # Calculate the number of responses processed so far
                    step = batch_id * inference_batch_size + response_id
                    log_response(response, step=step, lim=10)
                write_lines(output_file, processed_lines, escape_newline=True)
                save_to_json(metadata_file, metadata)
                pbar.update(len(batch))

    def generation_with_resume(
        self, output_file: str, input_file: str, metadata: dict, metadata_file: str
    ):
        """
        Writes generated output to file, resuming from last line generated.
        """
        # Read all the input lines and store them in a list
        input_lines = read_lines(input_file, unescape_newline=True)
        # update input lines, given the already processed lines; store this information
        input_lines, processed_lines, metadata, num_processed_lines, total_lines = (
            self.assess_progress(input_lines, output_file, metadata, metadata_file)
        )
        # perform the generation to a file
        self.generate_to_file(
            input_lines,
            processed_lines,
            num_processed_lines,
            total_lines,
            output_file,
            metadata,
            metadata_file,
        )

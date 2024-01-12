import os
from abc import ABC, abstractmethod

from loguru import logger
from tower_eval.utils import (
    get_num_processed_lines,
    log_response,
    read_lines,
    write_lines,
)
from tqdm import tqdm


class Generator(ABC):
    """Abstract class defining a shared interface for all the generators (OpenAI models as well as our internal LLMs)"""

    def __init__(self, **kwargs) -> None:
        self.server = None
        self.run_async = True
        self.batch_size = 16

    def generate(self, prompt: str, **kwargs):
        """
        The function that given the prompt sends the inference requtest to the model and gets the output.
        """
        pass

    @abstractmethod
    def _generate(self):
        """ """
        pass

    @abstractmethod
    def _batch_generate(self):
        """ """
        pass

    @staticmethod
    @abstractmethod
    def model_name() -> None:
        """Model name to be called for inference."""
        pass

    def generation_with_resume(
        self,
        output_file: str,
        input_file: str,
    ):
        """
        Writes generated output to file, resuming from last line generated.
        """
        # Read all the input lines and store them in a list
        input_lines = read_lines(input_file, unescape_newline=True)
        total_lines = len(input_lines)
        if os.path.exists(output_file):
            processed_lines = read_lines(output_file, unescape_newline=True)
        else:
            processed_lines = []
        num_processed_lines = get_num_processed_lines(output_file)
        assert (
            num_processed_lines <= total_lines
        ), f"MORE PROCESSED LINES ({num_processed_lines}) THAN INPUT LINES ({total_lines})!"
        # Skip the lines already processed
        input_lines = input_lines[num_processed_lines:]
        input_lines = [input_line.strip() for input_line in input_lines]
        inference_batch_size = self.batch_size
        if self.run_async:
            # special batch_size case for vllm to pass all strings at once and let the model handle it
            if self.batch_size == -1:
                # handle the case where input lines is finished
                inference_batch_size = max(len(input_lines), 1)
            with tqdm(initial=num_processed_lines, total=total_lines) as pbar:
                for batch_id in range(0, len(input_lines), inference_batch_size):
                    batch = input_lines[batch_id : batch_id + inference_batch_size]
                    responses = self._batch_generate(batch)
                    for response_id, response in enumerate(responses):
                        processed_lines.append(response.strip())
                        # Calculate the number of responses processed so far
                        step = batch_id * inference_batch_size + response_id
                        log_response(response, step=step, lim=10)
                    write_lines(output_file, processed_lines, escape_newline=True)
                    pbar.update(len(batch))
        else:
            for i, input_line in enumerate(
                tqdm(input_lines, initial=num_processed_lines, total=total_lines)
            ):
                response = self._generate(input_line)
                processed_lines.append(response.strip())
                write_lines(output_file, processed_lines, escape_newline=True)
                log_response(response, step=i, lim=10)

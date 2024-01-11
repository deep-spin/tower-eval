import asyncio
from typing import List

import nest_asyncio
from loguru import logger
from text_generation import AsyncClient, Client
from tower_eval.models.exceptions import GenerationException
from tower_eval.models.inference_handler import Generator
from tower_eval.models.tgi.server import Server


class TGI(Generator):
    """TGI Generate Wrapper.

    Args:
        api_base (str, optional): TGI Server URL. Defaults to "http://127.0.0.1:8080"
        retry_max_attempts (int, optional): Maximum number of retries. Defaults to 1.
        retry_max_interval (int, optional): Maximum interval between retries. Defaults to 10.
        retry_min_interval (int, optional): Minimum interval between retries. Defaults to 4.
        retry_multiplier (int, optional): Multiplier for the retry interval. Defaults to 1.
    """

    def __init__(
        self,
        api_base: str = None,
        conda_path: str = None,
        port: str = "2233",
        max_tokens: int = 1024,
        stop_sequences: list = ["\n", "\\n", "</s>"],
        do_sample: bool = False,
        top_k: int = None,
        seed: int = None,
        venv: str = "tgi-env",
        gpu: str = "0",
        retries: int = 10,
        wait_interval: int = 10,
        run_async: bool = True,
        batch_size: int = 16,
        max_input_length: str = "4096",
        max_total_length: str = "6000",
        quantize: str = "none",  # can be "bitsandbytes"
        debug: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens  # actually max new tokens
        self.stop_sequences = stop_sequences
        self.do_sample = do_sample
        self.top_k = top_k
        self.seed = seed
        self.api_base = api_base
        self.conda_path = conda_path
        self.model_dir = kwargs.get("model_dir")
        self.port = port
        self.venv = venv
        self.gpu = gpu
        self.retries = retries
        self.wait_interval = wait_interval
        self.run_async = run_async
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_total_length = max_total_length
        self.quantize = quantize
        self.debug = debug
        # If the server url is given, use the existing server, otherwise try to launch it locally
        self.server = Server(
            api_base=self.api_base,
            conda_path=self.conda_path,
            model_dir=self.model_dir,
            port=self.port,
            venv=self.venv,
            gpu=self.gpu,
            retries=self.retries,
            wait_interval=self.wait_interval,
            max_input_length=self.max_input_length,
            max_total_length=self.max_total_length,
            quantize=self.quantize,
            debug=debug,
        )
        self.api_base = self.server.setup_server()
        # Set the client
        if self.run_async:
            nest_asyncio.apply()
            self.client = AsyncClient(self.api_base, timeout=120)
        else:
            self.client = Client(self.api_base, timeout=120)

    def _generate(self, input_line: str) -> str:
        """It calls the Chat completion function of OpenAI.

        Args:
            input_line (str): The prompt for the model


        Returns:
            str: Returns the generated response. For now, it is only the generated text without any additional information.
        """
        try:
            response = self.client.generate(
                input_line,
                max_new_tokens=self.max_tokens,
                do_sample=self.do_sample,
                top_k=self.top_k,
                seed=self.seed,
                stop_sequences=self.stop_sequences,
            )
        except Exception as e:
            raise GenerationException(str(e))

        return response.generated_text.strip("</s>").strip().strip("\\n")

    def _batch_generate(self, input_lines: List[str]) -> List[str]:
        """It calls the Async client of TGI.

        Args:
            input_lines (List[str]): The input prompts for the model


        Returns:
            str: Returns the generated responses.
        """
        try:
            responses = asyncio.run(self._async_batch_generate(batch=input_lines))
            responses = [
                response.generated_text.strip("</s>").strip().strip("\\n")
                for response in responses
            ]
        except Exception as e:
            raise GenerationException(str(e))

        return responses

    async def _async_batch_generate(self, batch):
        return await asyncio.gather(
            *[
                self.client.generate(
                    sample,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.do_sample,
                    top_k=self.top_k,
                    seed=self.seed,
                    stop_sequences=self.stop_sequences,
                )
                for sample in batch
            ]
        )

    @staticmethod
    def model_name():
        return "tgi"

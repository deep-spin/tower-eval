import os
import subprocess
from time import sleep

import psutil
import requests
from loguru import logger
from tower_eval.models.inference_handler import Generator


class Server:
    """The Class responsible for creating the TGI Server

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
        model_dir: str = None,
        port: str = "8080",
        venv: str = "tgi-env",
        gpu: str = "0",
        retries: int = 1,
        wait_interval=5,
        max_input_length: str = "4096",
        max_total_length: str = "6000",
        quantize: str = "none",  # can be "bitsandbytes"
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.api_base = api_base
        self.conda_path = conda_path
        self.model_dir = model_dir
        self.port = port
        self.venv = venv
        self.gpu = gpu
        self.retries = retries
        self.wait_interval = wait_interval
        self.max_input_length = max_input_length
        self.max_total_length = max_total_length
        self.debug = debug
        if quantize not in ["gptq", "bitsandbytes"]:
            logger.opt(ansi=True).warning(
                f"<red>Quantization method {quantize} not supported.</red>Not quantizing instead."
            )
            quantize = "none"
        self.quantize = quantize

    def setup_server(self):
        if self.api_base:
            logger.opt(ansi=True).info(
                f"Using the pre-launched server at {self.api_base}"
            )
            self.is_local = False
        else:
            if self.model_dir:
                logger.opt(ansi=True).info(
                    f"Creating the server for <yellow>{self.model_dir}</yellow> "
                    + f"on port <yellow>{self.port}</yellow> "
                    + f"It will run on <yellow>gpu: {self.gpu}</yellow>"
                )
                self._create_server()
                self.api_base = f"http://127.0.0.1:{self.port}"
                if self.is_server_up(self.api_base):
                    logger.opt(ansi=True).info(f"Server launched at {self.api_base}")
                else:
                    logger.opt(ansi=True).info(f"Creating the server failed")
                    logger.opt(ansi=True).warning(
                        f"This is not normal; set the <yellow>debug</yellow> argument to <blue>True</blue> in your config file to enable debugging of the server creation command."
                    )
                    exit(1)
                self.is_local = True
            else:
                logger.opt(ansi=True).info(
                    f'To launch the server locally you need to specify the "model_dir" path'
                )
                exit(1)

        return self.api_base

    def _create_server(self):
        if self.debug:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.PIPE, subprocess.PIPE
        try:
            current_dir = os.getcwd()
            if self.quantize != "none":
                logger.opt(colors=True).warning(
                    f"<red>Quantizing model using {self.quantize}</red>"
                )
            self.process = subprocess.Popen(
                [
                    "bash",
                    f"{current_dir}/tower_eval/models/tgi/run_server.sh",
                    "-m",
                    self.model_dir,
                    "-p",
                    self.port,
                    "-e",
                    self.venv,
                    "-g",
                    self.gpu,
                    "-i",
                    self.max_input_length,
                    "-t",
                    self.max_total_length,
                    "-c",
                    self.conda_path,
                    "-q",
                    self.quantize,
                ],
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as e:
            print(f"Error running the script: {e}")
        return

    def is_server_up(self, server_address):
        for i in range(self.retries):
            logger.opt(ansi=True).info(
                f"Checking server, attempt: {i+1}/{self.retries}"
            )
            try:
                response = requests.get(server_address)
                if response.status_code == 200:
                    return True
            except:
                sleep(self.wait_interval)
        return False

    def close_server(self):
        if self.is_local:
            logger.opt(ansi=True).info("Terminating the server")
            for child in psutil.Process(self.process.pid).children(recursive=True):
                logger.opt(ansi=True).info(f"<red>Terminating {child.name()}</red>")
                child.kill()
            logger.opt(ansi=True).info("Server terminated")

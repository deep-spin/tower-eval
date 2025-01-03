# -*- coding: utf-8 -*-
import os

import anthropic

from tower_eval.models.inference_handler import Generator
from tower_eval.utils import generate_with_retries


class Anthropic(Generator):
    """Anthropic API wrapper."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        retry_max_attempts: int = 1,
        retry_max_interval: int = 10,
        retry_min_interval: int = 4,
        retry_multiplier: int = 1,
        stop_sequences: list[str] = [],
        system_prompt: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run_async = False  # only sync calls are supported
        # Set anthropic settings
        model = kwargs.get("model", model)
        temperature = kwargs.get("temperature", temperature)
        top_p = kwargs.get("top_p", top_p)
        max_tokens = kwargs.get("max_tokens", max_tokens)
        stop_sequences = kwargs.get("stop_sequences", stop_sequences)
        system_prompt = system_prompt
        self.model_args = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop_sequences": stop_sequences,
        }
        if system_prompt is not None:
            self.model_args["system"] = system_prompt

        # Generations object / retry args
        self.client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=os.environ.get("ANTHROPIC_API_KEY", api_key),
        )
        self.retry_max_attempts = kwargs.get("retry_max_attempts", retry_max_attempts)
        self.retry_max_interval = kwargs.get("retry_max_interval", retry_max_interval)
        self.retry_min_interval = kwargs.get("retry_min_interval", retry_min_interval)
        self.retry_multiplier = kwargs.get("retry_multiplier", retry_multiplier)

    def _generate(self, input_line: str) -> str:
        """It calls the Chat completion function of anthropic.

        Args:
            prompt (str): Prompt for the anthropic model


        Returns:
            str: Returns the response used.
        """
        prompt = {"messages": [{"role": "user", "content": input_line}]}
        response = generate_with_retries(
            retry_function=self.client.messages.create,
            model_args=self.model_args | prompt,
            retry_max_attempts=self.retry_max_attempts,
            retry_multiplier=self.retry_multiplier,
            retry_min_interval=self.retry_min_interval,
            retry_max_interval=self.retry_max_interval,
        )

        response = response.content[0].text
        return response

    @staticmethod
    def model_name():
        return "anthropic"

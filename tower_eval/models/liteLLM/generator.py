# -*- coding: utf-8 -*-
import litellm
from tower_eval.models.inference_handler import Generator

import logging
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

class LiteLLM(Generator):
    """LiteLLM completion Wrapper.
    """

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        frequency_penalty: float = 0.0,
        retry_max_attempts: int = 1,
        stop_sequences: list[str] = [],
        run_async: bool = False,
        system_prompt: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run_async = False  # only sync calls are supported
        model = kwargs.get("model", model)
        temperature = kwargs.get("temperature", temperature)
        top_p = kwargs.get("top_p", top_p)
        max_tokens = kwargs.get("max_tokens", max_tokens)
        frequency_penalty = kwargs.get("frequency_penalty", frequency_penalty)
        stop_sequences = kwargs.get("stop_sequences", stop_sequences)
        self.system_prompt = system_prompt
        self.run_async = run_async
        self.model_args = {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        self.retry_max_attempts = kwargs.get("retry_max_attempts", retry_max_attempts)


    def _generate(self, input_line: str) -> str:
        """Generate response using LiteLLM.

        Args:
            input_line (str): Input prompt for the model

        Returns:
            str: Generated response text
        """
        # Build messages for chat completion
        messages = []
        # Add system message if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            
        # Add user message
        messages.append({"role": "user", "content": input_line})
        
        # Prepare arguments for litellm.completion
        completion_args = self.model_args.copy()
        completion_args["messages"] = messages
        response = litellm.completion(
            **completion_args,
            num_retries=self.retry_max_attempts,
        )
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response = response.choices[0].message.content
            if response is None:
                return ""
            else:
                return response
        else:
            raise ValueError("Invalid response format from LiteLLM")

    @staticmethod
    def model_name():
        return "litellm"
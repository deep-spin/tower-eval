# -*- coding: utf-8 -*-
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

import base64

import vertexai.preview.generative_models as generative_models

from tower_eval.models.exceptions import GenerationException
from tower_eval.models.inference_handler import Generator
from tower_eval.utils import generate_with_retries
from tower_eval.models.vertexAI import API_TYPE
from loguru import logger


class VertexAI(Generator):
    """Google's Vertex AI Wrapper.

    Args:
        model: the name of the model to use for the inference (default: gemini-pro)
        temprature: the temprature
        top_p: Defines the cumulative probability cutoff for token selection.
        max_tokens: determines the maximum number of tokens the model is supposed to generate.
        api_type (str, optional): OpenAI API Type. Defaults to OpenAI.
        retry_max_attempts (int, optional): Maximum number of retries. Defaults to 1.
        retry_max_interval (int, optional): Maximum interval between retries. Defaults to 10.
        retry_min_interval (int, optional): Minimum interval between retries. Defaults to 4.
        retry_multiplier (int, optional): Multiplier for the retry interval. Defaults to 1.
    """

    def __init__(
        self,
        model: str = "gemini-pro",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        candidate_count: int = 1,
        retry_max_attempts: int = 1,
        retry_max_interval: int = 10,
        retry_min_interval: int = 4,
        retry_multiplier: int = 1,
        run_async: bool = False,
        system_prompt: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run_async = run_async
        self.retry_max_attempts = retry_max_attempts
        self.retry_max_interval = retry_max_interval
        self.retry_min_interval = retry_min_interval
        self.retry_multiplier = retry_multiplier
        # Gimini and PaLM models need to be called through different model APIs with some small differences.
        # But, to avoid having two different inference endpoint in Tower-Eval we decided to handle both of them here.
        self.model_type = API_TYPE.get(model)
        if self.model_type == "gemini-1.5":
            try:
                project=kwargs["project"]
                location=kwargs["location"]
            except:
                logger.opt(colors=True).error(f"<red>For Gemeni-1.5 models you need to provide \"project\" and \"location\" in your config file.</red>")
            vertexai.init(project=project, location=location)
            from vertexai.generative_models import GenerativeModel
            logger.info(f"Using the following system prompt: {system_prompt}")
            self.model = GenerativeModel(model, system_instruction=system_prompt)
            self.inference_function = self.model.generate_content
            self.model_args = {
                "generation_config": {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        elif self.model_type == "gemini":
            from vertexai.preview.generative_models import GenerativeModel
            self.model = GenerativeModel(model, system_instruction=system_prompt)
            self.inference_function = self.model.generate_content
            self.model_args = {
                "generation_config": {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        elif self.model_type == "palm":
            logger.info(f"Model \"{model}\" doesn't support system prompt. So, running the inference with user prompt only.")
            self.model = TextGenerationModel.from_pretrained(model)
            self.inference_function = self.model.predict
            self.model_args = {
                "candidate_count": candidate_count,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            logger.opt(colors=True).info(
                f"<red>Model {model} is not supported by Vertex AI.</red>"
            )
            exit(1)

    def _generate(self, input_line: str) -> str:
        """It calls the generate_content() function of VertexAI.

        Args:
            input_line (str): Prompt for the model

        Returns:
            str: Returns the response used.
        """
        try:
            if self.model_type in ["gemini", "gemini-1.5"]:
                prompt = {"contents": input_line}
            elif self.model_type == "palm":
                prompt = {"prompt": input_line}

            responses = generate_with_retries(
                retry_function=self.inference_function,
                model_args=self.model_args | prompt,
                retry_max_attempts=self.retry_max_attempts,
                retry_multiplier=self.retry_multiplier,
                retry_min_interval=self.retry_min_interval,
                retry_max_interval=self.retry_max_interval,
            )
        except Exception as e:
            raise GenerationException(str(e))

        return responses.text

    @staticmethod
    def model_name():
        return "vertex-ai"

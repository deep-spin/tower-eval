# -*- coding: utf-8 -*-
import os

from loguru import logger
import deepl
from deepl import DocumentTranslationException, DeepLException

import logging
logging.getLogger('deepl').setLevel(logging.WARNING)
from tower_eval.models.inference_handler import Generator
from tower_eval.utils import generate_with_retries


class DeepL(Generator):
    """DeepL Wrapper.

    Args:
        api_key (str): DeepL API Key
        model (str): Specifies which DeepL model should be used for translation. 
            - "latency_optimized": the classic translation model of DeepL with lower latency that support all language pairs; default value
            - "quality_optimized": uses higher latency, improved quality “next-gen” translation models, which support only a subset of language pairs;
                                    if a language pair that is not supported by next-gen models is included in the request, it will fail.
            - "prefer_quality_optimized": prioritizes use of higher latency, improved quality “next-gen” translation models, which support only a subset of DeepL languages;
                                    if a request includes a language pair not supported by next-gen models, the request will fall back to latency_optimized classic models)
            Check this link for more information:  https://developers.deepl.com/docs/api-reference/translate#request-body-descriptions
        retry_max_attempts (int, optional): Maximum number of retries. Defaults to 1.
        retry_max_interval (int, optional): Maximum interval between retries. Defaults to 10.
        retry_min_interval (int, optional): Minimum interval between retries. Defaults to 4.
        retry_multiplier (int, optional): Multiplier for the retry interval. Defaults to 1.
    """

    def __init__(
        self,
        api_key: str = os.environ["DEEPL_API_KEY"],
        model: str = "latency_optimized",
        retry_max_attempts: int = 1,
        retry_max_interval: int = 10,
        retry_min_interval: int = 4,
        retry_multiplier: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.run_async = False  # only sync calls are supported
        # Set openai settings
        self.model_type = kwargs.get("model", model)
        self.retry_max_attempts = kwargs.get("retry_max_attempts", retry_max_attempts)
        self.retry_max_interval = kwargs.get("retry_max_interval", retry_max_interval)
        self.retry_min_interval = kwargs.get("retry_min_interval", retry_min_interval)
        self.retry_multiplier = kwargs.get("retry_multiplier", retry_multiplier)
        self.client = deepl.Translator(api_key)
        
        self.src_lang_map = {
            "es-latam": "es",
            "en-gb": "en",
            "en-us": "en",
            "en-uk": "en",
            "pt-br": "pt",
            "zh-tw": "zh",
            "zh-cn": "zh",
        }
        
        self.trg_lang_map = {
            "pt": "pt-pt",
            "en": "en-us",
            "es-latam": "es",
            "zh": "zh-hans",
            "zh-cn": "zh-hans",
            "zh-tw": "zh-hant",
        }

    def normalize_languages(self):
        # Valid languages per vendor
        valid_src_lang = self.src_lang_map.get(self.source_language.lower(), self.source_language)
        valid_trg_lang = self.trg_lang_map.get(self.target_language.lower(), self.target_language)

        if valid_src_lang != self.source_language:
            logger.warning(
                f"Source language ({self.source_language}) not supported by DeepL. " f"Using {valid_src_lang} instead"
            )
        if valid_trg_lang != self.target_language:
            logger.warning(
                f"Target language ({self.target_language}) not supported by DeepL. " f"Using {valid_trg_lang} instead"
            )

        self.source_language = valid_src_lang
        self.target_language = valid_trg_lang

    def _generate(self, input_line: str, context: str = None, formality: str = "default") -> str:
        """It calls the translate_text function of DeepL.

        Args:
            input_line (str): The text to be translated.
            context (str, optional): The context for the translation. Defaults to None.


        Returns:
            str: Returns the translated text.
        """
        self.normalize_languages()
        
        try:
            response = generate_with_retries(
                retry_function=self.client.translate_text,
                model_args={"text": input_line,
                            "context": context,
                            "source_lang": self.source_language, 
                            "target_lang": self.target_language,
                            "formality": formality,
                            "model_type": self.model_type},
                retry_max_attempts=self.retry_max_attempts,
                retry_multiplier=self.retry_multiplier,
                retry_min_interval=self.retry_min_interval,
                retry_max_interval=self.retry_max_interval,
            )
        except DocumentTranslationException as e:
            # If an error occurs during document translation after the document was
            # already uploaded, a DocumentTranslationException is raised. The
            # document_handle property contains the document handle that may be used to
            # later retrieve the document from the server, or contact DeepL support.
            doc_id = e.document_handle.id
            doc_key = e.document_handle.key
            logger.opt(colors=True).error(f"<red>Error after uploading ${e}, id: ${doc_id} key: ${doc_key}</red>")
        except DeepLException as e:
            # Errors during upload raise a DeepLException
            logger.opt(colors=True).error(f"<red>{e}</red>")

        response = response.text
        return response

    @staticmethod
    def model_name():
        return "deepl"

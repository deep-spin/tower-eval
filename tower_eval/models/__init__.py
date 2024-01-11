try:
    from tower_eval.models.openAI.generator import OpenAI
    from tower_eval.models.tgi.generator import TGI
    from tower_eval.models.vllm.generator import VLLM

    __all__ = [OpenAI, TGI, VLLM]

    available_models = {model.model_name(): model for model in __all__}

except ModuleNotFoundError:
    from tower_eval.models.openAI.generator import OpenAI
    from tower_eval.models.tgi.generator import TGI

    __all__ = [OpenAI, TGI]

    available_models = {model.model_name(): model for model in __all__}

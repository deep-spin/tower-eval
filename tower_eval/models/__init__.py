from tower_eval.models.anthropic.generator import Anthropic
from tower_eval.models.openAI.generator import OpenAI
from tower_eval.models.tgi.generator import TGI
from tower_eval.models.vertexAI.generator import VertexAI
from tower_eval.models.vllm.generator import VLLM

available_models = {
    model.model_name(): model for model in [OpenAI, TGI, VLLM, VertexAI, Anthropic]
}

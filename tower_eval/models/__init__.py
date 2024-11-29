from tower_eval.models.anthropic.generator import Anthropic
from tower_eval.models.cohere.generator import Cohere
from tower_eval.models.hf.generator import HF
from tower_eval.models.openAI.generator import OpenAI
from tower_eval.models.seq2seq.generator import Seq2Seq
from tower_eval.models.vertexAI.generator import VertexAI
from tower_eval.models.vllm.generator import VLLM

available_models = {
    model.model_name(): model
    for model in [OpenAI, VLLM, VertexAI, Anthropic, Cohere, Seq2Seq, HF]
}

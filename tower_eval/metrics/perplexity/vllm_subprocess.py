import vllm
from jsonargparse import CLI
from loguru import logger

from tower_eval.metrics.perplexity.metric import Perplexity
from tower_eval.utils import tokenize_text


def main(gold_data_path: str, model_id: str, max_model_context: int):
    # load data
    gold_data = Perplexity._handle_inputs(gold_data_path)
    references = gold_data["text"]

    # vllm
    llm = vllm.LLM(model=model_id, enforce_eager=True, gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()
    sampling_params = vllm.SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=1
    )
    tokenized_prompts = tokenize_text(references, tokenizer)
    truncated_prompts = Perplexity.truncate_prompts(
        tokenized_prompts, max_model_context
    )
    logger.warning(
        f"Truncating prompts to fit model max context of {max_model_context}"
    )
    model_output = llm.generate(
        prompt_token_ids=truncated_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    perplexities, mean_perplexity = Perplexity.get_perplexity_from_vllm_output(
        model_output
    )

    print(f"""PERPLEXITIES: {perplexities}\n PERPLEXITY: {mean_perplexity}""")


if __name__ == "__main__":
    CLI([main], as_positional=False)

import re

from datasets import load_dataset
from jsonargparse import CLI
from vllm import LLM, SamplingParams


def extract_first_word(input_string):
    # Define a regular expression pattern to match the first word
    pattern = r"\b\w+\b"

    # Use the findall function from the re module to find all matches
    matches = re.findall(pattern, input_string)

    if matches:
        return matches[0]
    else:
        return ""


def main(model_dir: str):
    # Load the model
    model = LLM(model_dir)
    s = SamplingParams(temperature=0.0, max_tokens=10)
    for subset in ["generated", "handwritten"]:
        dataset = load_dataset("NOVA-vision-language/calame-pt", subset)["train"]
        input_lines = dataset["sentence"]
        gold_words = dataset["last_word"]
        # generate
        model_outputs = model.generate(input_lines, sampling_params=s, use_tqdm=True)
        # replicate calame script
        generations = [
            output.outputs[0].text.replace("\n", "") for output in model_outputs
        ]
        # Extract first predicted word
        predicted_last_words = [extract_first_word(g).strip() for g in generations]
        correct_predictions = [
            1 if p.lower() == g.lower() else 0
            for p, g in zip(predicted_last_words, gold_words)
        ]
        accuracy = 100 * sum(correct_predictions) / len(correct_predictions)
        print(f"Accuracy for {subset} subset: {accuracy}")


if __name__ == "__main__":
    CLI([main], as_positional=False)

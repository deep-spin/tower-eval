import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import spacy
import yaml
from loguru import logger
from mosestokenizer import MosesDetokenizer, MosesTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

PathInput = Union[str, Path]
PATTERN_OPEN_TAG = r"<\s*([\w]+)\s*>\s*"
PATTERN_CLOSE_TAG = r"\s*<\s*\/\s*([\w]+)\s*>"
PATTERN_SHOT_NAME = r"^([0-9]+)[-_]shot.*$"

SPACY_LANGUAGE_TO_MODEL = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "it": "it_core_news_sm",
    "pl": "pl_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "zh": "zh_core_web_sm",
    "ko": "ko_core_news_sm",
}


def args_to_dict(args, prefix: str, strip_prefix: bool = False):
    """Filters argparse's `Namespace` into dictionary with arguments
    beginning with the given prefix."""
    prefix += "_"
    d = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix) or k in ["language", "tokenize", "lowercase"]:
            k = k.replace(prefix, "") if strip_prefix else k
            d[k] = v
    return d


def detokenize(lines: list[str]) -> list[str]:
    detokenizer = TreebankWordDetokenizer()
    detokenized_lines = [detokenizer.detokenize(line.split()) for line in lines]
    return detokenized_lines


def detokenize_moses(lines: list[str], lang: str) -> list[str]:
    detokenizer = MosesDetokenizer(lang)
    detokenized_lines = [detokenizer(line.split()) for line in lines]
    return detokenized_lines


def tokenize_spacy(lines: list[str], language, keep_xml=False) -> list[str]:
    spacy_model = SPACY_LANGUAGE_TO_MODEL[language]
    try:
        tokenizer = spacy.load(spacy_model, disable=["tagger", "parser", "ner"])
    except OSError:
        logger.warning(f"Spacy model {spacy_model} not found. Trying to download...")
        download_proc = subprocess.run(
            ["python", "-m", "spacy", "download", spacy_model]
        )
        tokenizer = spacy.load(spacy_model, disable=["tagger", "parser", "ner"])

    tokenized = list(tokenizer.pipe(lines))
    tokenized = [[token.text for token in sentence] for sentence in tokenized]
    output_lines = [" ".join(sentence) for sentence in tokenized]
    if keep_xml:
        output_lines = [
            re.sub(PATTERN_OPEN_TAG, r"<\1>", output_line)
            for output_line in output_lines
        ]
        output_lines = [
            re.sub(PATTERN_CLOSE_TAG, r"</\1>", output_line)
            for output_line in output_lines
        ]
    return output_lines


def parse_yaml_config(config_path):
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    return config_data


def get_task_args(args: dict) -> dict:
    source = args.get("source")
    hypothesis = args.get("hypothesis")
    references = args.get("references")

    task_args = {
        "hypothesis": hypothesis,
        "references": references,
    }
    if source:
        task_args.update({"source": source})
    return task_args


def load_json_file(
    file_path: PathInput,
) -> list:
    """
    Function to to load json files.

    :param file_path: either the handler to the file storing the hypotheses.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl_file(
    file_path: PathInput,
) -> list:
    """
    Function to parse error span jsonl files.
    All inputs will be returned as list of dictionaries.

    :param file_path: either the handler to the file storing the hypotheses.
        Must be a jsonl file where each line is a dict which has at least the following keys:
        {"mt": <mt>, "annotations": [{"start": <start>, "end": <end>, "severity": <severity>}, ...]}
    :return:
        -  List of dictionaries
    """
    with open(file_path, "r") as f:
        json_list = list(f)

    return [json.loads(l) for l in json_list]


def list_to_dict(l: list[dict[str, str]]) -> dict[str, list[str]]:
    """
    Assumes lists in dict have the same length.
    """
    d = {}
    for k in l[0].keys():
        d[k] = [d[k] for d in l]
    return d


def dict_to_records(d: dict[str, list[str]]) -> list[dict[str, str]]:
    """
    Assumes lists in dict have the same length.
    """
    records = []
    for i in range(len(list(d.values())[0])):
        records.append({k: v[i] for k, v in d.items()})

    return records


def write_lines(
    path: PathInput,
    lines: Iterable[str],
    escape_newline: bool = False,
) -> None:
    """Writes lines to a file.

    Lines can be escaped, meaning \n is transformed to \\n.

    Args:
        path: The path to the file.
        lines: The lines to write.
        escape_newline: Whether to escape newlines.
    """
    # make dir, if not exists
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if escape_newline:
        lines = (l.replace("\n", "\\n") for l in lines)
    with open(path, "w") as f:
        f.writelines((f"{l}\n" for l in lines))


def combine_metrics_args(task_metric: dict, subtask_metric: dict) -> dict:
    eval_metric = task_metric.copy()
    for metric_arg, metric_arg_value in subtask_metric.items():
        eval_metric[metric_arg] = metric_arg_value
    return eval_metric


def generate_with_retries(
    retry_function,
    model_args,
    retry_max_attempts=1,
    retry_multiplier=1,
    retry_max_interval=10,
    retry_min_interval=4,
):
    retryer = Retrying(
        stop=stop_after_attempt(retry_max_attempts),
        wait=wait_random_exponential(
            multiplier=retry_multiplier, max=retry_max_interval, min=retry_min_interval
        ),
        reraise=True,
    )
    return retryer(retry_function, **model_args)


def sample_strings_from_list(strings: list[str], num_samples: int) -> list[str]:
    """ """
    return np.random.choice(strings, num_samples).tolist()


def read_lines(path: PathInput, unescape_newline: bool = False) -> List[str]:
    """Reads lines from a file.
    Lines can be unescapped, meaning \\n is transformed to \n.
    Args:
        path: The path to the file.
        unescape_newline: Whether to unescape newlines.
    Returns:
        The lines in the file."""
    with open(path) as f:
        lines = [l[:-1] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines


def make_dir_if_not_exists(output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def log_response(response: str, step: int, lim: int = 10) -> None:
    """
    Logs the response from the model.
    """
    if step < lim:
        if "<" in response:
            response = response.replace("<", "\\<")
            logger.opt(colors=True).info(
                f"<red> Escaped tags in below response to not throw colorizing error. </red>"
            )
        logger.opt(colors=True).info(f"Response {step}: {response.strip()}")


def get_num_processed_lines(output_file: str, **kwargs):
    """
    This function is used to write the output from the last line of the input file.
    """
    if os.path.exists(output_file):
        processed_lines = read_lines(output_file, unescape_newline=False)
        num_processed_lines = len(processed_lines)
        if num_processed_lines > 0:
            logger.opt(ansi=True).warning(
                f"<red>Resuming generation from line {num_processed_lines}.</red>"
            )
    else:
        num_processed_lines = 0

    return num_processed_lines


def load_data_to_records(
    data_path: str,
) -> list[dict[str, str]]:
    """ """
    # prompt data file can either be a jsonl or a csv
    data = load_jsonl_file(data_path)

    return data


def save_to_json(
    save_location: PathInput,
    data: dict,
) -> None:
    """ """
    if isinstance(save_location, str):
        save_location = Path(save_location)
    save_location.parent.mkdir(parents=True, exist_ok=True)
    with open(save_location, "w") as output_path:
        json.dump(data, output_path, indent=4)


def get_sacrebleu_segment_scores(
    hypotheses: list[str],
    references,
    method,
) -> list[float]:
    """ """
    segment_scores = []
    for h, r in zip(hypotheses, references):
        segment_score = method.sentence_score(h, r)
        segment_scores.append(segment_score.score)
    return segment_scores


def get_eval_args_given_task(
    eval_args: Dict[str, str],
    task_name: str,
    data_dir: Path,
    subtask: str,
    output_dir: Path,
    model_type: str,
    model_name: str,
) -> Dict[str, str]:
    """ """
    if task_name == "gec":
        eval_args["references_m2"] = data_dir / task_name / subtask / "test_corpus.m2"
    generations_dir = Path(str(output_dir).replace("evaluations", "generations"))
    eval_args["hypothesis_path"] = (
        generations_dir
        / task_name
        / subtask
        / model_type
        / model_name
        / "generation.txt"
    )
    eval_args["gold_data_path"] = data_dir / task_name / subtask / "test.jsonl"
    # add language argument to eval_args as it is needed in some metrics
    language = subtask.split(".")[1]
    if "-" in language:
        language = language.split("-")[1]
    eval_args["language"] = language

    return eval_args


def text_to_label(
    text: str,
    text_type: str,
    label_space: List[str] = None,
    return_is_random: bool = False,
) -> int:
    is_random = False
    if text_type == "categorical":
        label = int(text.strip())
        is_random = False
    # if hypothesis is text
    elif text_type == "text":
        label = np.random.randint(len(label_space))
        is_random = True
        for l in label_space:
            if re.search(l, text, re.IGNORECASE) is not None:
                label = label_space.index(l)
                is_random = False
                break
    if return_is_random:
        return label, is_random
    else:
        return label

import os
from typing import Callable

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tower_eval.utils import load_data_to_records


def random_retrieval(
    examples: list[dict[str, str]],
    n_shots: int,
    total_examples: int,
    **kwargs,
) -> list[list[dict[str, str]]]:
    examples_idxs = np.random.choice(len(examples), total_examples)
    examples = [examples[i] for i in examples_idxs]

    # split_idxs are the indices where we should split the examples
    # If n_shots = 2, then split_idxs = [2, 4, 6]
    # meaning the first 2 examples go for instance 1, the next 2 for instance 2, etc.
    examples_per_instance = []
    for i in range(0, total_examples - n_shots + 1, n_shots):
        examples_per_instance.append(examples[i : i + n_shots])

    return [list(examples) for examples in examples_per_instance]


def ordered_retrieval(
    examples: list[dict[str, str]],
    n_shots: int,
    total_examples: int,
    **kwargs,
) -> list[list[dict[str, str]]]:
    examples_per_instance = []
    for _ in range(0, total_examples - n_shots + 1, n_shots):
        examples_per_instance.append(examples[:n_shots])

    return [list(examples) for examples in examples_per_instance]


def get_similar_examples(n_examples, encoder, index, seed_sentence):
    # encode the seed sentence
    question_embedding = encoder.encode(seed_sentence)

    # search for similar examples
    dists, I = index.search(
        torch.FloatTensor(question_embedding).unsqueeze(0), n_examples
    )
    idxs = I[0]

    return idxs, dists


def similarity_retrieval(
    test_set: list[dict[str, str]],
    examples: list[dict[str, str]],
    n_shots: int,
    **kwargs,
) -> list[list[dict[str, str]]]:
    encoder = SentenceTransformer(
        "sentence-transformers/LaBSE",
        #   device="cpu"
    )

    # load faiss index
    index_file = os.path.join(kwargs["index_path"], "knn.index")
    index = faiss.read_index(index_file)

    selected_examples = []
    # i = 0
    for test_setence in test_set:
        selected_examples_per_instance = []
        # get similar examples
        idxs, dists = get_similar_examples(n_shots, encoder, index, test_setence["src"])
        for idx in idxs:
            selected_examples_per_instance.append(examples[idx])
        selected_examples.append(selected_examples_per_instance)
    return selected_examples


def force_label_balance_retrieval(
    examples: list[dict[str, str]],
    n_shots: int,
    total_examples: int,
    task: str,
    n_positive: int = 1,
    retrieval: str = "random",
    **kwargs,
) -> list[list[dict[str, str]]]:
    """
    Gets few shot examples for APE such that, for each instance, a minimum of positive examples (for which no PE is required) are included in the prompt.
    Once example pools are separated, the remaining examples are sampled randomly or ordered, depending on method choice.
    """
    assert n_positive < n_shots, "n_positive should be less than or equal to n_shots."
    n_negative = n_shots - n_positive
    total_positive_examples = n_positive * (total_examples // n_shots)
    total_negative_examples = n_negative * (total_examples // n_shots)
    # iterate over examples to split them between positive and negative ones
    positive_examples, negative_examples = get_positive_negative_examples_from_task(
        examples, task
    )
    # sample positive examples
    _retrieval_func = get_fewshot_retrieval_method(retrieval)
    positive_examples = _retrieval_func(
        positive_examples, n_positive, total_positive_examples
    )
    negative_examples = _retrieval_func(
        negative_examples, n_negative, total_negative_examples
    )
    # combine positive and negative examples
    out_examples = []
    for positive, negative in zip(positive_examples, negative_examples):
        joined_examples = positive + negative
        np.random.shuffle(joined_examples)
        # shuffle the joined list to avoid having all positive examples at the beginning
        out_examples.append(joined_examples)
    return out_examples


def get_positive_negative_examples_from_task(examples: list[dict[str, str]], task: str):
    """
    Gets positive and negative examples from a list of examples, given a task.
    """
    positive_examples = []
    negative_examples = []
    # each "e" is an example inside examples, which corresponds to a row in the raw data's dataframe
    positive_label_condition: Callable[[dict[str, str]], bool] = None
    if task in ["ape"]:
        positive_label_condition = lambda e: e["mt"] == e["ref"]
    elif task in ["paraphrase_identification", "word_sense_disambiguation"]:
        positive_label_condition = lambda e: e["answer"] == "Yes"
    elif task in ["gec"]:
        positive_label_condition = lambda e: e["src"] == e["ref"]
    else:
        raise NotImplementedError(
            f"Retrieval with forced label balance is not implemented for task {task}."
        )
    for e in examples:
        if positive_label_condition(e):
            positive_examples.append(e)
        else:
            negative_examples.append(e)
    return positive_examples, negative_examples


def get_fewshot_retrieval_method(method: str) -> Callable:
    """Returns a few shot retrieval function, given a method name. Handles exception when method name is not implemented"""
    available_fewshot_retrieval_methods = {
        "random": random_retrieval,
        "ordered": ordered_retrieval,
        "force_label_balance": force_label_balance_retrieval,
        "similarity": similarity_retrieval,
    }
    if method is not None:
        try:
            fewshot_retrieval_method = available_fewshot_retrieval_methods[method]
        except KeyError as e:
            e(
                f"{method} fewshot retrieval method is not implemented. Please choose from {list(available_fewshot_retrieval_methods.keys())}."
            )
    else:
        fewshot_retrieval_method = None
    return fewshot_retrieval_method


def load_few_shot_data(
    test_set: list[dict[str, str]],
    fewshot_data_path: str,
    n_fewshots: int,
    total_shots: int,
    fewshot_retrieval_method: str,
    task: str,
    index_path: str = None,
    fewshot_retrieval_args: dict = {},
) -> list[list[dict[str, str]]]:
    """
    Loads fewshot data from json or txt file and returns a list of fewshot examples, where each item is a list of examples
    pertaining to a single data instance.
    """
    # raw data file must be a jsonl
    fewshot_data = load_data_to_records(fewshot_data_path)
    # choose method of fewshot retrieval
    _fewshot_retrieval_func = get_fewshot_retrieval_method(fewshot_retrieval_method)
    fewshot_retrieval_args["task"] = task
    fewshot_retrieval_args["index_path"] = index_path
    fewshot_examples_list = _fewshot_retrieval_func(
        test_set=test_set,
        examples=fewshot_data,
        n_shots=n_fewshots,
        total_examples=total_shots,
        **fewshot_retrieval_args,
    )

    return fewshot_examples_list

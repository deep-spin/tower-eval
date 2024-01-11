import argparse
import os
import random
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tower_eval.utils import load_data_to_records, sample_strings_from_list, write_lines


def index_data(
    datastore_filename: str,
    datastore_indexed_path: Path,
    task_name: str = "task",
    subtask_name: str = "subtask",
) -> None:
    """
    :param datastore_filename: The input csv of json file name to be encoded and used as the datastore
    :param datastore_indexed_path: The output index file that will be used to retrieve similar samples from.
    :param index_columns: the columns to be indexed.
    """
    logger.opt(colors=True).info(f"========================================")
    logger.opt(colors=True).info(
        f"Indexing the data of task: <yellow> {task_name} </yellow>, subtask: <green> {subtask_name} </green> <blue> {datastore_filename} </blue>"
    )
    datastore_indexed_keys_path = os.path.join(datastore_indexed_path, "keys")
    datastore_indexed_values_path = os.path.join(datastore_indexed_path, "values")

    if not os.path.exists(datastore_indexed_path):
        os.makedirs(datastore_indexed_path)
    df = pd.read_csv(datastore_filename, encoding="utf8")

    sources = df["src"]
    targets = df["ref"]

    # create numpy memmaps to store keys and values
    dstore_keys = np.memmap(
        datastore_indexed_keys_path, dtype=np.float16, mode="w+", shape=(len(df), 768)
    )
    dstore_vals = np.memmap(
        datastore_indexed_values_path, dtype=np.int32, mode="w+", shape=(len(df), 1)
    )

    # initialize LaBSE encoder
    encoder = SentenceTransformer("sentence-transformers/LaBSE")

    print("getting embeddings")
    idx = 0
    number_vals = 0
    while idx < len(sources):
        sources_i = sources[idx : idx + 100000]

        # encode questions
        source_embeddings = encoder.encode(sources_i, batch_size=5000)

        keys = source_embeddings
        vals = [[i + idx] for i in range(len(sources_i))]

        # store keys and values
        dstore_keys[idx : (idx + len(vals))] = keys.astype(np.float16)
        dstore_vals[idx : (idx + len(vals))] = np.array(vals).astype(np.int32)

        idx += 100000
        number_vals += len(vals)

        if idx % 1000000 == 0:
            print(idx, len(sources))

    logger.opt(colors=True).info("Indexing the data with FAISS")
    if len(sources) < 10000:
        index = faiss.IndexFlatL2(768)
        index.add(torch.tensor(dstore_keys.astype(np.float32)))
    else:
        quantizer = faiss.IndexFlatL2(768)
        index = faiss.IndexIVFPQ(quantizer, 768, 2, 64, 8)
        index.nprobe = 32

        logger.opt(colors=True).info("Training the index clusters")
        np.random.seed(42)

        # If we have more than 1M samples, randomly sample 1M of them to train the clustering algorithem.
        random_sample = np.random.choice(
            np.arange(number_vals), size=[min(1000000, number_vals)], replace=False
        )

        index.train(dstore_keys[random_sample].astype(np.float32))

        logger.opt(colors=True).info("Add the keys to the FAISS Index")
        start = 0
        while start < number_vals:
            end = min(number_vals, start + 100000)
            to_add = dstore_keys[start:end].copy()
            index.add_with_ids(
                np.array(to_add).astype("float32"), np.arange(start, end)
            )
            start += 100000

            if start % 1000000 == 0:
                print(start, len(sources))

    # save index
    faiss.write_index(index, os.path.join(datastore_indexed_path, "knn.index"))

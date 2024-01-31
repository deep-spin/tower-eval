import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

def index_data(
    datastore_filename: str,
    datastore_indexed_path: Path,
    task_name: str = "task",
    subtask_name: str = "subtask",
    jsonl: bool =False,
    batch_size: int = 1000,
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

    if not os.path.exists(datastore_indexed_path):
        os.makedirs(datastore_indexed_path)
    
    if jsonl:
        df = pd.read_json(datastore_filename, lines=True)
    else:    
        df = pd.read_csv(datastore_filename, encoding="utf8")
    source_sentences = df['src'].to_list()

    # initialize LaBSE encoder
    encoder = SentenceTransformer("sentence-transformers/LaBSE")
    embeddings = encoder.encode(source_sentences, batch_size=batch_size)
    embeddings = np.asarray(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(datastore_indexed_path, "knn.index"))
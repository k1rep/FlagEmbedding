import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from FlagEmbedding import FlagModel

import chromadb

logger = logging.getLogger(__name__)

@dataclass
class Args:
    encoder: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )

    query_data: str = field(
        default="namespace-Pt/msmarco-corpus",
        metadata={'help': 'queries and their positive passages for evaluation'}
    )
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )


def search(model: FlagModel, queries: datasets.Dataset, collection, k: int = 100, batch_size: int = 256,
           max_length: int = 512):
    """
    1. Encode queries into dense embeddings;
    2. Search through Chroma index
    """
    query_embeddings = model.encode_queries(queries["text"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        all_scores.extend([res["distances"] for res in results])
        all_indices.extend([res["ids"] for res in results])

    return all_scores, all_indices


if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    model = FlagModel(args.encoder,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: "
                      if args.add_instruction else None,
                      use_fp16=args.fp16
                      )

    query_data = datasets.load_dataset('json', data_files=args.query_data)
    client = chromadb.Client()
    collection = client.get_collection(name="corpus")

    scores, indices = search(
        model=model,
        queries=query_data,
        collection=collection,
        k=args.k,
        batch_size=args.batch_size,
        max_length=args.max_query_length
    )

    print(scores, indices)

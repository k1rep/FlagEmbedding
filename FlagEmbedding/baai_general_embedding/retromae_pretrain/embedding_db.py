import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from FlagEmbedding import FlagModel

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

    corpus_data: str = field(
        default="code_search_net",
        metadata={'help': 'candidate passages'}
    )
    query_data: str = field(
        default="namespace-Pt/msmarco-corpus",
        metadata={'help': 'queries and their positive passages for evaluation'}
    )

    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )


def index(model: FlagModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int = 512,
          index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False,
          load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)

    else:
        corpus_embeddings = model.encode_corpus(corpus["whole_func_string"], batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]

        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings

    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    model = FlagModel(args.encoder,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: "
                      if args.add_instruction else None,
                      use_fp16=args.fp16
                      )
    t_corpus = datasets.load_dataset(args.corpus_data)
    corpus = datasets.concatenate_datasets([t_corpus["train"], t_corpus["validation"], t_corpus["test"]])
    index(model, corpus, batch_size=args.batch_size, save_path=args.save_path, save_embedding=args.save_embedding,
          load_embedding=args.load_embedding)

import logging
import datasets
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
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )


def index(model: FlagModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int = 512):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create chroma index;
    """
    corpus_embeddings = model.encode_corpus(corpus["whole_func_string"], batch_size=batch_size, max_length=max_length)

    client = chromadb.Client()
    collection = client.create_collection(name="corpus")

    logger.info("Adding embeddings to Chroma...")
    for idx, embedding in enumerate(tqdm(corpus_embeddings, desc="Indexing")):
        collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{"id": idx}],
            ids=[str(idx)]
        )
    return collection


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
    collection = index(model, corpus, batch_size=args.batch_size)

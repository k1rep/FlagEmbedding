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

    corpus_data: str = field(
        default="code_search_net",
        metadata={'help': 'candidate passages'}
    )

    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )


def index(model: FlagModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int = 512):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create chroma index;
    """
    corpus_embeddings = model.encode_corpus(corpus["whole_func_string"], batch_size=batch_size, max_length=max_length)

    client = chromadb.PersistentClient()
    collection = client.create_collection(name="corpus")

    logger.info("Adding embeddings to Chroma...")
    for idx, (embedding, func_string) in tqdm(enumerate(zip(corpus_embeddings, corpus["whole_func_string"])),
                                              desc="Indexing to Chroma", total=len(corpus_embeddings)):
        try:
            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{"id": idx, "func_string": func_string}],
                ids=[str(idx)]
            )
        except Exception as e:
            logger.error(f"在索引 {idx} 处添加嵌入向量时出错: {e}")
    return collection


if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    model = FlagModel(args.encoder,
                      use_fp16=args.fp16
                      )
    tr_corpus = datasets.load_dataset(args.corpus_data, split="train")
    v_corpus = datasets.load_dataset(args.corpus_data, split="validation")
    te_corpus = datasets.load_dataset(args.corpus_data, split="test")
    corpus = datasets.concatenate_datasets([tr_corpus, v_corpus, te_corpus])
    collection = index(model, corpus, batch_size=args.batch_size)

"""
1. get query as input
2. embed query into vector
3. search the most similar code snippet in the vector db
4. combine the query and retrieved code snippet into a full content
5. generate response with the content
"""
import logging

import chromadb

from retromae_pretrain import search_db
import generator
from FlagEmbedding import FlagModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_embedding(query: str, model: FlagModel) -> list:
    return model.encode(query).tolist()


def get_prompt(query: str, retrieved_res: list) -> dict:
    full_content = f"Query: {query}\nCode: {retrieved_res[0]}"
    message = {'content': full_content, 'role': 'user'}
    return message


if __name__ == '__main__':
    embedding_model = FlagModel('BAAI/bge-base-en-v1.5', use_fp16=True)
    generator = generator.CodeLlamaGenerator()
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection(name="corpus")
    logger.info("Loading RAG with embedding model: BAAI/bge-base-en-v1.5, vector database: chroma, and llm: "
                "codellama:13b")
    top_k = 50
    while True:
        print('Input: ')
        query = input('')
        logger.info(f"get query: {query}")
        _, _, res = search_db.search(model=embedding_model, query=query, collection=collection,
                                     batch_size=256, max_length=512, k=top_k)
        logger.info(f"embedding query: {res}")
        prompt = get_prompt(query, res)
        logger.info(f"prompt: {prompt}")
        response = generator.generate(prompt)['message']['content']
        logger.info(f"response: {response}")
        print(response)

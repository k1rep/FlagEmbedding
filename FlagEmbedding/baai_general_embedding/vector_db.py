from abc import ABC, abstractmethod
import chromadb


class VectorDatabaseInterface(ABC):
    @abstractmethod
    def add_vector(self, vector, meta):
        pass

    @abstractmethod
    def add_vectors(self, vectors, meta):
        pass

    @abstractmethod
    def search(self, query_vector, top_k=5):
        pass


class ChromaDatabase(VectorDatabaseInterface):
    def __init__(self, host='localhost', port=8000, collection_name='vectors'):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_vector(self, vector, meta):
        vector_id = str(self.collection.count())
        self.collection.add(ids=[vector_id], embeddings=[vector], metadatas=[meta])

    def add_vectors(self, vectors, metas):
        collection_size = self.collection.count()
        ids = [str(i) for i in range(collection_size, collection_size+len(vectors))]
        self.collection.add(ids=ids, embeddings=vectors, metadatas=metas)

    def update_vector(self, vector_id, vector, meta):
        self.collection.update(ids=[vector_id], embeddings=[vector], metadatas=[meta])

    def search(self, query_embeddings, n_results=1):
        results = self.collection.query(query_embeddings=query_embeddings, n_results=n_results)
        return [{'id': results['ids'][i],'meta': results['metadatas'][i], 'distance': results['distances'][i]}
                for i in range(len(results['ids']))
                ]


if __name__ == '__main__':
    test_embedding1 = [0, 0, 0, 1]
    meta_1 = {'source': 'update_1'}
    test_embedding2 = [0, 0, 1, 0]
    meta_2 = {'source': 't_2'}
    test_embedding3 = [0, 1, 0, 1]
    meta_3 = {'source': 't_3'}
    test_embedding4 = [1, 0, 0, 0]
    meta_4 = {'source': 't_4'}
    db = ChromaDatabase()
    db.update_vector('0', test_embedding1, meta_1)
    res = db.search([[0, 0, 0, 1]], 1)
    print(res)
    db.add_vectors([test_embedding2, test_embedding3, test_embedding4], [meta_2, meta_3, meta_4])
    res = db.search([[0, 0, 1, 1]], n_results=2)
    print(res)
    print(type(res))

import chromadb
from .base import VectorStore

class ChromaStore(VectorStore):
    def __init__(self, path: str, collection_name: str, hnsw_space: str = "cosine"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.hnsw_space = hnsw_space
        self.collection = None

    def get_or_create_collection(self, embedding_fn, collection_name: str):
        if self.collection is None:
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": self.hnsw_space}
            )
        return self.collection

    def add_documents(self, documents, ids, metadatas):
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def query(self, query_text: str, embedding_fn, k: int = 6):
        return self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = None
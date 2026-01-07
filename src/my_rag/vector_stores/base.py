from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

class VectorStore(ABC):
    @abstractmethod
    def get_or_create_collection(self, embedding_fn, collection_name: str):
        pass

    @abstractmethod
    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[Dict]):
        pass

    @abstractmethod
    def query(self, query_text: str, embedding_fn, k: int = 6):
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def delete_collection(self):
        pass
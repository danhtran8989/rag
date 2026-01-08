# import chromadb
# from .base import VectorStore

# class ChromaStore(VectorStore):
#     def __init__(self, path: str, collection_name: str, hnsw_space: str = "cosine"):
#         self.client = chromadb.PersistentClient(path=path)
#         self.collection_name = collection_name
#         self.hnsw_space = hnsw_space
#         self.collection = None

#     def get_or_create_collection(self, embedding_fn, collection_name: str):
#         if self.collection is None:
#             try:
#                 self.client.delete_collection(collection_name)
#             except:
#                 pass
#             self.collection = self.client.create_collection(
#                 name=collection_name,
#                 embedding_function=embedding_fn,
#                 metadata={"hnsw:space": self.hnsw_space}
#             )
#         return self.collection

#     def add_documents(self, documents, ids, metadatas):
#         self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

#     def query(self, query_text: str, embedding_fn, k: int = 6):
#         return self.collection.query(
#             query_texts=[query_text],
#             n_results=k,
#             include=["documents", "distances", "metadatas"]
#         )

#     def count(self) -> int:
#         return self.collection.count() if self.collection else 0

#     def delete_collection(self):
#         try:
#             self.client.delete_collection(self.collection_name)
#         except:
#             pass
#         self.collection = None

# src/my_rag/vector_stores/chroma_store.py
import chromadb
from .base import VectorStore
from typing import Optional


class ChromaStore(VectorStore):
    def __init__(self, path: str, collection_name: str, hnsw_space: str = "cosine"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.hnsw_space = hnsw_space
        self.collection = None

    def get_or_create_collection(self, embedding_fn, collection_name: str):
        """
        Lấy collection hiện có nếu tồn tại, chỉ tạo mới nếu chưa có.
        KHÔNG BAO GIỜ tự động xóa collection cũ!
        """
        if self.collection is None:
            # Thử lấy collection hiện có trước
            try:
                self.collection = self.client.get_collection(
                    name=collection_name
                )
                print(f"Collection '{collection_name}' đã tồn tại → tải lại (count: {self.collection.count()})")
            except Exception:
                # Nếu không tồn tại → tạo mới
                print(f"Không tìm thấy collection '{collection_name}' → tạo mới")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn,
                    metadata={"hnsw:space": self.hnsw_space}
                )
        return self.collection

    def add_documents(self, documents, ids, metadatas):
        if self.collection is None:
            raise RuntimeError("Collection chưa được khởi tạo. Gọi get_or_create_collection trước.")
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, query_text: str, embedding_fn, k: int = 6):
        if self.collection is None:
            raise RuntimeError("Collection chưa được khởi tạo.")
        return self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def delete_collection(self):
        """Chỉ gọi khi thực sự muốn xóa toàn bộ (rebuild)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Đã xóa collection '{self.collection_name}'")
        except Exception as e:
            print(f"Không thể xóa collection (có thể chưa tồn tại): {e}")
        self.collection = None

    def reset(self):
        """Alias để rõ nghĩa khi rebuild"""
        self.delete_collection()
        self.collection = None
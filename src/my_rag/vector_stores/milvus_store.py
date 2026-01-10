# from pymilvus import MilvusClient, DataType
# from pymilvus.milvus_client import MilvusClient as MC
# from typing import List
# import numpy as np
# from .base import VectorStore

# class MilvusStore(VectorStore):
#     def __init__(self, uri: str, collection_name: str, user: str = None, password: str = None):
#         self.uri = uri
#         self.collection_name = collection_name
#         self.client = MilvusClient(uri=uri, user=user or "", password=password or "")
#         self.dim = None
#         self.collection_exists = False

#     def get_or_create_collection(self, embedding_fn, collection_name: str):
#         if not self.client.has_collection(collection_name):
#             # Infer dimension from a dummy embedding
#             dummy_emb = embedding_fn(["dummy"])[0]
#             self.dim = len(dummy_emb)
#             schema = {
#                 "fields": [
#                     {"name": "id", "type": DataType.VARCHAR, "is_primary": True, "max_length": 100},
#                     {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.dim}},
#                     {"name": "source", "type": DataType.VARCHAR, "max_length": 512},
#                 ]
#             }
#             index_params = {
#                 "metric_type": "COSINE",
#                 "index_type": "IVF_FLAT",
#                 "params": {"nlist": 128}
#             }
#             self.client.create_collection(
#                 collection_name=collection_name,
#                 schema=schema,
#                 index_params=[{"field_name": "vector", **index_params}]
#             )
#         self.collection_exists = True
#         return self

#     def add_documents(self, documents: List[str], ids: List[str], metadatas: List[dict]):
#         embeddings = embedding_fn(documents)
#         data = [
#             {"id": id_, "vector": emb, "source": meta.get("source", "")}
#             for id_, emb, meta in zip(ids, embeddings, metadatas)
#         ]
#         self.client.insert(collection_name=self.collection_name, data=data)

#     def query(self, query_text: str, embedding_fn, k: int = 6):
#         query_emb = embedding_fn([query_text])[0]
#         results = self.client.search(
#             collection_name=self.collection_name,
#             data=[query_emb],
#             limit=k,
#             output_fields=["source", "id"]
#         )
#         docs, dists, metas = [], [], []
#         for hit in results[0]:
#             entity = hit["entity"]
#             docs.append(entity.get("text", ""))  # Milvus doesn't store text by default → we store it
#             dists.append(hit["distance"])
#             metas.append({"source": entity.get("source", "?")})
#         # Since we didn't store text, return empty docs and rely on metadata if needed
#         return {"documents": [docs], "distances": [dists], "metadatas": [metas]}

#     def count(self) -> int:
#         if self.client.has_collection(self.collection_name):
#             res = self.client.get_collection_stats(self.collection_name)
#             return int(res.get("row_count", 0))
#         return 0

#     def delete_collection(self):
#         if self.client.has_collection(self.collection_name):
#             self.client.drop_collection(self.collection_name)
#         self.collection_exists = False

###########################################
# src/my_rag/vector_stores/milvus.py
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
from .base import VectorStore


class MilvusStore(VectorStore):
    def __init__(
        self,
        uri: str = "./milvus_rag.db",
        collection_name: str = "rag_docs",
        metric_type: str = "COSINE",
        user: str = None,
        password: str = None,
    ):
        self.client = MilvusClient(
            uri=uri,
            user=user or "",
            password=password or ""
        )
        self.collection_name = collection_name
        self.metric_type = metric_type
        self._dimension: Optional[int] = None

    def get_or_create_collection(self, embedding_fn, collection_name: str = None):
        """Create collection if not exists, infer dimension from embedding function"""
        if collection_name:
            self.collection_name = collection_name

        if self.client.has_collection(self.collection_name):
            collection_info = self.client.describe_collection(self.collection_name)
            self._dimension = collection_info["params"]["dim"]
            self.client.load_collection(self.collection_name)
            return self

        # Infer dimension
        dummy_embedding = embedding_fn(["dummy text"])[0]
        self._dimension = len(dummy_embedding)

        schema = {
            "fields": [
                {"name": "id", "type": DataType.VARCHAR, "is_primary": True, "max_length": 100},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": self._dimension}},
                {"name": "text", "type": DataType.VARCHAR, "params": {"max_length": 65535}},
                {"name": "source", "type": DataType.VARCHAR, "params": {"max_length": 512}},
            ]
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type=self.metric_type,
            auto_id=False
        )

        # Modern & convenient index (2025/2026 style)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type
        )
        self.client.create_index(self.collection_name, index_params)
        self.client.load_collection(self.collection_name)

        return self

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict]
    ):
        if not documents:
            return

        if len(documents) != len(ids) or len(documents) != len(metadatas):
            raise ValueError("documents, ids, and metadatas must have the same length")

        embeddings = embedding_fn(documents)  # assuming embedding_fn is in scope or injected

        data = [
            {
                "id": str(id_),  # Milvus VARCHAR primary key
                "vector": emb,
                "text": text,
                "source": meta.get("source", "unknown")[:512]
            }
            for id_, emb, text, meta in zip(ids, embeddings, documents, metadatas)
        ]

        self.client.insert(self.collection_name, data)

    def query(self, query_text: str, embedding_fn, k: int = 6) -> Dict:
        if self._dimension is None:
            raise RuntimeError("Collection not initialized. Call get_or_create_collection first.")

        query_emb = embedding_fn([query_text])[0]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=["text", "source", "id"],
            search_params={"metric_type": self.metric_type}
        )[0]

        return {
            "documents": [hit["entity"]["text"] for hit in results],
            "distances": [hit["distance"] for hit in results],
            "metadatas": [{"source": hit["entity"]["source"]} for hit in results],
            "ids": [hit["entity"]["id"] for hit in results]
        }

    def count(self) -> int:
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
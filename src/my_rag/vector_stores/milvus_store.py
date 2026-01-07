from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client import MilvusClient as MC
from typing import List
import numpy as np
from .base import VectorStore

class MilvusStore(VectorStore):
    def __init__(self, uri: str, collection_name: str, user: str = None, password: str = None):
        self.uri = uri
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri, user=user or "", password=password or "")
        self.dim = None
        self.collection_exists = False

    def get_or_create_collection(self, embedding_fn, collection_name: str):
        if not self.client.has_collection(collection_name):
            # Infer dimension from a dummy embedding
            dummy_emb = embedding_fn(["dummy"])[0]
            self.dim = len(dummy_emb)
            schema = {
                "fields": [
                    {"name": "id", "type": DataType.VARCHAR, "is_primary": True, "max_length": 100},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": self.dim}},
                    {"name": "source", "type": DataType.VARCHAR, "max_length": 512},
                ]
            }
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=[{"field_name": "vector", **index_params}]
            )
        self.collection_exists = True
        return self

    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[dict]):
        embeddings = embedding_fn(documents)
        data = [
            {"id": id_, "vector": emb, "source": meta.get("source", "")}
            for id_, emb, meta in zip(ids, embeddings, metadatas)
        ]
        self.client.insert(collection_name=self.collection_name, data=data)

    def query(self, query_text: str, embedding_fn, k: int = 6):
        query_emb = embedding_fn([query_text])[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=["source", "id"]
        )
        docs, dists, metas = [], [], []
        for hit in results[0]:
            entity = hit["entity"]
            docs.append(entity.get("text", ""))  # Milvus doesn't store text by default → we store it
            dists.append(hit["distance"])
            metas.append({"source": entity.get("source", "?")})
        # Since we didn't store text, return empty docs and rely on metadata if needed
        return {"documents": [docs], "distances": [dists], "metadatas": [metas]}

    def count(self) -> int:
        if self.client.has_collection(self.collection_name):
            res = self.client.get_collection_stats(self.collection_name)
            return int(res.get("row_count", 0))
        return 0

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.collection_exists = False
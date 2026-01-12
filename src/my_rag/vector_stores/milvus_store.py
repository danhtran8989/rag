# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
from .base import VectorStore


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store using MilvusClient (2025-2026 recommended style)
    - Uses simplified create_collection (dimension + metric_type)
    - Uses IndexParams + prepare_index_params for index creation
    - Stores text + source + dynamic fields
    """

    def __init__(
        self,
        uri: str = "./milvus_rag.db",
        collection_name: str = "rag_docs",
        metric_type: str = "COSINE",
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.client = MilvusClient(
            uri=uri,
            user=user or "",
            password=password or ""
        )
        self.collection_name = collection_name
        self.metric_type = metric_type
        self._dimension: Optional[int] = None
        self._initialized = False  # ← Optional: helps rag_system know if init happened

    def get_or_create_collection(
        self,
        embedding_fn,  # callable: list[str] → list[list[float]]
        collection_name: Optional[str] = None
    ):
        if collection_name:
            self.collection_name = collection_name

        if self.client.has_collection(self.collection_name):
            info = self.client.describe_collection(self.collection_name)
            for field in info.get("fields", []):
                if field.get("type") == str(DataType.FLOAT_VECTOR):
                    self._dimension = field["params"].get("dim")
                    break
            if self._dimension is None:
                raise RuntimeError("Could not detect vector dimension in existing collection")
            
            self.client.load_collection(self.collection_name)
            self._initialized = True
            return self

        # Infer dimension
        dummy_embedding = embedding_fn(["dummy text"])[0]
        self._dimension = len(dummy_embedding)

        # Create collection - modern simplified API
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self._dimension,
            metric_type=self.metric_type,
            primary_field_name="id",
            vector_field_name="vector",
            auto_id=False,
            enable_dynamic_field=True,  # ← allows extra metadata fields
        )

        # Create index - correct 2.4+/2.5+ way
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",          # auto-optimizes (great default)
            metric_type=self.metric_type,
            # params={}                      # optional for AUTOINDEX
            # index_name="vector_idx"        # optional
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        self.client.load_collection(self.collection_name)
        self._initialized = True
        return self

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        embedding_fn=None,
    ):
        if not documents:
            return

        if len(documents) != len(ids) != len(metadatas):
            raise ValueError("documents, ids, metadatas must have same length")

        if embedding_fn is None:
            raise ValueError("embedding_fn required for add_documents")

        embeddings = embedding_fn(documents)

        data = [
            {
                "id": str(id_),  # VARCHAR PK needs string
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
            }
            for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]

        self.client.insert(collection_name=self.collection_name, data=data)

    def query(
        self,
        query_text: str,
        embedding_fn,
        k: int = 6,
        output_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Call get_or_create_collection() first")

        query_emb = embedding_fn([query_text])[0]

        fields = output_fields or ["text", "source", "id"]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=fields,
            search_params={"metric_type": self.metric_type}
        )[0]

        return {
            "documents": [hit["entity"].get("text", "") for hit in results],
            "distances": [hit["distance"] for hit in results],
            "metadatas": [{"source": hit["entity"].get("source", "?")} for hit in results],
            "ids": [hit["entity"].get("id", "") for hit in results],
        }

    def count(self) -> int:
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self._dimension = None
        self._initialized = False
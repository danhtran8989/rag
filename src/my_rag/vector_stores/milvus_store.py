# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
from .base import VectorStore


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store implementation using MilvusClient (2025-2026 style)
    - Uses simplified create_collection API (no raw schema dict)
    - AUTOINDEX for best performance with zero tuning
    - Stores text + source metadata
    """

    def __init__(
        self,
        uri: str = "./milvus_rag.db",           # or "http://localhost:19530"
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

    def get_or_create_collection(
        self,
        embedding_fn,                       # callable that takes list[str] → list[list[float]]
        collection_name: Optional[str] = None
    ):
        """
        Create collection if it doesn't exist.
        Uses modern simplified API (dimension + metric_type directly).
        """
        if collection_name:
            self.collection_name = collection_name

        if self.client.has_collection(self.collection_name):
            # Get existing dimension
            info = self.client.describe_collection(self.collection_name)
            for field in info.get("fields", []):
                if field["type"] == str(DataType.FLOAT_VECTOR):
                    self._dimension = field["params"]["dim"]
                    break
            if self._dimension is None:
                raise RuntimeError("Could not determine vector dimension from existing collection")
            
            self.client.load_collection(self.collection_name)
            return self

        # Infer dimension from embedding function
        dummy_embedding = embedding_fn(["dummy text"])[0]
        self._dimension = len(dummy_embedding)

        # ───────────────────────────────────────────────────────────────
        # Modern recommended way (2025-2026) - NO schema dictionary!
        # ───────────────────────────────────────────────────────────────
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self._dimension,
            metric_type=self.metric_type,
            primary_field_name="id",
            vector_field_name="vector",
            auto_id=False,                    # We provide custom string IDs
            enable_dynamic_field=True,        # Allows extra fields without schema
        )
        # ───────────────────────────────────────────────────────────────

        # Create strong default index - AUTOINDEX is excellent in recent versions
        self.client.create_index(
            collection_name=self.collection_name,
            field_name="vector",
            index_params={
                "index_type": "AUTOINDEX",
                "metric_type": self.metric_type,
                "params": {}   # let Milvus auto-tune
            }
        )

        self.client.load_collection(self.collection_name)
        return self

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        embedding_fn=None,              # optional - can be passed here instead of globally
    ):
        if not documents:
            return

        if len(documents) != len(ids) or len(documents) != len(metadatas):
            raise ValueError("documents, ids, and metadatas must have the same length")

        # If embedding_fn not passed, you should have it available in your context
        if embedding_fn is None:
            raise ValueError("embedding_fn is required to generate embeddings")

        embeddings = embedding_fn(documents)

        data = [
            {
                "id": str(id_),                     # must be string for VARCHAR PK
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
                # You can add more fields here → dynamic fields will accept them
            }
            for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

    def query(
        self,
        query_text: str,
        embedding_fn,
        k: int = 6,
        output_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        Returns: {"documents": [...], "distances": [...], "metadatas": [...], "ids": [...]}
        """
        if self._dimension is None:
            raise RuntimeError("Collection not initialized. Call get_or_create_collection first.")

        if embedding_fn is None:
            raise ValueError("embedding_fn is required for query")

        query_emb = embedding_fn([query_text])[0]

        default_fields = ["text", "source", "id"]
        fields = output_fields if output_fields is not None else default_fields

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=fields,
            search_params={"metric_type": self.metric_type}
        )[0]  # first query vector results

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

    def clear_collection(self):
        """Delete all entities but keep the collection structure"""
        if self.client.has_collection(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                filter="id != ''"  # delete everything
            )
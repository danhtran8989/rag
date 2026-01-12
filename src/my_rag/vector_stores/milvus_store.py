# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional, Callable
from .base import VectorStore


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store implementation (compatible with pymilvus 2.4.x - 2.6.x)
    Features:
    - Uses simplified create_collection API (auto int64 PK + auto_id=True)
    - AUTOINDEX for excellent performance with zero tuning
    - Stores text + source + supports dynamic fields
    - embedding_fn can be stored at init or passed per operation
    """

    def __init__(
        self,
        uri: str = "./milvus_rag.db",
        collection_name: str = "rag_docs",
        metric_type: str = "COSINE",
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.client = MilvusClient(
            uri=uri,
            user=user or "",
            password=password or ""
        )
        self.collection_name = collection_name
        self.metric_type = metric_type.upper()  # Ensure COSINE/L2/IP
        self.embedding_fn = embedding_fn
        self._dimension: Optional[int] = None
        self._initialized = False

    def get_or_create_collection(
        self,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        collection_name: Optional[str] = None
    ):
        if collection_name:
            self.collection_name = collection_name

        # Allow overriding/storing embedding function
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn

        if self.client.has_collection(self.collection_name):
            info = self.client.describe_collection(self.collection_name)
            for field in info.get("fields", []):
                if field.get("type") == str(DataType.FLOAT_VECTOR):
                    self._dimension = field["params"].get("dim")
                    break
            
            if self._dimension is None:
                raise RuntimeError("Could not detect vector field dimension in existing collection")

            self.client.load_collection(self.collection_name)
            self._initialized = True
            return self

        if self.embedding_fn is None:
            raise ValueError("embedding_fn is required to infer dimension")

        # Infer dimension from dummy call
        dummy_embedding = self.embedding_fn(["dummy text"])[0]
        self._dimension = len(dummy_embedding)

        # Modern simplified creation - auto int64 PK + auto_id=True by default
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self._dimension,
            metric_type=self.metric_type,
            enable_dynamic_field=True,           # Very useful for extra metadata
        )

        # Create strong default index (AUTOINDEX = excellent in 2025-2026)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
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
        ids: List[str],                        # ← kept for compatibility, but NOT used as PK
        metadatas: List[Dict[str, Any]],
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Insert documents.
        Note: Primary key is auto-generated (int64) by Milvus
        Your original ids can be stored in dynamic field 'original_id' if needed
        """
        if not documents:
            return

        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length")

        ef = embedding_fn or self.embedding_fn
        if ef is None:
            raise ValueError("embedding_fn required (pass it or set during __init__)")

        embeddings = ef(documents)

        data = [
            {
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
                # Optional: store your original id as extra field
                # "original_id": str(original_id),
            }
            for emb, doc, original_id, meta
            in zip(embeddings, documents, ids, metadatas)
        ]

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

    def query(
        self,
        query_text: str,
        embedding_fn: Callable[[List[str]], List[List[float]]],
        k: int = 6,
        output_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Call get_or_create_collection() first")

        query_emb = embedding_fn([query_text])[0]

        fields = output_fields or ["text", "source"]

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
            "ids": [hit["id"] for hit in results],           # ← Milvus auto-generated int64 ids
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

    def clear(self):
        """Delete all vectors but keep collection schema"""
        if self.client.has_collection(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                filter="id >= 0"   # delete everything
            )
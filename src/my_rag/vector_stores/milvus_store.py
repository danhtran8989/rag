# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from typing import List, Dict, Any, Optional, Callable
from .base import VectorStore


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store (pymilvus 2.6.x compatible - Jan 2026)
    - Uses explicit CollectionSchema for reliable auto_id=True behavior
    - Auto-generated int64 primary key ("id")
    - No need to provide "id" in insert data
    - Supports dynamic fields for extra metadata
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
        self.metric_type = metric_type.upper()
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

        if embedding_fn is not None:
            self.embedding_fn = embedding_fn

        if self.client.has_collection(self.collection_name):
            info = self.client.describe_collection(self.collection_name)
            for field in info.get("fields", []):
                if field.get("type") == str(DataType.FLOAT_VECTOR):
                    self._dimension = field["params"].get("dim")
                    break

            if self._dimension is None:
                raise RuntimeError("Could not detect vector dimension")

            self.client.load_collection(self.collection_name)
            self._initialized = True
            return self

        if self.embedding_fn is None:
            raise ValueError("embedding_fn required to infer dimension")

        dummy_embedding = self.embedding_fn(["dummy text"])[0]
        self._dimension = len(dummy_embedding)

        # ───────────────────────────────────────────────────────────────
        # Explicit schema - most reliable way (recommended in 2026)
        # ───────────────────────────────────────────────────────────────
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,               # Milvus auto-generates IDs
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._dimension
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=512
                ),
            ],
            description="RAG documents collection",
            enable_dynamic_field=True
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type=self.metric_type
        )
        # ───────────────────────────────────────────────────────────────

        # Create index
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
        ids: List[str],  # ← ignored for PK, can store as extra field if needed
        metadatas: List[Dict[str, Any]],
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        if not documents:
            return

        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have same length")

        ef = embedding_fn or self.embedding_fn
        if ef is None:
            raise ValueError("embedding_fn required")

        embeddings = ef(documents)

        data = [
            {
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
                # Optional: keep your original id if you want traceability
                # "original_id": str(original_id),
            }
            for emb, doc, original_id, meta in zip(embeddings, documents, ids, metadatas)
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

        # return {
        #     "documents": [hit["entity"].get("text", "") for hit in results],
        #     "distances": [hit["distance"] for hit in results],
        #     "metadatas": [{"source": hit["entity"].get("source", "?")} for hit in results],
        #     "ids": [hit["id"] for hit in results],  # auto-generated int64 ids
        # }

        return {
            "documents": [[hit["entity"].get("text", "") for hit in results]],  # ← nested
            "distances": [[hit["distance"] for hit in results]],               # ← nested
            "metadatas": [[{"source": hit["entity"].get("source", "?")} for hit in results]],
            "ids": [[hit["id"] for hit in results]],
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
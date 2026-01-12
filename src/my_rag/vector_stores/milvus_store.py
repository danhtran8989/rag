# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Base interface (you probably already have this)"""
    def get_or_create_collection(self, *args, **kwargs):
        raise NotImplementedError

    def add_documents(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store using MilvusClient (pymilvus 2.4+ / 2.5+ compatible - 2025/2026)
    - Uses explicit CollectionSchema
    - Auto-generated int64 primary key ("id")
    - No need to provide "id" when inserting
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

    def _get_vector_dimension(self) -> int:
        """Get vector dimension from existing collection or raise error"""
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(f"Collection {self.collection_name} does not exist yet")

        info = self.client.describe_collection(self.collection_name)
        for field in info.get("fields", []):
            if field.get("type") == str(DataType.FLOAT_VECTOR):
                dim = field["params"].get("dim")
                if dim is not None:
                    return int(dim)
        raise RuntimeError("Could not detect vector dimension from collection schema")

    def get_or_create_collection(
        self,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> 'MilvusStore':
        """
        Create collection if it doesn't exist, or load it if it does.
        Returns self for chaining.
        """
        if collection_name:
            self.collection_name = collection_name

        if embedding_fn is not None:
            self.embedding_fn = embedding_fn

        # Already exists → just load and get dimension
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists. Loading...")
            self._dimension = self._get_vector_dimension()
            self.client.load_collection(self.collection_name)
            self._initialized = True
            return self

        # Need to create new collection
        if self.embedding_fn is None and dimension is None:
            raise ValueError("embedding_fn or explicit dimension is required to create collection")

        # Infer dimension if not provided
        if dimension is None:
            if self.embedding_fn is None:
                raise ValueError("embedding_fn required to infer dimension")
            dummy_embedding = self.embedding_fn(["dummy text for dimension detection"])[0]
            dimension = len(dummy_embedding)
            logger.info(f"Inferred embedding dimension: {dimension}")

        self._dimension = dimension

        # ───────────────────────────────────────────────────────────────
        # Explicit schema - recommended way in 2025/2026
        # ───────────────────────────────────────────────────────────────
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
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
        logger.info(f"Created collection '{self.collection_name}' with dim={self._dimension}")

        # Create automatic index (AUTOINDEX is usually good choice in recent versions)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200}  # optional tuning
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        self.client.load_collection(self.collection_name)
        self._initialized = True
        logger.info(f"Collection '{self.collection_name}' created, indexed and loaded.")
        return self

    def is_ready(self) -> bool:
        """Check if collection is usable"""
        return (
            self._initialized
            and self.client.has_collection(self.collection_name)
            and self.client.get_collection_stats(self.collection_name).get("row_count", 0) >= 0
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[Any]] = None,           # ← ignored for PK, but can be stored in dynamic field
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        if not documents:
            return

        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have same length")

        ef = embedding_fn or self.embedding_fn
        if ef is None:
            raise ValueError("embedding function required")

        embeddings = ef(documents)

        data = [
            {
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
                # You can add original_id if you want traceability
                # "original_id": ids[i] if ids else None,
                **meta  # dynamic fields for all other metadata
            }
            for i, (emb, doc, meta) in enumerate(zip(embeddings, documents, metadatas))
        ]

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        logger.info(f"Inserted {len(data)} documents into '{self.collection_name}'")

    def query(
        self,
        query_text: str,
        embedding_fn: Callable[[List[str]], List[List[float]]],
        k: int = 6,
        output_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Collection not ready. Call get_or_create_collection() first")

        query_emb = embedding_fn([query_text])[0]

        fields = output_fields or ["text", "source"]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=fields,
            search_params={"metric_type": self.metric_type, "params": {"ef": 128}}
        )[0]  # first query result list

        return {
            "documents": [hit["entity"].get("text", "") for hit in results],
            "distances": [hit["distance"] for hit in results],
            "metadatas": [{"source": hit["entity"].get("source", "?")} for hit in results],
            "ids": [hit["id"] for hit in results],           # auto-generated int64 ids
        }

    def count(self) -> int:
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        self._dimension = None
        self._initialized = False
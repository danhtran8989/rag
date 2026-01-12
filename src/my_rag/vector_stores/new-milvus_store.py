# src/my_rag/vector_stores/milvus_store.py
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Base interface (you probably already have this elsewhere)"""
    def get_or_create_collection(self, *args, **kwargs):
        raise NotImplementedError

    def add_documents(self, *args, **kwargs):
        raise NotImplementedError

    def query(self, *args, **kwargs):
        raise NotImplementedError


class MilvusStore(VectorStore):
    """
    Modern Milvus vector store using MilvusClient (pymilvus 2.4+ / 2.5+ compatible)
    - Auto-generated int64 primary key ("id")
    - Supports dynamic fields
    - Robust metadata handling
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
        if collection_name:
            self.collection_name = collection_name

        if embedding_fn is not None:
            self.embedding_fn = embedding_fn

        # Already exists → load and get dimension
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' exists → loading...")
            self._dimension = self._get_vector_dimension()
            self.client.load_collection(self.collection_name)
            self._initialized = True
            return self

        # Create new collection
        if self.embedding_fn is None and dimension is None:
            raise ValueError("Need embedding_fn or explicit dimension to create collection")

        # Infer dimension if needed
        if dimension is None:
            dummy_embedding = self.embedding_fn(["dummy text"])[0]
            dimension = len(dummy_embedding)
            logger.info(f"Auto-detected embedding dimension: {dimension}")

        self._dimension = dimension

        # Create explicit schema
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            ],
            description="RAG documents collection",
            enable_dynamic_field=True
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type=self.metric_type
        )

        # Create index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200}
        )
        self.client.create_index(self.collection_name, index_params)

        self.client.load_collection(self.collection_name)
        self._initialized = True
        logger.info(f"Created & loaded collection '{self.collection_name}' (dim={self._dimension})")
        return self

    def is_ready(self) -> bool:
        return (
            self._initialized
            and self.client.has_collection(self.collection_name)
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Any],  # Accept Any to handle common mistake gracefully
        ids: Optional[List[Any]] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        if not documents:
            return

        # ─── Handle common mistake: metadatas as list of strings ───────
        if metadatas and isinstance(metadatas[0], str):
            logger.warning(
                "metadatas is list of strings instead of dicts → "
                "converting to simple metadata with 'source' field"
            )
            metadatas = [{"source": src} for src in metadatas]

        # Now we expect list of dicts
        if not all(isinstance(m, dict) for m in metadatas):
            raise ValueError("metadatas must be list of dictionaries (after auto-conversion)")

        if len(documents) != len(metadatas):
            raise ValueError(
                f"documents ({len(documents)}) and metadatas ({len(metadatas)}) length mismatch"
            )

        ef = embedding_fn or self.embedding_fn
        if ef is None:
            raise ValueError("embedding function required")

        embeddings = ef(documents)

        data = []
        for emb, doc, meta in zip(embeddings, documents, metadatas):
            entry = {
                "vector": emb,
                "text": doc,
                "source": str(meta.get("source", "unknown"))[:512],
            }
            # Add all other metadata fields as dynamic fields
            entry.update({k: v for k, v in meta.items() if k != "source" and k != "text"})
            data.append(entry)

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

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=k,
            output_fields=fields,
            search_params={"metric_type": self.metric_type, "params": {"ef": 128}}
        )[0]

        return {
            "documents": [hit["entity"].get("text", "") for hit in res],
            "distances": [hit["distance"] for hit in res],
            "metadatas": [hit["entity"] for hit in res],  # contains source + dynamic fields
            "ids": [hit["id"] for hit in res],
        }

    def count(self) -> int:
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def delete_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        self._dimension = None
        self._initialized = False
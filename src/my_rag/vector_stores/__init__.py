from .base import VectorStore
from .chroma_store import ChromaStore
from .milvus_store import MilvusStore
from .pgvector_store import PGVectorStore

def get_vector_store(store_type: str = "chroma", **kwargs) -> VectorStore:
    store_type = store_type.lower()
    if store_type == "chroma":
        return ChromaStore(**kwargs)
    elif store_type == "milvus":
        return MilvusStore(**kwargs)
    elif store_type == "pgvector":
        return PGVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store: {store_type}")
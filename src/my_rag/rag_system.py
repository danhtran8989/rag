# src/my_rag/rag_system.py
import torch
from chromadb.utils import embedding_functions
from typing import List, Tuple
from .config import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_DEFAULT, VECTOR_DB_CONFIG
)
from .text_extraction import extract_text
from .chunking import chunk_text
from .utils import ensure_ollama_models
from .vector_stores import get_vector_store
import ollama

class RAGSystem:
    def __init__(self, llm_models: List[str]):
        ensure_ollama_models(llm_models)
        self.vector_store = None
        self.embedding_fn = None
        self.current_embedding_model = None
        self.indexed_files = set()
        self.store_type = VECTOR_DB_DEFAULT

    def _get_embedding_fn(self, embedding_model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if embedding_model_name != self.current_embedding_model or self.embedding_fn is None:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
                device=device,
                normalize_embeddings=True,
                trust_remote_code=True
            )
            self.current_embedding_model = embedding_model_name
        return self.embedding_fn

    def get_or_create_collection(self, embedding_model_name: str, uploaded_files: List[str], vector_db_type: str = None):
        self.store_type = vector_db_type or VECTOR_DB_DEFAULT
        embedding_fn = self._get_embedding_fn(embedding_model_name)

        config = VECTOR_DB_CONFIG[self.store_type]
        self.vector_store = get_vector_store(self.store_type, **config)

        files_changed = set(uploaded_files or []) != self.indexed_files
        if files_changed or self.vector_store.count() == 0:
            self.vector_store.delete_collection()

            collection = self.vector_store.get_or_create_collection(
                embedding_fn=embedding_fn,
                collection_name=config["collection_name"]
            )

            if uploaded_files:
                chunks, ids, metadatas = [], [], []
                for file_path in uploaded_files:
                    filename = os.path.basename(file_path)
                    print(f"📄 Đang xử lý: {filename}")
                    text = extract_text(file_path)
                    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
                        chunk_id = f"{filename}_chunk_{i:04d}"
                        chunks.append(chunk)
                        ids.append(chunk_id)
                        metadatas.append({"source": filename})

                if chunks:
                    self.vector_store.add_documents(chunks, ids, metadatas)
                    print(f"✅ Đã index {len(chunks)} chunks vào {self.store_type.upper()}")

            self.indexed_files = set(uploaded_files or [])

    def retrieve(self, query: str, k: int = 6) -> List[Tuple[str, float, dict]]:
        if not self.vector_store or self.vector_store.count() == 0:
            return []
        results = self.vector_store.query(query, self.embedding_fn, k=k)
        return [(doc, 1.0 - (dist or 0), meta)
                for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0])]

    # build_prompt and generate_answer remain the same
    # ... (copy from previous version)
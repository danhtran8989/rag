# src/my_rag/rag_system.py
import torch
# from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Generator, Dict, Optional
import hashlib
import os
import ollama

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_DEFAULT,
    VECTOR_DB_CONFIG
)
from .text_extraction import extract_text
from .pdf2structure_text import get_chunks
from .chunking import chunk_text
from .utils import ensure_ollama_models
from .vector_stores import get_vector_store


def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Tính hash SHA-256 của file để phát hiện thay đổi nội dung."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


class RAGSystem:
    """
    Singleton RAG System để đảm bảo chỉ có một instance duy nhất,
    giữ trạng thái vector store và indexed files giữa các request.
    """
    _instance = None
    _initialized = False

    def __new__(cls, llm_models: List[str] = None):
        if cls._instance is None:
            cls._instance = super(RAGSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self, llm_models: List[str] = None):
        if not RAGSystem._initialized:
            if llm_models:
                ensure_ollama_models(llm_models)

            self.vector_store = None
            self.embedding_fn = None
            self.current_embedding_model = None
            self.indexed_files: Dict[str, str] = {}  # {file_path: hash}
            self.store_type = VECTOR_DB_DEFAULT

            # Khởi tạo vector store với config mặc định
            config = VECTOR_DB_CONFIG[self.store_type]
            self.vector_store = get_vector_store(self.store_type, **config)

            RAGSystem._initialized = True
            print("RAGSystem singleton instance created and initialized.")

    def _get_embedding_fn(self, embedding_model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cpu_device = "cpu"
        if embedding_model_name != self.current_embedding_model or self.embedding_fn is None:
            print(f"Loading new embedding model: {embedding_model_name}")
            # self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            #     # model_name_or_path=embedding_model_name,   # or just embedding_model_name
            #     # device=cpu_device,
            #     # # normalize_embeddings=True,
            #     # trust_remote_code=True,
            #     model_name=embedding_model_name,
            #     device=cpu_device,
            #     # normalize_embeddings=True,
            #     trust_remote_code=True
            # )

            self.embedding_fn = SentenceTransformer(
                model_name_or_path=embedding_model_name,   # or just embedding_model_name
                device=cpu_device,
                # # normalize_embeddings=True,
                trust_remote_code=True,
            )
            self.current_embedding_model = embedding_model_name
        return self.embedding_fn

    def get_or_create_collection(
        self,
        embedding_model_name: str,
        uploaded_files: List[str],
        vector_db_type: str = None,
    ):
        """Tạo hoặc tái sử dụng collection dựa trên thay đổi file / db type / embedding model."""
        # Chuẩn hóa và kiểm tra thay đổi vector db type
        vector_db_type = vector_db_type.lower() if vector_db_type else self.store_type
        db_changed = vector_db_type != self.store_type

        if db_changed:
            print(f"Vector DB type CHANGED: {self.store_type.upper()} → {vector_db_type.upper()}")
            self.store_type = vector_db_type
            config = VECTOR_DB_CONFIG[self.store_type]
            self.vector_store = get_vector_store(self.store_type, **config)

        embedding_fn = self._get_embedding_fn(embedding_model_name)
        embedding_changed = embedding_model_name != self.current_embedding_model

        config = VECTOR_DB_CONFIG[self.store_type]

        # Tính hash các file hiện tại
        current_hashes: Dict[str, str] = {}
        valid_files = []
        for file_path in (uploaded_files or []):
            if os.path.exists(file_path):
                file_hash = get_file_hash(file_path)
                current_hashes[file_path] = file_hash
                valid_files.append(file_path)
            else:
                print(f"File không tồn tại (có thể đã bị xóa): {file_path}")

        files_changed = current_hashes != self.indexed_files

        # Điều kiện rebuild toàn bộ collection
        need_rebuild = (
            db_changed or
            embedding_changed or
            files_changed or
            len(self.indexed_files) == 0
        )

        if need_rebuild:
            reasons = []
            if db_changed: reasons.append("vector DB type changed")
            if embedding_changed: reasons.append("embedding model changed")
            if files_changed: reasons.append("files changed/added/removed")
            if len(self.indexed_files) == 0: reasons.append("first time indexing")

            print("Rebuilding collection because: " + ", ".join(reasons))

            # Xóa collection cũ (an toàn với cả Chroma & Milvus)
            self.vector_store.delete_collection()

            # Tạo collection mới
            self.vector_store.get_or_create_collection(
                embedding_fn=embedding_fn,
                collection_name=config["collection_name"]
            )

            # Index lại toàn bộ tài liệu
            chunks, ids, metadatas = [], [], []
            for file_path in valid_files:
                filename = os.path.basename(file_path)
                print(f"Đang xử lý: {filename}")
                # text = extract_text(file_path)
                # chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                current_chunks = get_chunks(file_path)

                for i, chunk in enumerate(current_chunks):
                    chunk_id = f"{filename}_chunk_{i:04d}"
                    chunks.append(chunk)
                    ids.append(chunk_id)
                    metadatas.append({"source": file_path})

            if chunks:
                self.vector_store.add_documents(chunks, ids, metadatas, embedding_fn=embedding_fn)
                print(f"Đã index {len(chunks)} chunks vào {self.store_type.upper()}")

            # Cập nhật trạng thái
            self.indexed_files = current_hashes.copy()

        else:
            print("Không cần rebuild - giữ nguyên collection hiện tại")
            # Đảm bảo collection đã được load (an toàn với cả hai loại db)
            self.vector_store.get_or_create_collection(
                embedding_fn=embedding_fn,
                collection_name=config["collection_name"]
            )

    def retrieve(self, query: str, k: int = 6) -> List[Tuple[str, float, dict]]:
        if not self.vector_store or self.vector_store.count() == 0:
            print("Vector store chưa sẵn sàng hoặc collection rỗng.")
            return []

        results = self.vector_store.query(
            query_text=query,
            embedding_fn=self.embedding_fn,
            k=k
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        # Chuẩn hóa distance thành similarity (nếu cần)
        return [
            (doc, 1.0 - (dist or 0), meta)
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            )
        ]

    def build_prompt(
        self,
        query: str,
        context_items: List[Tuple[str, float, dict]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        context_texts = [
            f"[Nguồn: {os.path.basename(metadata['source'])}]\n{chunk}"
            for chunk, score, metadata in context_items
        ]
        context_block = "\n\n".join(context_texts)

        system_instruction = (
            "Bạn là một trợ lý thông minh, chính xác và trung thực. "
            "Hãy trả lời câu hỏi dựa CHỈ vào thông tin trong NGỮ CẢNH dưới đây. "
            "Trích dẫn nguồn khi sử dụng thông tin từ tài liệu. "
            "Nếu thông tin không có trong ngữ cảnh, hãy trả lời rõ ràng: "
            "\"Tôi không biết\" hoặc \"Thông tin không đủ để trả lời\"."
        )

        prompt_parts = []

        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n")

        user_message = f"{system_instruction}\n\nNGỮ CẢNH:\n{context_block}\n\nCÂU HỎI: {query}"
        prompt_parts.append(f"<start_of_turn>user\n{user_message}<end_of_turn>\n")
        prompt_parts.append("<start_of_turn>model")

        return "".join(prompt_parts)

    def stream_answer(
        self,
        query: str,
        k: int = 6,
        model: str = "llama3",
        params: dict = None
    ) -> Generator[str, None, None]:
        context = self.retrieve(query, k=k)
        prompt = self.build_prompt(query, context)

        options = {
            "temperature": params.get("temperature", 0.7) if params else 0.7,
            "top_k": params.get("top_k", 40) if params else 40,
            "top_p": params.get("top_p", 0.9) if params else 0.9,
            "repeat_penalty": params.get("repeat_penalty", 1.1) if params else 1.1,
        }
        if params and params.get("max_tokens") and params["max_tokens"] > 0:
            options["num_predict"] = params["max_tokens"]

        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options=options,
        )

        for chunk in stream:
            yield chunk["message"]["content"]
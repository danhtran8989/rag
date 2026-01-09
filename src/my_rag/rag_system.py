# src/my_rag/rag_system.py
import torch
from chromadb.utils import embedding_functions
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

            # Khởi tạo các thuộc tính cơ bản
            self.vector_store = None
            self.embedding_fn = None
            self.current_embedding_model = None
            self.indexed_files: Dict[str, str] = {}  # {file_path: hash}
            self.store_type = VECTOR_DB_DEFAULT

            # Tạo vector_store ngay từ đầu với config mặc định
            config = VECTOR_DB_CONFIG[self.store_type]
            self.vector_store = get_vector_store(self.store_type, **config)

            RAGSystem._initialized = True
            print("RAGSystem singleton instance created and initialized.")

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

    def get_or_create_collection(
        self,
        embedding_model_name: str,
        uploaded_files: List[str],
        vector_db_type: str = None,
    ):
        """Tạo hoặc tái sử dụng collection dựa trên thay đổi file."""
        # Cập nhật store_type nếu có thay đổi
        if vector_db_type:
            self.store_type = vector_db_type.lower()
            config = VECTOR_DB_CONFIG[self.store_type]
            self.vector_store = get_vector_store(self.store_type, **config)

        embedding_fn = self._get_embedding_fn(embedding_model_name)
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

        print(f"Indexed files (before): {self.indexed_files}")
        print(f"Current file hashes: {current_hashes}")

        # Kiểm tra có thay đổi không
        files_changed = False

        # File mới hoặc thay đổi nội dung
        for fp, new_hash in current_hashes.items():
            if self.indexed_files.get(fp) != new_hash:
                files_changed = True
                print(f"File mới hoặc đã thay đổi: {os.path.basename(fp)}")

        # File bị xóa khỏi danh sách
        for old_fp in list(self.indexed_files.keys()):
            if old_fp not in current_hashes:
                files_changed = True
                print(f"File bị xóa khỏi danh sách: {os.path.basename(old_fp)}")

        # Điều kiện rebuild: lần đầu HOẶC có thay đổi file
        need_rebuild = files_changed or len(self.indexed_files) == 0

        if need_rebuild:
            print("Rebuild collection (lần đầu hoặc có thay đổi file)")
            self.vector_store.delete_collection()

            # Tạo collection mới
            self.vector_store.get_or_create_collection(
                embedding_fn=embedding_fn,
                collection_name=config["collection_name"]
            )

            # Index tài liệu
            chunks, ids, metadatas = [], [], []
            for file_path in valid_files:
                filename = os.path.basename(file_path)
                print(f"Đang xử lý: {filename}")
                text = extract_text(file_path)
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
                    chunk_id = f"{filename}_chunk_{i:04d}"
                    chunks.append(chunk)
                    ids.append(chunk_id)
                    metadatas.append({"source": file_path})

            if chunks:
                self.vector_store.add_documents(chunks, ids, metadatas)
                print(f"Đã index {len(chunks)} chunks vào {self.store_type.upper()}")

            # Cập nhật trạng thái
            self.indexed_files = current_hashes.copy()
        else:
            print("Không có thay đổi → Giữ nguyên collection hiện tại")
            # Quan trọng: load lại collection nếu chưa có (do restart kernel hoặc lần đầu không rebuild)
            if self.vector_store.collection is None:
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

        return [
            (doc, 1.0 - (dist or 0), meta)
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            )
        ]

#     def build_prompt(self, query: str, context_items: List[Tuple[str, float, dict]]) -> str:
#         context_text = "\n\n".join([
#             f"[Nguồn: {os.path.basename(m['source'])}]: {c}"
#             for c, s, m in context_items
#         ])
#         prompt = f"""Bạn là một trợ lý thông minh và chính xác. Hãy trả lời câu hỏi dựa CHỈ vào thông tin ngữ cảnh dưới đây.
# Nếu không có thông tin liên quan trong ngữ cảnh, hãy trả lời "Tôi không biết" hoặc "Thông tin không có trong tài liệu".

# NGỮ CẢNH:
# {context_text}

# CÂU HỎI: {query}
# TRẢ LỜI:"""
#         return prompt

    def build_prompt(
        self,
        query: str,
        context_items: List[Tuple[str, float, dict]],
        conversation_history: Optional[List[Dict[str, str]]] = None  # Optional history for multi-turn
    ) -> str:
        """
        Build a properly formatted prompt for Gemma 3 IT models.
        
        Args:
            query: Current user question
            context_items: List of (chunk_text, score, metadata)
            conversation_history: Previous messages in OpenAI-style format 
                                 [{"role": "user"/"assistant", "content": "..."}]
        """
        # Build context block with sources
        context_texts = [
            f"[Nguồn: {os.path.basename(metadata['source'])}]\n{chunk}"
            for chunk, score, metadata in context_items
        ]
        context_block = "\n\n".join(context_texts)

        # System-like instructions (must be in first user turn for Gemma 3)
        system_instruction = (
            "Bạn là một trợ lý thông minh, chính xác và trung thực. "
            "Hãy trả lời câu hỏi dựa CHỈ vào thông tin trong NGỮ CẢNH dưới đây. "
            "Trích dẫn nguồn khi sử dụng thông tin từ tài liệu. "
            "Nếu thông tin không có trong ngữ cảnh, hãy trả lời rõ ràng: "
            "\"Tôi không biết\" hoặc \"Thông tin không đủ để trả lời\"."
        )

        # Start building the formatted prompt
        prompt_parts = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n")

        # Add the current turn: combine system instruction + context + query in the user message
        user_message = f"{system_instruction}\n\nNGỮ CẢNH:\n{context_block}\n\nCÂU HỎI: {query}"

        prompt_parts.append(f"<start_of_turn>user\n{user_message}<end_of_turn>\n")
        
        # Crucial: End with model's turn so it starts generating
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
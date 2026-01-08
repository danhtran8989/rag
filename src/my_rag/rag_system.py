# src/my_rag/rag_system.py
import torch
from chromadb.utils import embedding_functions
from typing import List, Tuple, Generator, Dict
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
    """Tính hash MD5 của file để phát hiện thay đổi nội dung."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            md5.update(block)
    return md5.hexdigest()


class RAGSystem:
    def __init__(self, llm_models: List[str]):
        ensure_ollama_models(llm_models)
        self.vector_store = None
        self.embedding_fn = None
        self.current_embedding_model = None
        
        # Lưu trữ hash của các file đã index: {file_path: hash}
        self.indexed_files: Dict[str, str] = {}
        
        self.store_type = VECTOR_DB_DEFAULT
        self.llm_model = None
        self.gen_params = {}

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
        self.store_type = vector_db_type or VECTOR_DB_DEFAULT
        embedding_fn = self._get_embedding_fn(embedding_model_name)
        config = VECTOR_DB_CONFIG[self.store_type]
        self.vector_store = get_vector_store(self.store_type, **config)

        # Tính hash hiện tại của các file được upload
        current_hashes: Dict[str, str] = {}
        valid_files = []
        for file_path in (uploaded_files or []):
            if os.path.exists(file_path):
                file_hash = get_file_hash(file_path)
                current_hashes[file_path] = file_hash
                valid_files.append(file_path)
            else:
                print(f"⚠️ File không tồn tại (có thể đã bị xóa): {file_path}")
        print(f"Indexed files: {self.indexed_files}")
        print(f"Current hashes: {current_hashes}")
        # Kiểm tra xem có thay đổi nào không
        files_changed = False

        # File mới hoặc thay đổi nội dung
        for fp, new_hash in current_hashes.items():
            old_hash = self.indexed_files.get(fp)
            if old_hash != new_hash:
                print(f"Old hash: {old_hash}, New hash: {new_hash} for file {fp}")
                files_changed = True
                print(f"📄 File mới hoặc đã thay đổi: {os.path.basename(fp)}")

        # File bị xóa khỏi danh sách upload
        for old_fp in list(self.indexed_files.keys()):
            if old_fp not in current_hashes:
                files_changed = True
                print(f"🗑️ File bị xóa khỏi danh sách: {os.path.basename(old_fp)}")

        # Nếu collection rỗng hoặc có thay đổi → rebuild toàn bộ
        if files_changed or self.vector_store.count() == 0:
            print("files_changed:", files_changed)
            print("🔄 Phát hiện thay đổi → Xóa collection cũ và tái index...")
            self.vector_store.delete_collection()
            self.vector_store.get_or_create_collection(
                embedding_fn=embedding_fn,
                collection_name=config["collection_name"]
            )

            chunks, ids, metadatas = [], [], []
            for file_path in valid_files:
                filename = os.path.basename(file_path)
                print(f"📄 Đang xử lý: {filename}")
                text = extract_text(file_path)
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
                    chunk_id = f"{filename}_chunk_{i:04d}"
                    chunks.append(chunk)
                    ids.append(chunk_id)
                    metadatas.append({"source": file_path})  # Lưu full path để dễ debug

            if chunks:
                self.vector_store.add_documents(chunks, ids, metadatas)
                print(f"✅ Đã index {len(chunks)} chunks vào {self.store_type.upper()}")

            # Cập nhật trạng thái đã index
            self.indexed_files = current_hashes
        else:
            print("✅ Không có thay đổi trong tài liệu → Giữ nguyên collection hiện tại.")

    def retrieve(self, query: str, k: int = 6) -> List[Tuple[str, float, dict]]:
        if not self.vector_store or self.vector_store.count() == 0:
            return []
        results = self.vector_store.query(query, self.embedding_fn, k=k)
        if not results or not results.get("documents"):
            return []

        return [
            (doc, 1.0 - (dist or 0), meta)
            for doc, dist, meta in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0]
            )
        ]

    def build_prompt(self, query: str, context_items: List[Tuple[str, float, dict]]) -> str:
        context_text = "\n\n".join([
            f"[Nguồn: {os.path.basename(m['source'])}]: {c}"
            for c, s, m in context_items
        ])
        prompt = f"""Bạn là một trợ lý thông minh. Sử dụng thông tin ngữ cảnh dưới đây để trả lời câu hỏi. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.

NGỮ CẢNH:
{context_text}

CÂU HỎI: {query}
TRẢ LỜI:"""
        return prompt

    def stream_answer(self, query: str, k: int, model: str, params: dict) -> Generator[str, None, None]:
        """Stream câu trả lời từ Ollama dựa trên ngữ cảnh đã retrieve."""
        context = self.retrieve(query, k=k)
        prompt = self.build_prompt(query, context)

        options = {
            "temperature": params.get("temperature", 0.7),
            "top_k": params.get("top_k", 40),
            "top_p": params.get("top_p", 0.9),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
        }
        if params.get("max_tokens") and params["max_tokens"] > 0:
            options["num_predict"] = params["max_tokens"]

        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options=options,
        )

        for chunk in stream:
            yield chunk["message"]["content"]
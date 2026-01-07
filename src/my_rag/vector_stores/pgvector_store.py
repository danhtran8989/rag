from langchain.vectorstores import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from .base import VectorStore
from sqlalchemy import create_engine
import os

class PGVectorStore(VectorStore):
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.vectorstore = None

    def get_or_create_collection(self, embedding_fn, collection_name: str):
        # Langchain PGVector handles collection as table name
        model_name = embedding_fn.model_name if hasattr(embedding_fn, "model_name") else "default"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = PGVector(
            connection_string=self.connection_string,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        return self.vectorstore

    def add_documents(self, documents, ids, metadatas):
        self.vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, embedding_fn, k: int = 6):
        docs = self.vectorstore.similarity_search_with_score(query_text, k=k)
        documents = [doc.page_content for doc, _ in docs]
        distances = [score for _, score in docs]
        metadatas = [doc.metadata for doc, _ in docs]
        return {
            "documents": [documents],
            "distances": [distances],
            "metadatas": [metadatas]
        }

    def count(self) -> int:
        return self.vectorstore._collection_count()

    def delete_collection(self):
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(f'DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;')
            conn.execute(f'DROP TABLE IF EXISTS langchain_pg_collection CASCADE;')
        self.vectorstore = None
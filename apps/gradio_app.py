# apps/gradio_app.py
import gradio as gr
import os
import sys
import argparse

# Append the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.my_rag.config import (
    LLM_MODELS,
    DEFAULT_LLM,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING,
    hyperparams,
    VECTOR_DB_DEFAULT,
)
from src.my_rag.rag_system import RAGSystem

# Global RAG system instance
rag_system = RAGSystem(LLM_MODELS)

# Load custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "gradio_app", "static", "gradio_app.css")
custom_css = ""
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        custom_css = f.read()


def print_retrieved_chunks(query: str, k: int, embedding_model: str):
    try:
        if rag_system.vector_store is None:
            print("Vector store chưa được khởi tạo.")
            return

        if rag_system.vector_store.count() == 0:
            print("Collection rỗng — chưa có tài liệu nào được index.")
            return

        embedding_fn = rag_system._get_embedding_fn(embedding_model)

        results = rag_system.vector_store.query(
            query_text=query,
            embedding_fn=embedding_fn,
            k=k
        )

        if not results or not results.get("documents"):
            print("Không tìm thấy chunk nào phù hợp với query này.")
            return

        docs = results["documents"]
        dists = results["distances"]
        metas = results["metadatas"]

        # ───────────────────────────────────────────────────────────────
        # Normalize to flat lists - more reliable way
        # ───────────────────────────────────────────────────────────────
        if isinstance(docs, list) and docs and isinstance(docs[0], list):
            # Chroma style - nested
            docs = docs[0]
            dists = dists[0] if dists and isinstance(dists[0], list) else dists
            metas = metas[0] if metas and isinstance(metas[0], list) else metas
        # else: already flat (Milvus, or future stores)

        if not docs:
            print("Không có kết quả nào.")
            return

        print("\n" + "=" * 80)
        print(f"RETRIEVED CHUNKS FOR QUERY: \"{query}\"")
        print(f"Embedding model: {embedding_model} | Top K: {k}")
        print("=" * 80)

        # Now we can safely zip
        for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas), 1):
            source = meta.get("source", "Unknown source")
            print(f"\nChunk {i} | Distance: {dist:.4f} | Source: {os.path.basename(source)}")
            # print("-" * 60)
            # print(doc.strip())
            # print("-" * 60)

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback
        traceback.print_exc()


def chat_with_docs(
    files,
    message,
    llm_model,
    embedding_model,
    vector_db,
    history,
    temperature,
    top_k,
    top_p,
    repeat_penalty,
    max_tokens,
    retrieval_k,
):
    if not message.strip():
        yield history, ""
        return

    # 1. Handle file uploads and re-indexing
    file_paths = [f.name for f in files] if files else []
    rag_system.get_or_create_collection(
        embedding_model_name=embedding_model,
        uploaded_files=file_paths,
        vector_db_type=vector_db.lower(),
    )

    # 2. Prepare History
    history = history[:] if history else []
    history.append([message, ""])

    # 3. Prepare generation parameters
    gen_params = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "max_tokens": int(max_tokens) if max_tokens > 0 else None,
    }

    # 4. PRINT RETRIEVED CHUNKS BEFORE INFERENCE
    print_retrieved_chunks(
        query=message,
        k=retrieval_k,
        embedding_model=embedding_model,
    )

    # 5. Stream the LLM response
    try:
        accumulated_response = ""
        for text_chunk in rag_system.stream_answer(
            query=message,
            k=retrieval_k,
            model=llm_model,
            params=gen_params
        ):
            accumulated_response += text_chunk
            history[-1][1] = accumulated_response
            yield history, ""
    except Exception as e:
        error_msg = f"Lỗi: {str(e)}"
        history[-1][1] = error_msg
        yield history, ""


def update_status(files):
    if not files:
        return "<div class='status'>Chưa có tài liệu nào</div>"
    names = [os.path.basename(f.name) for f in files]
    truncated = ", ".join(names[:4]) + ("..." if len(names) > 4 else "")
    return f"<div class='status'>Đã tải {len(names)} tài liệu: {truncated}</div>"


def create_demo():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RAG Chatbot Pro") as demo:
        gr.HTML("<h1 style='text-align: center;'>RAG Chatbot Pro</h1>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Cài đặt")
                file_upload = gr.File(label="Tải tài liệu", file_count="multiple", type="filepath")
                llm_dropdown = gr.Dropdown(choices=LLM_MODELS, value=DEFAULT_LLM, label="Model LLM")
                emb_dropdown = gr.Dropdown(choices=EMBEDDING_MODELS, value=DEFAULT_EMBEDDING, label="Embedding")
                vector_db_dropdown = gr.Dropdown(choices=["chroma", "milvus"], value=VECTOR_DB_DEFAULT, label="Vector DB")

                retrieval_k = gr.Slider(1, 100, value=5, step=1, label="Số lượng chunk (K)")

                with gr.Accordion("Thông số Gen", open=False):
                    temperature = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                    top_k_slider = gr.Slider(1, 100, value=40, label="Top K")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, label="Top P")
                    repeat_penalty = gr.Slider(0.0, 2.0, value=1.1, label="Repeat Penalty")
                    max_tokens = gr.Number(value=-1, label="Max Tokens (-1 = vô hạn)")

                status = gr.Markdown("<div class='status'>Chưa có tài liệu nào</div>")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600, bubble_full_width=False)

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Nhập câu hỏi tại đây...",
                        container=False,
                        scale=7,
                        show_label=False
                    )
                    send_btn = gr.Button("Gửi", variant="primary")

                gr.Markdown("### 💡 Ví dụ câu hỏi")
                examples = gr.Examples(
                    examples=[
                        ["how to prepare Oracle Application Object Library"],
                        ["tóm tắt các tài liệu Oracle"],
                    ],
                    inputs=msg,
                    cache_examples=False,
                )

        # Events
        file_upload.change(update_status, inputs=file_upload, outputs=status)

        input_args = [
            file_upload, msg, llm_dropdown, emb_dropdown, vector_db_dropdown,
            chatbot, temperature, top_k_slider, top_p, repeat_penalty, max_tokens, retrieval_k
        ]

        send_btn.click(chat_with_docs, inputs=input_args, outputs=[chatbot, msg])
        msg.submit(chat_with_docs, inputs=input_args, outputs=[chatbot, msg])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(server_port=args.port, share=args.share, debug=args.debug)
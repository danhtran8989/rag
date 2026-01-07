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

# Track current LLM settings to avoid unnecessary rebuilds
current_llm_config = {
    "model": DEFAULT_LLM,
    "temperature": hyperparams["generation"]["temperature"],
    "top_k": hyperparams["generation"]["top_k"],
    "top_p": hyperparams["generation"]["top_p"],
    "repeat_penalty": hyperparams["generation"]["repeat_penalty"],
    "max_tokens": hyperparams["generation"]["max_tokens"] if hyperparams["generation"]["max_tokens"] > 0 else None,
}

# Load custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "gradio_app.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        custom_css = f.read()
else:
    custom_css = ""


def rebuild_llm_if_needed(model, temperature, top_k, top_p, repeat_penalty, max_tokens):
    """
    Rebuild the LLM and chain only if parameters have changed.
    max_tokens: int or None; if <= 0 or None → unlimited (no num_predict)
    """
    global current_llm_config
    # Normalize max_tokens: None or <= 0 means unlimited
    effective_max_tokens = None if (max_tokens is None or max_tokens <= 0) else int(max_tokens)

    new_config = {
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "max_tokens": effective_max_tokens,
    }

    # Only rebuild if something changed
    if new_config != current_llm_config:
        from langchain_community.llms import Ollama

        params = {
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        if effective_max_tokens is not None:
            params["num_predict"] = effective_max_tokens

        # Update the LLM
        rag_system.llm = Ollama(**params)
        rag_system.current_llm_model = model

        # Rebuild the RAG chain with the new LLM
        rag_system.chain = rag_system._create_rag_chain()

        # Update tracking
        current_llm_config = new_config.copy()


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
    """
    Main chat function – handles file upload, indexing, LLM updates, and streaming.
    """
    if not message.strip():
        yield history, ""
        return

    # Handle file uploads and re-indexing
    file_paths = [f.name for f in files] if files else []
    if file_paths:
        rag_system.get_or_create_collection(
            embedding_model_name=embedding_model,
            uploaded_files=file_paths,
            vector_db_type=vector_db.lower(),
        )

    # Ensure history is mutable and append user message
    history = history[:] if history else []
    history.append([message, ""])

    # Update LLM only if parameters changed
    rebuild_llm_if_needed(
        model=llm_model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
    )

    # === CRITICAL FIX HERE ===
    # Your custom ChromaStore likely has .vectorstore (the real LangChain Chroma instance)
    # If the attribute name is different (e.g., .db, .vectordb), change it accordingly.
    # retriever = rag_system.vector_store.vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    retriever = rag_system.vector_store.db.as_retriever(search_kwargs={"k": retrieval_k})

    input_data = {"input": message}  # Change to "question" if your chain expects that key

    try:
        accumulated = ""
        for chunk in rag_system.chain.stream(input_data):
            if isinstance(chunk, dict):
                text = chunk.get("answer") or chunk.get("output") or chunk.get("response") or ""
            else:
                text = str(chunk)
            accumulated += text
            history[-1][1] = accumulated
            yield history, ""
    except Exception as e:
        error_msg = f"Lỗi khi tạo phản hồi: {str(e)}"
        history[-1][1] = error_msg
        yield history, ""

    yield history, ""


def update_status(files):
    """Show uploaded document status."""
    if not files:
        return "<div class='status'>Chưa có tài liệu nào</div>"
    names = [os.path.basename(f.name) for f in files]
    truncated = ", ".join(names[:4]) + ("..." if len(names) > 4 else "")
    return f"<div class='status'>Đã tải {len(names)} tài liệu: {truncated}</div>"


def create_demo():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RAG Chatbot Pro") as demo:
        gr.HTML(
            """
            <h1 style="text-align: center;">RAG Chatbot Pro</h1>
            <p class='markdown' style="text-align: center;">Tải lên tài liệu (PDF, DOCX, XLSX, TXT) và hỏi bất kỳ câu gì về nội dung của chúng!</p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Cài đặt cơ bản")
                file_upload = gr.File(
                    label="Tải lên tài liệu",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".xlsx", ".xls", ".txt"],
                    type="filepath",
                )
                llm_dropdown = gr.Dropdown(
                    choices=LLM_MODELS,
                    value=DEFAULT_LLM,
                    label="Model LLM",
                    info="Chọn model Ollama",
                )
                emb_dropdown = gr.Dropdown(
                    choices=EMBEDDING_MODELS,
                    value=DEFAULT_EMBEDDING,
                    label="Embedding Model",
                    info="Thay đổi sẽ rebuild index",
                )
                vector_db_dropdown = gr.Dropdown(
                    choices=["chroma", "milvus", "pgvector"],
                    value=VECTOR_DB_DEFAULT,
                    label="Vector Database",
                )

                gr.Markdown("### ⚙️ Cài đặt nâng cao")
                retrieval_k = gr.Slider(
                    minimum=hyperparams["retrieval"]["min_k"],
                    maximum=hyperparams["retrieval"]["max_k"],
                    value=hyperparams["retrieval"]["default_k"],
                    step=1,
                    label="Top K Retrieval (số chunk)",
                )

                with gr.Accordion("Generation Parameters (Ollama)", open=False):
                    temperature = gr.Slider(0.0, 2.0, value=hyperparams["generation"]["temperature"], step=0.05, label="Temperature")
                    top_k_slider = gr.Slider(1, 100, value=hyperparams["generation"]["top_k"], step=1, label="Top K")
                    top_p = gr.Slider(0.0, 1.0, value=hyperparams["generation"]["top_p"], step=0.01, label="Top P")
                    repeat_penalty = gr.Slider(0.0, 2.0, value=hyperparams["generation"]["repeat_penalty"], step=0.05, label="Repeat Penalty")
                    max_tokens = gr.Number(
                        value=-1,
                        label="Max Tokens",
                        info="-1 hoặc để trống = không giới hạn số token",
                    )

                status = gr.Markdown("<div class='status'>Chưa có tài liệu nào</div>")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=620,
                    avatar_images=("Người dùng", "Bot"),
                    bubble_full_width=False,
                    show_label=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Nhập câu hỏi",
                        placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu?",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Gửi", variant="primary", scale=1)

        # Events
        file_upload.change(update_status, inputs=file_upload, outputs=status)

        send_btn.click(
            chat_with_docs,
            inputs=[
                file_upload,
                msg,
                llm_dropdown,
                emb_dropdown,
                vector_db_dropdown,
                chatbot,
                temperature,
                top_k_slider,
                top_p,
                repeat_penalty,
                max_tokens,
                retrieval_k,
            ],
            outputs=[chatbot, msg],
        )

        msg.submit(
            chat_with_docs,
            inputs=[
                file_upload,
                msg,
                llm_dropdown,
                emb_dropdown,
                vector_db_dropdown,
                chatbot,
                temperature,
                top_k_slider,
                top_p,
                repeat_penalty,
                max_tokens,
                retrieval_k,
            ],
            outputs=[chatbot, msg],
        )

        gr.Markdown("### Gợi ý câu hỏi")
        gr.Examples(
            examples=[
                ["Tóm tắt nội dung chính của tài liệu"],
                ["Summary the documents in English?"],
                ["Những điểm chính trong báo cáo là gì?"],
            ],
            inputs=msg,
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch RAG Chatbot Pro Gradio App")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name (0.0.0.0 for external)")
    parser.add_argument("--auth", nargs=2, metavar=('username', 'password'), help="Basic auth: --auth user pass")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = create_demo()
    auth = tuple(args.auth) if args.auth else None
    demo.launch(
        share=args.share,
        debug=args.debug,
        server_port=args.port,
        server_name=args.server_name,
        auth=auth,
    )
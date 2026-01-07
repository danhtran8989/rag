# apps/gradio_app.py

import gradio as gr
import os
import sys

# Append the current directory to sys.path
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

# Initialise the RAG system (loads Ollama models on start)
rag_system = RAGSystem(LLM_MODELS)

# Load custom CSS from static file
CSS_PATH = os.path.join(os.path.dirname(__file__), "gradio_app", "static", "gradio_app.css")
with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()


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
    Main chat function called on every message or file upload.
    """
    file_paths = [f.name for f in files] if files else []

    # Re-index if files changed or vector DB / embedding model changed
    if file_paths:
        rag_system.get_or_create_collection(
            embedding_model_name=embedding_model,
            uploaded_files=file_paths,
            vector_db_type=vector_db.lower(),
        )

    # Append user message to history
    history.append([message, ""])

    # Stream the answer
    for response_chunk in rag_system.generate_answer(
        query=message,
        llm_model=llm_model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        retrieval_k=retrieval_k,
    ):
        history[-1][1] = response_chunk
        yield history, ""

    # Final yield to clear the textbox
    yield history, ""


def update_status(files):
    """Update the status markdown showing uploaded documents."""
    if not files:
        return "<div class='status'>📭 Chưa có tài liệu nào</div>"

    names = [os.path.basename(f.name) for f in files]
    truncated = ", ".join(names[:4]) + ("..." if len(names) > 4 else "")
    return f"<div class='status'>📚 Đã tải {len(names)} tài liệu: {truncated}</div>"


def create_demo():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RAG Chatbot Pro") as demo:
        gr.HTML(
            """
            <h1>🤖 RAG Chatbot Pro</h1>
            <p class='markdown'>Tải lên tài liệu (PDF, DOCX, XLSX, TXT) và hỏi bất kỳ câu gì về nội dung của chúng!</p>
            """
        )

        with gr.Row():
            # Left column – settings
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Cài đặt cơ bản")

                file_upload = gr.File(
                    label="📂 Tải lên tài liệu",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".xlsx", ".xls", ".txt"],
                    type="filepath",
                )

                llm_dropdown = gr.Dropdown(
                    choices=LLM_MODELS,
                    value=DEFAULT_LLM,
                    label="🧠 Model LLM",
                    info="Đã tự động tải nếu cần",
                )

                emb_dropdown = gr.Dropdown(
                    choices=EMBEDDING_MODELS,
                    value=DEFAULT_EMBEDDING,
                    label="🔍 Embedding Model",
                    info="Thay đổi sẽ rebuild index",
                )

                vector_db_dropdown = gr.Dropdown(
                    choices=["chroma", "milvus", "pgvector"],
                    value=VECTOR_DB_DEFAULT,
                    label="🗄️ Vector Database",
                    info="Chọn cơ sở dữ liệu vector (thay đổi sẽ rebuild index)",
                )

                gr.Markdown("### ⚙️ Cài đặt nâng cao")

                retrieval_k = gr.Slider(
                    minimum=hyperparams["retrieval"]["min_k"],
                    maximum=hyperparams["retrieval"]["max_k"],
                    value=hyperparams["retrieval"]["default_k"],
                    step=1,
                    label="🔢 Top K Retrieval (số chunk)",
                )

                with gr.Accordion("Generation Parameters (Ollama)", open=False):
                    temperature = gr.Slider(
                        0.0, 2.0, value=hyperparams["generation"]["temperature"], step=0.05, label="🌡️ Temperature"
                    )
                    top_k_slider = gr.Slider(1, 100, value=hyperparams["generation"]["top_k"], step=1, label="🔝 Top K")
                    top_p = gr.Slider(0.0, 1.0, value=hyperparams["generation"]["top_p"], step=0.01, label="📈 Top P")
                    repeat_penalty = gr.Slider(
                        0.0, 2.0, value=hyperparams["generation"]["repeat_penalty"], step=0.05, label="🔁 Repeat Penalty"
                    )
                    max_tokens = gr.Number(
                        value=hyperparams["generation"]["max_tokens"], label="📏 Max Tokens (-1 = không giới hạn)"
                    )

                status = gr.Markdown("<div class='status'>📭 Chưa có tài liệu nào</div>")

            # Right column – chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=620,
                    avatar_images=("🤓", "🤖"),
                    bubble_full_width=False,
                    show_label=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="💬 Nhập câu hỏi",
                        placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu?",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("🚀 Gửi", variant="primary", scale=1)

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

        gr.Markdown("### 💡 Gợi ý câu hỏi")
        gr.Examples(
            examples=[
                ["Tóm tắt nội dung chính của tài liệu"],
                ["Summary documents in English?"],
            ],
            inputs=msg,
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(debug=True)
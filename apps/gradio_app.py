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
CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "gradio_app.css")
custom_css = ""
if os.path.exists(CSS_PATH):
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

    # 4. Stream response using the fixed RAGSystem method
    try:
        accumulated_response = ""
        # Call the new stream_answer method we created in rag_system.py
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
                vector_db_dropdown = gr.Dropdown(choices=["chroma", "milvus", "pgvector"], value=VECTOR_DB_DEFAULT, label="Vector DB")
                
                retrieval_k = gr.Slider(1, 20, value=5, step=1, label="Số lượng chunk (K)")
                
                with gr.Accordion("Thông số Gen", open=False):
                    temperature = gr.Slider(0.0, 2.0, value=0.7, label="Temperature")
                    top_k_slider = gr.Slider(1, 100, value=40, label="Top K")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, label="Top P")
                    repeat_penalty = gr.Slider(0.0, 2.0, value=1.1, label="Repeat Penalty")
                    max_tokens = gr.Number(value=-1, label="Max Tokens (-1 = vô hạn)")

                status = gr.Markdown("<div class='status'>Chưa có tài liệu nào</div>")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600, bubble_full_width=False)
                msg = gr.Textbox(placeholder="Nhập câu hỏi tại đây...", container=False, scale=7)
                send_btn = gr.Button("Gửi", variant="primary")

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
    args = parser.parse_args()
    
    demo = create_demo()
    demo.launch(server_port=args.port)
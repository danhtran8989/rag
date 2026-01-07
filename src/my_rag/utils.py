import ollama

def ensure_ollama_models(models: list):
    print("🔍 Kiểm tra và tải model Ollama nếu cần...")
    for model in models:
        try:
            ollama.show(model)
            print(f"✓ Đã có: {model}")
        except Exception:
            print(f"⬇ Đang tải: {model} (có thể mất vài phút)...")
            ollama.pull(model)
            print(f"✓ Hoàn tất tải: {model}")
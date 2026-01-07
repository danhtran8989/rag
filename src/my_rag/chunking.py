from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at sentence end
            while end > start and text[end-1] not in ".!?\n":
                end -= 1
            if end == start:  # No sentence break found
                end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks
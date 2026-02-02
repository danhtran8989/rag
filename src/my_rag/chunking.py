from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    # chunks, start = [], 0
    # while start < len(text):
    #     end = min(start + chunk_size, len(text))
    #     if end < len(text):
    #         # Try to break at sentence end
    #         while end > start and text[end-1] not in ".!?\n":
    #             end -= 1
    #         if end == start:  # No sentence break found
    #             end = min(start + chunk_size, len(text))
    #     chunk = text[start:end].strip()
    #     if chunk:
    #         chunks.append(chunk)
    #     start = end - overlap if end < len(text) else end

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks = text_splitter.split_text(text)

    return chunks


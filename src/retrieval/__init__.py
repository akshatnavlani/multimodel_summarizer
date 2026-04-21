"""
src/retrieval/__init__.py
--------------------------
Retrieval subpackage — MiniLM embeddings + FAISS IndexFlatIP search.
"""

from src.retrieval.embedder    import embed_paper, embed_query
from src.retrieval.faiss_index import build_index, search_index, add_to_index, FAISSIndex

__all__ = [
    "embed_paper",
    "embed_query",
    "build_index",
    "search_index",
    "add_to_index",
    "FAISSIndex",
]

"""
src/summarization/__init__.py
------------------------------
Summarization subpackage — multimodal-aware LLM summarization.
"""

from src.summarization.summarizer import Summarizer, summarize

__all__ = [
    "Summarizer",
    "summarize",
]

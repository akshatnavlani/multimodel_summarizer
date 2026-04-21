"""
src/pipeline/__init__.py
-------------------------
Pipeline subpackage — end-to-end PDF summarization + XAI orchestration.
"""

from src.pipeline.run_pipeline import run_pipeline, run_pipeline_batch

__all__ = [
    "run_pipeline",
    "run_pipeline_batch",
]

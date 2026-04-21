"""
src/evaluation/__init__.py
---------------------------
Evaluation subpackage — ROUGE, retrieval, performance metrics.
"""

from src.evaluation.evaluator import (
    Evaluator,
    evaluate_summary,
    evaluate_retrieval,
    evaluate_pipeline,
)

__all__ = [
    "Evaluator",
    "evaluate_summary",
    "evaluate_retrieval",
    "evaluate_pipeline",
]

"""
src/xai/__init__.py
--------------------
XAI subpackage — sentence attribution + modality contribution ratio.
"""

from src.xai.explainer import Explainer, explain

__all__ = [
    "Explainer",
    "explain",
]

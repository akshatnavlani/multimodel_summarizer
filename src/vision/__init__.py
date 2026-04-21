"""
src/vision/__init__.py
-----------------------
Vision subpackage — BLIP-2 figure captioning + Deplot chart extraction.
"""

from src.vision.figure_understander import describe_figures, describe_figures_batch
from src.vision.chart_extractor     import extract_charts, extract_charts_batch

__all__ = [
    "describe_figures",
    "describe_figures_batch",
    "extract_charts",
    "extract_charts_batch",
]

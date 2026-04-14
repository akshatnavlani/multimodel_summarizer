"""
src/ingestion/__init__.py
-------------------------
Ingestion sub-package: PDF loading and corpus collection utilities.
"""

from src.ingestion.pdf_loader import PDFLoader, load_pdf, generate_paper_id

__all__ = ["PDFLoader", "load_pdf", "generate_paper_id"]

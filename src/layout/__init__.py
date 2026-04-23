"""
src/layout/__init__.py
-----------------------
Layout sub-package: document element detection and region cropping.
"""

from src.layout.layout_parser import LayoutParser, parse_layout

__all__ = ["LayoutParser", "parse_layout"]

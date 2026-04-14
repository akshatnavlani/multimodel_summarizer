"""
config/__init__.py
------------------
Makes `config` a Python package and exposes the two primary helpers.

Typical import pattern across the project:
    from config import get_settings, get_project_paths

Or, for finer control:
    from config.settings import Settings, get_settings
    from config.paths   import get_project_paths, get_paper_paths
"""

from config.settings import Settings, get_settings
from config.paths import (
    get_project_paths,
    get_paper_paths,
    ensure_paper_dirs,
    paths_to_str_dict,
)

__all__ = [
    "Settings",
    "get_settings",
    "get_project_paths",
    "get_paper_paths",
    "ensure_paper_dirs",
    "paths_to_str_dict",
]

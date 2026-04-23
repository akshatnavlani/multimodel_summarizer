"""
config/paths.py
---------------
Centralised, environment-aware path management.

Rules:
  - Uses pathlib exclusively (no os.path).
  - Works on Google Colab, Kaggle, and local machines without modification.
  - All directories are created on first access.
  - No hardcoded absolute paths anywhere in the codebase — import from here.

Usage:
    from config.paths import get_project_paths
    paths = get_project_paths()
    print(paths["summaries"])          # Path object, dir guaranteed to exist
    print(paths["summaries"] / "paper_001_summary.json")
"""

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Root detection  (Colab / Kaggle / local — all handled)
# ---------------------------------------------------------------------------

def _detect_project_root() -> Path:
    """
    Resolve the project root directory across all supported environments.

    Priority:
      1. Env var PROJECT_ROOT  (explicit override, useful for Colab mounts)
      2. Google Colab  → /content/drive/MyDrive/<PROJECT_NAME>  (if Drive mounted)
                       → /content/<PROJECT_NAME>  (fallback if Drive not mounted)
      3. Kaggle        → /kaggle/working/<PROJECT_NAME>
      4. Local         → parent of this file's directory  (config/ → project root)
    """
    # 1. Explicit override
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # 2. Google Colab
    try:
        import google.colab  # noqa: F401
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists():
            return drive_path / "multimodal_xai_summarizer"
        return Path("/content/multimodal_xai_summarizer")
    except ImportError:
        pass

    # 3. Kaggle
    if Path("/kaggle/input").exists():
        return Path("/kaggle/working/multimodal_xai_summarizer")

    # 4. Local: config/paths.py lives at  <root>/config/paths.py
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Directory definitions
# ---------------------------------------------------------------------------

def _build_path_map(root: Path) -> Dict[str, Path]:
    """
    Return a flat dictionary mapping logical names → Path objects.

    All paths are absolute and derived from `root`.
    No directory is created here; creation is handled separately.
    """
    data = root / "data"
    evaluation = data / "evaluation"
    extracted = data / "extracted"
    figures = data / "figures"
    tables = data / "tables"
    embeddings = data / "embeddings"
    summaries = data / "summaries"
    xai_outputs = data / "xai_outputs"

    return {
        # ── Project root ────────────────────────────────────────────────
        "root": root,

        # ── Source code  (read-only at runtime) ─────────────────────────
        "config": root / "config",
        "src": root / "src",
        "app": root / "app",
        "tests": root / "tests",

        # ── Data subtree ────────────────────────────────────────────────
        "data": data,
        "raw_pdfs": data / "raw_pdfs",
        "extracted": extracted,
        "figures": figures,
        "tables": tables,
        "embeddings": embeddings,
        "summaries": summaries,
        "xai_outputs": xai_outputs,

        # ── XAI sub-directories ──────────────────────────────────────────
        "attention_maps": xai_outputs / "attention_maps",
        "shap_outputs": xai_outputs / "shap",
        "lime_outputs": xai_outputs / "lime",
        "mcr_outputs": xai_outputs / "mcr",

        # ── Evaluation ──────────────────────────────────────────────────
        "evaluation": evaluation,
        "results": evaluation / "results",
        "reports": evaluation / "reports",

        # ── Logs ────────────────────────────────────────────────────────
        "logs": root / "logs",

        # ── Model / weight caches (avoids re-downloading) ───────────────
        "model_cache": root / ".model_cache",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_project_paths(create_dirs: bool = True) -> Dict[str, Path]:
    """
    Return a dictionary of all project paths.

    Parameters
    ----------
    create_dirs : bool
        When True (default) every directory in the map is created if it
        does not already exist.  Set to False only in unit tests that do
        not need real directories.

    Returns
    -------
    dict
        Mapping of logical name → :class:`pathlib.Path`.

    Example
    -------
    >>> paths = get_project_paths()
    >>> summary_file = paths["summaries"] / "paper_001.json"
    >>> print(summary_file)
    /content/multimodal_xai_summarizer/data/summaries/paper_001.json
    """
    root = _detect_project_root()
    paths = _build_path_map(root)

    if create_dirs:
        _ensure_dirs(paths)

    return paths


def get_paper_paths(paper_id: str, base_paths: Dict[str, Path]) -> Dict[str, Path]:
    """
    Return per-paper file paths derived from the global path map.

    Parameters
    ----------
    paper_id   : str  — e.g. "2401.00123"
    base_paths : dict — from get_project_paths()

    Returns
    -------
    dict
        All expected input/output file paths for one paper.
        None of these paths are guaranteed to exist yet.
    """
    pid = paper_id  # shorthand

    return {
        # ── Raw input ───────────────────────────────────────────────────
        "pdf": base_paths["raw_pdfs"] / f"{pid}.pdf",

        # ── Ingestion outputs ────────────────────────────────────────────
        "metadata": base_paths["extracted"] / f"{pid}_metadata.json",
        "pages_dir": base_paths["extracted"] / pid / "pages",  # page PNGs

        # ── Layout ──────────────────────────────────────────────────────
        "layout_json": base_paths["extracted"] / f"layout_{pid}.json",
        "layout": base_paths["extracted"] / f"layout_{pid}.json",  # alias

        # ── Text / OCR ───────────────────────────────────────────────────
        "text_json": base_paths["extracted"] / f"text_{pid}.json",
        "text_elements": base_paths["extracted"] / f"text_{pid}.json",  # alias

        # ── Tables ──────────────────────────────────────────────────────
        "tables_dir": base_paths["tables"] / pid,
        "tables_json": base_paths["extracted"] / f"tables_{pid}.json",

        # ── Figures ──────────────────────────────────────────────────────
        "figures_dir": base_paths["figures"] / pid,
        "fig_descriptions": base_paths["figures"] / f"{pid}_fig_descriptions.json",
        "chart_data": base_paths["figures"] / f"{pid}_chart_data.json",

        # ── Embeddings ───────────────────────────────────────────────────
        "embeddings_pkl": base_paths["embeddings"] / f"{pid}_embeddings.pkl",
        "faiss_index": base_paths["embeddings"] / f"{pid}.faiss",
        "chunk_metadata": base_paths["embeddings"] / f"{pid}_chunks.json",

        # ── Summary ──────────────────────────────────────────────────────
        "summary_json": base_paths["summaries"] / f"{pid}_summary.json",

        # ── XAI outputs ──────────────────────────────────────────────────
        "shap_json": base_paths["shap_outputs"] / f"{pid}_shap.json",
        "lime_json": base_paths["lime_outputs"] / f"{pid}_lime.json",
        "mcr_json": base_paths["mcr_outputs"] / f"{pid}_mcr.json",
        "mcr_pie_png": base_paths["mcr_outputs"] / f"{pid}_modality_pie.png",
        "attention_maps_dir": base_paths["attention_maps"] / pid,
        "explanation_json": base_paths["xai_outputs"] / f"explanation_{pid}.json",
    }


# ---------------------------------------------------------------------------
# Directory creation helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(paths: Dict[str, Path]) -> None:
    """Create all directories in the path map (no-op if they already exist)."""
    created: list[str] = []
    for name, p in paths.items():
        # Only create directories for paths that look like directories
        # (i.e., no file suffix).  File paths inside get_paper_paths are
        # not in this map, so this is safe.
        if not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))

    if created:
        logger.debug("[paths] Ensured %d directories exist.", len(created))


def ensure_paper_dirs(paper_paths: Dict[str, Path]) -> None:
    """Create per-paper directories (pages_dir, figures_dir, etc.)."""
    dir_keys = (
        "pages_dir",
        "tables_dir",
        "figures_dir",
        "attention_maps_dir",
    )
    for key in dir_keys:
        paper_paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility helpers used across modules
# ---------------------------------------------------------------------------

def paths_to_str_dict(paths: Dict[str, Path]) -> Dict[str, str]:
    """
    Convert a path dictionary to a JSON-serialisable string dictionary.

    Useful for logging, debugging, and writing manifests.
    """
    return {k: str(v) for k, v in paths.items()}


def resolve_relative(base: Path, relative: str) -> Path:
    """
    Safely join a relative path string onto a base Path.

    Prevents directory traversal attacks in user-supplied filenames.

    Parameters
    ----------
    base     : Path — trusted base directory
    relative : str  — potentially untrusted relative path

    Returns
    -------
    Path — absolute path guaranteed to be inside `base`

    Raises
    ------
    ValueError if the resolved path escapes `base`
    """
    resolved = (base / relative).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(
            f"Path traversal detected: '{relative}' escapes base '{base}'"
        )
    return resolved


# ---------------------------------------------------------------------------
# Example usage (run as script for a quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  Project Paths Sanity Check")
    print("=" * 60)

    paths = get_project_paths(create_dirs=True)

    print(f"\n[paths] Root detected: {paths['root']}")
    print(f"[paths] Total path entries: {len(paths)}")
    print()
    print("[paths] Full path map (JSON):")
    print(json.dumps(paths_to_str_dict(paths), indent=2))

    # Per-paper paths example
    print()
    print("[paths] Example per-paper paths for paper '2401.12345':")
    paper_paths = get_paper_paths("2401.12345", paths)
    ensure_paper_dirs(paper_paths)
    print(json.dumps({k: str(v) for k, v in paper_paths.items()}, indent=2))

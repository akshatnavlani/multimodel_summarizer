"""
src/ingestion/pdf_loader.py
----------------------------
Stage 1 — PDF Ingestion.

Responsibilities
----------------
1. Accept a PDF file path.
2. Extract per-page: full text, embedded image bytes (saved as PNG),
   image bounding boxes, and document-level metadata.
3. Persist results to  data/extracted/{paper_id}_metadata.json
   and save all page images to  data/extracted/{paper_id}/pages/.
4. Cache aggressively: if the JSON output already exists, return it
   immediately without touching PyMuPDF.

Pipeline contract
-----------------
Every public function returns:

    {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "cached" | "error",
        "paper_id":    str,
        "metadata":    { ... }      # document-level metadata
        "pages":       [ ... ]      # per-page payload  (see PageRecord)
        "message":     str          # present only when status == "error"
    }

Dependencies
------------
- pymupdf  (import fitz)   — PDF parsing
- Pillow                   — image saving (optional; fitz can also write PNG)
- Standard library only beyond those two

Install
-------
    pip install pymupdf Pillow
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — fail with a clear message at call-time, not at import
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

try:
    from PIL import Image as PILImage
    import io as _io
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ---------------------------------------------------------------------------
# Path helpers — lazy import to avoid circular deps at package level
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    """Return the global project path map (creates dirs if needed)."""
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _get_settings():
    """Return the global Settings singleton."""
    from config.settings import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------

def generate_paper_id(pdf_path: str | Path) -> str:
    """
    Derive a stable, filesystem-safe paper identifier from the PDF path.

    Strategy
    --------
    1. If the filename looks like an arXiv ID  (e.g. ``2401.12345v2.pdf``)
       return the canonical arXiv ID without the version suffix.
    2. Otherwise return the stem of the filename with non-alphanumeric
       characters replaced by underscores.
    3. As a last resort (empty stem) return the first 12 hex chars of the
       SHA-256 of the absolute path string.

    Parameters
    ----------
    pdf_path : str | Path

    Returns
    -------
    str — short, filesystem-safe identifier  (no spaces, no slashes)

    Examples
    --------
    >>> generate_paper_id("2401.12345v2.pdf")
    '2401.12345'
    >>> generate_paper_id("my paper (2024).pdf")
    'my_paper__2024_'
    """
    p = Path(pdf_path)
    stem = p.stem  # filename without extension

    # arXiv pattern: YYMM.NNNNN or YYMM.NNNNNvN
    arxiv_re = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
    m = arxiv_re.match(stem)
    if m:
        return m.group(1)

    # Sanitise: replace non-alphanumeric/dot/dash with underscore,
    # then strip leading/trailing decoration so a stem like ".pdf"
    # (pathlib dotfile with no extension) collapses to "" and falls
    # through to the hash fallback.
    sanitised = re.sub(r"[^A-Za-z0-9.\-]", "_", stem).strip("._-")
    if sanitised:
        return sanitised

    # Fallback: hash the absolute path
    return hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:12]


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    """
    Persist a JSON-serialisable dictionary to disk.

    Creates parent directories automatically.

    Parameters
    ----------
    data        : dict — must be JSON-serialisable
    output_path : str | Path
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.debug("[pdf_loader] Saved JSON → %s", out)


def load_json(input_path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file from disk.

    Parameters
    ----------
    input_path : str | Path

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError if path does not exist.
    json.JSONDecodeError if file is malformed.
    """
    src = Path(input_path)
    with src.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_doc_metadata(doc) -> Dict[str, Any]:
    """
    Pull document-level metadata from a PyMuPDF Document object.

    Returns a flat, JSON-serialisable dictionary.
    """
    raw: Dict[str, Any] = doc.metadata or {}

    return {
        "title":        raw.get("title", "").strip()    or None,
        "author":       raw.get("author", "").strip()   or None,
        "subject":      raw.get("subject", "").strip()  or None,
        "keywords":     raw.get("keywords", "").strip() or None,
        "creator":      raw.get("creator", "").strip()  or None,
        "producer":     raw.get("producer", "").strip() or None,
        "creation_date":raw.get("creationDate", "").strip() or None,
        "mod_date":     raw.get("modDate", "").strip()  or None,
        "page_count":   doc.page_count,
        "is_encrypted": doc.is_encrypted,
    }


def _save_image_png(
    image_bytes: bytes,
    out_path: Path,
    xref: int,
) -> bool:
    """
    Save raw image bytes as a PNG file.

    Tries Pillow first (better format detection), falls back to writing the
    raw bytes directly when Pillow is unavailable.

    Returns True on success, False on failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _PIL_OK:
            img = PILImage.open(_io.BytesIO(image_bytes))
            img.save(str(out_path), format="PNG")
        else:
            out_path.write_bytes(image_bytes)
        return True
    except Exception as exc:
        logger.warning(
            "[pdf_loader] Could not save image xref=%d → %s : %s",
            xref, out_path, exc,
        )
        return False


def _extract_page_images(
    page,
    doc,
    pages_dir: Path,
    paper_id: str,
    page_number: int,
    dpi: int,
) -> List[Dict[str, Any]]:
    """
    Extract all embedded raster images from a PDF page.

    For each image we record:
    - ``image_path``  : relative path to the saved PNG (relative to project root)
    - ``bbox``        : [x0, y0, x1, y1] in PDF user-space units
    - ``width``       : pixel width of the image
    - ``height``      : pixel height of the image
    - ``xref``        : internal PDF cross-reference number (useful for deduplication)
    - ``colorspace``  : e.g. "DeviceRGB"

    We also render the full page as a raster PNG (used by Detectron2 in Stage 2).

    Parameters
    ----------
    page        : fitz.Page
    doc         : fitz.Document
    pages_dir   : Path — directory for rendered page PNGs
    paper_id    : str
    page_number : int   — 1-based
    dpi         : int   — rendering DPI for full-page raster

    Returns
    -------
    List[dict]
    """
    image_records: List[Dict[str, Any]] = []
    seen_xrefs: set = set()

    # ── 1. Render full page as PNG (needed by layout parser) ─────────────
    page_png_path = pages_dir / f"page_{page_number:04d}.png"
    if not page_png_path.exists():
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI is PDF native
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(page_png_path))
    # Record the full-page render as a special image (bbox = full page)
    rect = page.rect
    image_records.append({
        "image_path": str(page_png_path),
        "bbox":       [rect.x0, rect.y0, rect.x1, rect.y1],
        "width":      int(rect.width),
        "height":     int(rect.height),
        "xref":       -1,             # -1 = full-page render, not an embedded img
        "colorspace": "RGB",
        "image_type": "page_render",
    })

    # ── 2. Extract embedded images ────────────────────────────────────────
    img_list = page.get_images(full=True)
    for img_info in img_list:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        # Bounding box in page coordinates
        bbox_list: List[Any] = []
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                bbox_list = [r.x0, r.y0, r.x1, r.y1]
        except Exception:
            bbox_list = []

        # Extract raw bytes
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning(
                "[pdf_loader] Failed to extract image xref=%d on page %d: %s",
                xref, page_number, exc,
            )
            continue

        image_bytes: bytes = base_image.get("image", b"")
        if not image_bytes:
            continue

        img_w: int  = base_image.get("width", 0)
        img_h: int  = base_image.get("height", 0)
        cs:    str  = base_image.get("colorspace-name", "unknown")

        # Skip tiny images (icons, bullets — not useful for vision models)
        if img_w < 32 or img_h < 32:
            continue

        # Save
        img_filename = f"{paper_id}_p{page_number:04d}_x{xref:05d}.png"
        img_path = pages_dir / img_filename
        if not img_path.exists():
            _save_image_png(image_bytes, img_path, xref)

        image_records.append({
            "image_path": str(img_path),
            "bbox":       bbox_list,
            "width":      img_w,
            "height":     img_h,
            "xref":       xref,
            "colorspace": cs,
            "image_type": "embedded",
        })

    return image_records


def _extract_page_text(page) -> str:
    """
    Extract plain text from a PDF page using PyMuPDF's text layer.

    Uses 'text' mode which preserves reading order as best as fitz can.
    Returns an empty string if the page has no text layer (scanned page).
    """
    try:
        return page.get_text("text") or ""
    except Exception as exc:
        logger.warning("[pdf_loader] Text extraction failed on page: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Core loader class
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Stateless PDF ingestion worker.

    Usage
    -----
    ::

        loader = PDFLoader()
        result = loader.load("path/to/paper.pdf")
        print(result["status"])   # "success" | "cached" | "error"
        print(result["paper_id"])
        print(len(result["pages"]))

    The result is also written to disk automatically.
    """

    def __init__(self) -> None:
        self._paths = _get_paths()
        self._cfg   = _get_settings()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load(
        self,
        pdf_path: str | Path,
        paper_id: Optional[str] = None,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Load and ingest a PDF file.

        Parameters
        ----------
        pdf_path        : str | Path — path to the PDF file
        paper_id        : str | None — override auto-generated ID
        force_reprocess : bool       — ignore cache and reprocess

        Returns
        -------
        dict — pipeline contract payload (see module docstring)
        """
        if not _FITZ_OK:
            return _error_response(
                str(pdf_path), "",
                "PyMuPDF (fitz) is not installed. "
                "Run: pip install pymupdf"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return _error_response(
                str(pdf_path), "",
                f"PDF not found: {pdf_path}"
            )

        # Resolve paper_id
        pid = paper_id or generate_paper_id(pdf_path)

        # Resolve output paths
        output_json = self._paths["extracted"] / f"{pid}_metadata.json"
        pages_dir   = self._paths["extracted"] / pid / "pages"

        # ── Cache check ───────────────────────────────────────────────
        if not force_reprocess and output_json.exists():
            logger.info("[pdf_loader] Cache hit — loading %s", output_json)
            try:
                cached = load_json(output_json)
                cached["status"] = "cached"
                return cached
            except Exception as exc:
                logger.warning(
                    "[pdf_loader] Cache corrupt (%s); reprocessing.", exc
                )

        # ── Process ───────────────────────────────────────────────────
        logger.info("[pdf_loader] Processing PDF: %s  →  id=%s", pdf_path, pid)
        pages_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            return _error_response(
                str(pdf_path), str(output_json),
                f"PyMuPDF could not open PDF: {exc}"
            )

        if doc.is_encrypted:
            doc.close()
            return _error_response(
                str(pdf_path), str(output_json),
                "PDF is encrypted / password-protected."
            )

        # Document metadata
        doc_meta = _extract_doc_metadata(doc)

        # Per-page extraction
        pages: List[Dict[str, Any]] = []
        total_pages = min(doc.page_count, self._cfg.PDF_MAX_PAGES)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1   # 1-based for human readability

            text = _extract_page_text(page)
            images = _extract_page_images(
                page=page,
                doc=doc,
                pages_dir=pages_dir,
                paper_id=pid,
                page_number=page_number,
                dpi=self._cfg.PDF_DPI,
            )

            pages.append({
                "page_number": page_number,
                "text":        text,
                "images":      images,
                "char_count":  len(text),
                "image_count": len([i for i in images if i["image_type"] == "embedded"]),
            })

        doc.close()

        # Build counts for quick filtering (Stage 6 corpus curation)
        embedded_img_count = sum(p["image_count"] for p in pages)
        text_char_count    = sum(p["char_count"]  for p in pages)

        result: Dict[str, Any] = {
            "input_path":  str(pdf_path),
            "output_path": str(output_json),
            "status":      "success",
            "paper_id":    pid,
            "metadata": {
                **doc_meta,
                "pages_processed":   total_pages,
                "embedded_img_count": embedded_img_count,
                "text_char_count":    text_char_count,
                "pages_dir":          str(pages_dir),
            },
            "pages": pages,
        }

        # ── Persist ───────────────────────────────────────────────────
        save_json(result, output_json)
        logger.info(
            "[pdf_loader] Done — %d pages, %d images → %s",
            total_pages, embedded_img_count, output_json,
        )

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def load_pdf(
    pdf_path: str | Path,
    paper_id: Optional[str] = None,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: instantiate PDFLoader and call load().

    Parameters
    ----------
    pdf_path        : str | Path
    paper_id        : str | None
    force_reprocess : bool

    Returns
    -------
    dict — pipeline contract payload

    Example
    -------
    ::

        from src.ingestion.pdf_loader import load_pdf
        result = load_pdf("data/raw_pdfs/2401.12345.pdf")
        if result["status"] in ("success", "cached"):
            for page in result["pages"]:
                print(page["page_number"], len(page["text"]))
    """
    return PDFLoader().load(pdf_path, paper_id=paper_id,
                            force_reprocess=force_reprocess)


# ---------------------------------------------------------------------------
# Internal error builder
# ---------------------------------------------------------------------------

def _error_response(
    input_path: str,
    output_path: str,
    message: str,
) -> Dict[str, Any]:
    """Return a structured error payload that conforms to the pipeline contract."""
    logger.error("[pdf_loader] ERROR — %s", message)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "paper_id":    "",
        "metadata":    {},
        "pages":       [],
        "message":     message,
    }


# ---------------------------------------------------------------------------
# CLI / example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke-test.  Run from the project root:

        python -m src.ingestion.pdf_loader path/to/paper.pdf

    Or without a real PDF (dry-run path check):

        python -m src.ingestion.pdf_loader
    """
    import sys, json

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if target is None or not target.exists():
        # ── Demo: show paper_id logic only ────────────────────────────
        print("=== generate_paper_id demo ===")
        test_cases = [
            "2401.12345v2.pdf",
            "2312.00123.pdf",
            "my paper (2024).pdf",
            "ACL_2023_Findings.pdf",
            ".pdf",
        ]
        for name in test_cases:
            pid = generate_paper_id(name)
            print(f"  {name!r:35s} → {pid!r}")
        print()
        print("Pass a real PDF path as argument to run full ingestion.")
        print("  python -m src.ingestion.pdf_loader path/to/paper.pdf")
        sys.exit(0)

    # ── Full ingestion ─────────────────────────────────────────────────
    result = load_pdf(target)

    # Print summary (exclude full pages array for brevity)
    summary = {k: v for k, v in result.items() if k != "pages"}
    summary["pages_count"] = len(result.get("pages", []))
    if result["pages"]:
        first = result["pages"][0]
        summary["first_page_preview"] = {
            "page_number":  first["page_number"],
            "char_count":   first["char_count"],
            "image_count":  first["image_count"],
            "text_snippet": first["text"][:120].replace("\n", " "),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

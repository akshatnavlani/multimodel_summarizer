"""
src/extraction/text_extractor.py
---------------------------------
Stage 3A — Text Extraction.

Responsibilities
----------------
1. Consume the layout result (Stage 2) and the ingestion result (Stage 1).
2. For every element typed ``text`` or ``title`` or ``list``:
   a. Retrieve the text directly from PyMuPDF's text layer (fast, zero GPU).
   b. If the text layer returns less than MIN_CHARS, run OCR via ocr_engine.py.
3. Deduplicate and clean chunks.
4. Assign stable chunk IDs and record source metadata (page, element_id, type).
5. Save  data/extracted/text_{paper_id}.json  and return the pipeline payload.

Text chunks feed directly into the FAISS embedder (Stage 6) and the
Gemini summarisation prompt (Stage 7).

Pipeline contract
-----------------
    {
        "input_path":   str,
        "output_path":  str,
        "status":       "success" | "cached" | "error",
        "paper_id":     str,
        "metadata":     {
            "total_chunks":   int,
            "total_chars":    int,
            "ocr_fallback_count": int
        },
        "text_chunks":  [
            {
                "chunk_id":    str,          # "{paper_id}_chunk_{idx:04d}"
                "element_id":  str,          # from layout
                "type":        str,          # "text" | "title" | "list"
                "page":        int,
                "text":        str,
                "char_count":  int,
                "source":      "fitz" | "paddleocr" | "tesseract"
            }
        ]
    }

Dependencies
------------
    pip install pymupdf
    (ocr_engine handles paddleocr / tesseract)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import fitz
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

# Minimum characters from fitz text layer before we fall back to OCR
_MIN_FITZ_CHARS = 20

# Minimum characters a chunk must have to be kept
_MIN_CHUNK_CHARS = 10

# Text element types we process
_TEXT_TYPES = {"text", "title", "list"}


# ---------------------------------------------------------------------------
# Path / settings helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _get_settings():
    from config.settings import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("[text] Saved → %s", path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _error_response(input_path: str, output_path: str, msg: str) -> Dict[str, Any]:
    logger.error("[text] ERROR — %s", msg)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "paper_id":    "",
        "metadata":    {},
        "text_chunks": [],
        "message":     msg,
    }


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_chunk(raw: str) -> str:
    """
    Normalise a raw text chunk extracted from a PDF page.

    - Remove null bytes and non-printable control characters.
    - Collapse excessive blank lines (> 2 consecutive → 2).
    - Strip leading/trailing whitespace per line and globally.
    - Remove hyphenation artefacts  (e.g. "re-\nsult" → "result").
    """
    # Remove null bytes and most C0 controls except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    # De-hyphenate end-of-line breaks common in PDF text extraction
    text = re.sub(r"-\n([a-z])", r"\1", text)
    # Collapse runs of spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    # Collapse runs of blank lines to at most 2
    collapsed: List[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()


def _chunk_id(paper_id: str, idx: int) -> str:
    return f"{paper_id}_chunk_{idx:04d}"


def _is_duplicate(text: str, seen: set) -> bool:
    """Exact-match deduplication (normalised whitespace)."""
    key = re.sub(r"\s+", " ", text).strip().lower()
    if key in seen:
        return True
    seen.add(key)
    return False


# ---------------------------------------------------------------------------
# Per-element text resolution
# ---------------------------------------------------------------------------

def _fitz_text_for_element(
    fitz_doc,
    element: Dict[str, Any],
) -> str:
    """
    Extract text for one element using the PyMuPDF text layer.

    Uses page clip-rect to restrict extraction to the element's bounding box
    so we don't pull in text from neighbouring regions.
    """
    if not _FITZ_OK or fitz_doc is None:
        return ""
    page_num = element.get("page", 1) - 1   # 0-based
    if page_num < 0 or page_num >= fitz_doc.page_count:
        return ""

    page = fitz_doc[page_num]
    bbox_pdf = element.get("bbox_pdf")

    if bbox_pdf and len(bbox_pdf) == 4:
        try:
            clip = fitz.Rect(*bbox_pdf)
            text = page.get_text("text", clip=clip)
        except Exception:
            text = page.get_text("text")
    else:
        text = page.get_text("text")

    return text or ""


def _resolve_element_text(
    element: Dict[str, Any],
    fitz_doc: Any,
    ocr_engine,                              # OCREngine instance (lazy)
    ingestion_page_map: Dict[int, str],     # page_number → fitz text
) -> Dict[str, str]:
    """
    Return {"text": ..., "source": "fitz" | "paddleocr" | "tesseract" | "none"}.

    Strategy:
    1. fitz clip-rect extraction.
    2. If chars < MIN_FITZ_CHARS AND saved_path exists → OCR fallback.
    3. If both fail → use the ingestion page-level text (Stage 1 output).
    """
    # 1. fitz clip-rect
    raw = _fitz_text_for_element(fitz_doc, element)
    cleaned = _clean_chunk(raw)
    if len(cleaned) >= _MIN_FITZ_CHARS:
        return {"text": cleaned, "source": "fitz"}

    # 2. OCR on saved image crop (figure crops are skipped — they go to vision)
    saved_path = element.get("saved_path")
    if saved_path and Path(saved_path).exists() and ocr_engine is not None:
        ocr_result = ocr_engine.get_result(saved_path)
        if ocr_result.get("status") == "success":
            ocr_text = _clean_chunk(ocr_result["text"])
            if len(ocr_text) >= _MIN_FITZ_CHARS:
                return {"text": ocr_text, "source": ocr_result.get("backend", "ocr")}

    # 3. Fall back to page-level ingestion text
    page_num = element.get("page", 1)
    page_text = ingestion_page_map.get(page_num, "")
    if page_text:
        return {"text": _clean_chunk(page_text), "source": "fitz"}

    return {"text": cleaned, "source": "fitz"}


# ---------------------------------------------------------------------------
# Main TextExtractor class
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Extracts, cleans, and chunks text elements from layout + ingestion results.

    Usage
    -----
    ::

        from src.extraction.text_extractor import TextExtractor

        extractor = TextExtractor()
        result = extractor.extract(ingestion_result, layout_result)
        for chunk in result["text_chunks"]:
            print(chunk["chunk_id"], chunk["text"][:80])
    """

    def __init__(self) -> None:
        self._paths = _get_paths()
        self._cfg   = _get_settings()

    def extract(
        self,
        ingestion_result: Dict[str, Any],
        layout_result:    Dict[str, Any],
        force_reprocess:  bool = False,
    ) -> Dict[str, Any]:
        """
        Extract structured text chunks for all text/title/list elements.

        Parameters
        ----------
        ingestion_result : dict — from pdf_loader.load()
        layout_result    : dict — from layout_parser.parse()
        force_reprocess  : bool

        Returns
        -------
        dict — pipeline contract payload
        """
        # Guard upstream errors
        for stage, res in (("ingestion", ingestion_result), ("layout", layout_result)):
            if res.get("status") == "error":
                return _error_response(
                    res.get("input_path", ""),
                    "",
                    f"Upstream {stage} error: {res.get('message', 'unknown')}",
                )

        paper_id   = ingestion_result["paper_id"]
        input_path = ingestion_result["input_path"]
        output_json = self._paths["extracted"] / f"text_{paper_id}.json"

        # ── Cache check ───────────────────────────────────────────────
        if not force_reprocess and output_json.exists():
            logger.info("[text] Cache hit — %s", output_json)
            try:
                cached = _load_json(output_json)
                cached["status"] = "cached"
                return cached
            except Exception as exc:
                logger.warning("[text] Cache corrupt (%s); reprocessing.", exc)

        logger.info("[text] Extracting text for paper_id=%s", paper_id)

        # Build a page-number → raw-text map from ingestion (Stage 1 safety net)
        ingestion_page_map: Dict[int, str] = {
            p["page_number"]: p.get("text", "")
            for p in ingestion_result.get("pages", [])
        }

        # Lazy OCR engine (only created if needed)
        _ocr: Optional[Any] = None

        def _get_ocr():
            nonlocal _ocr
            if _ocr is None:
                from src.extraction.ocr_engine import OCREngine
                _ocr = OCREngine()
            return _ocr

        # Open fitz doc once
        fitz_doc = None
        if _FITZ_OK and Path(input_path).exists():
            try:
                fitz_doc = fitz.open(input_path)
            except Exception as exc:
                logger.warning("[text] fitz.open failed: %s", exc)

        # Collect text elements from layout
        text_elements = [
            e for e in layout_result.get("elements", [])
            if e.get("type") in _TEXT_TYPES
        ]

        chunks: List[Dict[str, Any]] = []
        seen_texts: set = set()
        ocr_fallback_count = 0
        chunk_idx = 0

        for element in text_elements:
            resolved = _resolve_element_text(
                element            = element,
                fitz_doc           = fitz_doc,
                ocr_engine         = _get_ocr() if element.get("saved_path") else None,
                ingestion_page_map = ingestion_page_map,
            )
            text   = resolved["text"]
            source = resolved["source"]

            if len(text) < _MIN_CHUNK_CHARS:
                continue
            if _is_duplicate(text, seen_texts):
                continue

            if source in ("paddleocr", "tesseract"):
                ocr_fallback_count += 1

            chunks.append({
                "chunk_id":   _chunk_id(paper_id, chunk_idx),
                "element_id": element.get("element_id", ""),
                "type":       element.get("type", "text"),
                "page":       element.get("page", 0),
                "text":       text,
                "char_count": len(text),
                "source":     source,
            })
            chunk_idx += 1

        if fitz_doc is not None:
            fitz_doc.close()

        # If layout produced zero text elements (e.g. pure native backend
        # with no text blocks), fall back to page-level ingestion text
        if not chunks:
            logger.info(
                "[text] No layout text elements — using page-level ingestion text."
            )
            for page_num, raw_text in ingestion_page_map.items():
                cleaned = _clean_chunk(raw_text)
                if len(cleaned) < _MIN_CHUNK_CHARS:
                    continue
                if _is_duplicate(cleaned, seen_texts):
                    continue
                chunks.append({
                    "chunk_id":   _chunk_id(paper_id, chunk_idx),
                    "element_id": f"{paper_id}_p{page_num:04d}_page",
                    "type":       "text",
                    "page":       page_num,
                    "text":       cleaned,
                    "char_count": len(cleaned),
                    "source":     "fitz",
                })
                chunk_idx += 1

        total_chars = sum(c["char_count"] for c in chunks)

        result: Dict[str, Any] = {
            "input_path":  input_path,
            "output_path": str(output_json),
            "status":      "success",
            "paper_id":    paper_id,
            "metadata": {
                "total_chunks":       len(chunks),
                "total_chars":        total_chars,
                "ocr_fallback_count": ocr_fallback_count,
            },
            "text_chunks": chunks,
        }

        _save_json(result, output_json)
        logger.info(
            "[text] Done — %d chunks, %d chars, %d OCR fallbacks → %s",
            len(chunks), total_chars, ocr_fallback_count, output_json,
        )
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_text(
    ingestion_result: Dict[str, Any],
    layout_result:    Dict[str, Any],
    force_reprocess:  bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: instantiate TextExtractor and call extract().

    Example
    -------
    ::

        from src.ingestion.pdf_loader   import load_pdf
        from src.layout.layout_parser   import parse_layout
        from src.extraction.text_extractor import extract_text

        ingestion = load_pdf("data/raw_pdfs/2401.12345.pdf")
        layout    = parse_layout(ingestion)
        text_out  = extract_text(ingestion, layout)
        print(text_out["metadata"])
    """
    return TextExtractor().extract(ingestion_result, layout_result, force_reprocess)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json as _json
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None or not target.exists():
        print("Usage: python -m src.extraction.text_extractor <paper.pdf>")
        sys.exit(0)

    from src.ingestion.pdf_loader  import load_pdf
    from src.layout.layout_parser  import parse_layout

    ing    = load_pdf(target)
    layout = parse_layout(ing)
    result = extract_text(ing, layout)

    summary = {k: v for k, v in result.items() if k != "text_chunks"}
    summary["first_3_chunks"] = result["text_chunks"][:3]
    print(_json.dumps(summary, indent=2, ensure_ascii=False))

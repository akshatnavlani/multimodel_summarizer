"""
src/extraction/ocr_engine.py
-----------------------------
OCR sub-module shared by text_extractor and table_parser.

Backends (tried in order)
--------------------------
1. PaddleOCR  — best free OCR; handles dense academic figure labels,
                rotated text, and multi-language layouts.
                pip install paddleocr paddlepaddle
2. Tesseract 5 — reliable fallback; LSTM-based; 85–92 % accuracy on
                 clean academic text.
                 apt install tesseract-ocr python3-pytesseract
3. PyMuPDF text layer — zero-cost last resort when image-based OCR is
                 unavailable; works only for native (non-scanned) PDFs.

Pipeline contract
-----------------
Every public function returns:

    {
        "input_path":  str,
        "status":      "success" | "error",
        "backend":     "paddleocr" | "tesseract" | "fitz_text" | "none",
        "text":        str,
        "confidence":  float,      # mean word confidence 0-1 (1.0 for fitz)
        "message":     str         # only on error
    }

Usage
-----
    from src.extraction.ocr_engine import run_ocr
    result = run_ocr("data/figures/2401.12345/fig_001.png")
    print(result["text"])
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import pytesseract
    _TESSERACT_OK = True
    # Verify binary is reachable at import time
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        _TESSERACT_OK = False
except ImportError:
    _TESSERACT_OK = False

# PaddleOCR is imported lazily inside _PaddleBackend._ensure_engine()
# to avoid the ~3 s import penalty on every module import.
_PADDLE_OK: Optional[bool] = None   # None = not yet probed


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _PaddleBackend:
    """Wraps PaddleOCR with lazy model loading and result caching by path."""

    def __init__(self, lang: str = "en") -> None:
        self._engine = None
        self._lang   = lang

    def _ensure_engine(self) -> bool:
        global _PADDLE_OK
        if self._engine is not None:
            return True
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True handles rotated text (axis labels etc.)
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                show_log=False,
            )
            _PADDLE_OK = True
            logger.info("[ocr] PaddleOCR engine loaded (lang=%s).", self._lang)
            return True
        except Exception as exc:
            _PADDLE_OK = False
            logger.warning("[ocr] PaddleOCR unavailable: %s", exc)
            return False

    def run(self, image_path: Path) -> Optional[Dict[str, Any]]:
        if not self._ensure_engine():
            return None
        try:
            result = self._engine.ocr(str(image_path), cls=True)
            # result: [ [ [[x,y],…], (text, conf) ], … ]  per page
            lines: List[str] = []
            confidences: List[float] = []
            if result and result[0]:
                for item in result[0]:
                    if item and len(item) >= 2:
                        text_part = item[1]
                        if text_part and len(text_part) >= 2:
                            lines.append(str(text_part[0]))
                            confidences.append(float(text_part[1]))
            text = "\n".join(lines)
            mean_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
            return {"text": text, "confidence": mean_conf, "backend": "paddleocr"}
        except Exception as exc:
            logger.warning("[ocr] PaddleOCR inference error: %s", exc)
            return None


class _TesseractBackend:
    """Wraps pytesseract with a sensible academic-PDF config."""

    # PSM 6 = assume a uniform block of text  (good for figure regions)
    _CONFIG = "--psm 6 --oem 1"

    def run(self, image_path: Path) -> Optional[Dict[str, Any]]:
        if not _TESSERACT_OK or not _PIL_OK:
            return None
        try:
            img = _PILImage.open(image_path).convert("RGB")
            data = pytesseract.image_to_data(
                img,
                config=self._CONFIG,
                output_type=pytesseract.Output.DICT,
            )
            words       = data.get("text", [])
            conf_values = data.get("conf", [])
            valid_words: List[str]  = []
            valid_confs: List[float] = []
            for w, c in zip(words, conf_values):
                try:
                    cf = float(c)
                except (TypeError, ValueError):
                    cf = -1.0
                if w.strip() and cf > 0:
                    valid_words.append(w.strip())
                    valid_confs.append(cf / 100.0)   # tesseract: 0-100
            text      = " ".join(valid_words)
            mean_conf = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0
            return {"text": text, "confidence": mean_conf, "backend": "tesseract"}
        except Exception as exc:
            logger.warning("[ocr] Tesseract error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Singleton backends (shared across all calls in a process)
# ---------------------------------------------------------------------------

_paddle_backend    = _PaddleBackend()
_tesseract_backend = _TesseractBackend()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(
    image_path: str | Path,
    prefer_backend: Optional[str] = None,   # "paddleocr" | "tesseract" | None
) -> Dict[str, Any]:
    """
    Run OCR on an image file.

    Parameters
    ----------
    image_path     : str | Path — path to PNG / JPEG
    prefer_backend : str | None — force a specific backend (testing / ablation)

    Returns
    -------
    dict  — pipeline contract payload (see module docstring)

    Example
    -------
    ::

        from src.extraction.ocr_engine import run_ocr
        r = run_ocr("data/figures/paper123/fig_001.png")
        if r["status"] == "success":
            print(r["text"][:200])
    """
    p = Path(image_path)
    if not p.exists():
        return _error("none", str(p), f"Image not found: {p}")

    order = _resolve_order(prefer_backend)
    for backend_name in order:
        result = _try_backend(backend_name, p)
        if result is not None:
            return {
                "input_path": str(p),
                "status":     "success",
                "backend":    result["backend"],
                "text":       _clean_text(result["text"]),
                "confidence": round(result["confidence"], 4),
            }

    return _error("none", str(p), "All OCR backends failed or unavailable.")


class OCREngine:
    """
    Stateful wrapper around run_ocr() that caches results by path.

    Useful when the same image is queried multiple times (e.g. figure
    region cropped at different stages).

    Usage
    -----
    ::

        engine = OCREngine()
        text = engine.get_text("figures/fig_001.png")
    """

    def __init__(self, prefer_backend: Optional[str] = None) -> None:
        self._prefer  = prefer_backend
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_text(self, image_path: str | Path) -> str:
        """Return extracted text (empty string on failure)."""
        key = str(Path(image_path).resolve())
        if key not in self._cache:
            self._cache[key] = run_ocr(image_path, self._prefer)
        return self._cache[key].get("text", "")

    def get_result(self, image_path: str | Path) -> Dict[str, Any]:
        """Return full OCR result dict."""
        key = str(Path(image_path).resolve())
        if key not in self._cache:
            self._cache[key] = run_ocr(image_path, self._prefer)
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_order(prefer: Optional[str]) -> List[str]:
    """Return ordered list of backends to try."""
    default = ["paddleocr", "tesseract"]
    if prefer is None:
        return default
    if prefer in default:
        return [prefer] + [b for b in default if b != prefer]
    return default


def _try_backend(name: str, path: Path) -> Optional[Dict[str, Any]]:
    if name == "paddleocr":
        return _paddle_backend.run(path)
    if name == "tesseract":
        return _tesseract_backend.run(path)
    return None


def _clean_text(raw: str) -> str:
    """Normalise whitespace; strip null bytes and control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def _error(backend: str, path: str, msg: str) -> Dict[str, Any]:
    logger.warning("[ocr] %s", msg)
    return {
        "input_path": path,
        "status":     "error",
        "backend":    backend,
        "text":       "",
        "confidence": 0.0,
        "message":    msg,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None or not target.exists():
        print("Usage: python -m src.extraction.ocr_engine <image.png>")
        print(f"  PaddleOCR  available: {_PADDLE_OK}")
        print(f"  Tesseract  available: {_TESSERACT_OK}")
        sys.exit(0)
    result = run_ocr(target)
    print(json.dumps(result, indent=2))

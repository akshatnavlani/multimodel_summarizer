"""
src/layout/layout_parser.py
----------------------------
Stage 2 — Document Layout Parsing.

Responsibilities
----------------
1. Convert PDF pages to raster images (pdftoppm → PyMuPDF fallback).
2. Detect bounding boxes for TEXT / TITLE / LIST / FIGURE / TABLE elements.
3. Crop and persist figure patches  →  data/figures/{paper_id}/
4. Crop and persist table patches   →  data/tables/{paper_id}/
5. Write a unified layout JSON      →  data/extracted/layout_{paper_id}.json
6. Return the pipeline-contract payload.

Detection back-ends (tried in order)
--------------------------------------
Tier 1  –  layoutparser  +  PubLayNet Detectron2 model
           pip install "layoutparser[layoutmodels]" detectron2
Tier 2  –  raw Detectron2  +  PubLayNet weights
           pip install detectron2
Tier 3  –  PyMuPDF-native blocks / tables / images  (zero extra deps)
           Always available; uses fitz text-block analysis,
           fitz.Page.find_tables(), and embedded image rects.

Pipeline contract
-----------------
Every public function returns:

    {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "cached" | "error",
        "paper_id":    str,
        "backend":     "layoutparser" | "detectron2" | "pymupdf_native",
        "metadata":    {
            "total_elements": int,
            "element_counts": {"text": int, "figure": int, "table": int, ...}
        },
        "elements": [
            {
                "element_id":  str,         # "{paper_id}_p{page}_t{type}_{idx}"
                "type":        str,         # text | title | list | figure | table
                "page":        int,         # 1-based
                "bbox":        [x0,y0,x1,y1],   # image-space pixels
                "bbox_pdf":    [x0,y0,x1,y1],   # PDF user-space points (if available)
                "score":       float,       # detection confidence (1.0 for fallback)
                "saved_path":  str | null   # only for figure / table crops
            }
        ]
    }

Dependencies
------------
    pip install pymupdf Pillow
    # For full ML detection (Colab T4):
    pip install "layoutparser[layoutmodels]"
    # Detectron2 install varies by CUDA version — see:
    #   https://detectron2.readthedocs.io/tutorials/install.html
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import fitz                 # PyMuPDF — always required
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import layoutparser as lp
    _LP_OK = True
except ImportError:
    _LP_OK = False

try:
    import detectron2           # noqa: F401 — presence check only
    _D2_OK = True
except ImportError:
    _D2_OK = False

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False


# ---------------------------------------------------------------------------
# PubLayNet label map (shared by both layoutparser and raw Detectron2 paths)
# ---------------------------------------------------------------------------
PUBLAYNET_LABEL_MAP: Dict[int, str] = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
}

# Element types that get cropped and saved to disk
_VISUAL_TYPES = {"figure", "table"}

# Detectron2 / layoutparser model specs
_LP_MODEL_SPEC = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
_LP_MODEL_EXTRA_CFG = [
    "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80,
]
# Lighter model as fallback (faster on CPU)
_LP_MODEL_SPEC_LIGHT = "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config"

# Raw Detectron2 PubLayNet weights (Facebook AI Research CDN)
_D2_WEIGHTS_URL = (
    "https://huggingface.co/ds4sd/PubLayNet/resolve/main/"
    "model_final_68b7b0.pkl"
)
_D2_CONFIG_URL = (
    "https://raw.githubusercontent.com/facebookresearch/detectron2/"
    "main/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)


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
# Page-to-image conversion (Stage pre-work)
# ---------------------------------------------------------------------------

def _render_page_pdftoppm(
    pdf_path: Path,
    page_number: int,
    out_dir: Path,
    dpi: int,
) -> Optional[Path]:
    """
    Render a single PDF page to PNG using pdftoppm.

    Parameters
    ----------
    pdf_path    : Path — source PDF
    page_number : int  — 1-based page index
    out_dir     : Path — directory to write the PNG
    dpi         : int  — output resolution

    Returns
    -------
    Path to the saved PNG, or None on failure.
    """
    if not shutil.which("pdftoppm"):
        return None

    out_stem = out_dir / f"page_{page_number:04d}"
    cmd = [
        "pdftoppm",
        "-png",
        "-r", str(dpi),
        "-f", str(page_number),
        "-l", str(page_number),
        "-singlefile",
        str(pdf_path),
        str(out_stem),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=60
        )
        if result.returncode != 0:
            logger.warning(
                "[layout] pdftoppm failed on page %d: %s",
                page_number, result.stderr.decode()[:200],
            )
            return None
        out_file = out_dir / f"page_{page_number:04d}.png"
        return out_file if out_file.exists() else None
    except Exception as exc:
        logger.warning("[layout] pdftoppm exception page %d: %s", page_number, exc)
        return None


def _render_page_fitz(
    pdf_path: Path,
    page_number: int,
    out_dir: Path,
    dpi: int,
) -> Optional[Path]:
    """
    Render a single PDF page to PNG using PyMuPDF (fitz).

    Fallback for when pdftoppm is unavailable.
    """
    if not _FITZ_OK:
        return None
    out_file = out_dir / f"page_{page_number:04d}.png"
    if out_file.exists():
        return out_file
    try:
        doc  = fitz.open(str(pdf_path))
        page = doc[page_number - 1]  # 0-based
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_file))
        doc.close()
        return out_file
    except Exception as exc:
        logger.warning(
            "[layout] fitz render failed page %d: %s", page_number, exc
        )
        return None


def get_page_image(
    pdf_path: Path,
    page_number: int,
    pages_dir: Path,
    dpi: int,
) -> Optional[Path]:
    """
    Return path to a rasterised PNG for `page_number`.

    Checks the cache first (already rendered by pdf_loader Stage 1),
    then tries pdftoppm, then fitz.
    """
    # Check if pdf_loader already rendered this page
    cached = pages_dir / f"page_{page_number:04d}.png"
    if cached.exists():
        return cached

    pages_dir.mkdir(parents=True, exist_ok=True)

    path = _render_page_pdftoppm(pdf_path, page_number, pages_dir, dpi)
    if path:
        return path
    return _render_page_fitz(pdf_path, page_number, pages_dir, dpi)


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------

def _crop_and_save(
    page_image_path: Path,
    bbox: List[float],
    element_type: str,
    paper_id: str,
    page_number: int,
    element_idx: int,
    figures_dir: Path,
    tables_dir: Path,
) -> Optional[str]:
    """
    Crop a region from the page image and save it.

    Returns the path of the saved crop (str), or None if the element type
    does not produce a saved crop (e.g. plain text).
    """
    if element_type not in _VISUAL_TYPES:
        return None
    if not _PIL_OK:
        logger.warning("[layout] Pillow not available — skipping crop save.")
        return None

    out_dir = figures_dir if element_type == "figure" else tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{paper_id}_p{page_number:04d}_{element_type}_{element_idx:03d}.png"
    out_path = out_dir / out_filename

    if out_path.exists():
        return str(out_path)

    try:
        img = PILImage.open(page_image_path).convert("RGB")
        w, h = img.size
        x0 = max(0, int(bbox[0]))
        y0 = max(0, int(bbox[1]))
        x1 = min(w, int(bbox[2]))
        y1 = min(h, int(bbox[3]))
        if x1 <= x0 or y1 <= y0:
            logger.warning("[layout] Degenerate crop bbox %s on page %d", bbox, page_number)
            return None
        crop = img.crop((x0, y0, x1, y1))
        crop.save(str(out_path))
        return str(out_path)
    except Exception as exc:
        logger.warning("[layout] Crop failed for %s page %d idx %d: %s",
                       element_type, page_number, element_idx, exc)
        return None


# ---------------------------------------------------------------------------
# Element ID builder
# ---------------------------------------------------------------------------

def _make_element_id(paper_id: str, page: int, etype: str, idx: int) -> str:
    return f"{paper_id}_p{page:04d}_{etype}_{idx:03d}"


# ---------------------------------------------------------------------------
# Tier 1 — layoutparser + PubLayNet Detectron2
# ---------------------------------------------------------------------------

class _LayoutParserBackend:
    """
    Wraps the `layoutparser` library with a PubLayNet Detectron2 model.

    The model is loaded lazily on the first call to detect() and then
    cached in this instance.
    """

    def __init__(self, score_threshold: float = 0.80) -> None:
        self._model = None
        self._score_threshold = score_threshold

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not _LP_OK:
            return False
        try:
            logger.info("[layout] Loading layoutparser PubLayNet model …")
            self._model = lp.models.Detectron2LayoutModel(
                config_path=_LP_MODEL_SPEC,
                label_map=PUBLAYNET_LABEL_MAP,
                extra_config=[
                    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                    self._score_threshold,
                ],
            )
            logger.info("[layout] layoutparser model loaded.")
            return True
        except Exception as exc:
            logger.warning(
                "[layout] layoutparser model (heavy) failed: %s. "
                "Trying lighter model.", exc
            )
            try:
                self._model = lp.models.Detectron2LayoutModel(
                    config_path=_LP_MODEL_SPEC_LIGHT,
                    label_map=PUBLAYNET_LABEL_MAP,
                    extra_config=[
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        self._score_threshold,
                    ],
                )
                logger.info("[layout] layoutparser light model loaded.")
                return True
            except Exception as exc2:
                logger.warning("[layout] layoutparser light model also failed: %s", exc2)
                return False

    def detect(self, page_image_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Run detection on a rendered page image.

        Returns list of raw element dicts, or None on failure.
        """
        if not self._ensure_model():
            return None
        if not _PIL_OK:
            return None
        try:
            image = PILImage.open(page_image_path).convert("RGB")
            layout = self._model.detect(image)  # returns lp.Layout
            elements = []
            for block in layout:
                coord = block.coordinates  # (x1, y1, x2, y2)
                elements.append({
                    "type":  block.type.lower() if block.type else "text",
                    "bbox":  [float(c) for c in coord],
                    "score": float(block.score) if block.score is not None else 1.0,
                })
            return elements
        except Exception as exc:
            logger.warning("[layout] layoutparser detection failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Tier 2 — raw Detectron2 + PubLayNet weights
# ---------------------------------------------------------------------------

class _Detectron2Backend:
    """
    Uses Detectron2 directly with downloaded PubLayNet weights.

    Falls back to this when layoutparser is unavailable but detectron2 is.
    """

    def __init__(self, score_threshold: float = 0.80) -> None:
        self._predictor = None
        self._score_threshold = score_threshold

    def _ensure_predictor(self) -> bool:
        if self._predictor is not None:
            return True
        if not _D2_OK or not _NP_OK:
            return False
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.model_zoo import model_zoo

            cfg = get_cfg()
            # Base Mask-RCNN X-101 config
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
                )
            )
            cfg.MODEL.WEIGHTS = _D2_WEIGHTS_URL
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._score_threshold
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(PUBLAYNET_LABEL_MAP)
            cfg.MODEL.DEVICE = "cuda" if _cuda_available() else "cpu"

            self._predictor = DefaultPredictor(cfg)
            logger.info("[layout] Raw Detectron2 predictor loaded (device=%s).",
                        cfg.MODEL.DEVICE)
            return True
        except Exception as exc:
            logger.warning("[layout] Detectron2 backend setup failed: %s", exc)
            return False

    def detect(self, page_image_path: Path) -> Optional[List[Dict[str, Any]]]:
        if not self._ensure_predictor() or not _NP_OK:
            return None
        if not _PIL_OK:
            return None
        try:
            image = np.array(PILImage.open(page_image_path).convert("RGB"))
            # Detectron2 expects BGR
            image_bgr = image[:, :, ::-1]
            outputs = self._predictor(image_bgr)
            instances = outputs["instances"].to("cpu")
            boxes  = instances.pred_boxes.tensor.numpy().tolist()
            scores = instances.scores.numpy().tolist()
            labels = instances.pred_classes.numpy().tolist()

            elements = []
            for box, score, label in zip(boxes, scores, labels):
                elements.append({
                    "type":  PUBLAYNET_LABEL_MAP.get(label, "text"),
                    "bbox":  [float(c) for c in box],
                    "score": float(score),
                })
            return elements
        except Exception as exc:
            logger.warning("[layout] Detectron2 detection failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Tier 3 — PyMuPDF-native layout analysis (zero extra deps)
# ---------------------------------------------------------------------------

class _PyMuPDFNativeBackend:
    """
    Derives layout elements directly from PDF structure using PyMuPDF.

    Element types:
    - text   → fitz text blocks  (type == 0)
    - figure → embedded image rects  +  non-text blocks
    - table  → fitz.Page.find_tables()  (fitz >= 1.23)

    All bounding boxes are converted from PDF points to pixels at `dpi`.
    """

    def detect_on_page(
        self,
        fitz_doc,
        page_number: int,           # 1-based
        dpi: int,
    ) -> List[Dict[str, Any]]:
        """
        Return detected elements for one page.

        Parameters
        ----------
        fitz_doc    : fitz.Document
        page_number : int (1-based)
        dpi         : int

        Returns
        -------
        list of element dicts (same schema as other tiers)
        """
        if not _FITZ_OK:
            return []

        page = fitz_doc[page_number - 1]
        scale = dpi / 72.0           # 72 pt/inch is PDF native

        elements: List[Dict[str, Any]] = []

        # ── Tables (fitz.Page.find_tables, fitz >= 1.23) ──────────────
        table_rects: List[Any] = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                r = tab.bbox            # fitz.Rect or tuple
                rect = fitz.Rect(r)
                img_bbox = _pdf_rect_to_img(rect, scale)
                table_rects.append(rect)
                elements.append({
                    "type":    "table",
                    "bbox":    img_bbox,
                    "bbox_pdf": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "score":   1.0,
                })
        except (AttributeError, Exception):
            # find_tables not available in older fitz versions
            pass

        # ── Figures — embedded image rects ────────────────────────────
        figure_rects: List[Any] = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    img_bbox = _pdf_rect_to_img(r, scale)
                    # Skip tiny images (icons, decorations)
                    if (r.width < 20) or (r.height < 20):
                        continue
                    # Skip if it overlaps a detected table
                    if any(_rects_overlap(r, tr, threshold=0.5) for tr in table_rects):
                        continue
                    figure_rects.append(r)
                    elements.append({
                        "type":    "figure",
                        "bbox":    img_bbox,
                        "bbox_pdf": [r.x0, r.y0, r.x1, r.y1],
                        "score":   1.0,
                    })
            except Exception:
                continue

        # ── Text blocks ────────────────────────────────────────────────
        blocks = page.get_text("blocks")
        # blocks: (x0, y0, x1, y1, "text", block_no, block_type)
        # block_type 0 = text, 1 = image
        for block in blocks:
            if len(block) < 6:
                continue
            x0, y0, x1, y1 = block[:4]
            block_text: str = block[4] if len(block) > 4 else ""
            btype: int = int(block[6]) if len(block) > 6 else 0

            if btype == 1:
                # image block — already handled above or use as figure hint
                continue

            rect = fitz.Rect(x0, y0, x1, y1)

            # Skip text overlapping a table (it will be parsed by table parser)
            if any(_rects_overlap(rect, tr, threshold=0.7) for tr in table_rects):
                continue
            # Skip text overlapping a figure
            if any(_rects_overlap(rect, fr, threshold=0.7) for fr in figure_rects):
                continue

            stripped = block_text.strip()
            if not stripped:
                continue

            img_bbox = _pdf_rect_to_img(rect, scale)
            # Heuristic title detection: short line, capitalised, near top of page
            is_title = (
                len(stripped) < 120
                and stripped[0].isupper()
                and y0 < page.rect.height * 0.35
            )
            etype = "title" if is_title else "text"

            elements.append({
                "type":    etype,
                "bbox":    img_bbox,
                "bbox_pdf": [x0, y0, x1, y1],
                "score":   1.0,
            })

        return elements


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _pdf_rect_to_img(rect, scale: float) -> List[float]:
    """Convert a fitz.Rect (PDF points) to pixel bbox at `scale`."""
    return [
        rect.x0 * scale,
        rect.y0 * scale,
        rect.x1 * scale,
        rect.y1 * scale,
    ]


def _rects_overlap(r1, r2, threshold: float = 0.5) -> bool:
    """Return True if r1 overlaps r2 by more than `threshold` of r1's area."""
    ix0 = max(r1.x0, r2.x0)
    iy0 = max(r1.y0, r2.y0)
    ix1 = min(r1.x1, r2.x1)
    iy1 = min(r1.y1, r2.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    r1_area    = (r1.x1 - r1.x0) * (r1.y1 - r1.y0)
    if r1_area == 0:
        return False
    return (inter_area / r1_area) > threshold


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Main LayoutParser class
# ---------------------------------------------------------------------------

class LayoutParser:
    """
    Document layout parsing orchestrator.

    Selects the best available backend at construction time,
    then processes all pages of a paper.

    Usage
    -----
    ::

        from src.layout.layout_parser import LayoutParser
        from src.ingestion.pdf_loader  import load_pdf

        ingestion = load_pdf("data/raw_pdfs/2401.12345.pdf")
        parser    = LayoutParser()
        layout    = parser.parse(ingestion)

        print(layout["status"])          # "success"
        print(layout["backend"])         # "pymupdf_native" | "layoutparser" | ...
        print(len(layout["elements"]))
    """

    def __init__(self) -> None:
        self._paths   = _get_paths()
        self._cfg     = _get_settings()

        # Initialise all three tiers (model loading is lazy)
        self._lp_backend  = _LayoutParserBackend(
            score_threshold=self._cfg.LAYOUT_SCORE_THRESHOLD
        ) if _LP_OK else None

        self._d2_backend  = _Detectron2Backend(
            score_threshold=self._cfg.LAYOUT_SCORE_THRESHOLD
        ) if (_D2_OK and not _LP_OK) else None

        self._pymupdf_backend = _PyMuPDFNativeBackend()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(
        self,
        ingestion_result: Dict[str, Any],
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse layout for all pages of a paper.

        Parameters
        ----------
        ingestion_result : dict — output of PDFLoader.load()
        force_reprocess  : bool — ignore cache

        Returns
        -------
        dict — pipeline contract payload
        """
        # Validate ingestion result
        if ingestion_result.get("status") == "error":
            return _error_response(
                ingestion_result.get("input_path", ""),
                "",
                f"Upstream ingestion error: {ingestion_result.get('message', 'unknown')}",
            )

        paper_id   = ingestion_result["paper_id"]
        input_path = ingestion_result["input_path"]
        output_json = self._paths["extracted"] / f"layout_{paper_id}.json"

        # ── Cache check ─────────────────────────────────────────────
        if not force_reprocess and output_json.exists():
            logger.info("[layout] Cache hit — %s", output_json)
            try:
                cached = _load_json(output_json)
                cached["status"] = "cached"
                return cached
            except Exception as exc:
                logger.warning("[layout] Cache corrupt (%s); reprocessing.", exc)

        # ── Choose backend ──────────────────────────────────────────
        backend_name = self._pick_backend()
        logger.info("[layout] Processing %s with backend=%s", paper_id, backend_name)

        # ── Process pages ───────────────────────────────────────────
        pages_dir   = self._paths["extracted"] / paper_id / "pages"
        figures_dir = self._paths["figures"] / paper_id
        tables_dir  = self._paths["tables"]  / paper_id

        pages_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        pdf_path     = Path(input_path)
        ingested_pages = ingestion_result.get("pages", [])
        total_pages  = len(ingested_pages)

        all_elements: List[Dict[str, Any]] = []
        element_counts: Dict[str, int]     = {}
        global_idx: Dict[str, int]         = {}

        # Open fitz doc once for the native backend (avoids re-opening per page)
        fitz_doc = None
        if backend_name == "pymupdf_native" and _FITZ_OK and pdf_path.exists():
            try:
                fitz_doc = fitz.open(str(pdf_path))
            except Exception as exc:
                logger.warning("[layout] fitz.open failed: %s", exc)

        for page_info in ingested_pages:
            page_number: int = page_info["page_number"]

            raw_elements = self._detect_page(
                backend_name   = backend_name,
                pdf_path       = pdf_path,
                page_number    = page_number,
                pages_dir      = pages_dir,
                fitz_doc       = fitz_doc,
            )

            if raw_elements is None:
                # Hard fallback to native if primary failed
                if fitz_doc is not None:
                    raw_elements = self._pymupdf_backend.detect_on_page(
                        fitz_doc, page_number, self._cfg.PDF_DPI
                    )
                    logger.debug(
                        "[layout] Used pymupdf_native fallback for page %d", page_number
                    )
                else:
                    raw_elements = []

            # Process each element: assign IDs, crop, save
            for raw in raw_elements:
                etype = raw["type"]
                idx   = global_idx.get(etype, 0)
                global_idx[etype] = idx + 1
                element_counts[etype] = element_counts.get(etype, 0) + 1

                eid   = _make_element_id(paper_id, page_number, etype, idx)
                bbox  = raw["bbox"]

                # Crop and save visual elements
                saved_path: Optional[str] = None
                page_img_path = pages_dir / f"page_{page_number:04d}.png"
                if page_img_path.exists():
                    saved_path = _crop_and_save(
                        page_image_path = page_img_path,
                        bbox            = bbox,
                        element_type    = etype,
                        paper_id        = paper_id,
                        page_number     = page_number,
                        element_idx     = idx,
                        figures_dir     = figures_dir,
                        tables_dir      = tables_dir,
                    )

                record: Dict[str, Any] = {
                    "element_id": eid,
                    "type":       etype,
                    "page":       page_number,
                    "bbox":       bbox,
                    "bbox_pdf":   raw.get("bbox_pdf", []),
                    "score":      raw.get("score", 1.0),
                    "saved_path": saved_path,
                }
                all_elements.append(record)

        if fitz_doc is not None:
            fitz_doc.close()

        result: Dict[str, Any] = {
            "input_path":  input_path,
            "output_path": str(output_json),
            "status":      "success",
            "paper_id":    paper_id,
            "backend":     backend_name,
            "metadata": {
                "total_elements": len(all_elements),
                "pages_processed": total_pages,
                "element_counts":  element_counts,
                "figures_dir":     str(figures_dir),
                "tables_dir":      str(tables_dir),
            },
            "elements": all_elements,
        }

        _save_json(result, output_json)
        logger.info(
            "[layout] Done — %d elements across %d pages → %s",
            len(all_elements), total_pages, output_json,
        )
        return result

    # ------------------------------------------------------------------
    # Backend selection and per-page dispatch
    # ------------------------------------------------------------------

    def _pick_backend(self) -> str:
        """Return the name of the best available detection backend."""
        if _LP_OK:
            return "layoutparser"
        if _D2_OK:
            return "detectron2"
        return "pymupdf_native"

    def _detect_page(
        self,
        backend_name: str,
        pdf_path: Path,
        page_number: int,
        pages_dir: Path,
        fitz_doc: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run detection for a single page with the chosen backend.

        Returns
        -------
        list of raw element dicts, or None on failure (triggers fallback).
        """
        if backend_name == "pymupdf_native":
            if fitz_doc is None:
                return None
            return self._pymupdf_backend.detect_on_page(
                fitz_doc, page_number, self._cfg.PDF_DPI
            )

        # For ML-based backends we need the rendered page image
        page_img_path = get_page_image(
            pdf_path    = pdf_path,
            page_number = page_number,
            pages_dir   = pages_dir,
            dpi         = self._cfg.PDF_DPI,
        )
        if page_img_path is None:
            logger.warning(
                "[layout] Could not render page %d — falling back to native.",
                page_number,
            )
            return None

        if backend_name == "layoutparser" and self._lp_backend:
            return self._lp_backend.detect(page_img_path)

        if backend_name == "detectron2" and self._d2_backend:
            return self._d2_backend.detect(page_img_path)

        return None  # triggers caller fallback


# ---------------------------------------------------------------------------
# Convenience module-level function
# ---------------------------------------------------------------------------

def parse_layout(
    ingestion_result: Dict[str, Any],
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: instantiate LayoutParser and call parse().

    Parameters
    ----------
    ingestion_result : dict — from PDFLoader.load()
    force_reprocess  : bool

    Returns
    -------
    dict — pipeline contract payload

    Example
    -------
    ::

        from src.ingestion.pdf_loader  import load_pdf
        from src.layout.layout_parser  import parse_layout

        result  = load_pdf("data/raw_pdfs/2401.12345.pdf")
        layout  = parse_layout(result)

        figures = [e for e in layout["elements"] if e["type"] == "figure"]
        tables  = [e for e in layout["elements"] if e["type"] == "table"]
        print(f"{len(figures)} figures, {len(tables)} tables detected.")
    """
    return LayoutParser().parse(ingestion_result, force_reprocess=force_reprocess)


# ---------------------------------------------------------------------------
# I/O utilities (thin wrappers — keep layout module self-contained)
# ---------------------------------------------------------------------------

def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("[layout] Saved → %s", path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _error_response(
    input_path: str,
    output_path: str,
    message: str,
) -> Dict[str, Any]:
    logger.error("[layout] ERROR — %s", message)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "paper_id":    "",
        "backend":     "none",
        "metadata":    {},
        "elements":    [],
        "message":     message,
    }


# ---------------------------------------------------------------------------
# CLI / example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke-test.  Run from project root:

        python -m src.layout.layout_parser path/to/paper.pdf
    """
    import sys
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if target is None or not target.exists():
        print("Usage: python -m src.layout.layout_parser path/to/paper.pdf")
        print()
        print("Available backends:")
        print(f"  layoutparser  : {'✓' if _LP_OK else '✗ (pip install layoutparser[layoutmodels])'}")
        print(f"  detectron2    : {'✓' if _D2_OK else '✗ (pip install detectron2)'}")
        print(f"  pymupdf_native: {'✓' if _FITZ_OK else '✗ (pip install pymupdf)'}")
        sys.exit(0)

    # Run full pipeline from PDF
    from src.ingestion.pdf_loader import load_pdf

    print(f"[1/2] Ingesting {target} …")
    ingestion = load_pdf(target)
    if ingestion["status"] == "error":
        print(f"Ingestion error: {ingestion['message']}")
        sys.exit(1)

    print(f"[2/2] Parsing layout …")
    layout = parse_layout(ingestion)

    # Print summary
    summary = {k: v for k, v in layout.items() if k != "elements"}
    summary["sample_elements"] = layout["elements"][:5]
    print(_json.dumps(summary, indent=2))

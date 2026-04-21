"""
src/vision/chart_extractor.py
------------------------------
Deplot based chart-to-table extraction for scientific PDF figures.

Deplot (google/deplot) converts chart images into linearized data tables,
enabling numerical data extraction from bar charts, line graphs, pie charts, etc.

Pipeline contract:
    INPUT  → ingestion_result (Dict), layout_result (Dict),
             figure_result (Dict, optional), force_reprocess (bool)
    OUTPUT → {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "error",
        "metadata":    {
            "paper_id":         str,
            "total_charts":     int,
            "extracted_charts": int,
            "failed_charts":    int,
            "device_used":      str,
            "model_used":       str,
            "charts": [
                {
                    "chart_id":    str,
                    "data":        str,   # linearized table from Deplot
                    "vqa_fallback":str,   # BLIP-2 VQA if Deplot fails
                    "page":        int,
                    "bbox":        list,
                    "crop_path":   str,
                    "source":      str,
                    "failed":      bool,
                }
            ]
        }
    }

Caching:
    - Checks data/figures/{paper_id}_chart_data.json before running Deplot.
    - Saves after processing each paper.

Chart detection strategy:
    - Primary: uses figure crops already extracted by figure_understander.
    - Heuristic: figures with aspect ratio typical of charts (wide/short).
    - Fallback: treats ALL figure elements as potential charts.

Usage:
    from src.vision.chart_extractor import extract_charts
    result = extract_charts(ingestion_result, layout_result, figure_result)
    for chart in result["metadata"]["charts"]:
        print(chart["chart_id"], chart["data"][:100])
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model references
# ---------------------------------------------------------------------------
_deplot_processor = None
_deplot_model     = None
_blip2_processor  = None   # for VQA fallback
_blip2_model      = None
_device_used      = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _get_settings():
    from config.settings import get_settings
    return get_settings()


def _paper_id_from_ingestion(ingestion_result: Dict[str, Any]) -> str:
    pid = ingestion_result.get("paper_id") or ingestion_result.get(
        "metadata", {}
    ).get("paper_id", "")
    if not pid:
        pid = Path(ingestion_result.get("input_path", "unknown.pdf")).stem
    return pid


def _cache_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["figures"] / f"{paper_id}_chart_data.json"


def _error_response(input_path: str, message: str, output_path: str = "") -> Dict[str, Any]:
    logger.error("[chart_extractor] %s", message)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "message":     message,
        "metadata":    {},
    }


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model_cache_dir(base_paths: Dict[str, Path]) -> str:
    cache = base_paths.get("model_cache", Path(".model_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


# ---------------------------------------------------------------------------
# Deplot model loading
# ---------------------------------------------------------------------------

def _load_deplot(device: str, model_cache_dir: str) -> Tuple[Any, Any]:
    """
    Load Deplot (Pix2Struct) processor and model.

    Deplot is lightweight (~1.3B params) and runs comfortably on T4 in fp16.
    On CPU it uses fp32 and is slower but functional.
    """
    global _deplot_processor, _deplot_model

    if _deplot_processor is not None:
        logger.info("[chart_extractor] Deplot already loaded — reusing.")
        return _deplot_processor, _deplot_model

    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

    MODEL_ID = "google/deplot"
    logger.info("[chart_extractor] Loading Deplot (%s) on %s …", MODEL_ID, device)

    processor = Pix2StructProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=model_cache_dir,
    )

    if device == "cuda":
        import torch
        model = Pix2StructForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=model_cache_dir,
        ).to(device)
    else:
        model = Pix2StructForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=model_cache_dir,
        )
        model.eval()

    _deplot_processor = processor
    _deplot_model     = model
    logger.info("[chart_extractor] Deplot loaded on %s.", device)
    return processor, model


# ---------------------------------------------------------------------------
# BLIP-2 VQA fallback loader
# ---------------------------------------------------------------------------

def _load_blip2_vqa(device: str, model_cache_dir: str) -> Tuple[Any, Any]:
    """Load BLIP-2 for VQA fallback when Deplot fails."""
    global _blip2_processor, _blip2_model

    if _blip2_processor is not None:
        return _blip2_processor, _blip2_model

    # Try to reuse already-loaded model from figure_understander
    try:
        from src.vision.figure_understander import _blip2_processor as _fp
        from src.vision.figure_understander import _blip2_model     as _fm
        if _fp is not None:
            logger.info("[chart_extractor] Reusing BLIP-2 from figure_understander.")
            _blip2_processor = _fp
            _blip2_model     = _fm
            return _blip2_processor, _blip2_model
    except ImportError:
        pass

    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    MODEL_ID  = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(MODEL_ID, cache_dir=model_cache_dir)

    if device == "cuda":
        try:
            model = Blip2ForConditionalGeneration.from_pretrained(
                MODEL_ID, load_in_8bit=True, device_map="auto", cache_dir=model_cache_dir
            )
        except Exception:
            model = Blip2ForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, cache_dir=model_cache_dir
            ).to(device)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, cache_dir=model_cache_dir
        )
        model.eval()

    _blip2_processor = processor
    _blip2_model     = model
    return processor, model


# ---------------------------------------------------------------------------
# Chart detection heuristic
# ---------------------------------------------------------------------------

def _is_likely_chart(bbox: List[float], caption: str = "") -> bool:
    """
    Heuristic: decide if a figure is likely a chart/graph.

    Rules:
      - Aspect ratio: width > height (most charts are wider than tall)
      - Caption keywords: chart, graph, figure, plot, bar, line, pie, table
    """
    chart_keywords = {
        "chart", "graph", "plot", "bar", "line", "pie", "histogram",
        "scatter", "curve", "figure", "fig", "accuracy", "loss",
        "performance", "results", "comparison",
    }

    if len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect = w / max(h, 1)
        if aspect > 0.8:   # wider than tall → likely a chart
            return True

    if caption:
        words = set(caption.lower().split())
        if words & chart_keywords:
            return True

    return False


def _collect_chart_candidates(
    figure_elements: List[Dict],
    figure_result: Optional[Dict],
) -> List[Dict]:
    """
    Build the list of chart candidates to process with Deplot.

    If figure_result is available, enrich with captions for heuristic.
    Otherwise, treat all figure elements as candidates.
    """
    # Build caption lookup from figure_result
    caption_map: Dict[str, str] = {}
    if figure_result and figure_result.get("status") in ("success", "cached"):
        figs = figure_result.get("metadata", {}).get("figures", [])
        for f in figs:
            caption_map[f.get("element_id", f.get("figure_id", ""))] = f.get("caption", "")

    candidates = []
    for el in figure_elements:
        bbox     = el.get("bbox", el.get("bbox_pdf", []))
        elem_id  = el.get("element_id", el.get("id", ""))
        caption  = caption_map.get(elem_id, "")
        crop_path = el.get("crop_path", "")

        candidates.append({
            "element_id": elem_id,
            "bbox":       bbox,
            "page":       el.get("page", 1),
            "caption":    caption,
            "crop_path":  crop_path,
            "is_chart":   _is_likely_chart(bbox, caption),
        })

    return candidates


# ---------------------------------------------------------------------------
# Figure crop helper (for elements without existing crops)
# ---------------------------------------------------------------------------

def _ensure_crop(
    el: Dict,
    pdf_path: Path,
    figures_dir: Path,
    paper_id: str,
    chart_idx: int,
    dpi_scale: float = 2.0,
) -> Optional[str]:
    """Crop a figure from PDF if crop_path is missing or doesn't exist."""
    crop_path = el.get("crop_path", "")
    if crop_path and Path(crop_path).exists():
        return crop_path

    try:
        import fitz
        bbox     = el.get("bbox", el.get("bbox_pdf", []))
        page_num = max(0, el.get("page", 1) - 1)

        if len(bbox) != 4:
            return None

        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < 10 or h < 10:
            return None

        chart_dir = figures_dir / paper_id
        chart_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            doc.close()
            return None

        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = doc[page_num].get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
        doc.close()

        if pix.width == 0 or pix.height == 0:
            return None

        out = chart_dir / f"{paper_id}_chart_{chart_idx:04d}.png"
        pix.save(str(out))
        return str(out)

    except Exception as e:
        logger.warning("[chart_extractor] Crop failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Deplot extraction
# ---------------------------------------------------------------------------

def _run_deplot_batch(
    chart_items: List[Dict],
    processor: Any,
    model: Any,
    device: str,
    batch_size: int = 2,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """
    Run Deplot on a batch of chart crops.

    Deplot's prompt is 'Generate underlying data table of the figure below:'.
    Returns list of items with 'data' key added.
    """
    from PIL import Image
    import torch

    DEPLOT_PROMPT = "Generate underlying data table of the figure below:"
    results = []

    for i in range(0, len(chart_items), batch_size):
        batch = chart_items[i: i + batch_size]
        images = []
        valid  = []

        for item in batch:
            cp = item.get("crop_path", "")
            if cp and Path(cp).exists():
                try:
                    img = Image.open(cp).convert("RGB")
                    if img.size[0] > 0 and img.size[1] > 0:
                        images.append(img)
                        valid.append(item)
                        continue
                except Exception as e:
                    logger.warning("[chart_extractor] Cannot open %s: %s", cp, e)
            # Invalid image — mark as failed
            item["data"]   = "[deplot failed: invalid image]"
            item["failed"] = True
            results.append(item)

        if not images:
            continue

        try:
            inputs = processor(
                images=images,
                text=[DEPLOT_PROMPT] * len(images),
                return_tensors="pt",
                max_patches=512,
            )

            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            decoded = processor.batch_decode(generated, skip_special_tokens=True)

            for item, table_str in zip(valid, decoded):
                item["data"]   = table_str.strip()
                item["failed"] = False
                results.append(item)
                logger.debug(
                    "[chart_extractor] Chart %s → %d chars of table data.",
                    item.get("chart_id", "?"), len(table_str),
                )

        except Exception as e:
            logger.error("[chart_extractor] Deplot batch %d failed: %s", i // batch_size, e)
            for item in valid:
                item["data"]   = f"[deplot failed: {str(e)[:80]}]"
                item["failed"] = True
                results.append(item)

    return results


# ---------------------------------------------------------------------------
# VQA fallback
# ---------------------------------------------------------------------------

def _vqa_fallback(
    chart_items: List[Dict],
    blip2_proc: Any,
    blip2_model: Any,
    device: str,
) -> List[Dict]:
    """
    For charts where Deplot failed, use BLIP-2 VQA to extract a text description.
    """
    from PIL import Image
    import torch

    QUESTION = "What values does this chart show?"

    for item in chart_items:
        if not item.get("failed", False):
            continue

        cp = item.get("crop_path", "")
        if not cp or not Path(cp).exists():
            item["vqa_fallback"] = "[no image available]"
            continue

        try:
            image  = Image.open(cp).convert("RGB")
            inputs = blip2_proc(images=image, text=QUESTION, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = blip2_model.generate(**inputs, max_new_tokens=80)
            answer = blip2_proc.decode(out[0], skip_special_tokens=True).strip()
            item["vqa_fallback"] = answer
            logger.debug("[chart_extractor] VQA fallback for %s: %s", item.get("chart_id"), answer)
        except Exception as e:
            item["vqa_fallback"] = f"[vqa failed: {str(e)[:60]}]"

    return chart_items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_charts(
    ingestion_result: Dict[str, Any],
    layout_result: Dict[str, Any],
    figure_result: Optional[Dict[str, Any]] = None,
    force_reprocess: bool = False,
    batch_size: int = 2,
    max_new_tokens: int = 512,
    use_vqa_fallback: bool = True,
    process_all_figures: bool = False,
) -> Dict[str, Any]:
    """
    Extract structured table data from chart figures using Deplot.

    Parameters
    ----------
    ingestion_result    : Dict — output of load_pdf()
    layout_result       : Dict — output of parse_layout()
    figure_result       : Dict — output of describe_figures() (optional, enriches heuristic)
    force_reprocess     : bool — ignore cache
    batch_size          : int  — Deplot batch size (reduce to 1 if OOM)
    max_new_tokens      : int  — max table tokens from Deplot
    use_vqa_fallback    : bool — run BLIP-2 VQA on failed charts
    process_all_figures : bool — if True, run Deplot on all figures, not just chart-like ones

    Returns
    -------
    Pipeline contract dict.
    """
    # ── Guard upstream errors ────────────────────────────────────────────────
    for stage, res in (("ingestion", ingestion_result), ("layout", layout_result)):
        if res.get("status") == "error":
            return _error_response(
                res.get("input_path", ""),
                f"Upstream {stage} failed: {res.get('message','')}",
            )

    input_path = ingestion_result.get("input_path", "")
    paper_id   = _paper_id_from_ingestion(ingestion_result)
    base_paths = _get_paths()
    out_path   = _cache_path(paper_id, base_paths)

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_reprocess and out_path.exists():
        try:
            cached = json.loads(out_path.read_text())
            logger.info("[chart_extractor] Cache hit for %s.", paper_id)
            return {
                "input_path":  input_path,
                "output_path": str(out_path),
                "status":      "success",
                "cached":      True,
                "metadata":    cached,
            }
        except Exception as e:
            logger.warning("[chart_extractor] Cache corrupt (%s) — reprocessing.", e)

    # ── Extract figure elements ──────────────────────────────────────────────
    elements = layout_result.get("metadata", {}).get(
        "elements", layout_result.get("elements", [])
    )
    figure_elements = [e for e in elements if e.get("type") == "figure"]

    if not figure_elements:
        logger.warning("[chart_extractor] No figure elements for %s.", paper_id)
        result_meta = {
            "paper_id":         paper_id,
            "total_charts":     0,
            "extracted_charts": 0,
            "failed_charts":    0,
            "device_used":      "none",
            "model_used":       "none",
            "charts":           [],
        }
        out_path.write_text(json.dumps(result_meta, indent=2))
        return {
            "input_path":  input_path,
            "output_path": str(out_path),
            "status":      "success",
            "metadata":    result_meta,
        }

    # ── Collect chart candidates ─────────────────────────────────────────────
    candidates = _collect_chart_candidates(figure_elements, figure_result)

    if process_all_figures:
        to_process = candidates
    else:
        to_process = [c for c in candidates if c["is_chart"]]
        if not to_process:
            logger.info(
                "[chart_extractor] No chart-like figures detected for %s — "
                "falling back to all %d figures.", paper_id, len(candidates)
            )
            to_process = candidates

    # ── Ensure all candidates have crop paths ───────────────────────────────
    pdf_path = Path(input_path)
    for idx, item in enumerate(to_process):
        if not item.get("crop_path") or not Path(item["crop_path"]).exists():
            item["crop_path"] = _ensure_crop(
                el         = item,
                pdf_path   = pdf_path,
                figures_dir= base_paths["figures"],
                paper_id   = paper_id,
                chart_idx  = idx,
            ) or ""

    # Add chart_id
    for idx, item in enumerate(to_process):
        item["chart_id"] = f"{paper_id}_chart_{idx:04d}"

    # ── Load Deplot ──────────────────────────────────────────────────────────
    device    = _detect_device()
    cache_dir = _get_model_cache_dir(base_paths)
    t0        = time.time()

    try:
        deplot_proc, deplot_model = _load_deplot(device, cache_dir)
    except Exception as e:
        return _error_response(input_path, f"Deplot load failed: {e}", str(out_path))

    # ── Run Deplot ───────────────────────────────────────────────────────────
    processed = _run_deplot_batch(
        chart_items   = to_process,
        processor     = deplot_proc,
        model         = deplot_model,
        device        = device,
        batch_size    = batch_size,
        max_new_tokens= max_new_tokens,
    )

    # ── VQA fallback for failed charts ───────────────────────────────────────
    n_failed = sum(1 for c in processed if c.get("failed", False))
    if use_vqa_fallback and n_failed > 0:
        logger.info(
            "[chart_extractor] Running VQA fallback on %d failed charts.", n_failed
        )
        try:
            blip2_proc, blip2_model = _load_blip2_vqa(device, cache_dir)
            processed = _vqa_fallback(processed, blip2_proc, blip2_model, device)
        except Exception as e:
            logger.warning("[chart_extractor] VQA fallback load failed: %s", e)

    # ── Build output ─────────────────────────────────────────────────────────
    charts_out = []
    n_ok = 0
    for item in processed:
        failed = item.get("failed", False)
        if not failed:
            n_ok += 1
        charts_out.append({
            "chart_id":    item.get("chart_id", ""),
            "element_id":  item.get("element_id", ""),
            "data":        item.get("data", ""),
            "vqa_fallback":item.get("vqa_fallback", ""),
            "page":        item.get("page", -1),
            "bbox":        item.get("bbox", []),
            "crop_path":   item.get("crop_path", ""),
            "source":      item.get("source", "heuristic"),
            "is_chart":    item.get("is_chart", True),
            "failed":      failed,
        })

    elapsed = time.time() - t0
    logger.info(
        "[chart_extractor] %s: %d/%d charts in %.1fs.",
        paper_id, n_ok, len(charts_out), elapsed,
    )

    result_meta = {
        "paper_id":         paper_id,
        "total_charts":     len(to_process),
        "extracted_charts": n_ok,
        "failed_charts":    len(charts_out) - n_ok,
        "device_used":      device,
        "model_used":       "google/deplot",
        "elapsed_seconds":  round(elapsed, 2),
        "charts":           charts_out,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(result_meta, indent=2))
    logger.info("[chart_extractor] Saved → %s", out_path)

    return {
        "input_path":  input_path,
        "output_path": str(out_path),
        "status":      "success",
        "metadata":    result_meta,
    }


def extract_charts_batch(
    paper_list: List[Tuple[Dict, Dict, Optional[Dict]]],
    force_reprocess: bool = False,
    batch_size: int = 2,
) -> List[Dict[str, Any]]:
    """
    Process multiple papers sharing a single loaded Deplot model.

    Parameters
    ----------
    paper_list : list of (ingestion_result, layout_result, figure_result|None)
    """
    results = []
    for ing, lay, fig_res in paper_list:
        r = extract_charts(
            ingestion_result=ing,
            layout_result=lay,
            figure_result=fig_res,
            force_reprocess=force_reprocess,
            batch_size=batch_size,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.ingestion.pdf_loader      import load_pdf
    from src.layout.layout_parser      import parse_layout
    from src.vision.figure_understander import describe_figures

    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw_pdfs/")
        sys.exit(1)

    test_pdf = pdfs[0]
    print(f"Testing on: {test_pdf}")

    ing = load_pdf(test_pdf)
    lay = parse_layout(ing)
    fig = describe_figures(ing, lay)

    result = extract_charts(ing, lay, figure_result=fig, force_reprocess=True, batch_size=1)

    print(f"\nStatus:    {result['status']}")
    print(f"Output:    {result['output_path']}")
    meta = result.get("metadata", {})
    print(f"Charts:    {meta.get('extracted_charts')} / {meta.get('total_charts')}")
    print(f"Failed:    {meta.get('failed_charts')}")
    print(f"Device:    {meta.get('device_used')}")
    print(f"Time:      {meta.get('elapsed_seconds')}s")
    print(f"\nFirst 3 chart data samples:")
    for c in meta.get("charts", [])[:3]:
        status = "✅" if not c["failed"] else "⚠️ VQA"
        fallback = f" → VQA: {c['vqa_fallback'][:60]}" if c["failed"] and c.get("vqa_fallback") else ""
        print(f"  {status} [{c['chart_id']}] p{c['page']}: {c['data'][:80]}{fallback}")

"""
src/vision/figure_understander.py
----------------------------------
BLIP-2 based figure captioning for scientific PDF figures.

Pipeline contract:
    INPUT  → ingestion_result (Dict), layout_result (Dict), force_reprocess (bool)
    OUTPUT → {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "error",
        "metadata":    {
            "paper_id":         str,
            "total_figures":    int,
            "processed_figures":int,
            "failed_figures":   int,
            "device_used":      str,
            "model_used":       str,
            "figures":          [{"figure_id", "caption", "page", "bbox", "source"}]
        }
    }

Caching:
    - Checks data/figures/{paper_id}_fig_descriptions.json before running BLIP-2.
    - Saves immediately after each paper so GPU quota loss doesn't lose work.
    - Model weights cached to .model_cache/ to avoid re-downloading.

Performance:
    - 8-bit quantization on GPU (T4 / P100).
    - CPU fp32 fallback with smaller batch size.
    - Batch processing: configurable batch_size.

Usage:
    from src.vision.figure_understander import describe_figures
    result = describe_figures(ingestion_result, layout_result)
    print(result["metadata"]["figures"][0]["caption"])
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when model is actually loaded
# ---------------------------------------------------------------------------
_blip2_processor = None
_blip2_model     = None
_device          = None


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
    """Extract a clean paper_id from the ingestion result."""
    pid = ingestion_result.get("paper_id") or ingestion_result.get(
        "metadata", {}
    ).get("paper_id", "")
    if not pid:
        raw = ingestion_result.get("input_path", "unknown.pdf")
        pid = Path(raw).stem
    return pid


def _cache_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["figures"] / f"{paper_id}_fig_descriptions.json"


def _error_response(input_path: str, message: str, output_path: str = "") -> Dict[str, Any]:
    logger.error("[figure_understander] %s", message)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "message":     message,
        "metadata":    {},
    }


# ---------------------------------------------------------------------------
# Device + model loading
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_model_cache_dir(base_paths: Dict[str, Path]) -> str:
    cache = base_paths.get("model_cache", Path(".model_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def _load_blip2(device: str, model_cache_dir: str) -> Tuple[Any, Any, str]:
    """
    Load BLIP-2 processor and model.

    Strategy:
      - GPU  → 8-bit quantization via bitsandbytes (saves ~6 GB VRAM)
      - CPU  → fp32, no quantization (slow but works on any machine)

    Returns (processor, model, actual_model_name)
    """
    global _blip2_processor, _blip2_model, _device

    if _blip2_processor is not None:
        logger.info("[figure_understander] BLIP-2 already loaded — reusing.")
        return _blip2_processor, _blip2_model, _device

    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    logger.info("[figure_understander] Loading BLIP-2 (%s) on %s …", MODEL_ID, device)

    processor = Blip2Processor.from_pretrained(
        MODEL_ID,
        cache_dir=model_cache_dir,
    )

    if device == "cuda":
        try:
            import torch
            model = Blip2ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                load_in_8bit=True,
                device_map="auto",
                cache_dir=model_cache_dir,
            )
            logger.info("[figure_understander] BLIP-2 loaded in 8-bit on GPU.")
        except Exception as e:
            logger.warning(
                "[figure_understander] 8-bit load failed (%s) — falling back to fp16.", e
            )
            import torch
            model = Blip2ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                cache_dir=model_cache_dir,
            ).to(device)
    else:
        import torch
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            cache_dir=model_cache_dir,
        )
        model.eval()
        logger.info("[figure_understander] BLIP-2 loaded in fp32 on CPU.")

    _blip2_processor = processor
    _blip2_model     = model
    _device          = device
    return processor, model, MODEL_ID


# ---------------------------------------------------------------------------
# Figure crop extraction
# ---------------------------------------------------------------------------

def _extract_figure_crops(
    pdf_path: Path,
    figure_elements: List[Dict[str, Any]],
    figures_dir: Path,
    paper_id: str,
    dpi_scale: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Crop each figure bounding box from the PDF and save as PNG.

    Returns list of dicts: {figure_id, crop_path, page, bbox, element_id}
    """
    import fitz  # PyMuPDF

    crops = []
    paper_fig_dir = figures_dir / paper_id
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi_scale, dpi_scale)

    for idx, el in enumerate(figure_elements):
        bb       = el.get("bbox", el.get("bbox_pdf", []))
        page_num = max(0, el.get("page", 1) - 1)
        elem_id  = el.get("element_id", el.get("id", f"fig_{idx:04d}"))
        fig_id   = f"{paper_id}_fig_{idx:04d}"

        if len(bb) != 4:
            logger.warning("[figure_understander] Element %s has no valid bbox — skipped.", elem_id)
            continue

        if page_num >= len(doc):
            logger.warning("[figure_understander] Page %d out of range for %s.", page_num, paper_id)
            continue

        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0

        # Skip degenerate / tiny boxes
        if w < 20 or h < 20:
            logger.debug("[figure_understander] Skipping tiny box %s (%dx%d).", fig_id, w, h)
            continue

        try:
            page = doc[page_num]
            pix  = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))

            if pix.width == 0 or pix.height == 0:
                logger.debug("[figure_understander] Zero-size pixmap for %s — skipped.", fig_id)
                continue

            crop_path = paper_fig_dir / f"{fig_id}.png"
            pix.save(str(crop_path))

            crops.append({
                "figure_id":  fig_id,
                "element_id": elem_id,
                "crop_path":  str(crop_path),
                "page":       page_num + 1,
                "bbox":       bb,
                "source":     "layout_detection",
            })
        except Exception as e:
            logger.error("[figure_understander] Crop failed for %s: %s", fig_id, e)

    doc.close()
    logger.info("[figure_understander] Extracted %d figure crops for %s.", len(crops), paper_id)
    return crops


# ---------------------------------------------------------------------------
# BLIP-2 captioning
# ---------------------------------------------------------------------------

def _caption_batch(
    crop_dicts: List[Dict[str, Any]],
    processor: Any,
    model: Any,
    device: str,
    batch_size: int = 4,
    max_new_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Run BLIP-2 captioning on a list of crop dicts.
    Returns the same list with 'caption' key added.
    """
    from PIL import Image
    import torch

    results = []

    for i in range(0, len(crop_dicts), batch_size):
        batch = crop_dicts[i: i + batch_size]
        images = []

        for item in batch:
            try:
                img = Image.open(item["crop_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("[figure_understander] Cannot open %s: %s", item["crop_path"], e)
                images.append(Image.new("RGB", (224, 224), color=(200, 200, 200)))

        try:
            inputs = processor(images=images, return_tensors="pt")

            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    min_length=5,
                )

            captions = processor.batch_decode(generated, skip_special_tokens=True)

            for item, caption in zip(batch, captions):
                item["caption"] = caption.strip()
                results.append(item)

            logger.debug(
                "[figure_understander] Batch %d/%d captioned.",
                i // batch_size + 1,
                (len(crop_dicts) + batch_size - 1) // batch_size,
            )

        except Exception as e:
            logger.error("[figure_understander] BLIP-2 batch failed: %s", e)
            for item in batch:
                item["caption"] = f"[captioning failed: {str(e)[:60]}]"
                results.append(item)

    return results


# ---------------------------------------------------------------------------
# VQA fallback for failed Deplot charts
# ---------------------------------------------------------------------------

def _vqa_caption(
    crop_path: str,
    question: str,
    processor: Any,
    model: Any,
    device: str,
) -> str:
    """Run BLIP-2 in VQA mode on a single image."""
    from PIL import Image
    import torch

    try:
        image  = Image.open(crop_path).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"[vqa failed: {str(e)[:60]}]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_figures(
    ingestion_result: Dict[str, Any],
    layout_result: Dict[str, Any],
    force_reprocess: bool = False,
    batch_size: int = 4,
    max_new_tokens: int = 100,
    dpi_scale: float = 2.0,
) -> Dict[str, Any]:
    """
    Caption all figure elements detected in layout_result using BLIP-2.

    Parameters
    ----------
    ingestion_result : Dict  — output of load_pdf()
    layout_result    : Dict  — output of parse_layout()
    force_reprocess  : bool  — ignore cache and rerun
    batch_size       : int   — images per BLIP-2 forward pass (reduce if OOM)
    max_new_tokens   : int   — max caption length
    dpi_scale        : float — pixmap scale for figure crops (2.0 = ~150 DPI)

    Returns
    -------
    Pipeline contract dict with metadata.figures list.
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
            logger.info("[figure_understander] Cache hit for %s.", paper_id)
            return {
                "input_path":  input_path,
                "output_path": str(out_path),
                "status":      "success",
                "cached":      True,
                "metadata":    cached,
            }
        except Exception as e:
            logger.warning("[figure_understander] Cache corrupt (%s) — reprocessing.", e)

    # ── Extract figure elements from layout ──────────────────────────────────
    elements = layout_result.get("metadata", {}).get(
        "elements", layout_result.get("elements", [])
    )
    figure_elements = [e for e in elements if e.get("type") == "figure"]

    if not figure_elements:
        logger.warning("[figure_understander] No figure elements found for %s.", paper_id)
        result_meta = {
            "paper_id":          paper_id,
            "total_figures":     0,
            "processed_figures": 0,
            "failed_figures":    0,
            "device_used":       "none",
            "model_used":        "none",
            "figures":           [],
        }
        out_path.write_text(json.dumps(result_meta, indent=2))
        return {
            "input_path":  input_path,
            "output_path": str(out_path),
            "status":      "success",
            "metadata":    result_meta,
        }

    # ── Crop figures from PDF ────────────────────────────────────────────────
    pdf_path = Path(input_path)
    t0 = time.time()

    crops = _extract_figure_crops(
        pdf_path       = pdf_path,
        figure_elements= figure_elements,
        figures_dir    = base_paths["figures"],
        paper_id       = paper_id,
        dpi_scale      = dpi_scale,
    )

    if not crops:
        return _error_response(input_path, f"All figure crops failed for {paper_id}.", str(out_path))

    # ── Load model ───────────────────────────────────────────────────────────
    device    = _detect_device()
    cache_dir = _get_model_cache_dir(base_paths)

    try:
        processor, model, model_id = _load_blip2(device, cache_dir)
    except Exception as e:
        return _error_response(input_path, f"BLIP-2 load failed: {e}", str(out_path))

    # ── Caption ──────────────────────────────────────────────────────────────
    captioned = _caption_batch(
        crop_dicts    = crops,
        processor     = processor,
        model         = model,
        device        = device,
        batch_size    = batch_size,
        max_new_tokens= max_new_tokens,
    )

    # ── Build output records ─────────────────────────────────────────────────
    figures_out = []
    failed      = 0
    for item in captioned:
        caption = item.get("caption", "")
        if caption.startswith("[captioning failed"):
            failed += 1
        figures_out.append({
            "figure_id":  item["figure_id"],
            "element_id": item.get("element_id", ""),
            "caption":    caption,
            "page":       item.get("page", -1),
            "bbox":       item.get("bbox", []),
            "crop_path":  item.get("crop_path", ""),
            "source":     item.get("source", "layout_detection"),
        })

    elapsed = time.time() - t0
    logger.info(
        "[figure_understander] %s: %d figures in %.1fs (%.1fs/fig).",
        paper_id, len(figures_out), elapsed,
        elapsed / max(len(figures_out), 1),
    )

    result_meta = {
        "paper_id":          paper_id,
        "total_figures":     len(figure_elements),
        "processed_figures": len(figures_out),
        "failed_figures":    failed,
        "device_used":       device,
        "model_used":        "Salesforce/blip2-opt-2.7b",
        "elapsed_seconds":   round(elapsed, 2),
        "figures":           figures_out,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(result_meta, indent=2))
    logger.info("[figure_understander] Saved → %s", out_path)

    return {
        "input_path":  input_path,
        "output_path": str(out_path),
        "status":      "success",
        "metadata":    result_meta,
    }


def describe_figures_batch(
    paper_list: List[Tuple[Dict, Dict]],
    force_reprocess: bool = False,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """
    Process multiple papers sharing a single loaded BLIP-2 model.

    Parameters
    ----------
    paper_list : list of (ingestion_result, layout_result) tuples

    Returns
    -------
    List of pipeline contract dicts, one per paper.
    """
    results = []
    for ing, lay in paper_list:
        r = describe_figures(
            ingestion_result=ing,
            layout_result=lay,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Quick smoke test — uses first PDF in data/raw_pdfs/
    import glob

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.ingestion.pdf_loader   import load_pdf
    from src.layout.layout_parser   import parse_layout

    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw_pdfs/")
        sys.exit(1)

    test_pdf = pdfs[0]
    print(f"Testing on: {test_pdf}")

    ing = load_pdf(test_pdf)
    if ing.get("status") not in ("success", "cached"):
        print(f"Ingestion failed: {ing.get('message')}")
        sys.exit(1)

    lay = parse_layout(ing)
    if lay.get("status") not in ("success", "cached"):
        print(f"Layout failed: {lay.get('message')}")
        sys.exit(1)

    result = describe_figures(ing, lay, force_reprocess=True, batch_size=2)

    print(f"\nStatus:  {result['status']}")
    print(f"Output:  {result['output_path']}")
    meta = result.get("metadata", {})
    print(f"Figures: {meta.get('processed_figures')} / {meta.get('total_figures')}")
    print(f"Device:  {meta.get('device_used')}")
    print(f"Time:    {meta.get('elapsed_seconds')}s")
    print(f"\nFirst 3 captions:")
    for fig in meta.get("figures", [])[:3]:
        print(f"  [{fig['figure_id']}] p{fig['page']}: {fig['caption']}")

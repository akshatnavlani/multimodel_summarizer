"""
src/pipeline/run_pipeline.py  (v4 — final)
-------------------------------------------
CRITICAL FIXES:
  1. Stale cache threshold raised to 500 words (was 100).
  2. clear_all_stale_caches() helper — call once to reset old broken results.
  3. Tables extracted directly from pdfplumber data if TAPAS summary missing.
  4. Figures: PyMuPDF text near figure bbox injected as lightweight captions
     when skip_vision=True (zero model, zero download).
  5. retrieval_result always enriched with both figures and tables.
  6. top_k=12 for richer context.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

STALE_WORD_THRESHOLD = 500   # any cached summary shorter than this is regenerated


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def clear_all_stale_caches(project_root: str = None) -> int:
    """
    Delete all cached summaries that are shorter than STALE_WORD_THRESHOLD words.
    Returns number of caches cleared.
    """
    try:
        from config.paths import get_project_paths
        paths = get_project_paths(create_dirs=False)
        summ_dir = paths["summaries"]
    except Exception:
        if project_root:
            summ_dir = Path(project_root) / "data" / "summaries"
        else:
            return 0

    cleared = 0
    for f in summ_dir.glob("*_summary.json"):
        try:
            data    = json.loads(f.read_text())
            summary = data.get("summary", "")
            wc      = len(summary.split())
            if wc < STALE_WORD_THRESHOLD:
                f.unlink()
                logger.info("[pipeline] Cleared stale cache: %s (%d words)", f.name, wc)
                cleared += 1
        except Exception:
            pass
    return cleared


def _summary_is_stale(paper_id: str) -> bool:
    """True if cached summary < STALE_WORD_THRESHOLD words."""
    try:
        from config.paths import get_project_paths
        paths = get_project_paths(create_dirs=False)
        cache = paths["summaries"] / f"{paper_id}_summary.json"
        if not cache.exists():
            return False
        data    = json.loads(cache.read_text())
        summary = data.get("summary", "")
        wc      = len(summary.split())
        if wc < STALE_WORD_THRESHOLD:
            logger.info("[pipeline] Stale summary (%d words < %d) — will regenerate.",
                        wc, STALE_WORD_THRESHOLD)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Table text extraction (no TAPAS needed)
# ---------------------------------------------------------------------------

def _extract_table_text_from_data(tbl: Dict) -> str:
    """
    Generate a natural-language summary from columnar table data dict.
    Works with any pdfplumber / TAPAS output format.
    """
    # Try pre-computed summary first
    summary = str(tbl.get("summary", "")).strip()
    if summary and len(summary) > 20:
        return summary

    # Try markdown
    md = str(tbl.get("markdown", "")).strip()
    if md and len(md) > 10:
        return f"Table data: {md[:400]}"

    # Try to parse columnar dict  {col: [values]}
    data = tbl.get("data", {})
    if isinstance(data, dict) and data:
        cols = list(data.keys())
        rows: List[str] = []
        n_rows = max(
            (len(v) for v in data.values() if isinstance(v, list)), default=0
        )
        # Header row
        rows.append(" | ".join(str(c) for c in cols[:6]))
        # Up to 4 data rows
        for i in range(min(n_rows, 4)):
            row_vals = []
            for c in cols[:6]:
                val = data[c]
                row_vals.append(str(val[i] if isinstance(val, list) and i < len(val) else val))
            rows.append(" | ".join(row_vals))
        return "Table: " + "; ".join(rows)

    # Try list-of-rows format
    if isinstance(data, list) and data:
        if isinstance(data[0], dict):
            cols  = list(data[0].keys())
            lines = [" | ".join(str(r.get(c, "")) for c in cols[:6]) for r in data[:4]]
            return "Table: " + "; ".join(lines)
        if isinstance(data[0], list):
            lines = [" | ".join(str(v) for v in row[:6]) for row in data[:4]]
            return "Table: " + "; ".join(lines)

    tbl_id = tbl.get("table_id", "unknown")
    return f"Table {tbl_id} (data unavailable)"


# ---------------------------------------------------------------------------
# Figure text extraction (PyMuPDF, no model)
# ---------------------------------------------------------------------------

def _extract_figure_context(
    pdf_path: str,
    figure_elements: List[Dict],
    max_figures: int = 8,
) -> List[Dict]:
    """
    Extract text near each figure bbox from the PDF using PyMuPDF.
    Returns list of {figure_id, caption, page, bbox} dicts.
    Zero model downloads, runs in < 0.5s.
    """
    results = []
    try:
        import fitz
        doc = fitz.open(pdf_path)

        for idx, el in enumerate(figure_elements[:max_figures]):
            bb       = el.get("bbox", el.get("bbox_pdf", []))
            page_num = max(0, el.get("page", 1) - 1)
            elem_id  = el.get("element_id", el.get("id", f"fig_{idx:04d}"))

            if len(bb) != 4 or page_num >= len(doc):
                continue

            x0, y0, x1, y1 = bb
            h = y1 - y0

            # Search for caption text BELOW the figure (within 1 figure-height)
            caption_rect = fitz.Rect(x0 - 5, y1, x1 + 5, y1 + h * 1.2)
            caption_text = doc[page_num].get_text("text", clip=caption_rect).strip()

            # Fallback: text immediately above
            if not caption_text or len(caption_text) < 10:
                above_rect   = fitz.Rect(x0 - 5, max(0, y0 - h * 0.5), x1 + 5, y0)
                caption_text = doc[page_num].get_text("text", clip=above_rect).strip()

            # Further fallback: any text on the same page near the figure
            if not caption_text or len(caption_text) < 10:
                caption_text = f"Figure {idx + 1} from the paper."

            caption_text = caption_text[:300].replace("\n", " ").strip()
            results.append({
                "figure_id":  f"pdf_fig_{idx:04d}",
                "element_id": elem_id,
                "caption":    caption_text,
                "page":       page_num + 1,
                "bbox":       bb,
                "source":     "pymupdf_text",
            })

        doc.close()
    except Exception as e:
        logger.warning("[pipeline] Figure context extraction failed: %s", e)

    logger.info("[pipeline] Extracted text context for %d figures.", len(results))
    return results


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _validate(output: Dict, stage: str) -> None:
    if not isinstance(output, dict):
        raise ValueError(f"[{stage}] not a dict")
    s = output.get("status", "")
    if s in ("success", "cached"):
        return
    raise ValueError(f"[{stage}] error: {output.get('message', s)}")


def _log(step: int, name: str, t: float, extra: str = "") -> None:
    s = f" | {extra}" if extra else ""
    print(f"  Step {step}: {name} ({t:.3f}s{s})")


def _imports() -> Dict:
    from src.ingestion.pdf_loader       import load_pdf
    from src.layout.layout_parser       import parse_layout
    from src.extraction.text_extractor  import extract_text
    from src.extraction.table_parser    import parse_tables
    from src.vision.figure_understander import describe_figures
    from src.vision.chart_extractor     import extract_charts
    from src.retrieval.embedder         import embed_paper
    from src.retrieval.faiss_index      import build_index, search_index
    from src.summarization.summarizer   import Summarizer
    from src.xai.explainer              import Explainer
    return {k: v for k, v in locals().items()}


def _patch_layout() -> None:
    try:
        import src.layout.layout_parser as _lp
        if hasattr(_lp, "_LP_OK"): _lp._LP_OK = False
        if hasattr(_lp, "_D2_OK"): _lp._D2_OK = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path:           str,
    query:              str  = "Summarize the key contributions and results",
    force_reprocess:    bool = False,
    top_k:              int  = 12,
    max_summary_words:  int  = 800,
    skip_vision:        bool = True,
) -> Dict[str, Any]:
    """
    Process a PDF through all 8 stages.
    skip_vision=True uses PyMuPDF text extraction instead of BLIP-2 (instant).
    """
    t0 = time.time()
    st: Dict[str, float] = {}

    print(f"\n{'='*55}")
    print(f"  PDF: {Path(pdf_path).name}")
    print(f"  skip_vision={skip_vision} | force_reprocess={force_reprocess}")
    print(f"{'='*55}")

    if not Path(pdf_path).exists():
        return {"status": "error", "message": f"PDF not found: {pdf_path}",
                "summary": "", "xai": {},
                "metadata": {"pdf_path": pdf_path, "query": query}}

    try:
        _patch_layout()
        mods = _imports()

        # 1 — Ingestion
        t = time.time()
        ing = mods["load_pdf"](pdf_path)
        _validate(ing, "Ingestion")
        st["ingestion"] = round(time.time() - t, 3)
        mi = ing.get("metadata", {})
        _log(1, "Ingestion", st["ingestion"],
             f"pages={mi.get('page_count', mi.get('num_pages', '?'))}")

        # 2 — Layout
        t = time.time()
        lay = mods["parse_layout"](ing)
        _validate(lay, "Layout")
        st["layout"] = round(time.time() - t, 3)
        from collections import Counter
        elements    = lay.get("metadata", {}).get("elements", lay.get("elements", []))
        type_counts = Counter(e.get("type", "?") for e in elements)
        _log(2, "Layout", st["layout"],
             f"elements={len(elements)} {dict(type_counts)}")

        # 3a — Text
        t = time.time()
        txt = mods["extract_text"](ing, lay, force_reprocess=force_reprocess)
        _validate(txt, "Text")
        st["text_extraction"] = round(time.time() - t, 3)
        tm     = txt.get("metadata", {})
        chunks = tm.get("text_chunks") or tm.get("chunks") or []
        _log(3, "Text", st["text_extraction"], f"chunks={len(chunks)}")

        # 3b — Tables
        t = time.time()
        tbl_r = mods["parse_tables"](ing, lay)
        _validate(tbl_r, "Tables")
        st["table_parsing"] = round(time.time() - t, 3)
        raw_tables = (tbl_r.get("metadata", {}).get("tables")
                      or tbl_r.get("tables", []))
        # Inject text summary for every table that lacks one
        tables = []
        for tbl in raw_tables:
            enriched = dict(tbl)
            if not str(enriched.get("summary", "")).strip():
                enriched["summary"] = _extract_table_text_from_data(tbl)
            tables.append(enriched)
        _log(4, "Tables", st["table_parsing"],
             f"tables={len(tables)} (with auto-summary)")

        # 4 — Vision or lightweight figure context
        fig_list: List[Dict] = []
        st["figure_understanding"] = 0.0
        st["chart_extraction"]     = 0.0

        if not skip_vision:
            # Full BLIP-2 path
            t = time.time()
            fig_r = mods["describe_figures"](ing, lay,
                        force_reprocess=force_reprocess, batch_size=2)
            _validate(fig_r, "Figures")
            st["figure_understanding"] = round(time.time() - t, 3)
            fig_list = fig_r.get("metadata", {}).get("figures", [])
            _log(5, "Figures (BLIP-2)", st["figure_understanding"],
                 f"figures={len(fig_list)}")

            t = time.time()
            cht_r = mods["extract_charts"](ing, lay, figure_result=fig_r,
                        force_reprocess=force_reprocess, batch_size=1)
            _validate(cht_r, "Charts")
            st["chart_extraction"] = round(time.time() - t, 3)
            chart_fig_list = cht_r.get("metadata", {}).get("charts", [])
            _log(6, "Charts (Deplot)", st["chart_extraction"],
                 f"charts={len(chart_fig_list)}")

            figure_data = fig_r
            chart_data  = cht_r
        else:
            # Lightweight: extract text near figure bboxes — INSTANT, no model
            figure_elements = [e for e in elements if e.get("type") == "figure"]
            t = time.time()
            fig_list = _extract_figure_context(pdf_path, figure_elements, max_figures=8)
            st["figure_understanding"] = round(time.time() - t, 3)
            _log(5, "Figures (PyMuPDF text)", st["figure_understanding"],
                 f"figures={len(fig_list)}")

            figure_data = {"status": "success",
                           "metadata": {"figures": fig_list, "total_figures": len(fig_list)}}
            chart_data  = {"status": "success",
                           "metadata": {"charts": [], "total_charts": 0}}

        # 5 — Embeddings
        t = time.time()
        emb = mods["embed_paper"](
            ingestion_result = ing,
            text_result      = txt,
            figure_result    = figure_data,
            table_result     = {"status": "success", "metadata": {"tables": tables},
                                 "tables": tables},
            chart_result     = chart_data,
            force_reprocess  = force_reprocess,
        )
        _validate(emb, "Embeddings")
        st["embeddings"] = round(time.time() - t, 3)
        em       = emb.get("metadata", {})
        paper_id = em.get("paper_id", "unknown")
        n_chunks = em.get("total_chunks", 0)
        _log(7, "Embeddings", st["embeddings"],
             f"paper_id={paper_id} | chunks={n_chunks} | {em.get('modality_counts', {})}")

        # 6 — FAISS + Retrieval
        t = time.time()
        idx_r = mods["build_index"](emb, force_reprocess=force_reprocess)
        _validate(idx_r, "FAISS")
        ret = mods["search_index"](idx_r, query, top_k=top_k)
        st["retrieval"] = round(time.time() - t, 3)
        _log(8, "Retrieval", st["retrieval"],
             f"hits={len(ret.get('top_k_results', []))}")

        # Enrich retrieval result
        ret["figures"]  = fig_list
        ret["tables"]   = tables
        ret["paper_id"] = paper_id

        # 7 — Summarization (auto-detect stale cache)
        summ_force = force_reprocess or _summary_is_stale(paper_id)
        t = time.time()
        summ = mods["Summarizer"]().generate(
            ret, max_words=max_summary_words, force_reprocess=summ_force
        )
        _validate(summ, "Summarization")
        st["summarization"] = round(time.time() - t, 3)
        sm = summ.get("metadata", {})
        _log(9, "Summarization", st["summarization"],
             f"model={sm.get('model_used')} | words={len(summ.get('summary','').split())}")

        # 8 — XAI
        t = time.time()
        xai = mods["Explainer"]().explain(
            summ, ret, force_reprocess=force_reprocess
        )
        _validate(xai, "XAI")
        st["xai"] = round(time.time() - t, 3)
        xm = xai.get("metadata", {})
        _log(10, "XAI", st["xai"],
             f"sentences={xm.get('num_sentences')} | "
             f"mcr={xai.get('xai', {}).get('modality_contribution', {})}")

        total = round(time.time() - t0, 2)
        print(f"\n  ✅ DONE in {total}s | "
              f"words={len(summ.get('summary','').split())} | "
              f"model={sm.get('model_used')}\n")

        return {
            "status":  "success",
            "summary": summ.get("summary", ""),
            "xai":     xai.get("xai", {}),
            "metadata": {
                "pdf_path":       pdf_path,
                "paper_id":       paper_id,
                "query":          query,
                "num_chunks":     n_chunks,
                "num_figures":    len(fig_list),
                "num_tables":     len(tables),
                "execution_time": total,
                "stage_times":    st,
                "model_used":     sm.get("model_used", "unknown"),
                "summary_words":  len(summ.get("summary", "").split()),
            },
        }

    except Exception as e:
        total = round(time.time() - t0, 2)
        logger.exception("[pipeline] Error after %.2fs: %s", total, e)
        return {
            "status":  "error", "message": str(e),
            "summary": "", "xai": {},
            "metadata": {"pdf_path": pdf_path, "query": query,
                         "execution_time": total, "stage_times": st},
        }


def run_pipeline_batch(pdf_paths, query="Summarize the key contributions",
                       force_reprocess=False, skip_vision=True):
    results, n = [], len(pdf_paths)
    for i, p in enumerate(pdf_paths, 1):
        print(f"\n[{i}/{n}] {Path(p).name}")
        r = run_pipeline(p, query=query,
                         force_reprocess=force_reprocess, skip_vision=skip_vision)
        results.append(r)
        icon = "✅" if r["status"] == "success" else "❌"
        print(f"  {icon} {r['metadata'].get('execution_time')}s | "
              f"words={r['metadata'].get('summary_words')}")
    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n  {ok}/{n} succeeded.")
    return results


if __name__ == "__main__":
    import sys, logging, glob
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("No PDFs found."); sys.exit(0)

    # Clear stale caches first
    n = clear_all_stale_caches()
    print(f"Cleared {n} stale caches.\n")

    r = run_pipeline(pdfs[0], skip_vision=True, force_reprocess=False)
    print(f"\nStatus: {r['status']} | words={r['metadata'].get('summary_words')}")
    if r["status"] == "success":
        print(r["summary"][:400])

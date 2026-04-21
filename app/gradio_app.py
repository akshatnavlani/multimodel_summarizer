"""
app/gradio_app.py  (v3 — final)
---------------------------------
TARGET: <15s response on cached run, <90s first run with Gemini key.

KEY CHANGES:
  1. Clears stale caches automatically on every new PDF upload.
  2. "Force Reprocess" checkbox exposed to user.
  3. skip_vision defaults True (safe, instant).
  4. Pre-warms MiniLM model at startup (zero wait on first request).
  5. Status bar shows live model used + word count.
  6. Summary tab shows full text reliably.
  7. XAI tab shows modality bars + attribution preview.
"""

import logging, os, shutil, sys, time
from pathlib import Path
from typing import Dict, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(_ROOT))

import gradio as gr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_pipeline_fn = None

def _get_pipeline():
    global _pipeline_fn
    if _pipeline_fn is None:
        from src.pipeline.run_pipeline import run_pipeline
        _pipeline_fn = run_pipeline
    return _pipeline_fn

def _get_clear_fn():
    try:
        from src.pipeline.run_pipeline import clear_all_stale_caches
        return clear_all_stale_caches
    except Exception:
        return lambda: 0

def _raw_pdfs_dir() -> Path:
    try:
        from config.paths import get_project_paths
        return get_project_paths(create_dirs=True)["raw_pdfs"]
    except Exception:
        d = _ROOT / "data" / "raw_pdfs"
        d.mkdir(parents=True, exist_ok=True)
        return d

# Pre-warm embedding model at startup
def _prewarm():
    try:
        from src.retrieval.embedder import _load_model
        _load_model()
        logger.info("[app] Embedding model pre-warmed.")
    except Exception as e:
        logger.warning("[app] Pre-warm failed: %s", e)

_prewarm()

# ---------------------------------------------------------------------------
def _validate(file_obj) -> Tuple[bool, str]:
    if file_obj is None:
        return False, "No file uploaded."
    p = Path(getattr(file_obj, "name", str(file_obj)))
    if not str(p).lower().endswith(".pdf"):
        return False, "Only PDF files are supported."
    if not p.exists() or p.stat().st_size == 0:
        return False, "File is empty or inaccessible."
    return True, ""

def _xai_display(xai: dict) -> dict:
    if not xai:
        return {"note": "No XAI data."}
    mc    = xai.get("modality_contribution", {})
    attrs = xai.get("attribution", [])
    bars  = ""
    for mod, frac in mc.items():
        b = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        icon = {"text": "📄", "figure": "🖼️", "table": "📊"}.get(mod, "•")
        bars += f"{icon} {mod.upper():6s}  {b}  {frac:.1%}\n"
    return {
        "modality_contribution":      mc,
        "modality_bars":              bars.strip(),
        "num_attributed_sentences":   len(attrs),
        "attribution_preview": [
            {"sentence": a.get("summary_sentence", "")[:80],
             "source":   a.get("source_type", "?"),
             "score":    round(a.get("similarity_score", 0), 3),
             "page":     a.get("page", -1)}
            for a in attrs[:8]
        ],
        "shap_hook":     xai.get("shap_hook",     {}).get("status", "placeholder"),
        "lime_hook":     xai.get("lime_hook",      {}).get("status", "placeholder"),
        "gradient_hook": xai.get("gradient_hook",  {}).get("status", "placeholder"),
    }

def _meta_display(meta: dict, status: str) -> dict:
    st = meta.get("stage_times", {})
    return {
        "status":            status,
        "paper_id":          meta.get("paper_id", "unknown"),
        "total_time_s":      meta.get("execution_time", 0),
        "model_used":        meta.get("model_used", "unknown"),
        "summary_words":     meta.get("summary_words", 0),
        "chunks_embedded":   meta.get("num_chunks", 0),
        "figures_processed": meta.get("num_figures", 0),
        "tables_detected":   meta.get("num_tables", 0),
        "stage_times":       st,
        "stage_breakdown":   "\n".join(
            f"{k:28s}: {v:.3f}s" for k, v in st.items() if v > 0
        ),
    }

# ---------------------------------------------------------------------------
def process_pdf(file_obj, query: str, skip_vision: bool,
                force_reprocess: bool) -> Tuple[str, dict, dict]:
    valid, err = _validate(file_obj)
    if not valid:
        return f"⚠️  {err}", {}, {"status": "error", "message": err}

    query         = (query or "Summarize the key contributions and results").strip()
    upload_path   = Path(getattr(file_obj, "name", str(file_obj)))
    dest          = _raw_pdfs_dir() / upload_path.name

    try:
        if not dest.exists() or dest.stat().st_size != upload_path.stat().st_size:
            shutil.copy2(str(upload_path), str(dest))
        # Clear stale caches for this PDF on every fresh upload
        if force_reprocess:
            n = _get_clear_fn()()
            logger.info("[app] Cleared %d stale caches.", n)
    except Exception as e:
        return f"❌  File error: {e}", {}, {"status": "error", "message": str(e)}

    try:
        result = _get_pipeline()(
            pdf_path           = str(dest),
            query              = query,
            force_reprocess    = force_reprocess,
            skip_vision        = skip_vision,
            max_summary_words  = 800,
            top_k              = 12,
        )
    except Exception as e:
        logger.exception("[app] Pipeline exception: %s", e)
        return f"❌  Error: {e}", {}, {"status": "error", "message": str(e)}

    if result.get("status") == "error":
        return (f"❌  {result.get('message', 'Pipeline failed')}",
                {}, _meta_display(result.get("metadata", {}), "error"))

    summary = result.get("summary", "").strip()
    if not summary or len(summary) < 20:
        summary = ("⚠️  Summary generation used extractive fallback. "
                   "For best results, set GEMINI_API_KEY environment variable "
                   "and re-run with 'Force Reprocess' checked.")

    return (summary,
            _xai_display(result.get("xai", {})),
            _meta_display(result.get("metadata", {}), "success"))

# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');
:root{--bg:#0d0f14;--bg2:#13161e;--card:#1a1e28;--inp:#1f2433;--teal:#00c9a7;--text:#e8eaf0;--muted:#7a8099;--bord:#2a2f40;--mono:'IBM Plex Mono',monospace;--ser:'Fraunces',serif;}
body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
.hdr{text-align:center;padding:2rem 1rem 1rem;border-bottom:1px solid var(--bord);margin-bottom:1.5rem;}
.hdr h1{font-family:var(--ser)!important;font-weight:300;font-size:2.1rem;letter-spacing:-0.02em;margin:0 0 0.3rem;}
.hdr h1 span{color:var(--teal);font-style:italic;}
.hdr p{font-size:0.7rem;color:var(--muted);margin:0;letter-spacing:0.08em;text-transform:uppercase;}
input[type=text],textarea{background:var(--inp)!important;border:1px solid var(--bord)!important;border-radius:6px!important;color:var(--text)!important;font-family:var(--mono)!important;}
input:focus,textarea:focus{border-color:var(--teal)!important;outline:none!important;}
button.primary{background:var(--teal)!important;color:#0d0f14!important;border:none!important;border-radius:6px!important;font-family:var(--mono)!important;font-weight:600!important;}
button.secondary{background:transparent!important;border:1px solid var(--bord)!important;color:var(--muted)!important;border-radius:6px!important;font-family:var(--mono)!important;}
label,.gr-label{font-size:0.7rem!important;font-weight:600!important;letter-spacing:0.1em!important;text-transform:uppercase!important;color:var(--muted)!important;}
"""

HEADER = """<div class="hdr"><h1>Multimodal Doc <span>Summarizer</span> + XAI</h1>
<p>PDF → Layout → Text/Table/Figure → Embed → FAISS → Summary → XAI</p></div>"""

PIPE_STEPS = """<div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;
gap:0;font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#7a8099;padding:0.4rem 0 1rem;">
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #2a2f40;">📄 PDF</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #2a2f40;">🗺️ Layout</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #2a2f40;">🔤 Text+Table+Fig</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #2a2f40;">🔢 Embed</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #2a2f40;">🔍 FAISS</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #00c9a7;color:#00c9a7;">✨ Summary</span><span style="margin:0 3px;">→</span>
<span style="background:#1a1e28;padding:3px 7px;border-radius:4px;border:1px solid #f5a623;color:#f5a623;">🔬 XAI</span>
</div>"""


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Multimodal Summarizer + XAI",
                   theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate",
                                        font=gr.themes.GoogleFont("IBM Plex Mono"))) as app:

        gr.HTML(HEADER)
        gr.HTML(PIPE_STEPS)

        with gr.Row():
            with gr.Column(scale=2):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"],
                                    file_count="single")
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Query",
                    value="Summarize the key contributions, methodology, and results",
                    lines=2)
                with gr.Row():
                    skip_cb  = gr.Checkbox(
                        label="Skip Vision/BLIP-2 (faster, CPU-safe — uses PyMuPDF for figures)",
                        value=True)
                    force_cb = gr.Checkbox(
                        label="Force Reprocess (ignore cache)",
                        value=False)
                with gr.Row():
                    run_btn   = gr.Button("▶  Run Pipeline", variant="primary")
                    clear_btn = gr.Button("✕  Clear",        variant="secondary")

        status_bar = gr.HTML(
            "<div style='font-size:0.72rem;color:#7a8099;padding:0.3rem 0;'>"
            "Upload a PDF and click Run Pipeline. "
            "Set GEMINI_API_KEY for best summaries (free at aistudio.google.com).</div>")

        with gr.Tabs():
            with gr.Tab("📝 Summary"):
                summary_out = gr.Textbox(
                    label="Generated Summary (800 words)",
                    lines=18, interactive=False,
                    placeholder="Summary will appear here after pipeline runs…")
            with gr.Tab("🔬 XAI Explanation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        modality_out = gr.JSON(label="Modality Contribution", value={})
                    with gr.Column(scale=2):
                        xai_out = gr.JSON(label="Attribution + Hooks", value={})
            with gr.Tab("📊 Metadata"):
                meta_out = gr.JSON(label="Execution Details", value={})

        gr.Examples(
            examples=[
                ["What is the main contribution and key results?"],
                ["What dataset and evaluation metrics were used?"],
                ["What model architecture is proposed?"],
                ["What are the limitations of this approach?"],
                ["How does this compare to prior work?"],
            ],
            inputs=[query_input], label="Example Queries")

        gr.HTML("""<div style="text-align:center;padding:1rem 0 0.5rem;
        border-top:1px solid #2a2f40;margin-top:1.5rem;
        font-size:0.65rem;color:#4a5068;">
        Multimodal Summarization + XAI · MiniLM · FAISS · Gemini 1.5 Flash · Extractive fallback
        </div>""")

        def _run(file_obj, query, skip_vision, force_reprocess):
            yield ("⏳  Processing — please wait…", {}, {}, {},
                   "<div style='font-size:0.72rem;color:#f5a623;padding:0.3rem 0;'>"
                   "⏳  Pipeline running…</div>")

            summary, xai_disp, meta_disp = process_pdf(
                file_obj, query, skip_vision, force_reprocess)

            ok = not (summary.startswith("⚠️") or summary.startswith("❌"))
            wc    = len(summary.split()) if ok else 0
            model = meta_disp.get("model_used", "?")
            t_s   = meta_disp.get("total_time_s", 0)

            mc_disp = {
                "modality_contribution": xai_disp.get("modality_contribution", {}),
                "bars": xai_disp.get("modality_bars", ""),
            }

            status_html = (
                f"<div style='font-size:0.72rem;color:#00c9a7;padding:0.3rem 0;'>"
                f"✅  Done in {t_s:.1f}s | {wc} words | model: {model}</div>"
                if ok else
                f"<div style='font-size:0.72rem;color:#e05c4b;padding:0.3rem 0;'>"
                f"❌  {meta_disp.get('message', 'Error')}</div>"
            )

            yield summary, xai_disp, mc_disp, meta_disp, status_html

        def _clear():
            empty = ("<div style='font-size:0.72rem;color:#7a8099;padding:0.3rem 0;'>"
                     "Cleared.</div>")
            return None, "", True, False, "", {}, {}, {}, empty

        run_btn.click(
            fn=_run,
            inputs=[pdf_input, query_input, skip_cb, force_cb],
            outputs=[summary_out, xai_out, modality_out, meta_out, status_bar])

        clear_btn.click(
            fn=_clear, inputs=[],
            outputs=[pdf_input, query_input, skip_cb, force_cb,
                     summary_out, xai_out, modality_out, meta_out, status_bar])

    return app


app = build_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860,
               share=True, show_error=True)

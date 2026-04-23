"""
app/gradio_app.py  (v4 — FIXED)
---------------------------------
FIXES APPLIED:
  1. Removed dependency on hardcoded PROJECT_ROOT env var — uses auto-detection.
  2. Startup environment check: validates .env, GEMINI_API_KEY presence, __init__ files.
  3. GEMINI_MAX_OUTPUT_TOKENS raised to 4096 (was 512 — truncated summaries).
  4. Status bar now clearly shows: 'cached' vs 'fresh run', model name, word count.
  5. Added environment check panel in UI (collapsible Accordion).
  6. Corrected launch() call to always use server_name='0.0.0.0' so the app is
     reachable at http://localhost:7860 regardless of how the script is invoked.
  7. Increased PIPELINE_TIMEOUT to 300s (was 180s — insufficient for first run).
  8. Added clear warning when Gemini key is missing (ROUGE scores will be near-zero).

HOW TO RUN:
  1. Fix __init__ space bug:
       Linux/macOS: mv 'src/ingestion/__init__ .py' src/ingestion/__init__.py
                    mv 'src/layout/__init__ .py'    src/layout/__init__.py
    Windows PS:  Rename-Item 'src\\ingestion\\__init__ .py' '__init__.py'
              Rename-Item 'src\\layout\\__init__ .py'    '__init__.py'
  2. Fix your .env: remove PROJECT_ROOT / TRANSFORMERS_CACHE / HF_HOME Windows paths.
     Add:  GEMINI_API_KEY=your_key_here
  3. From the project root:  python app/gradio_app.py
  4. Open browser at:        http://localhost:7860
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# ── Project root: always resolved relative to this file ──────────────────────
# Never depend on PROJECT_ROOT env var — it breaks on other machines.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_ROOT / ".env"), override=True)
except Exception:
    pass

# Only set PROJECT_ROOT if it is missing or wrong (e.g. pointing to Windows path)
_env_root = os.environ.get("PROJECT_ROOT", "")
if not _env_root or not Path(_env_root).exists():
    os.environ["PROJECT_ROOT"] = str(_ROOT)

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Raise token limit at boot (settings singleton not yet created) ────────────
os.environ.setdefault("GEMINI_MAX_OUTPUT_TOKENS", "4096")
os.environ.setdefault("PIPELINE_TIMEOUT_SECONDS", "300")

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


# ── Environment diagnostics ───────────────────────────────────────────────────

def _run_env_check() -> Dict:
    """Check environment and return a dict of status items."""
    issues = []
    warnings = []
    ok = []

    # 1. __init__ files with space bug
    for pkg in ["ingestion", "layout"]:
        bad = _ROOT / "src" / pkg / "__init__ .py"
        good = _ROOT / "src" / pkg / "__init__.py"
        if bad.exists() and not good.exists():
            issues.append(f"src/{pkg}/__init__.py has a trailing space — rename it: "
                          f"mv 'src/{pkg}/__init__ .py' src/{pkg}/__init__.py")
        elif good.exists():
            ok.append(f"src/{pkg}/__init__.py ✓")

    # 2. GEMINI_API_KEY
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        warnings.append(
            "GEMINI_API_KEY is not set. The pipeline will use extractive fallback "
            "(ROUGE scores will be near-zero). Get a free key at aistudio.google.com."
        )
    else:
        ok.append("GEMINI_API_KEY is set ✓")

    # 3. PROJECT_ROOT sanity
    pr = os.environ.get("PROJECT_ROOT", "")
    if "\\" in pr and not Path(pr).exists():
        issues.append(
            f"PROJECT_ROOT={pr!r} appears to be a Windows path that does not exist. "
            "Comment it out in .env — auto-detection will handle it."
        )
    else:
        ok.append(f"PROJECT_ROOT={pr!r} ✓")

    # 4. requirements spot-check
    missing_pkgs = []
    for pkg in ["fitz", "pdfplumber", "sentence_transformers", "faiss", "gradio"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        issues.append(f"Missing packages: {missing_pkgs} — run: pip install -r requirements.txt")
    else:
        ok.append("Core packages installed ✓")

    return {"issues": issues, "warnings": warnings, "ok": ok}


def _env_check_html(check: Dict) -> str:
    lines = []
    for msg in check.get("issues", []):
        lines.append(f"<li style='color:#e05c4b;'>🔴 {msg}</li>")
    for msg in check.get("warnings", []):
        lines.append(f"<li style='color:#f5a623;'>🟡 {msg}</li>")
    for msg in check.get("ok", []):
        lines.append(f"<li style='color:#00c9a7;'>✅ {msg}</li>")
    if not lines:
        return "<p style='color:#00c9a7;'>All checks passed.</p>"
    return f"<ul style='font-size:0.72rem;font-family:monospace;padding-left:1rem;'>{''.join(lines)}</ul>"


# ── Pre-warm embedding model ──────────────────────────────────────────────────

def _prewarm():
    """Load MiniLM at startup so the first request doesn't stall."""
    try:
        from src.retrieval.embedder import _load_model
        _load_model()
        logger.info("[app] MiniLM pre-warmed.")
    except Exception as e:
        logger.warning("[app] Pre-warm failed (non-fatal): %s", e)


_prewarm()

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    mc = xai.get("modality_contribution", {})
    attrs = xai.get("attribution", [])
    bars = ""
    for mod, frac in mc.items():
        b = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        icon = {"text": "📄", "figure": "🖼️", "table": "📊"}.get(mod, "•")
        bars += f"{icon} {mod.upper():6s}  {b}  {frac:.1%}\n"
    return {
        "modality_contribution": mc,
        "modality_bars": bars.strip(),
        "num_attributed_sentences": len(attrs),
        "attribution_preview": [
            {
                "sentence": a.get("summary_sentence", "")[:80],
                "source": a.get("source_type", "?"),
                "score": round(a.get("similarity_score", 0), 3),
                "page": a.get("page", -1),
            }
            for a in attrs[:8]
        ],
        "shap_hook": xai.get("shap_hook", {}).get("status", "placeholder"),
        "lime_hook": xai.get("lime_hook", {}).get("status", "placeholder"),
        "gradient_hook": xai.get("gradient_hook", {}).get("status", "placeholder"),
    }


def _meta_display(meta: dict, status: str) -> dict:
    st = meta.get("stage_times", {})
    cached = meta.get("from_cache", False)
    return {
        "status": status,
        "cached": cached,
        "paper_id": meta.get("paper_id", "unknown"),
        "total_time_s": meta.get("execution_time", 0),
        "model_used": meta.get("model_used", "unknown"),
        "summary_words": meta.get("summary_words", 0),
        "chunks_embedded": meta.get("num_chunks", 0),
        "figures_processed": meta.get("num_figures", 0),
        "tables_detected": meta.get("num_tables", 0),
        "llm_error": meta.get("llm_error", ""),
        "gemini_rate_limited": meta.get("gemini_rate_limited", False),
        "gemini_cooldown_remaining_seconds": meta.get("gemini_cooldown_remaining_seconds", 0),
        "vision_warning": meta.get("vision_warning", ""),
        "vision_mode": meta.get("vision_mode", "unknown"),
        "stage_times": st,
        "stage_breakdown": "\n".join(
            f"{k:28s}: {v:.3f}s" for k, v in st.items() if v > 0
        ),
    }


# ── Core processing function ──────────────────────────────────────────────────

def process_pdf(
    file_obj,
    query: str,
    skip_vision: bool,
    force_reprocess: bool,
) -> Tuple[str, dict, dict]:
    valid, err = _validate(file_obj)
    if not valid:
        return f"⚠️  {err}", {}, {"status": "error", "message": err}

    query = (query or "Summarize the key contributions and results").strip()
    upload_path = Path(getattr(file_obj, "name", str(file_obj)))
    dest = _raw_pdfs_dir() / upload_path.name

    try:
        is_new_upload = (
            not dest.exists()
            or dest.stat().st_size != upload_path.stat().st_size
        )
        if is_new_upload:
            shutil.copy2(str(upload_path), str(dest))
        # Always clear stale caches on upload so new PDFs never get stuck
        n = _get_clear_fn()()
        if n > 0:
            logger.info("[app] Cleared %d stale caches.", n)
    except Exception as e:
        return f"❌  File error: {e}", {}, {"status": "error", "message": str(e)}

    try:
        result = _get_pipeline()(
            pdf_path=str(dest),
            query=query,
            force_reprocess=force_reprocess,
            skip_vision=skip_vision,
            max_summary_words=800,
            top_k=12,
        )
    except Exception as e:
        logger.exception("[app] Pipeline exception: %s", e)
        return f"❌  Error: {e}", {}, {"status": "error", "message": str(e)}

    if result.get("status") == "error":
        return (
            f"❌  {result.get('message', 'Pipeline failed')}",
            {},
            _meta_display(result.get("metadata", {}), "error"),
        )

    summary = result.get("summary", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    model_used = result.get("metadata", {}).get("model_used", "")

    if not summary or len(summary) < 20:
        summary = (
            "⚠️  Summary generation used extractive fallback — output quality is low.\n\n"
            "To get a proper structured summary:\n"
            "  1. Get a free Gemini API key at https://aistudio.google.com\n"
            "  2. Add it to your .env:  GEMINI_API_KEY=your_key\n"
            "  3. Re-run with 'Force Reprocess' checked."
        )
    elif "extractive" in model_used and (not gemini_key or gemini_key == "your_gemini_api_key_here"):
        summary = (
            "ℹ️  NOTE: Extractive fallback used (no GEMINI_API_KEY). "
            "ROUGE scores will be near-zero. This is expected behaviour, not a bug.\n\n"
            + summary
        )

    return (
        summary,
        _xai_display(result.get("xai", {})),
        _meta_display(result.get("metadata", {}), "success"),
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

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

HEADER = """<div class="hdr">
<h1>Multimodal Doc <span>Summarizer</span> + XAI</h1>
<p>PDF → Layout → Text/Table/Figure → Embed → FAISS → Summary → XAI</p>
</div>"""

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
    # Run env check once at build time
    env_check = _run_env_check()
    env_html = _env_check_html(env_check)
    has_issues = bool(env_check.get("issues"))
    has_warnings = bool(env_check.get("warnings"))

    with gr.Blocks(
        css=CSS,
        title="Multimodal Summarizer + XAI",
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
    ) as app:

        gr.HTML(HEADER)
        gr.HTML(PIPE_STEPS)

        # Environment check panel — open if there are issues
        with gr.Accordion(
            label="⚙️  Environment Check" + (" ⚠️" if has_issues or has_warnings else " ✅"),
            open=has_issues or has_warnings,
        ):
            gr.HTML(env_html)
            if not os.environ.get("GEMINI_API_KEY") or \
               os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
                gr.HTML(
                    "<div style='background:#1a1e28;border:1px solid #f5a623;border-radius:6px;"
                    "padding:0.7rem;font-size:0.7rem;color:#f5a623;margin-top:0.5rem;'>"
                    "🟡 <strong>No GEMINI_API_KEY found.</strong> The pipeline will use extractive "
                    "fallback — summaries will be raw text concatenations with near-zero ROUGE scores. "
                    "Get a free key at <a href='https://aistudio.google.com' target='_blank' "
                    "style='color:#00c9a7;'>aistudio.google.com</a> and add it to your .env file."
                    "</div>"
                )

        with gr.Row():
            with gr.Column(scale=2):
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    file_count="single",
                )
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Query",
                    value="Analyze this research paper",
                    lines=2,
                )
                with gr.Row():
                    skip_cb = gr.Checkbox(
                        label="Skip Vision/BLIP-2 (faster, CPU-safe — uses PyMuPDF for figures)",
                        value=True,
                    )
                    force_cb = gr.Checkbox(
                        label="Force Reprocess (ignore cache)",
                        value=False,
                    )
                with gr.Row():
                    run_btn = gr.Button("▶  Run Pipeline", variant="primary")
                    clear_btn = gr.Button("✕  Clear", variant="secondary")

        status_bar = gr.HTML(
            "<div style='font-size:0.72rem;color:#7a8099;padding:0.3rem 0;'>"
            "Upload a PDF and click Run Pipeline. "
            "Set GEMINI_API_KEY for best summaries (free at aistudio.google.com).</div>"
        )

        with gr.Tabs():
            with gr.Tab("📝 Summary"):
                summary_out = gr.Textbox(
                    label="Generated Summary (800 words)",
                    lines=18,
                    interactive=False,
                    placeholder="Summary will appear here after pipeline runs…",
                )
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
                ["Analyze this research paper"],
                ["What is the main contribution and what problem does it solve?"],
                ["What are the risks, costs, and future research directions?"],
                ["How does this compare to existing state-of-the-art approaches?"],
                ["What is the potential impact and who would benefit from this work?"],
            ],
            inputs=[query_input],
            label="Example Queries",
        )

        gr.HTML(
            """<div style="text-align:center;padding:1rem 0 0.5rem;
            border-top:1px solid #2a2f40;margin-top:1.5rem;
            font-size:0.65rem;color:#4a5068;">
            Multimodal Summarization + XAI · MiniLM · FAISS · Gemini 1.5 Flash · Extractive fallback
            </div>"""
        )

        # ── Event handlers ────────────────────────────────────────────────

        def _run(file_obj, query, skip_vision, force_reprocess):
            yield (
                "⏳  Processing — please wait…",
                {},
                {},
                {},
                "<div style='font-size:0.72rem;color:#f5a623;padding:0.3rem 0;'>"
                "⏳  Pipeline running…</div>",
            )

            summary, xai_disp, meta_disp = process_pdf(
                file_obj, query, skip_vision, force_reprocess
            )

            ok = meta_disp.get("status") != "error"
            wc = len(summary.split()) if ok else 0
            model = meta_disp.get("model_used", "?")
            t_s = meta_disp.get("total_time_s", 0)
            cached = meta_disp.get("cached", False)
            cache_tag = " [cached]" if cached else " [fresh]"
            cooldown_s = int(meta_disp.get("gemini_cooldown_remaining_seconds", 0) or 0)
            rate_limited = bool(meta_disp.get("gemini_rate_limited", False))
            llm_error = str(meta_disp.get("llm_error", "") or "")
            show_rate_limit_badge = rate_limited or ("429" in llm_error) or ("rate-limit" in llm_error.lower()) or ("quota" in llm_error.lower())
            vision_warning = str(meta_disp.get("vision_warning", "") or "")
            vision_mode = str(meta_disp.get("vision_mode", "") or "")

            mc_disp = {
                "modality_contribution": xai_disp.get("modality_contribution", {}),
                "bars": xai_disp.get("modality_bars", ""),
            }

            if ok:
                rate_limit_badge = ""
                if show_rate_limit_badge:
                    if cooldown_s > 0:
                        rate_limit_badge = (
                            f" <span style='color:#f5a623;'>| 🕒 Gemini rate-limited ({cooldown_s}s left)</span>"
                        )
                    else:
                        rate_limit_badge = " <span style='color:#f5a623;'>| ⚠️ Gemini rate-limited</span>"
                vision_badge = ""
                if vision_warning:
                    vision_badge = " <span style='color:#f5a623;'>| 🖼️ Vision fallback active</span>"
                elif vision_mode == "lightweight":
                    vision_badge = " <span style='color:#7a8099;'>| 🖼️ Vision skipped (lightweight mode)</span>"
                status_html = (
                    f"<div style='font-size:0.72rem;color:#00c9a7;padding:0.3rem 0;'>"
                    f"✅  Done in {t_s:.1f}s{cache_tag} | {wc} words | model: {model}{rate_limit_badge}{vision_badge}</div>"
                )
            else:
                status_html = (
                    f"<div style='font-size:0.72rem;color:#e05c4b;padding:0.3rem 0;'>"
                    f"❌  {meta_disp.get('message', 'Error')}</div>"
                )

            yield summary, xai_disp, mc_disp, meta_disp, status_html

        def _clear():
            empty = (
                "<div style='font-size:0.72rem;color:#7a8099;padding:0.3rem 0;'>"
                "Cleared.</div>"
            )
            return None, "", True, False, "", {}, {}, {}, empty

        run_btn.click(
            fn=_run,
            inputs=[pdf_input, query_input, skip_cb, force_cb],
            outputs=[summary_out, xai_out, modality_out, meta_out, status_bar],
        )

        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[
                pdf_input, query_input, skip_cb, force_cb,
                summary_out, xai_out, modality_out, meta_out, status_bar,
            ],
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
# Always runs the server — whether invoked as a script or imported by a notebook.
app = build_app()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Multimodal Doc Summarizer + XAI")
    print(f"  Project root : {_ROOT}")
    print(f"  Gemini key   : {'SET ✓' if os.environ.get('GEMINI_API_KEY') else 'NOT SET ⚠️'}")
    print(f"  Open browser : http://localhost:7860")
    print(f"{'='*60}\n")

    app.launch(
        server_name="0.0.0.0",  # reachable at localhost:7860
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,            # set True for a public ngrok URL
        show_error=True,
        quiet=False,
    )

"""
config/settings.py
------------------
Global configuration for the Multimodal Document Summarization + XAI pipeline.

All values are overridable via environment variables.
No hardcoded secrets. No hardcoded paths (see paths.py).

Usage:
    from config.settings import Settings
    cfg = Settings()
    print(cfg.BLIP2_MODEL_ID)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: read env var with a typed default
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))

def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, str(default)).lower()
    return raw in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Settings dataclass — one source of truth
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    Centralised, environment-aware configuration.

    All fields read from environment variables first, then fall back to
    sensible free-tier defaults.  Instantiate once and pass around.
    """

    # ------------------------------------------------------------------
    # Project identity
    # ------------------------------------------------------------------
    PROJECT_NAME: str = field(
        default_factory=lambda: _env("PROJECT_NAME", "multimodal-xai-summarizer")
    )

    # ------------------------------------------------------------------
    # Runtime environment  (colab | kaggle | local)
    # ------------------------------------------------------------------
    ENV: str = field(
        default_factory=lambda: _env("RUNTIME_ENV", _detect_env())
    )

    # ------------------------------------------------------------------
    # PDF Ingestion
    # ------------------------------------------------------------------
    PDF_DPI: int = field(
        default_factory=lambda: _env_int("PDF_DPI", 150)
    )
    PDF_MAX_PAGES: int = field(
        default_factory=lambda: _env_int("PDF_MAX_PAGES", 200)
    )
    PDF_MIN_FIGURES: int = field(
        default_factory=lambda: _env_int("PDF_MIN_FIGURES", 3)
    )
    PDF_MIN_TABLES: int = field(
        default_factory=lambda: _env_int("PDF_MIN_TABLES", 2)
    )

    # ------------------------------------------------------------------
    # Layout parsing  (Detectron2 + PubLayNet)
    # ------------------------------------------------------------------
    LAYOUT_MODEL_HUB: str = field(
        default_factory=lambda: _env(
            "LAYOUT_MODEL_HUB", "ds4sd/PubLayNet"
        )
    )
    LAYOUT_SCORE_THRESHOLD: float = field(
        default_factory=lambda: _env_float("LAYOUT_SCORE_THRESHOLD", 0.80)
    )
    LAYOUT_ELEMENT_TYPES: tuple = field(
        default=(
            "Text", "Title", "List", "Table", "Figure"
        )
    )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    OCR_PRIMARY: str = field(
        default_factory=lambda: _env("OCR_PRIMARY", "paddleocr")
    )
    OCR_FALLBACK: str = field(
        default_factory=lambda: _env("OCR_FALLBACK", "tesseract")
    )
    OCR_LANG: str = field(
        default_factory=lambda: _env("OCR_LANG", "en")
    )

    # ------------------------------------------------------------------
    # Vision models
    # ------------------------------------------------------------------
    BLIP2_MODEL_ID: str = field(
        default_factory=lambda: _env(
            "BLIP2_MODEL_ID", "Salesforce/blip2-opt-2.7b"
        )
    )
    BLIP2_LOAD_8BIT: bool = field(
        default_factory=lambda: _env_bool("BLIP2_LOAD_8BIT", True)
    )
    BLIP2_MAX_NEW_TOKENS: int = field(
        default_factory=lambda: _env_int("BLIP2_MAX_NEW_TOKENS", 128)
    )
    BLIP2_BATCH_SIZE: int = field(
        default_factory=lambda: _env_int("BLIP2_BATCH_SIZE", 4)
    )

    DEPLOT_MODEL_ID: str = field(
        default_factory=lambda: _env(
            "DEPLOT_MODEL_ID", "google/deplot"
        )
    )
    DEPLOT_MAX_NEW_TOKENS: int = field(
        default_factory=lambda: _env_int("DEPLOT_MAX_NEW_TOKENS", 512)
    )

    # ------------------------------------------------------------------
    # Table QA  (TAPAS)
    # ------------------------------------------------------------------
    TAPAS_MODEL_ID: str = field(
        default_factory=lambda: _env(
            "TAPAS_MODEL_ID", "google/tapas-base-finetuned-wtq"
        )
    )
    TAPAS_MAX_FACTS_PER_TABLE: int = field(
        default_factory=lambda: _env_int("TAPAS_MAX_FACTS_PER_TABLE", 3)
    )

    # ------------------------------------------------------------------
    # Embeddings + FAISS
    # ------------------------------------------------------------------
    EMBEDDER_MODEL_ID: str = field(
        default_factory=lambda: _env(
            "EMBEDDER_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    EMBEDDING_DIM: int = field(
        default_factory=lambda: _env_int("EMBEDDING_DIM", 384)
    )
    FAISS_TOP_K: int = field(
        default_factory=lambda: _env_int("FAISS_TOP_K", 10)
    )

    # ------------------------------------------------------------------
    # Summarization LLM
    # ------------------------------------------------------------------
    GEMINI_MODEL: str = field(
        default_factory=lambda: _env(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )
    )
    GEMINI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY")
    )
    GEMINI_MAX_OUTPUT_TOKENS: int = field(
        default_factory=lambda: _env_int("GEMINI_MAX_OUTPUT_TOKENS", 512)
    )
    GEMINI_TEMPERATURE: float = field(
        default_factory=lambda: _env_float("GEMINI_TEMPERATURE", 0.3)
    )
    GEMINI_RPM_LIMIT: int = field(
        default_factory=lambda: _env_int("GEMINI_RPM_LIMIT", 60)
    )
    GEMINI_RETRY_MAX: int = field(
        default_factory=lambda: _env_int("GEMINI_RETRY_MAX", 5)
    )
    GEMINI_RETRY_BASE_DELAY: float = field(
        default_factory=lambda: _env_float("GEMINI_RETRY_BASE_DELAY", 1.0)
    )

    # Fallback LLM (Groq)
    GROQ_API_KEY: Optional[str] = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY")
    )
    GROQ_MODEL: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "mixtral-8x7b-32768")
    )

    SUMMARY_MIN_WORDS: int = field(
        default_factory=lambda: _env_int("SUMMARY_MIN_WORDS", 200)
    )
    SUMMARY_MAX_WORDS: int = field(
        default_factory=lambda: _env_int("SUMMARY_MAX_WORDS", 350)
    )

    # ------------------------------------------------------------------
    # XAI — SHAP
    # ------------------------------------------------------------------
    SHAP_MAX_CHUNKS: int = field(
        default_factory=lambda: _env_int("SHAP_MAX_CHUNKS", 20)
    )
    SHAP_NSAMPLES: int = field(
        default_factory=lambda: _env_int("SHAP_NSAMPLES", 128)
    )
    SHAP_TOP_K: int = field(
        default_factory=lambda: _env_int("SHAP_TOP_K", 10)
    )

    # ------------------------------------------------------------------
    # XAI — LIME
    # ------------------------------------------------------------------
    LIME_NUM_FEATURES: int = field(
        default_factory=lambda: _env_int("LIME_NUM_FEATURES", 8)
    )
    LIME_NUM_SAMPLES: int = field(
        default_factory=lambda: _env_int("LIME_NUM_SAMPLES", 100)
    )

    # ------------------------------------------------------------------
    # XAI — Captum GradCAM
    # ------------------------------------------------------------------
    CAPTUM_UPSAMPLE_MODE: str = field(
        default_factory=lambda: _env("CAPTUM_UPSAMPLE_MODE", "bilinear")
    )
    CAPTUM_HEATMAP_DPI: int = field(
        default_factory=lambda: _env_int("CAPTUM_HEATMAP_DPI", 100)
    )
    CAPTUM_COLORMAP: str = field(
        default_factory=lambda: _env("CAPTUM_COLORMAP", "jet")
    )
    CAPTUM_ALPHA: float = field(
        default_factory=lambda: _env_float("CAPTUM_ALPHA", 0.5)
    )

    # ------------------------------------------------------------------
    # XAI — MCR (Modality Contribution Ratio)
    # ------------------------------------------------------------------
    MCR_TARGET_MIN: float = field(
        default_factory=lambda: _env_float("MCR_TARGET_MIN", 0.30)
    )
    MCR_PIE_COLORS: tuple = field(
        default=("#0B6E6E", "#7D3C98", "#D4570A")  # text, figure, table
    )

    # ------------------------------------------------------------------
    # Evaluation metrics targets
    # ------------------------------------------------------------------
    TARGET_ROUGE_L: float = field(
        default_factory=lambda: _env_float("TARGET_ROUGE_L", 0.25)
    )
    TARGET_BERTSCORE_F1: float = field(
        default_factory=lambda: _env_float("TARGET_BERTSCORE_F1", 0.85)
    )
    TARGET_XAI_TRUST_SCORE: float = field(
        default_factory=lambda: _env_float("TARGET_XAI_TRUST_SCORE", 3.5)
    )
    TARGET_SHAP_REF_ALIGN: float = field(
        default_factory=lambda: _env_float("TARGET_SHAP_REF_ALIGN", 0.70)
    )
    TARGET_LIME_ATTR_OVERLAP: float = field(
        default_factory=lambda: _env_float("TARGET_LIME_ATTR_OVERLAP", 0.60)
    )
    TARGET_CAPTUM_ATTN_REL: float = field(
        default_factory=lambda: _env_float("TARGET_CAPTUM_ATTN_REL", 0.65)
    )

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------
    ARXIV_DOMAINS: tuple = field(
        default=("cs.CL", "cs.CV", "cs.LG", "cs.AR", "cs.RO")
    )
    ARXIV_PAPERS_PER_DOMAIN: int = field(
        default_factory=lambda: _env_int("ARXIV_PAPERS_PER_DOMAIN", 20)
    )
    ANNOTATION_GOLD_WORDS: int = field(
        default_factory=lambda: _env_int("ANNOTATION_GOLD_WORDS", 200)
    )

    # ------------------------------------------------------------------
    # Gradio / deployment
    # ------------------------------------------------------------------
    GRADIO_SERVER_PORT: int = field(
        default_factory=lambda: _env_int("GRADIO_SERVER_PORT", 7860)
    )
    GRADIO_SHARE: bool = field(
        default_factory=lambda: _env_bool("GRADIO_SHARE", False)
    )
    PIPELINE_TIMEOUT_SECONDS: int = field(
        default_factory=lambda: _env_int("PIPELINE_TIMEOUT_SECONDS", 180)
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO")
    )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of all settings.

        Sensitive fields (API keys) are redacted.
        """
        import dataclasses
        raw = dataclasses.asdict(self)
        # Redact secrets
        for key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
            if raw.get(key):
                raw[key] = "***REDACTED***"
        # Convert tuples to lists for JSON compatibility
        for k, v in raw.items():
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw


# ---------------------------------------------------------------------------
# Environment auto-detection
# ---------------------------------------------------------------------------

def _detect_env() -> str:
    """
    Detect the current runtime environment.

    Returns one of: 'colab', 'kaggle', 'local'
    """
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"


# ---------------------------------------------------------------------------
# Module-level singleton (lazily created)
# ---------------------------------------------------------------------------

_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return the global Settings singleton.

    Call this instead of instantiating Settings() directly so that the
    object is shared across modules without re-reading env vars repeatedly.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# ---------------------------------------------------------------------------
# Example usage (run as script for a quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    cfg = get_settings()
    print(f"[settings] Environment detected : {cfg.ENV}")
    print(f"[settings] Project              : {cfg.PROJECT_NAME}")
    print(f"[settings] BLIP-2 model         : {cfg.BLIP2_MODEL_ID}")
    print(f"[settings] Gemini model         : {cfg.GEMINI_MODEL}")
    print(f"[settings] FAISS top-k          : {cfg.FAISS_TOP_K}")
    print(f"[settings] SHAP n_samples       : {cfg.SHAP_NSAMPLES}")
    print()
    print("[settings] Full config (JSON):")
    print(json.dumps(cfg.to_dict(), indent=2))

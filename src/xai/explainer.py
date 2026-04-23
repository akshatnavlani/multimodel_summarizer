"""
src/xai/explainer.py
---------------------
Explainability module for the Multimodal Document Summarization pipeline.

Explains WHY the generated summary was produced by:
  1. Sentence-level attribution  — maps each summary sentence to its
     most influential retrieved chunk (cosine similarity via MiniLM).
  2. Modality Contribution Ratio — what fraction of the summary came
     from text / figure / table sources.
  3. Placeholder hooks for SHAP, LIME, and gradient-based attribution
     (structure-ready for Step 8 extension).

Pipeline contract:
    INPUT  →
        summary_result   : Dict — output of Summarizer.generate()
        retrieval_result : Dict — output of search_index(), enriched with
                                  figures + tables lists
    OUTPUT →
    {
        "status": "success" | "error",
        "xai": {
            "attribution": [
                {
                    "sentence_index":  int,
                    "summary_sentence":str,
                    "source_chunk":    str,
                    "source_type":     "text" | "figure" | "table",
                    "source_id":       str,
                    "page":            int,
                    "similarity_score":float,
                }
            ],
            "modality_contribution": {
                "text":   float,   # fractions that sum to 1.0
                "figure": float,
                "table":  float,
            },
            "shap_hook":     None,   # placeholder
            "lime_hook":     None,   # placeholder
            "gradient_hook": None,   # placeholder
        },
        "metadata": {
            "paper_id":          str,
            "num_sentences":     int,
            "num_chunks":        int,
            "model_used":        str,
            "cached":            bool,
            "elapsed_seconds":   float,
        }
    }

Caching:
    - Saves to data/xai_outputs/{paper_id}_xai.json
    - Returns cached result if file exists and force_reprocess=False.

Performance:
    - Pure numpy cosine similarity — no GPU, no heavy models.
    - Reuses sentence-transformer model singleton from embedder if loaded;
      otherwise loads it on first call (cached across calls).

Usage:
    from src.xai.explainer import Explainer
    explainer  = Explainer()
    xai_result = explainer.explain(summary_result, retrieval_result)
    print(xai_result["xai"]["modality_contribution"])
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384
MIN_SENTENCE_LEN = 10   # characters — ignore very short fragments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _error_response(message: str) -> Dict[str, Any]:
    logger.error("[explainer] %s", message)
    return {
        "status":  "error",
        "message": message,
        "xai":     {},
        "metadata":{},
    }


def _cache_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["xai_outputs"] / f"{paper_id}_xai.json"


def _extract_paper_id(summary_result: Dict, retrieval_result: Dict) -> str:
    """Try multiple locations for paper_id."""
    pid = (
        summary_result.get("metadata", {}).get("paper_id")
        or retrieval_result.get("paper_id")
        or retrieval_result.get("metadata", {}).get("paper_id", "")
    )
    if not pid:
        results = retrieval_result.get("top_k_results", [])
        if results:
            cid = results[0].get("chunk_id", "")
            if "_" in cid:
                pid = "_".join(cid.split("_")[:2])
    return pid or "unknown_paper"


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """
    Split summary text into individual sentences.

    Uses a simple regex that handles common academic sentence endings
    (period, exclamation, question mark) while avoiding splits on
    abbreviations like 'Fig.', 'et al.', 'e.g.'.
    """
    # Protect common abbreviations from being split
    protected = text
    for abbr in ["Fig.", "Table.", "et al.", "e.g.", "i.e.", "vs.", "approx.", "Dr.", "Prof."]:
        protected = protected.replace(abbr, abbr.replace(".", "<!DOT!>"))

    # Split on sentence-ending punctuation followed by whitespace + capital
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Restore protected dots and filter short fragments
    sentences = []
    for s in raw_sentences:
        s = s.replace("<!DOT!>", ".").strip()
        if len(s) >= MIN_SENTENCE_LEN:
            sentences.append(s)

    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

# Module-level singleton — shared with embedder to avoid double loading
_sentence_model = None


def _get_embedding_model():
    """Load MiniLM model once; reuse across calls."""
    global _sentence_model

    if _sentence_model is not None:
        return _sentence_model

    # Try to reuse the model already loaded by embedder
    try:
        from src.retrieval.embedder import _embedding_model as _em
        if _em is not None:
            logger.info("[explainer] Reusing MiniLM from embedder module.")
            _sentence_model = _em
            return _sentence_model
    except ImportError:
        pass

    from sentence_transformers import SentenceTransformer
    logger.info("[explainer] Loading %s for XAI …", EMBEDDING_MODEL)
    _sentence_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    _sentence_model.max_seq_length = 256
    return _sentence_model


def _encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised float32 embeddings.
    Returns ndarray of shape (N, 384).
    """
    model = _get_embedding_model()
    vecs  = model.encode(
        texts,
        batch_size           = 32,
        convert_to_numpy     = True,
        normalize_embeddings = True,
        show_progress_bar    = False,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Cosine similarity (pure numpy — no sklearn dependency)
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(
    query_vecs: np.ndarray,   # (M, D) — sentence embeddings
    key_vecs:   np.ndarray,   # (N, D) — chunk embeddings
) -> np.ndarray:
    """
    Compute cosine similarity between every query and every key.

    Both inputs should be L2-normalised → dot product = cosine.
    Returns (M, N) float32 matrix of scores in [-1, 1].
    """
    return np.dot(query_vecs, key_vecs.T)   # (M, N)


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Shift scores to [0, 1] range using min-max normalisation."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-8:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _normalise_source_type(raw_type: str) -> str:
    """Map various type strings to canonical: text | figure | table."""
    t = raw_type.lower()
    if t in ("figure", "fig", "chart", "image"):
        return "figure"
    if t in ("table", "tbl"):
        return "table"
    return "text"


def _extract_chunks_from_retrieval(
    retrieval_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build a flat list of chunk dicts from the retrieval result.

    Reads from three sources:
      1. top_k_results — FAISS-retrieved chunks (may already include figure/table chunks)
      2. retrieval_result["figures"] — figure captions injected by the pipeline
      3. retrieval_result["tables"]  — table summaries injected by the pipeline

    Each chunk has: text, source_type, source_id, page.
    """
    chunks = []
    seen_ids: set = set()

    # 1. FAISS-retrieved chunks (text, and any embedded fig/table chunks)
    for item in retrieval_result.get("top_k_results", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cid = item.get("chunk_id", "")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunks.append({
            "text":        text,
            "source_type": _normalise_source_type(
                item.get("modality", item.get("type", "text"))
            ),
            "source_id":   item.get("source_id", cid),
            "page":        item.get("page", -1),
            "chunk_id":    cid,
        })

    # 2. Figure captions — always include even if not in top_k
    for fig in retrieval_result.get("figures", []):
        cap = str(fig.get("caption", "")).strip()
        if not cap or cap.startswith("[captioning failed"):
            continue
        fig_id = fig.get("figure_id", fig.get("element_id", ""))
        cid = f"{fig_id}_caption"
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunks.append({
            "text":        f"Figure {fig_id}: {cap}",
            "source_type": "figure",
            "source_id":   fig_id,
            "page":        fig.get("page", -1),
            "chunk_id":    cid,
        })

    # 3. Table summaries — always include even if not in top_k
    for tbl in retrieval_result.get("tables", []):
        sm = str(tbl.get("summary", tbl.get("markdown", ""))).strip()
        if not sm:
            continue
        tbl_id = tbl.get("table_id", "")
        cid = f"{tbl_id}_summary"
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunks.append({
            "text":        f"Table {tbl_id}: {sm}",
            "source_type": "table",
            "source_id":   tbl_id,
            "page":        tbl.get("page", -1),
            "chunk_id":    cid,
        })

    logger.info("[explainer] Attribution pool: %d chunks (%d from top_k, %d figures, %d tables).",
                len(chunks),
                len(retrieval_result.get("top_k_results", [])),
                len(retrieval_result.get("figures", [])),
                len(retrieval_result.get("tables", [])))
    return chunks


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _compute_attribution(
    sentences:      List[str],
    chunks:         List[Dict[str, Any]],
    chunk_embeddings: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    For each summary sentence, find the most similar retrieved chunk.

    Parameters
    ----------
    sentences        : list of sentence strings
    chunks           : list of chunk dicts (text, source_type, source_id, page)
    chunk_embeddings : pre-computed (N, 384) ndarray or None

    Returns
    -------
    List of attribution dicts, one per sentence.
    """
    if not sentences or not chunks:
        return []

    # Embed sentences
    sent_vecs = _encode_texts(sentences)          # (M, 384)

    # Embed chunks (or reuse)
    if chunk_embeddings is None or chunk_embeddings.shape[0] != len(chunks):
        chunk_texts = [c["text"] for c in chunks]
        chunk_vecs  = _encode_texts(chunk_texts)  # (N, 384)
    else:
        chunk_vecs = chunk_embeddings

    # Cosine similarity matrix (M sentences × N chunks)
    sim_matrix = _cosine_similarity_matrix(sent_vecs, chunk_vecs)   # (M, N)

    # Normalise to [0, 1] for interpretability
    norm_matrix = _normalise_scores(sim_matrix)

    attributions = []
    for sent_idx, sentence in enumerate(sentences):
        best_chunk_idx = int(np.argmax(sim_matrix[sent_idx]))
        best_chunk     = chunks[best_chunk_idx]
        norm_score     = float(norm_matrix[sent_idx, best_chunk_idx])

        # Truncate long source chunks for readability
        source_preview = best_chunk["text"][:300]

        attributions.append({
            "sentence_index":  sent_idx,
            "summary_sentence":sentence,
            "source_chunk":    source_preview,
            "source_type":     best_chunk["source_type"],
            "source_id":       best_chunk["source_id"],
            "page":            best_chunk["page"],
            "similarity_score":round(norm_score, 4),
        })

    return attributions


# ---------------------------------------------------------------------------
# Modality Contribution Ratio
# ---------------------------------------------------------------------------

def _compute_modality_contribution(
    attributions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute what fraction of summary sentences were attributed to
    each modality (text, figure, table).

    Returns a dict of fractions that sum to 1.0.
    Handles edge cases (no attributions, single modality, etc.).
    """
    if not attributions:
        return {"text": 1.0, "figure": 0.0, "table": 0.0}

    counts = {"text": 0, "figure": 0, "table": 0}
    for attr in attributions:
        st = attr.get("source_type", "text")
        if st in counts:
            counts[st] += 1
        else:
            counts["text"] += 1   # default to text for unknown types

    total = sum(counts.values())
    if total == 0:
        return {"text": 1.0, "figure": 0.0, "table": 0.0}

    return {k: round(v / total, 4) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# XAI placeholder hooks (structure-ready for SHAP / LIME)
# ---------------------------------------------------------------------------

def _shap_hook(
    summary: str,
    chunks:  List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    SHAP KernelExplainer hook — placeholder.

    Future implementation:
        - Treat retrieval as black-box function f(S) = mean cosine score
        - KernelExplainer samples chunk subsets (binary mask)
        - Computes Shapley values per chunk
        - Output: {"chunk_id": shap_value} dict

    Install: pip install shap
    """
    return {
        "status":  "placeholder",
        "message": "SHAP not yet implemented. "
                   "Install shap and implement KernelExplainer on retrieval scores.",
        "chunks_available": len(chunks),
    }


def _lime_hook(
    summary:  str,
    chunks:   List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    LIME TextExplainer hook — placeholder.

    Future implementation:
        - Treat each chunk as a 'token' in LIME's text representation
        - Perturb which chunks are present
        - Observe BERTScore change vs original summary sentence
        - Output: [(chunk_text, weight), ...] per sentence

    Install: pip install lime bert-score
    """
    return {
        "status":  "placeholder",
        "message": "LIME not yet implemented. "
                   "Install lime and bert-score, then implement LimeTextExplainer.",
        "chunks_available": len(chunks),
    }


def _gradient_hook(
    summary: str,
    chunks:  List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Gradient-based attribution hook — placeholder.

    Future implementation (Captum):
        - Use Integrated Gradients on sentence-transformer
        - Attribute summary token importance to input chunk tokens
        - Requires PyTorch + Captum

    Install: pip install captum
    """
    return {
        "status":  "placeholder",
        "message": "Gradient attribution not yet implemented. "
                   "Install captum and implement IntegratedGradients.",
        "chunks_available": len(chunks),
    }


# ---------------------------------------------------------------------------
# Explainer class
# ---------------------------------------------------------------------------

class Explainer:
    """
    Multimodal XAI explainer.

    Attributes each summary sentence to its most similar retrieved chunk
    and computes the overall modality contribution ratio.

    Provides placeholder hooks for SHAP, LIME, and gradient attribution
    ready for future implementation.

    Example
    -------
    explainer  = Explainer()
    xai_result = explainer.explain(summary_result, retrieval_result)
    for attr in xai_result["xai"]["attribution"]:
        print(attr["summary_sentence"], "→", attr["source_type"])
    """

    def __init__(self):
        self._base_paths = _get_paths()

    # ── Public API ───────────────────────────────────────────────────────────

    def explain(
        self,
        summary_result:   Dict[str, Any],
        retrieval_result: Dict[str, Any],
        force_reprocess:  bool = False,
    ) -> Dict[str, Any]:
        """
        Explain a generated summary using sentence-level attribution
        and modality contribution ratio.

        Parameters
        ----------
        summary_result   : Dict — output of Summarizer.generate()
        retrieval_result : Dict — output of search_index() (with top_k_results)
        force_reprocess  : bool — ignore cached result

        Returns
        -------
        XAI contract dict (JSON-serialisable).
        """
        # ── Validate inputs ──────────────────────────────────────────────────
        if not isinstance(summary_result, dict):
            return _error_response("summary_result must be a dict.")
        if not isinstance(retrieval_result, dict):
            return _error_response("retrieval_result must be a dict.")

        summary = summary_result.get("summary", "").strip()
        if not summary:
            return _error_response("summary_result['summary'] is empty.")

        top_k_results = retrieval_result.get("top_k_results", [])
        if not top_k_results:
            return _error_response("retrieval_result['top_k_results'] is empty.")

        paper_id   = _extract_paper_id(summary_result, retrieval_result)
        cache_file = _cache_path(paper_id, self._base_paths)

        # ── Cache check ──────────────────────────────────────────────────────
        if not force_reprocess and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                logger.info("[explainer] Cache hit for %s.", paper_id)
                cached.setdefault("metadata", {})["cached"] = True
                return cached
            except Exception as e:
                logger.warning("[explainer] Cache corrupt (%s) — recomputing.", e)

        t0 = time.time()

        # ── Build chunk list ─────────────────────────────────────────────────
        chunks = _extract_chunks_from_retrieval(retrieval_result)
        if not chunks:
            return _error_response("No valid chunks extracted from retrieval_result.")

        # ── Sentence segmentation ────────────────────────────────────────────
        sentences = _split_into_sentences(summary)
        logger.info("[explainer] %s: %d sentences, %d chunks.", paper_id, len(sentences), len(chunks))

        # ── Attribution ──────────────────────────────────────────────────────
        attributions = _compute_attribution(sentences, chunks)

        # ── Modality contribution ────────────────────────────────────────────
        modality_contribution = _compute_modality_contribution(attributions)

        elapsed = time.time() - t0

        # ── XAI placeholder hooks ────────────────────────────────────────────
        shap_hook     = _shap_hook(summary, chunks)
        lime_hook     = _lime_hook(summary, chunks)
        gradient_hook = _gradient_hook(summary, chunks)

        result = {
            "status": "success",
            "xai": {
                "attribution":           attributions,
                "modality_contribution": modality_contribution,
                "shap_hook":             shap_hook,
                "lime_hook":             lime_hook,
                "gradient_hook":         gradient_hook,
            },
            "metadata": {
                "paper_id":        paper_id,
                "num_sentences":   len(sentences),
                "num_chunks":      len(chunks),
                "model_used":      EMBEDDING_MODEL,
                "cached":          False,
                "elapsed_seconds": round(elapsed, 2),
            },
        }

        # ── Save ─────────────────────────────────────────────────────────────
        self._save(result, cache_file)
        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self, result: Dict[str, Any], path: Path) -> None:
        try:
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            logger.info("[explainer] Saved → %s", path)
        except Exception as e:
            logger.error("[explainer] Save failed: %s", e)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def explain(
    summary_result:   Dict[str, Any],
    retrieval_result: Dict[str, Any],
    force_reprocess:  bool = False,
) -> Dict[str, Any]:
    """Functional wrapper around Explainer.explain()."""
    return Explainer().explain(summary_result, retrieval_result, force_reprocess)


# ---------------------------------------------------------------------------
# Minimal test
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Unit tests using dummy data — no API keys or files required."""
    print("Running explainer tests …")

    dummy_summary_result = {
        "status":  "success",
        "summary": (
            "The paper proposes a transformer-based multimodal summarization method. "
            "Figure 1 illustrates the overall architecture with three processing stages. "
            "The model achieves a ROUGE-L score of 0.42 on the ArXiv benchmark. "
            "Table 1 shows a comparison with baseline text-only methods. "
            "The approach outperforms prior work by 12 ROUGE points."
        ),
        "metadata": {"paper_id": "test_paper_001"},
    }

    dummy_retrieval_result = {
        "query": "main contribution and results",
        "paper_id": "test_paper_001",
        "top_k_results": [
            {
                "rank": 1, "score": 0.91,
                "chunk_id": "test_paper_001_chunk_0000",
                "text": "We propose a transformer-based architecture for multimodal document summarization.",
                "type": "text", "modality": "text", "page": 1, "source_id": "sec_intro",
            },
            {
                "rank": 2, "score": 0.87,
                "chunk_id": "test_paper_001_fig_0001_caption",
                "text": "Figure 1: Overview of the three-stage multimodal pipeline architecture.",
                "type": "figure", "modality": "figure", "page": 2, "source_id": "fig_001",
            },
            {
                "rank": 3, "score": 0.84,
                "chunk_id": "test_paper_001_chunk_0010",
                "text": "Our model achieves ROUGE-L of 0.42 on the ArXiv summarization benchmark.",
                "type": "text", "modality": "text", "page": 5, "source_id": "sec_results",
            },
            {
                "rank": 4, "score": 0.79,
                "chunk_id": "test_paper_001_table_0001_summary",
                "text": "Table 1: Comparison showing our method (ROUGE-L=0.42) vs text-only baseline (0.30).",
                "type": "table", "modality": "table", "page": 6, "source_id": "tbl_001",
            },
            {
                "rank": 5, "score": 0.75,
                "chunk_id": "test_paper_001_chunk_0015",
                "text": "The proposed method outperforms all baselines by at least 12 ROUGE points.",
                "type": "text", "modality": "text", "page": 7, "source_id": "sec_conclusion",
            },
        ],
    }

    explainer  = Explainer()
    xai_result = explainer.explain(dummy_summary_result, dummy_retrieval_result, force_reprocess=True)

    # Test 1: top-level structure
    assert isinstance(xai_result, dict),     "Result must be dict"
    assert "status"   in xai_result,         "Missing 'status'"
    assert "xai"      in xai_result,         "Missing 'xai'"
    assert "metadata" in xai_result,         "Missing 'metadata'"
    assert xai_result["status"] == "success","Expected success"
    print("  ✅ Top-level structure valid")

    # Test 2: attribution
    xai = xai_result["xai"]
    assert "attribution" in xai,             "Missing attribution"
    attrs = xai["attribution"]
    assert len(attrs) > 0,                   "Attribution list empty"
    for key in ["sentence_index", "summary_sentence", "source_chunk",
                "source_type", "source_id", "page", "similarity_score"]:
        assert key in attrs[0],              f"Attribution missing key: {key}"
    print(f"  ✅ Attribution: {len(attrs)} sentences attributed")

    # Test 3: modality contribution sums to 1.0
    mc = xai["modality_contribution"]
    assert "text"   in mc, "Missing text in modality_contribution"
    assert "figure" in mc, "Missing figure in modality_contribution"
    assert "table"  in mc, "Missing table in modality_contribution"
    total = sum(mc.values())
    assert abs(total - 1.0) < 0.01, f"Modality contributions sum to {total}, expected 1.0"
    print(f"  ✅ Modality contribution sums to 1.0: {mc}")

    # Test 4: similarity scores in [0, 1]
    for attr in attrs:
        s = attr["similarity_score"]
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] range"
    print("  ✅ All similarity scores in [0, 1]")

    # Test 5: XAI hooks present (placeholders)
    assert "shap_hook"     in xai, "Missing shap_hook"
    assert "lime_hook"     in xai, "Missing lime_hook"
    assert "gradient_hook" in xai, "Missing gradient_hook"
    print("  ✅ XAI hooks (SHAP/LIME/gradient) present as placeholders")

    # Test 6: JSON serialisable
    json.dumps(xai_result)
    print("  ✅ Output is JSON-serialisable")

    # Test 7: error handling
    bad = explainer.explain({"summary": ""}, {"top_k_results": []})
    assert bad["status"] == "error", "Expected error for empty input"
    print("  ✅ Empty input → error returned correctly")

    # Test 8: metadata
    meta = xai_result["metadata"]
    for key in ["paper_id", "num_sentences", "num_chunks", "model_used",
                "cached", "elapsed_seconds"]:
        assert key in meta, f"Missing metadata key: {key}"
    print(f"  ✅ Metadata valid: {meta['num_sentences']} sentences, "
          f"{meta['num_chunks']} chunks, {meta['elapsed_seconds']}s")

    # Print sample attribution
    print("\n  Sample attributions:")
    for a in attrs[:3]:
        print(f"    [{a['sentence_index']}] {a['source_type']:6s} "
              f"(score={a['similarity_score']:.3f}) "
              f"| {a['summary_sentence'][:60]}")

    print("\n✅ All explainer tests passed.")


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import glob

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # Run unit tests first
    _run_tests()

    # Full pipeline smoke test
    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("\nNo PDFs found — skipping pipeline test.")
        sys.exit(0)

    from src.ingestion.pdf_loader       import load_pdf
    from src.layout.layout_parser       import parse_layout
    from src.extraction.text_extractor   import extract_text
    from src.extraction.table_parser     import parse_tables
    from src.retrieval.embedder          import embed_paper
    from src.retrieval.faiss_index       import build_index, search_index
    from src.summarization.summarizer    import Summarizer

    pdf = pdfs[0]
    ing = load_pdf(pdf)
    lay = parse_layout(ing)
    txt = extract_text(ing, lay, force_reprocess=True)
    tbl = parse_tables(ing, lay)
    emb = embed_paper(ing, txt, table_result=tbl, force_reprocess=False)
    idx = build_index(emb, force_reprocess=False)

    search_result = search_index(idx, "main contribution and results", top_k=8)
    search_result["tables"]   = tbl.get("metadata", {}).get("tables", [])
    search_result["paper_id"] = emb["metadata"]["paper_id"]

    summary_result = Summarizer().generate(search_result, max_words=300, force_reprocess=False)

    xai_result = Explainer().explain(summary_result, search_result, force_reprocess=True)

    print(f"\nStatus:     {xai_result['status']}")
    meta = xai_result["metadata"]
    print(f"Sentences:  {meta['num_sentences']}")
    print(f"Chunks:     {meta['num_chunks']}")
    print(f"Time:       {meta['elapsed_seconds']}s")
    print(f"\nModality contribution:")
    mc = xai_result["xai"]["modality_contribution"]
    for modality, fraction in mc.items():
        bar = "█" * int(fraction * 30)
        print(f"  {modality:6s}: {bar} {fraction:.1%}")

    print(f"\nSentence attributions:")
    for a in xai_result["xai"]["attribution"]:
        print(f"  [{a['sentence_index']}] → {a['source_type']:6s} "
              f"p{a['page']} score={a['similarity_score']:.3f}")
        print(f"       SENT:  {a['summary_sentence'][:80]}")
        print(f"       CHUNK: {a['source_chunk'][:80]}")
        print()

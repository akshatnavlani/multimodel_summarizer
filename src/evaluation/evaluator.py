"""
src/evaluation/evaluator.py
----------------------------
Evaluation module for the Multimodal Document Summarization + XAI pipeline.

Measures three dimensions:
  1. Summary Quality  — ROUGE-1/2/L, BLEU, BERTScore (optional)
  2. Retrieval Quality — source diversity, avg similarity, top-k coverage
  3. System Performance — total time, per-stage breakdown, efficiency score

Pipeline contract:
    All public methods return:
    {
        "status":  "success" | "error",
        "metrics": { "summary": {...}, "retrieval": {...}, "performance": {...} },
        "metadata": { "paper_id": str, "cached": bool, "elapsed_seconds": float }
    }

Caching:
    Results saved to data/evaluation/{paper_id}_eval.json.
    Returned on subsequent calls unless force_reprocess=True.

Usage:
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()

    # Evaluate summary quality against a reference
    result = evaluator.evaluate_summary(predicted, reference)

    # Evaluate full pipeline output in one call
    result = evaluator.evaluate_pipeline(pipeline_output, reference_summary)
    print(result["metrics"])
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUGE_TYPES   = ["rouge1", "rouge2", "rougeL"]
MIN_WORDS     = 10          # minimum word count for a valid summary
EFFICIENCY_TARGET_S = 120.0 # seconds — pipeline at or under this = score 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _error_response(message: str) -> Dict[str, Any]:
    logger.error("[evaluator] %s", message)
    return {
        "status":   "error",
        "message":  message,
        "metrics":  {},
        "metadata": {},
    }


def _eval_cache_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    eval_dir = base_paths.get("evaluation", base_paths["root"] / "evaluation" / "results")
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir / f"{paper_id}_eval.json"


def _extract_paper_id(pipeline_output: Dict[str, Any]) -> str:
    return (
        pipeline_output.get("metadata", {}).get("paper_id")
        or pipeline_output.get("paper_id", "unknown_paper")
    )


# ---------------------------------------------------------------------------
# ROUGE computation
# ---------------------------------------------------------------------------

def _compute_rouge(
    predicted: str,
    reference: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L using rouge-score library.

    Returns dict of {rouge_type: {precision, recall, fmeasure}}.
    Falls back to simple token-overlap if rouge-score is not installed.
    """
    predicted = predicted.strip()
    reference = reference.strip()

    if not predicted or not reference:
        return {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
                for rt in ROUGE_TYPES}

    # ── Primary: rouge-score library ─────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
        scores = scorer.score(reference, predicted)
        return {
            rt: {
                "precision": round(scores[rt].precision, 4),
                "recall":    round(scores[rt].recall,    4),
                "fmeasure":  round(scores[rt].fmeasure,  4),
            }
            for rt in ROUGE_TYPES
        }
    except ImportError:
        logger.warning("[evaluator] rouge-score not installed — using token overlap fallback.")

    # ── Fallback: simple unigram F1 ──────────────────────────────────────────
    pred_tokens = set(predicted.lower().split())
    ref_tokens  = set(reference.lower().split())
    overlap     = len(pred_tokens & ref_tokens)
    precision   = overlap / max(len(pred_tokens), 1)
    recall      = overlap / max(len(ref_tokens), 1)
    f1          = (2 * precision * recall / max(precision + recall, 1e-8))

    return {
        "rouge1": {"precision": round(precision,4), "recall": round(recall,4), "fmeasure": round(f1,4)},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0, "note": "fallback mode"},
        "rougeL": {"precision": round(precision,4), "recall": round(recall,4), "fmeasure": round(f1,4)},
    }


# ---------------------------------------------------------------------------
# BLEU computation
# ---------------------------------------------------------------------------

def _compute_bleu(predicted: str, reference: str) -> Dict[str, Any]:
    """
    Compute sentence-level BLEU using nltk.
    Returns {} with a note if nltk is unavailable (non-blocking).
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        # Download punkt tokenizer silently if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        ref_tokens  = reference.lower().split()
        pred_tokens = predicted.lower().split()
        smoothie    = SmoothingFunction().method4
        score       = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        return {"bleu": round(float(score), 4)}
    except ImportError:
        return {"bleu": None, "note": "nltk not installed — pip install nltk"}
    except Exception as e:
        return {"bleu": None, "note": f"BLEU failed: {str(e)[:60]}"}


# ---------------------------------------------------------------------------
# BERTScore computation
# ---------------------------------------------------------------------------

def _compute_bertscore(
    predicted: str,
    reference: str,
) -> Dict[str, Any]:
    """
    Compute BERTScore F1 using bert-score library (optional).
    Uses bert-base-uncased on CPU — ~10s first call, cached thereafter.
    Returns None values with a note if unavailable.
    """
    try:
        import bert_score
        P, R, F1 = bert_score.score(
            [predicted], [reference],
            lang    = "en",
            verbose = False,
            device  = "cpu",
        )
        return {
            "bertscore_precision": round(float(P[0]), 4),
            "bertscore_recall":    round(float(R[0]), 4),
            "bertscore_f1":        round(float(F1[0]), 4),
        }
    except ImportError:
        return {
            "bertscore_f1": None,
            "note": "bert-score not installed — pip install bert-score",
        }
    except Exception as e:
        return {"bertscore_f1": None, "note": f"BERTScore failed: {str(e)[:80]}"}


# ---------------------------------------------------------------------------
# Summary quality helpers
# ---------------------------------------------------------------------------

def _summary_stats(text: str) -> Dict[str, Any]:
    """Compute lightweight descriptive stats for a summary string."""
    words     = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "word_count":     len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1),
    }


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def _compute_retrieval_metrics(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute diversity, average similarity, and coverage from search results.

    Parameters
    ----------
    retrieval_result : Dict — output of search_index() with 'top_k_results'

    Returns
    -------
    {
        "avg_similarity_score": float,
        "max_similarity_score": float,
        "min_similarity_score": float,
        "source_distribution":  {"text": int, "figure": int, "table": int},
        "source_diversity_ratio": float,   # unique types / 3
        "top_k_retrieved":      int,
        "coverage_score":       float,     # has all 3 modalities → 1.0
    }
    """
    results = retrieval_result.get("top_k_results", [])

    if not results:
        return {
            "avg_similarity_score": 0.0,
            "max_similarity_score": 0.0,
            "min_similarity_score": 0.0,
            "source_distribution":  {"text": 0, "figure": 0, "table": 0},
            "source_diversity_ratio": 0.0,
            "top_k_retrieved":      0,
            "coverage_score":       0.0,
        }

    scores = [float(r.get("score", 0.0)) for r in results]

    # Source distribution
    dist: Dict[str, int] = {"text": 0, "figure": 0, "table": 0}
    for r in results:
        modality = r.get("modality", r.get("type", "text")).lower()
        if modality in ("figure", "chart", "image"):
            dist["figure"] += 1
        elif modality in ("table", "tbl"):
            dist["table"] += 1
        else:
            dist["text"] += 1

    unique_types = sum(1 for v in dist.values() if v > 0)

    return {
        "avg_similarity_score":   round(sum(scores) / max(len(scores), 1), 4),
        "max_similarity_score":   round(max(scores), 4),
        "min_similarity_score":   round(min(scores), 4),
        "source_distribution":    dist,
        "source_diversity_ratio": round(unique_types / 3, 4),
        "top_k_retrieved":        len(results),
        "coverage_score":         round(unique_types / 3, 4),
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_performance_metrics(pipeline_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute latency and efficiency score from pipeline metadata.

    efficiency_score = 1.0 if total_time <= EFFICIENCY_TARGET_S, else decays.
    """
    total_time  = float(pipeline_metadata.get("execution_time", 0.0))
    stage_times = pipeline_metadata.get("stage_times", {})

    # Efficiency score: 1.0 at or under target, graceful decay above
    if total_time <= 0:
        efficiency = 0.0
    elif total_time <= EFFICIENCY_TARGET_S:
        efficiency = 1.0
    else:
        # Decay: doubles target → 0.5, triples → 0.33
        efficiency = round(EFFICIENCY_TARGET_S / total_time, 4)

    # Identify bottleneck stage
    bottleneck = max(stage_times, key=stage_times.get) if stage_times else "unknown"

    return {
        "total_time_seconds":    round(total_time, 2),
        "efficiency_score":      efficiency,
        "target_time_seconds":   EFFICIENCY_TARGET_S,
        "bottleneck_stage":      bottleneck,
        "stage_times":           {k: round(v, 2) for k, v in stage_times.items()},
        "chunks_per_second":     round(
            pipeline_metadata.get("num_chunks", 0) / max(total_time, 0.001), 2
        ),
    }


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluation engine for the Multimodal Document Summarization + XAI pipeline.

    Methods
    -------
    evaluate_summary(predicted, reference)  — ROUGE / BLEU / BERTScore
    evaluate_retrieval(retrieval_result)    — diversity + similarity metrics
    evaluate_pipeline(pipeline_output, reference_summary)  — all three combined

    All methods return a JSON-serialisable dict with "status", "metrics",
    and "metadata" keys.

    Example
    -------
    evaluator = Evaluator()
    metrics   = evaluator.evaluate_pipeline(pipeline_output, reference)
    print(metrics["metrics"]["summary"]["rouge"]["rouge1"])
    """

    def __init__(self):
        self._base_paths = _get_paths()

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate_summary(
        self,
        predicted_summary:  str,
        reference_summary:  str,
        include_bertscore:  bool = False,
    ) -> Dict[str, Any]:
        """
        Compute summary quality metrics against a reference string.

        Parameters
        ----------
        predicted_summary : str  — generated summary to evaluate
        reference_summary : str  — gold/reference summary
        include_bertscore : bool — set True to run BERTScore (adds ~10s CPU)

        Returns
        -------
        {
            "status": "success",
            "metrics": {
                "summary": {
                    "rouge": {...},
                    "bleu":  {...},
                    "bertscore": {...},   # only if include_bertscore=True
                    "predicted_stats": {...},
                    "reference_stats": {...},
                }
            }
        }
        """
        if not predicted_summary or not predicted_summary.strip():
            return _error_response("predicted_summary is empty.")
        if not reference_summary or not reference_summary.strip():
            return _error_response("reference_summary is empty.")

        t0 = time.time()

        rouge_scores = _compute_rouge(predicted_summary, reference_summary)
        bleu_scores  = _compute_bleu(predicted_summary, reference_summary)

        summary_metrics: Dict[str, Any] = {
            "rouge":            rouge_scores,
            "bleu":             bleu_scores,
            "predicted_stats":  _summary_stats(predicted_summary),
            "reference_stats":  _summary_stats(reference_summary),
        }

        if include_bertscore:
            summary_metrics["bertscore"] = _compute_bertscore(
                predicted_summary, reference_summary
            )

        elapsed = round(time.time() - t0, 3)

        return {
            "status":  "success",
            "metrics": {"summary": summary_metrics},
            "metadata": {
                "elapsed_seconds": elapsed,
                "include_bertscore": include_bertscore,
                "cached": False,
            },
        }

    def evaluate_retrieval(
        self,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality from a search_index() result.

        Parameters
        ----------
        retrieval_result : Dict — output of search_index()

        Returns
        -------
        {
            "status": "success",
            "metrics": {
                "retrieval": {
                    "avg_similarity_score", "source_distribution",
                    "source_diversity_ratio", "coverage_score", ...
                }
            }
        }
        """
        if not isinstance(retrieval_result, dict):
            return _error_response("retrieval_result must be a dict.")

        t0 = time.time()
        retrieval_metrics = _compute_retrieval_metrics(retrieval_result)
        elapsed = round(time.time() - t0, 3)

        return {
            "status":  "success",
            "metrics": {"retrieval": retrieval_metrics},
            "metadata": {"elapsed_seconds": elapsed, "cached": False},
        }

    def evaluate_pipeline(
        self,
        pipeline_output:   Dict[str, Any],
        reference_summary: Optional[str] = None,
        include_bertscore: bool           = False,
        force_reprocess:   bool           = False,
    ) -> Dict[str, Any]:
        """
        Run all three evaluations (summary + retrieval + performance) from
        a single run_pipeline() output dict.

        Parameters
        ----------
        pipeline_output   : Dict — output of run_pipeline()
        reference_summary : str  — gold summary (optional; skips ROUGE if None)
        include_bertscore : bool — add BERTScore to summary metrics
        force_reprocess   : bool — ignore cached evaluation result

        Returns
        -------
        Full evaluation contract dict.
        """
        if not isinstance(pipeline_output, dict):
            return _error_response("pipeline_output must be a dict.")

        if pipeline_output.get("status") == "error":
            return _error_response(
                f"Pipeline output has error status: {pipeline_output.get('message', '')}"
            )

        paper_id   = _extract_paper_id(pipeline_output)
        cache_file = _eval_cache_path(paper_id, self._base_paths)

        # ── Cache check ──────────────────────────────────────────────────────
        if not force_reprocess and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                logger.info("[evaluator] Cache hit for %s.", paper_id)
                cached.setdefault("metadata", {})["cached"] = True
                return cached
            except Exception as e:
                logger.warning("[evaluator] Cache corrupt (%s) — recomputing.", e)

        t0 = time.time()

        all_metrics: Dict[str, Any] = {}

        # ── Summary metrics ──────────────────────────────────────────────────
        predicted = pipeline_output.get("summary", "")
        if predicted and reference_summary:
            summ_result      = self.evaluate_summary(
                predicted, reference_summary, include_bertscore
            )
            all_metrics["summary"] = summ_result.get("metrics", {}).get("summary", {})
        elif predicted:
            # No reference — only descriptive stats
            all_metrics["summary"] = {
                "note":            "No reference summary provided — ROUGE not computed.",
                "predicted_stats": _summary_stats(predicted),
            }
        else:
            all_metrics["summary"] = {"note": "No summary in pipeline output."}

        # ── Retrieval metrics ────────────────────────────────────────────────
        # Retrieval result may be embedded in xai or passed directly
        retrieval_result = pipeline_output.get("retrieval_result", {})
        if not retrieval_result:
            # Reconstruct a minimal retrieval_result from XAI attribution
            attribution = pipeline_output.get("xai", {}).get("attribution", [])
            if attribution:
                retrieval_result = {
                    "top_k_results": [
                        {
                            "text":     a.get("source_chunk", ""),
                            "modality": a.get("source_type", "text"),
                            "score":    a.get("similarity_score", 0.0),
                        }
                        for a in attribution
                    ]
                }

        if retrieval_result:
            retr_result          = self.evaluate_retrieval(retrieval_result)
            all_metrics["retrieval"] = retr_result.get("metrics", {}).get("retrieval", {})
        else:
            all_metrics["retrieval"] = {"note": "No retrieval result available."}

        # ── Performance metrics ──────────────────────────────────────────────
        pipeline_meta           = pipeline_output.get("metadata", {})
        all_metrics["performance"] = _compute_performance_metrics(pipeline_meta)

        # ── XAI quality metrics (bonus — derived from existing data) ─────────
        xai = pipeline_output.get("xai", {})
        if xai:
            mc = xai.get("modality_contribution", {})
            attribution = xai.get("attribution", [])
            avg_attribution_score = (
                sum(a.get("similarity_score", 0) for a in attribution)
                / max(len(attribution), 1)
            )
            all_metrics["xai_quality"] = {
                "modality_contribution":    mc,
                "num_attributed_sentences": len(attribution),
                "avg_attribution_score":    round(avg_attribution_score, 4),
                "multimodal_ratio":         round(
                    (mc.get("figure", 0) + mc.get("table", 0)), 4
                ),
            }

        elapsed = round(time.time() - t0, 2)

        result = {
            "status":  "success",
            "metrics": all_metrics,
            "metadata": {
                "paper_id":          paper_id,
                "has_reference":     reference_summary is not None,
                "include_bertscore": include_bertscore,
                "cached":            False,
                "elapsed_seconds":   elapsed,
            },
        }

        # ── Save ─────────────────────────────────────────────────────────────
        self._save(result, cache_file)
        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self, result: Dict[str, Any], path: Path) -> None:
        try:
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            logger.info("[evaluator] Saved → %s", path)
        except Exception as e:
            logger.error("[evaluator] Save failed: %s", e)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def evaluate_summary(
    predicted: str,
    reference: str,
    include_bertscore: bool = False,
) -> Dict[str, Any]:
    """Functional wrapper for Evaluator.evaluate_summary()."""
    return Evaluator().evaluate_summary(predicted, reference, include_bertscore)


def evaluate_retrieval(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    """Functional wrapper for Evaluator.evaluate_retrieval()."""
    return Evaluator().evaluate_retrieval(retrieval_result)


def evaluate_pipeline(
    pipeline_output:   Dict[str, Any],
    reference_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Functional wrapper for Evaluator.evaluate_pipeline()."""
    return Evaluator().evaluate_pipeline(pipeline_output, reference_summary)


# ---------------------------------------------------------------------------
# Minimal tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Unit tests with dummy data — no PDFs or models required."""
    import math
    print("Running evaluator tests …")

    # ── Test 1: evaluate_summary ─────────────────────────────────────────────
    pred = (
        "The paper proposes a transformer-based multimodal summarization method "
        "that processes text, figures, and tables jointly. It achieves ROUGE-L of 0.42 "
        "on the ArXiv benchmark. [Fig 1] shows the architecture. [Table 1] compares baselines."
    )
    ref = (
        "This work introduces a multimodal summarizer using transformers. "
        "The model reaches ROUGE-L 0.42 on ArXiv. Figure 1 illustrates the pipeline. "
        "Table 1 shows comparison with baselines."
    )

    evaluator = Evaluator()
    result    = evaluator.evaluate_summary(pred, ref)

    assert result["status"] == "success",    "Expected success"
    assert "metrics" in result,              "Missing metrics"
    assert "summary" in result["metrics"],   "Missing summary metrics"

    rouge = result["metrics"]["summary"]["rouge"]
    for rt in ["rouge1", "rouge2", "rougeL"]:
        assert rt in rouge, f"Missing {rt}"
        f = rouge[rt]["fmeasure"]
        assert 0.0 <= f <= 1.0, f"{rt} fmeasure={f} out of range"

    print(f"  ✅ ROUGE computed: R1={rouge['rouge1']['fmeasure']:.3f} "
          f"R2={rouge['rouge2']['fmeasure']:.3f} "
          f"RL={rouge['rougeL']['fmeasure']:.3f}")

    bleu = result["metrics"]["summary"]["bleu"].get("bleu")
    print(f"  ✅ BLEU: {bleu}")

    # ── Test 2: empty inputs return error ────────────────────────────────────
    err = evaluator.evaluate_summary("", ref)
    assert err["status"] == "error",         "Expected error for empty predicted"
    print("  ✅ Empty input → error")

    # ── Test 3: evaluate_retrieval ────────────────────────────────────────────
    dummy_retrieval = {
        "query": "main contribution",
        "top_k_results": [
            {"chunk_id": "c0", "text": "Method A",  "modality": "text",   "score": 0.91},
            {"chunk_id": "c1", "text": "Figure 1",  "modality": "figure", "score": 0.85},
            {"chunk_id": "c2", "text": "Table 1",   "modality": "table",  "score": 0.79},
            {"chunk_id": "c3", "text": "Method B",  "modality": "text",   "score": 0.72},
            {"chunk_id": "c4", "text": "Results",   "modality": "text",   "score": 0.68},
        ],
    }
    retr_result = evaluator.evaluate_retrieval(dummy_retrieval)

    assert retr_result["status"] == "success"
    rm = retr_result["metrics"]["retrieval"]
    assert "avg_similarity_score"   in rm
    assert "source_distribution"    in rm
    assert "source_diversity_ratio" in rm
    assert abs(rm["source_distribution"]["text"]   - 3) == 0
    assert abs(rm["source_distribution"]["figure"] - 1) == 0
    assert abs(rm["source_distribution"]["table"]  - 1) == 0
    assert rm["coverage_score"] == 1.0,  "Expected full coverage (3/3 modalities)"
    print(f"  ✅ Retrieval: avg_sim={rm['avg_similarity_score']} "
          f"diversity={rm['source_diversity_ratio']} "
          f"dist={rm['source_distribution']}")

    # ── Test 4: evaluate_pipeline ─────────────────────────────────────────────
    dummy_pipeline_output = {
        "status":  "success",
        "summary": pred,
        "xai": {
            "attribution": [
                {"summary_sentence": "S1", "source_type": "text",   "similarity_score": 0.91, "page": 1},
                {"summary_sentence": "S2", "source_type": "figure", "similarity_score": 0.85, "page": 2},
                {"summary_sentence": "S3", "source_type": "table",  "similarity_score": 0.79, "page": 3},
                {"summary_sentence": "S4", "source_type": "text",   "similarity_score": 0.72, "page": 4},
            ],
            "modality_contribution": {"text": 0.5, "figure": 0.25, "table": 0.25},
        },
        "metadata": {
            "paper_id":       "test_paper_001",
            "execution_time": 45.3,
            "num_chunks":     150,
            "stage_times": {
                "ingestion":       1.2,
                "layout":          2.1,
                "text_extraction": 3.4,
                "embeddings":      5.6,
                "retrieval":       0.8,
                "summarization":  28.0,
                "xai":             4.2,
            },
        },
    }

    pipeline_result = evaluator.evaluate_pipeline(
        dummy_pipeline_output, ref, force_reprocess=True
    )

    assert pipeline_result["status"] == "success"
    m = pipeline_result["metrics"]
    assert "summary"     in m
    assert "retrieval"   in m
    assert "performance" in m

    perf = m["performance"]
    assert perf["total_time_seconds"] == 45.3
    assert 0.0 <= perf["efficiency_score"] <= 1.0
    print(f"  ✅ Pipeline eval: efficiency={perf['efficiency_score']} "
          f"bottleneck={perf['bottleneck_stage']}")

    # ── Test 5: JSON serialisable ─────────────────────────────────────────────
    json.dumps(pipeline_result)
    print("  ✅ Output is JSON-serialisable")

    # ── Test 6: empty retrieval ───────────────────────────────────────────────
    empty_retr = evaluator.evaluate_retrieval({"top_k_results": []})
    assert empty_retr["status"] == "success"
    assert empty_retr["metrics"]["retrieval"]["top_k_retrieved"] == 0
    print("  ✅ Empty retrieval → zero metrics (no crash)")

    print("\n✅ All evaluator tests passed.")


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

    _run_tests()

    # Full pipeline smoke test
    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("\nNo PDFs — skipping pipeline evaluation test.")
        sys.exit(0)

    from src.pipeline.run_pipeline import run_pipeline

    result = run_pipeline(
        pdf_path    = pdfs[0],
        query       = "main contribution and results",
        skip_vision = True,
    )

    evaluator = Evaluator()
    metrics   = evaluator.evaluate_pipeline(result, force_reprocess=True)

    print(f"\nEvaluation Status: {metrics['status']}")
    m = metrics["metrics"]

    if "summary" in m:
        print("\nSUMMARY METRICS")
        print("-" * 40)
        for rt, scores in m["summary"].get("rouge", {}).items():
            print(f"  {rt:7s}: P={scores['precision']:.3f}  "
                  f"R={scores['recall']:.3f}  F={scores['fmeasure']:.3f}")
        bleu = m["summary"].get("bleu", {}).get("bleu")
        print(f"  BLEU   : {bleu}")
        ps = m["summary"].get("predicted_stats", {})
        print(f"  Words  : {ps.get('word_count')} | Sentences: {ps.get('sentence_count')}")

    if "retrieval" in m:
        print("\nRETRIEVAL METRICS")
        print("-" * 40)
        rm = m["retrieval"]
        print(f"  avg_similarity:  {rm.get('avg_similarity_score')}")
        print(f"  source_dist:     {rm.get('source_distribution')}")
        print(f"  diversity_ratio: {rm.get('source_diversity_ratio')}")

    if "performance" in m:
        print("\nPERFORMANCE METRICS")
        print("-" * 40)
        pm = m["performance"]
        print(f"  total_time:      {pm.get('total_time_seconds')}s")
        print(f"  efficiency_score:{pm.get('efficiency_score')}")
        print(f"  bottleneck:      {pm.get('bottleneck_stage')}")
        print(f"  stage_times:")
        for stage, t in pm.get("stage_times", {}).items():
            bar = "▪" * min(int(t * 3), 25)
            print(f"    {stage:30s}: {t:6.2f}s  {bar}")

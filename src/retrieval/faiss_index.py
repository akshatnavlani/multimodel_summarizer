"""
src/retrieval/faiss_index.py
-----------------------------
FAISS IndexFlatIP retrieval engine for the multimodal pipeline.

Builds a per-paper cosine similarity index over all embedded chunks
(text + figure captions + table summaries + chart data).

Key design:
    - IndexFlatIP (inner product) on L2-normalised vectors = cosine similarity
    - CPU-only (faiss-cpu) — no GPU FAISS needed for ≤10K chunks/paper
    - Per-paper index saved as .faiss binary + chunk metadata JSON

Pipeline contract:
    INPUT  → embed_result (Dict from embedder.embed_paper)
    OUTPUT → {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "error",
        "metadata": {
            "paper_id":        str,
            "index_size":      int,
            "embedding_dim":   int,
            "index_path":      str,
            "chunks_path":     str,
        }
    }

Search contract:
    search(query_str, top_k) → {
        "query":        str,
        "top_k_results": [
            {
                "rank":       int,
                "score":      float,
                "chunk_id":   str,
                "text":       str,
                "type":       str,
                "modality":   str,
                "page":       int,
                "source_id":  str,
            }
        ]
    }

Usage:
    from src.retrieval.faiss_index import build_index, search_index, FAISSIndex

    # Build from embed result
    index_result = build_index(embed_result)

    # Search
    hits = search_index(embed_result, "What is the main accuracy result?", top_k=5)
    for h in hits["top_k_results"]:
        print(h["rank"], h["score"], h["text"][:80])

    # OOP interface
    idx = FAISSIndex.from_embed_result(embed_result)
    hits = idx.search("attention mechanism", top_k=3)
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _paper_id_from_embed(embed_result: Dict[str, Any]) -> str:
    return embed_result.get("metadata", {}).get("paper_id", "unknown")


def _index_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["embeddings"] / f"{paper_id}.faiss"


def _chunks_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["embeddings"] / f"{paper_id}_chunks.json"


def _error_response(input_path: str, message: str, output_path: str = "") -> Dict[str, Any]:
    logger.error("[faiss_index] %s", message)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "message":     message,
        "metadata":    {},
    }


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def _import_faiss():
    """Import faiss with a clear error message if not installed."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        )


def _build_flat_ip_index(embeddings: np.ndarray) -> Any:
    """
    Build a FAISS IndexFlatIP from a (N, D) float32 array.

    IndexFlatIP with L2-normalised vectors gives cosine similarity.
    No training required — exact search, suitable for N ≤ 50K.
    """
    faiss = _import_faiss()

    n, d = embeddings.shape
    assert embeddings.dtype == np.float32, "Embeddings must be float32"

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    logger.info("[faiss_index] Built IndexFlatIP: %d vectors, dim=%d", n, d)
    return index


def _save_index(index: Any, path: Path) -> None:
    """Serialise FAISS index to disk."""
    faiss = _import_faiss()
    faiss.write_index(index, str(path))
    logger.info("[faiss_index] Index saved → %s", path)


def _load_index(path: Path) -> Any:
    """Load FAISS index from disk."""
    faiss = _import_faiss()
    index = faiss.read_index(str(path))
    logger.info("[faiss_index] Index loaded from %s (%d vectors)", path, index.ntotal)
    return index


# ---------------------------------------------------------------------------
# Public API — functional style
# ---------------------------------------------------------------------------

def build_index(
    embed_result: Dict[str, Any],
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """
    Build and save a FAISS index from an embed_paper result.

    Parameters
    ----------
    embed_result    : Dict — output of embedder.embed_paper()
    force_reprocess : bool — ignore cache

    Returns
    -------
    Pipeline contract dict.
    """
    if embed_result.get("status") == "error":
        return _error_response(
            embed_result.get("input_path", ""),
            f"Upstream embed failed: {embed_result.get('message','')}",
        )

    input_path = embed_result.get("input_path", "")
    paper_id   = _paper_id_from_embed(embed_result)
    base_paths = _get_paths()
    idx_path   = _index_path(paper_id, base_paths)
    cks_path   = _chunks_path(paper_id, base_paths)

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_reprocess and idx_path.exists() and cks_path.exists():
        try:
            index  = _load_index(idx_path)
            chunks = json.loads(cks_path.read_text())
            logger.info("[faiss_index] Cache hit for %s (%d vectors).", paper_id, index.ntotal)
            return {
                "input_path":  input_path,
                "output_path": str(idx_path),
                "status":      "success",
                "cached":      True,
                "metadata": {
                    "paper_id":      paper_id,
                    "index_size":    index.ntotal,
                    "embedding_dim": index.d,
                    "index_path":    str(idx_path),
                    "chunks_path":   str(cks_path),
                    "_index":        index,
                    "_chunks":       chunks,
                },
            }
        except Exception as e:
            logger.warning("[faiss_index] Cache corrupt (%s) — rebuilding.", e)

    # ── Get embeddings from result ───────────────────────────────────────────
    meta       = embed_result.get("metadata", {})
    embeddings = meta.get("embeddings")
    chunks     = meta.get("chunk_metadata", [])

    # If embeddings not in memory (e.g. loaded from cache), reload from pkl
    if embeddings is None:
        pkl_path = Path(embed_result.get("output_path", ""))
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                embeddings = pickle.load(f)
            logger.info("[faiss_index] Loaded embeddings from pkl: %s", pkl_path)
        else:
            return _error_response(
                input_path,
                f"No embeddings in result and pkl not found at {pkl_path}",
            )

    if not isinstance(embeddings, np.ndarray) or embeddings.shape[0] == 0:
        return _error_response(input_path, f"Empty or invalid embeddings for {paper_id}.")

    # ── Build index ──────────────────────────────────────────────────────────
    t0 = time.time()

    try:
        index = _build_flat_ip_index(embeddings)
    except Exception as e:
        return _error_response(input_path, f"FAISS build failed: {e}")

    # ── Save ─────────────────────────────────────────────────────────────────
    try:
        _save_index(index, idx_path)
        # Also ensure chunks JSON is on disk (embedder may have already saved it)
        if not cks_path.exists() and chunks:
            cks_path.write_text(json.dumps(chunks, indent=2))
    except Exception as e:
        logger.error("[faiss_index] Save failed: %s", e)

    elapsed = time.time() - t0
    logger.info("[faiss_index] %s: index built in %.2fs.", paper_id, elapsed)

    return {
        "input_path":  input_path,
        "output_path": str(idx_path),
        "status":      "success",
        "metadata": {
            "paper_id":      paper_id,
            "index_size":    index.ntotal,
            "embedding_dim": index.d,
            "index_path":    str(idx_path),
            "chunks_path":   str(cks_path),
            "elapsed_seconds": round(elapsed, 2),
            "_index":        index,
            "_chunks":       chunks,
        },
    }


def search_index(
    index_result_or_embed_result: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Search a built FAISS index with a natural language query.

    Accepts either a build_index result or an embed_paper result.
    Loads from disk if the in-memory index is not attached.

    Parameters
    ----------
    index_result_or_embed_result : Dict — output of build_index() or embed_paper()
    query                        : str  — natural language query
    top_k                        : int  — number of results to return

    Returns
    -------
    Search contract dict:
    {
        "query":         str,
        "top_k_results": [{rank, score, chunk_id, text, type, modality, page, source_id}]
    }
    """
    from src.retrieval.embedder import embed_query

    result = index_result_or_embed_result
    meta   = result.get("metadata", {})

    # Retrieve in-memory index if available
    index  = meta.get("_index")
    chunks = meta.get("_chunks") or meta.get("chunk_metadata", [])

    # Load from disk if not in memory
    if index is None:
        paper_id   = meta.get("paper_id", _paper_id_from_embed(result))
        base_paths = _get_paths()
        idx_path   = _index_path(paper_id, base_paths)
        cks_path   = _chunks_path(paper_id, base_paths)

        if not idx_path.exists():
            return {"query": query, "top_k_results": [], "error": f"Index not found: {idx_path}"}

        try:
            index = _load_index(idx_path)
            if not chunks and cks_path.exists():
                chunks = json.loads(cks_path.read_text())
        except Exception as e:
            return {"query": query, "top_k_results": [], "error": f"Index load failed: {e}"}

    if index.ntotal == 0:
        return {"query": query, "top_k_results": []}

    # ── Embed query ──────────────────────────────────────────────────────────
    query_vec = embed_query(query).reshape(1, -1)

    # ── Search ───────────────────────────────────────────────────────────────
    k       = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    # ── Build results ────────────────────────────────────────────────────────
    top_k_results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue

        chunk = chunks[idx] if idx < len(chunks) else {}

        top_k_results.append({
            "rank":      rank + 1,
            "score":     float(score),
            "chunk_id":  chunk.get("chunk_id",  f"chunk_{idx}"),
            "text":      chunk.get("text",      ""),
            "type":      chunk.get("type",      "unknown"),
            "modality":  chunk.get("modality",  "unknown"),
            "page":      chunk.get("page",      -1),
            "source_id": chunk.get("source_id", ""),
        })

    logger.debug(
        "[faiss_index] Query '%s' → top score=%.4f (%d results).",
        query[:50], scores[0][0] if len(scores[0]) > 0 else 0.0, len(top_k_results),
    )

    return {
        "query":        query,
        "top_k_results": top_k_results,
    }


def add_to_index(
    index_result: Dict[str, Any],
    new_embeddings: np.ndarray,
    new_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Add new vectors to an existing in-memory index.

    Parameters
    ----------
    index_result    : Dict        — output of build_index()
    new_embeddings  : np.ndarray  — (M, D) float32
    new_chunks      : list        — M chunk dicts

    Returns
    -------
    Updated index_result with larger index.
    """
    assert isinstance(new_embeddings, np.ndarray), "new_embeddings must be ndarray"
    assert new_embeddings.dtype == np.float32, "Must be float32"
    assert len(new_embeddings) == len(new_chunks), "Mismatch between embeddings and chunks"

    meta  = index_result.get("metadata", {})
    index = meta.get("_index")

    if index is None:
        return _error_response("", "No in-memory index to add to.")

    old_size = index.ntotal
    index.add(new_embeddings)
    meta["_chunks"] = (meta.get("_chunks") or []) + new_chunks
    meta["index_size"] = index.ntotal

    logger.info(
        "[faiss_index] add(): %d → %d vectors.", old_size, index.ntotal
    )

    # Persist updated index
    paper_id   = meta.get("paper_id", "unknown")
    base_paths = _get_paths()
    idx_path   = _index_path(paper_id, base_paths)
    try:
        _save_index(index, idx_path)
    except Exception as e:
        logger.warning("[faiss_index] Could not save after add: %s", e)

    return index_result


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Object-oriented wrapper around a FAISS IndexFlatIP.

    Provides add() and search() methods.
    Supports construction from an embed_paper result or direct from disk.

    Example
    -------
    idx  = FAISSIndex.from_embed_result(embed_result)
    hits = idx.search("What is the ROUGE score?", top_k=5)
    idx.add(more_embeddings, more_chunks)
    idx.save()
    """

    def __init__(
        self,
        paper_id:   str,
        index:      Any,
        chunks:     List[Dict[str, Any]],
        index_path: Path,
        chunks_path:Path,
    ):
        self.paper_id    = paper_id
        self._index      = index
        self._chunks     = chunks
        self._index_path = index_path
        self._chunks_path= chunks_path

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_embed_result(cls, embed_result: Dict[str, Any]) -> "FAISSIndex":
        """Build index from an embed_paper result dict."""
        result = build_index(embed_result, force_reprocess=False)
        if result["status"] != "success":
            raise RuntimeError(f"build_index failed: {result.get('message')}")
        m = result["metadata"]
        return cls(
            paper_id    = m["paper_id"],
            index       = m["_index"],
            chunks      = m["_chunks"],
            index_path  = Path(m["index_path"]),
            chunks_path = Path(m["chunks_path"]),
        )

    @classmethod
    def from_disk(cls, paper_id: str) -> "FAISSIndex":
        """Load a previously built index from disk."""
        base_paths  = _get_paths()
        idx_path    = _index_path(paper_id, base_paths)
        cks_path    = _chunks_path(paper_id, base_paths)

        if not idx_path.exists():
            raise FileNotFoundError(f"No index found at {idx_path}")

        index  = _load_index(idx_path)
        chunks = json.loads(cks_path.read_text()) if cks_path.exists() else []

        return cls(
            paper_id    = paper_id,
            index       = index,
            chunks      = chunks,
            index_path  = idx_path,
            chunks_path = cks_path,
        )

    # ── Core methods ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search the index with a natural language query.

        Returns the search contract dict:
        {"query": str, "top_k_results": [...]}
        """
        from src.retrieval.embedder import embed_query

        query_vec = embed_query(query).reshape(1, -1)
        k = min(top_k, self._index.ntotal)

        if k == 0:
            return {"query": query, "top_k_results": []}

        scores, indices = self._index.search(query_vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            chunk = self._chunks[idx] if idx < len(self._chunks) else {}
            results.append({
                "rank":      rank + 1,
                "score":     float(score),
                "chunk_id":  chunk.get("chunk_id",  f"chunk_{idx}"),
                "text":      chunk.get("text",      ""),
                "type":      chunk.get("type",      "unknown"),
                "modality":  chunk.get("modality",  "unknown"),
                "page":      chunk.get("page",      -1),
                "source_id": chunk.get("source_id", ""),
            })

        return {"query": query, "top_k_results": results}

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new vectors to the index in-place.

        Parameters
        ----------
        embeddings : np.ndarray (M, D) float32, L2-normalised
        chunks     : list of M chunk dicts
        """
        assert embeddings.dtype == np.float32
        assert len(embeddings) == len(chunks)
        old = self._index.ntotal
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info("[FAISSIndex] add(): %d → %d", old, self._index.ntotal)

    def save(self) -> None:
        """Persist index and chunk metadata to disk."""
        _save_index(self._index, self._index_path)
        self._chunks_path.write_text(json.dumps(self._chunks, indent=2))
        logger.info("[FAISSIndex] Saved: %s", self._index_path)

    def __len__(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSIndex(paper_id='{self.paper_id}', "
            f"vectors={self._index.ntotal}, "
            f"dim={self._index.d})"
        )


# ---------------------------------------------------------------------------
# Minimal tests
# ---------------------------------------------------------------------------

def _run_tests():
    """Sanity checks — no disk I/O required."""
    print("Running faiss_index tests …")

    from src.retrieval.embedder import _embed_chunks, _load_model, EMBEDDING_DIM

    model = _load_model()

    # Build dummy embeddings
    dummy_chunks = [
        {"chunk_id": "c0", "text": "The model achieves 94% accuracy on ROUGE-L.",
         "type": "text", "modality": "text", "page": 1, "source_id": "sec2"},
        {"chunk_id": "c1", "text": "Figure 1 shows the attention heatmap.",
         "type": "figure", "modality": "figure", "page": 2, "source_id": "fig1"},
        {"chunk_id": "c2", "text": "Table 2 compares BERT vs GPT-2 on F1.",
         "type": "table", "modality": "table", "page": 3, "source_id": "tbl2"},
        {"chunk_id": "c3", "text": "We use a transformer encoder with 12 layers.",
         "type": "text", "modality": "text", "page": 4, "source_id": "sec3"},
        {"chunk_id": "c4", "text": "The dataset contains 50K scientific papers.",
         "type": "text", "modality": "text", "page": 5, "source_id": "sec4"},
    ]

    embeddings = _embed_chunks(dummy_chunks, model)
    assert embeddings.shape == (5, EMBEDDING_DIM)
    print(f"  ✅ Dummy embeddings: {embeddings.shape}")

    # Build index
    index = _build_flat_ip_index(embeddings)
    assert index.ntotal == 5
    print(f"  ✅ IndexFlatIP built: {index.ntotal} vectors")

    # Wrap in FAISSIndex manually (no disk)
    faiss_idx = FAISSIndex(
        paper_id    = "test_paper",
        index       = index,
        chunks      = dummy_chunks,
        index_path  = Path("/tmp/test.faiss"),
        chunks_path = Path("/tmp/test_chunks.json"),
    )

    # Test search
    hits = faiss_idx.search("What is the accuracy result?", top_k=3)
    assert "top_k_results" in hits
    assert len(hits["top_k_results"]) == 3
    assert hits["top_k_results"][0]["rank"] == 1
    assert hits["top_k_results"][0]["score"] > 0.0
    print(f"  ✅ search(): top hit = '{hits['top_k_results'][0]['text'][:60]}' (score={hits['top_k_results'][0]['score']:.4f})")

    # Test add
    extra_chunks = [{"chunk_id": "c5", "text": "Conclusion: the method is efficient.",
                     "type": "text", "modality": "text", "page": 6, "source_id": "sec5"}]
    extra_emb = _embed_chunks(extra_chunks, model)
    faiss_idx.add(extra_emb, extra_chunks)
    assert len(faiss_idx) == 6
    print(f"  ✅ add(): index now has {len(faiss_idx)} vectors")

    # Verify top-k format
    hit = hits["top_k_results"][0]
    for key in ["rank", "score", "chunk_id", "text", "type", "modality", "page", "source_id"]:
        assert key in hit, f"Missing key: {key}"
    print(f"  ✅ Search contract keys validated")

    print("\n✅ All faiss_index tests passed.")
    return hits


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, glob
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # Run unit tests
    sample_hits = _run_tests()

    print("\n--- Sample search output ---")
    print(json.dumps({
        "query":        sample_hits["query"],
        "top_k_results": [
            {k: v for k, v in h.items() if k != "text"}
            | {"text_preview": h["text"][:60]}
            for h in sample_hits["top_k_results"]
        ]
    }, indent=2))

    # Full pipeline test
    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    if not pdfs:
        print("\nNo PDFs found — skipping pipeline test.")
        sys.exit(0)

    from src.ingestion.pdf_loader      import load_pdf
    from src.layout.layout_parser      import parse_layout
    from src.extraction.text_extractor  import extract_text
    from src.extraction.table_parser    import parse_tables
    from src.retrieval.embedder         import embed_paper

    pdf = pdfs[0]
    ing = load_pdf(pdf)
    lay = parse_layout(ing)
    txt = extract_text(ing, lay, force_reprocess=True)
    tbl = parse_tables(ing, lay)
    emb = embed_paper(ing, txt, table_result=tbl, force_reprocess=True)

    idx_result = build_index(emb, force_reprocess=True)
    print(f"\nbuild_index: {idx_result['status']} | size={idx_result['metadata']['index_size']}")

    QUERIES = [
        "What is the main contribution of this paper?",
        "What accuracy or F1 score was achieved?",
        "What dataset was used for evaluation?",
    ]

    for q in QUERIES:
        hits = search_index(idx_result, q, top_k=3)
        print(f"\nQuery: {q}")
        for h in hits["top_k_results"]:
            print(f"  [{h['rank']}] score={h['score']:.4f} | {h['modality']:6s} | {h['text'][:80]}")

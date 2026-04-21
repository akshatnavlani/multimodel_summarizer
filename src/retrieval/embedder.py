"""
src/retrieval/embedder.py  (v2)
--------------------------------
FIXES vs v1:
  1. Always returns chunks under BOTH 'text_chunks' AND 'chunks' keys in metadata
     so any consumer works regardless of which key it checks.
  2. Cache reload now validates that the loaded pkl has matching size to json.
  3. Chunk builders log a warning when they return 0 items (debugging aid).
  4. embed_query exposed at module level for direct import.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_embedding_model = None
_model_name_used = None

MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _paper_id_from(ingestion_result: Dict[str, Any]) -> str:
    pid = (
        ingestion_result.get("paper_id")
        or ingestion_result.get("metadata", {}).get("paper_id", "")
    )
    if not pid:
        pid = Path(ingestion_result.get("input_path", "unknown.pdf")).stem
    return pid


def _error_response(input_path: str, message: str, output_path: str = "") -> Dict[str, Any]:
    logger.error("[embedder] %s", message)
    return {
        "input_path": input_path, "output_path": output_path,
        "status": "error", "message": message, "metadata": {},
    }


def _embeddings_pkl_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["embeddings"] / f"{paper_id}_embeddings.pkl"


def _chunks_json_path(paper_id: str, base_paths: Dict[str, Path]) -> Path:
    return base_paths["embeddings"] / f"{paper_id}_chunks.json"


def _load_model(model_name: str = MODEL_NAME) -> Any:
    global _embedding_model, _model_name_used
    if _embedding_model is not None:
        return _embedding_model
    from sentence_transformers import SentenceTransformer
    logger.info("[embedder] Loading %s on CPU …", model_name)
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 256
    _embedding_model = model
    _model_name_used = model_name
    logger.info("[embedder] Model loaded. Dim=%d",
                model.get_sentence_embedding_dimension())
    return model


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def _build_text_chunks(text_result: Dict[str, Any], paper_id: str) -> List[Dict]:
    if not text_result or text_result.get("status") == "error":
        return []
    meta = text_result.get("metadata", {})
    # Support both key names used across versions
    raw  = (meta.get("text_chunks")
            or meta.get("chunks")
            or text_result.get("text_chunks")
            or text_result.get("chunks", []))

    if not raw:
        logger.warning("[embedder] text_result has no chunks for %s.", paper_id)

    chunks = []
    for item in raw:
        if isinstance(item, dict):
            text     = item.get("text", "")
            chunk_id = item.get("chunk_id", f"{paper_id}_text_{len(chunks):04d}")
            page     = item.get("page", item.get("page_num", -1))
            ctype    = item.get("type", "text")
        else:
            text     = str(item)
            chunk_id = f"{paper_id}_text_{len(chunks):04d}"
            page     = -1
            ctype    = "text"

        text = str(text).strip()
        if not text:
            continue
        chunks.append({
            "chunk_id":  chunk_id, "text": text, "type": ctype,
            "source_id": chunk_id, "page": page,  "modality": "text",
        })

    logger.info("[embedder] %s: %d text chunks.", paper_id, len(chunks))
    return chunks


def _build_figure_chunks(figure_result: Dict[str, Any], paper_id: str) -> List[Dict]:
    if not figure_result or figure_result.get("status") == "error":
        return []
    figures = figure_result.get("metadata", {}).get("figures", [])
    chunks  = []
    for fig in figures:
        caption = fig.get("caption", "").strip()
        if not caption or caption.startswith("[captioning failed"):
            continue
        fig_id = fig.get("figure_id", f"{paper_id}_fig_{len(chunks):04d}")
        chunks.append({
            "chunk_id":  f"{fig_id}_caption",
            "text":      f"Figure {fig_id}: {caption}",
            "type":      "figure", "source_id": fig_id,
            "page":      fig.get("page", -1), "modality": "figure",
            "bbox":      fig.get("bbox", []), "crop_path": fig.get("crop_path", ""),
        })
    logger.info("[embedder] %s: %d figure chunks.", paper_id, len(chunks))
    return chunks


def _build_table_chunks(table_result: Dict[str, Any], paper_id: str) -> List[Dict]:
    if not table_result or table_result.get("status") == "error":
        return []
    tables = (table_result.get("metadata", {}).get("tables")
              or table_result.get("tables", []))
    chunks = []
    for idx, tbl in enumerate(tables):
        tbl_id  = tbl.get("table_id", f"{paper_id}_table_{idx:04d}")
        summary = tbl.get("summary", "").strip()
        md      = tbl.get("markdown", "").strip()
        text    = summary if summary else md[:400]
        if not text:
            continue
        chunks.append({
            "chunk_id":  f"{tbl_id}_summary",
            "text":      f"Table {tbl_id}: {text}",
            "type":      "table", "source_id": tbl_id,
            "page":      tbl.get("page", -1), "modality": "table",
            "markdown":  md[:200],
        })
    logger.info("[embedder] %s: %d table chunks.", paper_id, len(chunks))
    return chunks


def _build_chart_chunks(chart_result: Dict[str, Any], paper_id: str) -> List[Dict]:
    if not chart_result or chart_result.get("status") == "error":
        return []
    charts = chart_result.get("metadata", {}).get("charts", [])
    chunks = []
    for c in charts:
        if c.get("failed", False):
            text = c.get("vqa_fallback", "").strip()
            if not text:
                continue
            text = f"Chart {c.get('chart_id','?')} (approximate): {text}"
        else:
            data = c.get("data", "").strip()
            if not data:
                continue
            text = f"Chart {c.get('chart_id','?')} data: {data[:600]}"
        chunks.append({
            "chunk_id":  f"{c.get('chart_id', f'{paper_id}_chart_{len(chunks):04d}')}_data",
            "text":      text, "type": "figure",
            "source_id": c.get("chart_id", ""),
            "page":      c.get("page", -1), "modality": "chart",
        })
    logger.info("[embedder] %s: %d chart chunks.", paper_id, len(chunks))
    return chunks


def _embed_chunks(chunks: List[Dict], model: Any, batch_size: int = 64) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_paper(
    ingestion_result:  Dict[str, Any],
    text_result:       Dict[str, Any],
    figure_result:     Optional[Dict[str, Any]] = None,
    table_result:      Optional[Dict[str, Any]] = None,
    chart_result:      Optional[Dict[str, Any]] = None,
    force_reprocess:   bool = False,
    batch_size:        int  = 64,
) -> Dict[str, Any]:
    if ingestion_result.get("status") == "error":
        return _error_response(
            ingestion_result.get("input_path", ""),
            f"Upstream ingestion failed: {ingestion_result.get('message','')}"
        )

    input_path = ingestion_result.get("input_path", "")
    paper_id   = _paper_id_from(ingestion_result)
    base_paths = _get_paths()
    pkl_path   = _embeddings_pkl_path(paper_id, base_paths)
    json_path  = _chunks_json_path(paper_id, base_paths)

    # Cache check — validate pkl and json are consistent
    if not force_reprocess and pkl_path.exists() and json_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                embeddings = pickle.load(f)
            chunks = json.loads(json_path.read_text())
            if len(embeddings) == len(chunks) and len(embeddings) > 0:
                logger.info(
                    "[embedder] Cache hit for %s — %d embeddings.", paper_id, len(embeddings)
                )
                return {
                    "input_path":  input_path,
                    "output_path": str(pkl_path),
                    "status":      "success",
                    "cached":      True,
                    "metadata": {
                        "paper_id":       paper_id,
                        "total_chunks":   len(chunks),
                        "embedding_dim":  EMBEDDING_DIM,
                        "model_used":     MODEL_NAME,
                        "device_used":    "cpu",
                        "embeddings":     embeddings,
                        # Expose under BOTH keys so any consumer works
                        "text_chunks":    chunks,
                        "chunks":         chunks,
                        "chunk_metadata": chunks,
                        "modality_counts": {
                            m: sum(1 for c in chunks if c.get("modality") == m)
                            for m in ("text", "figure", "table", "chart")
                        },
                    },
                }
            else:
                logger.warning(
                    "[embedder] Cache mismatch (emb=%d, json=%d) — reprocessing.",
                    len(embeddings), len(chunks)
                )
        except Exception as e:
            logger.warning("[embedder] Cache corrupt (%s) — reprocessing.", e)

    t0 = time.time()

    all_chunks: List[Dict] = []
    all_chunks.extend(_build_text_chunks(text_result,    paper_id))
    all_chunks.extend(_build_figure_chunks(figure_result, paper_id))
    all_chunks.extend(_build_table_chunks(table_result,   paper_id))
    all_chunks.extend(_build_chart_chunks(chart_result,   paper_id))

    if not all_chunks:
        return _error_response(
            input_path,
            f"No embeddable content for {paper_id}. "
            "text_result has 0 chunks; check text_extractor output.",
            str(pkl_path),
        )

    logger.info("[embedder] %s: %d total chunks to embed.", paper_id, len(all_chunks))

    try:
        model = _load_model(MODEL_NAME)
    except Exception as e:
        return _error_response(input_path, f"Model load failed: {e}", str(pkl_path))

    try:
        embeddings = _embed_chunks(all_chunks, model, batch_size=batch_size)
    except Exception as e:
        return _error_response(input_path, f"Embedding failed: {e}", str(pkl_path))

    elapsed = time.time() - t0
    logger.info("[embedder] %s: %d chunks in %.1fs.", paper_id, len(all_chunks), elapsed)

    # Serialisable chunk list (strip heavy types)
    chunks_serialisable = [
        {k: v for k, v in c.items()
         if isinstance(v, (str, int, float, list, bool, type(None)))}
        for c in all_chunks
    ]

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        json_path.write_text(json.dumps(chunks_serialisable, indent=2))
        logger.info("[embedder] Saved → %s", pkl_path)
    except Exception as e:
        logger.error("[embedder] Save failed: %s", e)

    modality_counts = {
        m: sum(1 for c in all_chunks if c.get("modality") == m)
        for m in ("text", "figure", "table", "chart")
    }

    return {
        "input_path":  input_path,
        "output_path": str(pkl_path),
        "status":      "success",
        "metadata": {
            "paper_id":        paper_id,
            "total_chunks":    len(all_chunks),
            "embedding_dim":   embeddings.shape[1] if len(embeddings) > 0 else EMBEDDING_DIM,
            "model_used":      MODEL_NAME,
            "device_used":     "cpu",
            "elapsed_seconds": round(elapsed, 2),
            "modality_counts": modality_counts,
            "embeddings":      embeddings,
            # Expose under BOTH keys
            "text_chunks":     chunks_serialisable,
            "chunks":          chunks_serialisable,
            "chunk_metadata":  chunks_serialisable,
        },
    }


def embed_query(query: str) -> np.ndarray:
    model = _load_model(MODEL_NAME)
    vec   = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    return vec[0].astype(np.float32)


def _run_tests() -> None:
    print("Running embedder tests …")

    model = _load_model()
    assert model is not None
    print("  ✅ Model loaded")

    from src.retrieval.embedder import _embed_chunks, EMBEDDING_DIM
    vecs = _embed_chunks([{"text": "Test sentence for embedding."}], model)
    assert vecs.shape == (1, EMBEDDING_DIM)
    assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-5
    print(f"  ✅ Embed shape={vecs.shape}, normalised")

    q = embed_query("What is the main contribution?")
    assert q.shape == (EMBEDDING_DIM,)
    print(f"  ✅ embed_query shape={q.shape}")

    dummy_text = {
        "status": "success",
        "metadata": {
            "text_chunks": [
                {"chunk_id": "p_001", "text": "We propose a new method.", "type": "text", "page": 1}
            ]
        }
    }
    tc = _build_text_chunks(dummy_text, "test")
    assert len(tc) == 1
    print(f"  ✅ _build_text_chunks: {len(tc)} chunk")

    print("\n✅ All embedder tests passed.")


if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _run_tests()

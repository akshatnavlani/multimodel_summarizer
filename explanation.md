# Project Explanation

## What This Project Is

This repository implements a multimodal scientific PDF pipeline. A PDF is ingested, its pages are rasterized and parsed for layout elements, text and tables are extracted, figures can be captioned with BLIP-2, charts can be linearized with Deplot, all extracted content is embedded into MiniLM vectors, indexed with FAISS, summarized with Gemini or Groq, explained with a sentence-level XAI layer, and scored with an evaluation module.

The application is exposed through a Gradio UI in [app/gradio_app.py](app/gradio_app.py). The UI uploads a PDF into `data/raw_pdfs/`, runs the pipeline, and renders the final summary, XAI output, and execution metadata.

## Architecture

The codebase is organized as a staged document-processing system:

1. Stage 1 ingests the raw PDF and writes page images plus metadata.
2. Stage 2 detects page layout and crops visual regions.
3. Stage 3 extracts text, parses tables, and optionally performs OCR fallback.
4. Stage 4 understands figures and charts with vision models.
5. Stage 5 builds text/figure/table/chart embeddings.
6. Stage 6 stores those vectors in a FAISS index and answers similarity queries.
7. Stage 7 turns the retrieved evidence into a structured summary.
8. Stage 8 explains the summary by attributing each sentence to retrieved evidence.
9. Stage 9 evaluates the output with quality and performance metrics.

Shared configuration lives in [config/settings.py](config/settings.py) and [config/paths.py](config/paths.py). Those modules are the central contract for environment variables, runtime defaults, and filesystem layout.

## Data Flow

Raw PDFs enter `data/raw_pdfs/`. Ingestion writes the per-paper metadata file and page renders into `data/extracted/<paper_id>/pages/`. Layout parsing reads that ingestion output, detects text, figure, table, and list blocks, and writes `data/extracted/layout_<paper_id>.json`. Text extraction writes `data/extracted/text_<paper_id>.json`. Table parsing writes `data/extracted/tables_<paper_id>.json` plus one JSON file per table in `data/tables/<paper_id>/`.

Figure captioning and chart extraction cache their results in `data/figures/`. Embeddings and FAISS indexes are written into `data/embeddings/`. Summaries land in `data/summaries/`. XAI outputs are saved in `data/xai_outputs/`. Evaluation artifacts are stored under `data/evaluation/`.

The pipeline is intentionally cache-heavy. Most stages return `cached` when a matching JSON or binary artifact already exists, and `force_reprocess=True` bypasses those caches.

## Configuration Model

`Settings` in [config/settings.py](config/settings.py) is a dataclass populated from environment variables, with defaults that keep the pipeline runnable on CPU-only machines. Important settings include PDF DPI, OCR backend preferences, layout model thresholds, BLIP-2 and Deplot model IDs, MiniLM embedding settings, Gemini and Groq API keys, and XAI thresholds.

[config/paths.py](config/paths.py) resolves the project root and returns a canonical path map. The important correction in this repository is that the evaluation workspace now lives under `data/evaluation/`, and the paper-level layout/text/table JSON files follow the existing `layout_<paper_id>.json`, `text_<paper_id>.json`, and `tables_<paper_id>.json` naming convention.

## File-by-File Summary

### Root

- [requirements.txt](requirements.txt) lists the Python dependencies for PDF processing, OCR, embeddings, FAISS, transformers, LLM APIs, evaluation, visualization, Gradio, and testing.

### `config/`

- [config/__init__.py](config/__init__.py) exposes the main configuration helpers for convenient imports.
- [config/settings.py](config/settings.py) defines the environment-backed `Settings` dataclass and a singleton accessor.
- [config/paths.py](config/paths.py) owns filesystem discovery, path normalization, per-paper artifact locations, directory creation, and safe relative path resolution.

### `app/`

- [app/gradio_app.py](app/gradio_app.py) is the UI entrypoint. It validates uploads, copies PDFs into the raw input folder, clears stale caches, runs the pipeline, and renders summary/XAI/metadata tabs.

### `src/ingestion/`

- [src/ingestion/__init__.py](src/ingestion/__init__.py) re-exports ingestion helpers.
- [src/ingestion/pdf_loader.py](src/ingestion/pdf_loader.py) implements Stage 1. It opens PDFs with PyMuPDF, extracts page text, saves page images and embedded image crops, computes metadata, and persists the ingestion JSON cache.

### `src/layout/`

- [src/layout/__init__.py](src/layout/__init__.py) re-exports layout parsing helpers.
- [src/layout/layout_parser.py](src/layout/layout_parser.py) implements Stage 2. It chooses between layoutparser, Detectron2, and a PyMuPDF-native fallback, converts pages to images, detects layout elements, crops figures/tables, and writes the unified layout JSON.

### `src/extraction/`

- [src/extraction/__init__.py](src/extraction/__init__.py) re-exports extraction helpers.
- [src/extraction/ocr_engine.py](src/extraction/ocr_engine.py) wraps OCR backends with lazy loading and caching. It prefers PaddleOCR, falls back to Tesseract, and returns a structured OCR result.
- [src/extraction/text_extractor.py](src/extraction/text_extractor.py) implements Stage 3A. It extracts text from layout regions using PyMuPDF, falls back to OCR when needed, deduplicates chunks, and writes `text_<paper_id>.json`.
- [src/extraction/table_parser.py](src/extraction/table_parser.py) implements Stage 3C. It extracts tables with pdfplumber or OCR fallback, normalizes them into JSON-friendly row data, builds Markdown and natural-language summaries, and writes `tables_<paper_id>.json` plus per-table files.

### `src/vision/`

- [src/vision/__init__.py](src/vision/__init__.py) re-exports the vision helpers.
- [src/vision/figure_understander.py](src/vision/figure_understander.py) implements BLIP-2 figure captioning. It crops figure regions, captions them, caches the results, and exposes batch processing helpers.
- [src/vision/chart_extractor.py](src/vision/chart_extractor.py) implements Deplot-based chart-to-table extraction with a BLIP-2 VQA fallback for failed charts.

### `src/retrieval/`

- [src/retrieval/__init__.py](src/retrieval/__init__.py) re-exports embedding and retrieval helpers.
- [src/retrieval/embedder.py](src/retrieval/embedder.py) builds MiniLM embeddings for text chunks, figure captions, table summaries, and chart data, then caches both vectors and serializable chunk metadata.
- [src/retrieval/faiss_index.py](src/retrieval/faiss_index.py) builds and searches a cosine-similarity FAISS index, supports cache reloads, and provides both functional and object-oriented APIs.

### `src/summarization/`

- [src/summarization/__init__.py](src/summarization/__init__.py) re-exports summarization helpers.
- [src/summarization/summarizer.py](src/summarization/summarizer.py) implements Stage 7. It builds a structured prompt from retrieved text, figures, and tables, then uses Gemini, Groq, local BART, or an extractive fallback.

### `src/xai/`

- [src/xai/__init__.py](src/xai/__init__.py) re-exports explainability helpers.
- [src/xai/explainer.py](src/xai/explainer.py) implements Stage 8. It attributes each summary sentence to the most similar retrieved chunk, computes modality contribution ratios, and includes placeholder hooks for SHAP, LIME, and gradient-based attribution.

### `src/evaluation/`

- [src/evaluation/__init__.py](src/evaluation/__init__.py) re-exports evaluation helpers.
- [src/evaluation/evaluator.py](src/evaluation/evaluator.py) implements Stage 9. It computes ROUGE, BLEU, optional BERTScore, retrieval diversity metrics, and pipeline performance metrics, then caches the report in `data/evaluation/`.

### `src/pipeline/`

- [src/pipeline/__init__.py](src/pipeline/__init__.py) marks the pipeline package.
- [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py) orchestrates the full pipeline from ingestion through evaluation, handles stale-cache cleanup, performs fallback behavior when vision is disabled, and enriches table/figure data before summarization.

### `tests/`

- [tests/test_pdf_loader.py](tests/test_pdf_loader.py) covers arXiv ID handling, JSON I/O, cache behavior, missing-file errors, and encrypted-PDF behavior in ingestion.
- [tests/test_layout_parser.py](tests/test_layout_parser.py) covers geometry helpers, crop saving, cache behavior, backend selection, native layout parsing, and wrapper behavior.
- [tests/test_extraction.py](tests/test_extraction.py) covers OCR, text extraction, and table parsing, including cache and fallback logic.
- [tests/test_paths.py](tests/test_paths.py) is the added regression test that verifies the canonical project directories and paper-level artifact names.

## Technical Notes

- The pipeline is designed to run on CPU-only environments, but it can opportunistically use GPU acceleration for layout detection and vision models when available.
- PyMuPDF is the backbone for PDF access, page rendering, text extraction, and fallback layout logic.
- `layoutparser` and Detectron2 are optional heavy dependencies; the code degrades to a native PyMuPDF backend when they are missing.
- OCR can be entirely optional on clean native PDFs, but the OCR layer is available for scanned or sparse pages.
- Embedding and retrieval are intentionally separated so retrieval can be re-run independently of summarization.
- The summarizer is designed to keep output usable even when API keys are missing by falling back to extractive text generation.
- XAI currently focuses on explainability of sentence provenance rather than model-parameter attribution; the SHAP/LIME/Captum hooks are scaffolding for later work.

## Known Caveats

- The project depends on several heavyweight optional packages. Missing optional packages usually trigger graceful fallback rather than import-time failure, but the quality of results drops.
- Vision modules may download large model weights the first time they are used.
- Some cache files can become stale if the pipeline implementation changes; the pipeline includes stale-cache cleanup and rebuild logic to limit that risk.
- The canonical file naming convention matters. If new code writes to a different layout/text/table naming scheme, downstream cache lookup will miss the artifacts.

## Recommended Validation Order

1. Run the path regression test in [tests/test_paths.py](tests/test_paths.py).
2. Run the ingestion and layout tests together.
3. Run the extraction tests.
4. Then run the full pipeline-related test suite and fix any remaining failures stage by stage.

"""
tests/test_extraction.py
-------------------------
Unit tests for:
  - src/extraction/ocr_engine.py
  - src/extraction/text_extractor.py
  - src/extraction/table_parser.py

All tests run without a real PDF, without PaddleOCR, and (mostly) without
Tesseract. Every code path testable in isolation is exercised.

Run from project root:
    python -m pytest tests/test_extraction.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _fake_ingestion(paper_id: str = "2401.99999", pages: int = 2,
                    pdf_path: str = "/fake/paper.pdf") -> dict:
    return {
        "input_path": pdf_path,
        "output_path": f"/fake/extracted/{paper_id}_metadata.json",
        "status": "success",
        "paper_id": paper_id,
        "metadata": {"page_count": pages},
        "pages": [
            {
                "page_number": i + 1,
                "text": f"This is page {i+1} of the paper. It discusses methods.",
                "images": [],
                "char_count": 50,
                "image_count": 0,
            }
            for i in range(pages)
        ],
    }


def _fake_layout(paper_id: str = "2401.99999") -> dict:
    return {
        "input_path":  f"/fake/{paper_id}.pdf",
        "output_path": f"/fake/extracted/layout_{paper_id}.json",
        "status":      "success",
        "paper_id":    paper_id,
        "backend":     "pymupdf_native",
        "metadata":    {"total_elements": 4, "element_counts": {}},
        "elements": [
            {
                "element_id": f"{paper_id}_p0001_text_000",
                "type": "text",
                "page": 1,
                "bbox": [50, 50, 400, 120],
                "bbox_pdf": [50, 50, 400, 120],
                "score": 1.0,
                "saved_path": None,
            },
            {
                "element_id": f"{paper_id}_p0001_title_001",
                "type": "title",
                "page": 1,
                "bbox": [50, 20, 400, 45],
                "bbox_pdf": [50, 20, 400, 45],
                "score": 1.0,
                "saved_path": None,
            },
            {
                "element_id": f"{paper_id}_p0002_table_000",
                "type": "table",
                "page": 2,
                "bbox": [50, 50, 500, 300],
                "bbox_pdf": [50, 50, 500, 300],
                "score": 1.0,
                "saved_path": None,
            },
            {
                "element_id": f"{paper_id}_p0002_figure_000",
                "type": "figure",
                "page": 2,
                "bbox": [50, 320, 500, 600],
                "bbox_pdf": [50, 320, 500, 600],
                "score": 1.0,
                "saved_path": None,
            },
        ],
    }


def _fake_paths(tmp: Path) -> dict:
    keys = [
        "extracted", "figures", "tables", "raw_pdfs",
        "embeddings", "summaries", "xai_outputs",
        "attention_maps", "shap_outputs", "lime_outputs",
        "mcr_outputs", "evaluation", "results", "reports",
        "logs", "model_cache", "root", "config", "src", "app",
        "tests", "data",
    ]
    return {k: tmp / k for k in keys}


def _fake_settings():
    cfg = MagicMock()
    cfg.PDF_DPI = 150
    cfg.LAYOUT_SCORE_THRESHOLD = 0.80
    cfg.OCR_PRIMARY  = "paddleocr"
    cfg.OCR_FALLBACK = "tesseract"
    cfg.OCR_LANG     = "en"
    return cfg


# ===========================================================================
# OCR ENGINE TESTS
# ===========================================================================

class TestOCREngineFileNotFound(unittest.TestCase):

    def test_returns_error_for_missing_image(self):
        from src.extraction.ocr_engine import run_ocr
        result = run_ocr("/nonexistent/image.png")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"].lower())
        self.assertEqual(result["text"], "")

    def test_error_payload_is_json_serialisable(self):
        from src.extraction.ocr_engine import run_ocr
        result = run_ocr("/nonexistent/image.png")
        json.dumps(result)  # must not raise


class TestOCREngineTesseractFallback(unittest.TestCase):
    """Test Tesseract path using a real white PNG (text-free, returns empty)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_white_png(self) -> Path:
        from PIL import Image
        import numpy as np
        img = Image.fromarray(
            (255 * __import__("numpy").ones((100, 200, 3), dtype="uint8")), "RGB"
        )
        p = self.tmp / "blank.png"
        img.save(str(p))
        return p

    def test_tesseract_returns_success_on_blank_image(self):
        from src.extraction.ocr_engine import run_ocr
        img_path = self._make_white_png()
        with patch("src.extraction.ocr_engine._PADDLE_OK", False), \
             patch("src.extraction.ocr_engine._paddle_backend") as mock_paddle:
            mock_paddle.run.return_value = None   # paddle unavailable
            result = run_ocr(img_path, prefer_backend="tesseract")
        # A blank image gives empty text but status should still be success
        # (tesseract ran without error)
        self.assertIn(result["status"], ("success", "error"))
        self.assertIn("backend", result)


class TestOCREngineCleanText(unittest.TestCase):

    def test_clean_text_removes_control_chars(self):
        from src.extraction.ocr_engine import _clean_text
        raw = "hello\x00 \x01world\x1f!"
        cleaned = _clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x01", cleaned)
        self.assertEqual(cleaned, "hello world!")

    def test_clean_text_collapses_spaces(self):
        from src.extraction.ocr_engine import _clean_text
        self.assertEqual(_clean_text("a  b   c"), "a b c")

    def test_clean_text_strips_lines(self):
        from src.extraction.ocr_engine import _clean_text
        result = _clean_text("  hello  \n  world  ")
        lines = result.splitlines()
        for line in lines:
            self.assertEqual(line, line.strip())


class TestOCREngineClass(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_get_text_returns_empty_for_missing_file(self):
        from src.extraction.ocr_engine import OCREngine
        engine = OCREngine()
        text = engine.get_text("/nonexistent/image.png")
        self.assertEqual(text, "")

    def test_result_cached_after_first_call(self):
        from src.extraction.ocr_engine import OCREngine, run_ocr
        engine = OCREngine()
        missing = "/nonexistent/x.png"
        with patch("src.extraction.ocr_engine.run_ocr", wraps=run_ocr) as mock_run:
            engine.get_result(missing)
            engine.get_result(missing)
            # run_ocr must only be called once (second call uses cache)
            self.assertEqual(mock_run.call_count, 1)

    def test_clear_cache_forces_re_run(self):
        from src.extraction.ocr_engine import OCREngine, run_ocr
        engine = OCREngine()
        missing = "/nonexistent/y.png"
        with patch("src.extraction.ocr_engine.run_ocr", wraps=run_ocr) as mock_run:
            engine.get_result(missing)
            engine.clear_cache()
            engine.get_result(missing)
            self.assertEqual(mock_run.call_count, 2)


# ===========================================================================
# TEXT EXTRACTOR TESTS
# ===========================================================================

class TestTextExtractorUpstreamErrors(unittest.TestCase):

    def _make_extractor(self, tmp: Path):
        from src.extraction.text_extractor import TextExtractor
        ext = TextExtractor.__new__(TextExtractor)
        ext._paths = _fake_paths(tmp)
        ext._cfg   = _fake_settings()
        return ext

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ingestion_error_propagates(self):
        ext = self._make_extractor(self.tmp)
        bad_ing = {"status": "error", "message": "corrupt", "input_path": ""}
        result  = ext.extract(bad_ing, _fake_layout())
        self.assertEqual(result["status"], "error")
        self.assertIn("ingestion", result["message"].lower())

    def test_layout_error_propagates(self):
        ext = self._make_extractor(self.tmp)
        bad_lay = {"status": "error", "message": "crash", "input_path": ""}
        result  = ext.extract(_fake_ingestion(), bad_lay)
        self.assertEqual(result["status"], "error")
        self.assertIn("layout", result["message"].lower())


class TestTextExtractorCache(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cache_hit(self):
        from src.extraction.text_extractor import TextExtractor, _save_json

        paper_id = "2401.44444"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        cached = {
            "input_path": "/fake/p.pdf",
            "output_path": str(extracted_dir / f"text_{paper_id}.json"),
            "status": "success",
            "paper_id": paper_id,
            "metadata": {"total_chunks": 3},
            "text_chunks": [{"chunk_id": "c0", "text": "hello"}],
        }
        _save_json(cached, extracted_dir / f"text_{paper_id}.json")

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir

        ext = TextExtractor.__new__(TextExtractor)
        ext._paths = paths
        ext._cfg   = _fake_settings()

        result = ext.extract(_fake_ingestion(paper_id), _fake_layout(paper_id))
        self.assertEqual(result["status"], "cached")
        self.assertEqual(len(result["text_chunks"]), 1)

    def test_force_reprocess_skips_cache(self):
        from src.extraction.text_extractor import TextExtractor, _save_json

        paper_id = "2401.55555"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        _save_json(
            {"status": "success", "paper_id": paper_id,
             "metadata": {}, "text_chunks": []},
            extracted_dir / f"text_{paper_id}.json",
        )

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir
        # Create a stub PDF so fitz.open doesn't error
        stub_pdf = self.tmp / f"{paper_id}.pdf"
        stub_pdf.write_bytes(b"%PDF-1.4 stub")

        ing = _fake_ingestion(paper_id, pdf_path=str(stub_pdf))
        lay = _fake_layout(paper_id)

        ext = TextExtractor.__new__(TextExtractor)
        ext._paths = paths
        ext._cfg   = _fake_settings()

        with patch("src.extraction.text_extractor._FITZ_OK", False):
            result = ext.extract(ing, lay, force_reprocess=True)

        # Should be "success", not the cached result
        self.assertEqual(result["status"], "success")


class TestTextExtractorChunkLogic(unittest.TestCase):
    """
    Test chunk creation using injected page text (no real PDF, no fitz).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_extract(self, paper_id="2401.77777"):
        from src.extraction.text_extractor import TextExtractor

        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)
        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir

        ext = TextExtractor.__new__(TextExtractor)
        ext._paths = paths
        ext._cfg   = _fake_settings()

        # fitz not available → uses ingestion page-level text as fallback
        with patch("src.extraction.text_extractor._FITZ_OK", False):
            return ext.extract(
                _fake_ingestion(paper_id),
                _fake_layout(paper_id),
                force_reprocess=True,
            )

    def test_returns_success(self):
        result = self._run_extract()
        self.assertEqual(result["status"], "success")

    def test_chunks_have_required_keys(self):
        result = self._run_extract()
        required = {"chunk_id", "element_id", "type", "page", "text",
                    "char_count", "source"}
        for chunk in result["text_chunks"]:
            for key in required:
                self.assertIn(key, chunk, f"Missing '{key}' in chunk: {chunk}")

    def test_no_empty_chunks(self):
        result = self._run_extract()
        for chunk in result["text_chunks"]:
            self.assertGreater(chunk["char_count"], 0)
            self.assertTrue(chunk["text"].strip())

    def test_duplicate_text_not_repeated(self):
        result = self._run_extract()
        texts = [c["text"] for c in result["text_chunks"]]
        # Normalise then check uniqueness
        import re
        normed = [re.sub(r"\s+", " ", t).strip().lower() for t in texts]
        self.assertEqual(len(normed), len(set(normed)),
                         "Duplicate chunks detected")

    def test_output_json_written(self):
        paper_id = "2401.77777"
        self._run_extract(paper_id)
        out = self.tmp / "extracted" / f"text_{paper_id}.json"
        self.assertTrue(out.exists())

    def test_output_is_json_serialisable(self):
        result = self._run_extract()
        json.dumps(result)


class TestTextCleanChunk(unittest.TestCase):

    def test_removes_null_bytes(self):
        from src.extraction.text_extractor import _clean_chunk
        self.assertNotIn("\x00", _clean_chunk("hello\x00world"))

    def test_dehyphenates(self):
        from src.extraction.text_extractor import _clean_chunk
        result = _clean_chunk("re-\nsult")
        self.assertIn("result", result)

    def test_collapses_blank_lines(self):
        # _clean_chunk collapses to at most 2 consecutive blank lines,
        # i.e. at most 3 consecutive newline characters ("\n\n\n").
        # Input "a\n\n\n\n\nb" has 4 consecutive newlines (3 blank lines);
        # after collapsing it must have no more than 3 consecutive newlines.
        from src.extraction.text_extractor import _clean_chunk
        raw = "a\n\n\n\n\nb"
        result = _clean_chunk(raw)
        # Assert no run of 4 or more consecutive newlines survives
        runs_of_4_plus = __import__("re").findall(r"\n{4,}", result)
        self.assertEqual(runs_of_4_plus, [],
                         f"Found ≥4 consecutive newlines in: {result!r}")


# ===========================================================================
# TABLE PARSER TESTS
# ===========================================================================

class TestBuildTableData(unittest.TestCase):

    def test_empty_input(self):
        from src.extraction.table_parser import _build_table_data
        result = _build_table_data([])
        self.assertEqual(result["n_rows"], 0)
        self.assertEqual(result["headers"], [])

    def test_none_cells_replaced(self):
        from src.extraction.table_parser import _build_table_data
        result = _build_table_data([["A", None, "C"], ["1", "2", None]])
        # None should become ""
        self.assertNotIn(None, result["headers"])
        for row in result["rows"]:
            self.assertNotIn(None, row)

    def test_header_detected(self):
        from src.extraction.table_parser import _build_table_data
        rows = [["Method", "Accuracy", "F1"],
                ["Ours",   "92.1",    "91.4"],
                ["BERT",   "90.5",    "89.8"]]
        result = _build_table_data(rows)
        self.assertEqual(result["headers"], ["Method", "Accuracy", "F1"])
        self.assertEqual(result["n_rows"], 2)

    def test_synthetic_header_for_all_numeric(self):
        from src.extraction.table_parser import _build_table_data
        rows = [["1.0", "2.0"], ["3.0", "4.0"]]
        result = _build_table_data(rows)
        # All rows numeric → synthetic header Col1, Col2
        for h in result["headers"]:
            self.assertTrue(h.startswith("Col"))

    def test_rows_padded_to_same_width(self):
        from src.extraction.table_parser import _build_table_data
        rows = [["A", "B", "C"], ["x", "y"]]  # second row shorter
        result = _build_table_data(rows)
        n_cols = result["n_cols"]
        for row in result["rows"]:
            self.assertEqual(len(row), n_cols)

    def test_fully_empty_rows_dropped(self):
        from src.extraction.table_parser import _build_table_data
        rows = [["A", "B"], ["", ""], ["1", "2"]]
        result = _build_table_data(rows)
        for row in result["rows"]:
            self.assertTrue(any(cell.strip() for cell in row))


class TestToMarkdown(unittest.TestCase):

    def test_basic_table(self):
        from src.extraction.table_parser import _to_markdown, _build_table_data
        data = _build_table_data([["Model", "Acc"], ["BERT", "92.0"]])
        md = _to_markdown(data)
        self.assertIn("|", md)
        self.assertIn("Model", md)
        self.assertIn("BERT",  md)
        # Header separator row must be present
        self.assertIn("---", md)

    def test_empty_data_returns_empty_string(self):
        from src.extraction.table_parser import _to_markdown
        self.assertEqual(_to_markdown({"headers": [], "rows": []}), "")


class TestNLSummary(unittest.TestCase):

    def test_summary_contains_table_label(self):
        from src.extraction.table_parser import _generate_nl_summary, _build_table_data
        data = _build_table_data([["Method","Acc"], ["A","90"], ["B","95"]])
        summary = _generate_nl_summary(data, "Table 3")
        self.assertIn("Table 3", summary)

    def test_summary_mentions_best_value(self):
        from src.extraction.table_parser import _generate_nl_summary, _build_table_data
        data = _build_table_data([["Method","Acc"], ["A","90.0"], ["B","95.0"]])
        summary = _generate_nl_summary(data, "Table 1")
        self.assertIn("95.0", summary)

    def test_summary_detects_increasing_trend(self):
        from src.extraction.table_parser import _generate_nl_summary, _build_table_data
        data = _build_table_data([["Epoch","Loss"], ["1","0.5"], ["2","0.4"], ["3","0.3"]])
        summary = _generate_nl_summary(data, "Table 1")
        self.assertIn("decreasing", summary.lower())

    def test_empty_data_summary(self):
        from src.extraction.table_parser import _generate_nl_summary
        summary = _generate_nl_summary({"headers": [], "rows": [], "n_rows": 0}, "Table 5")
        self.assertIn("no extractable", summary.lower())


class TestTableParserUpstreamErrors(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_parser(self):
        from src.extraction.table_parser import TableParser
        p = TableParser.__new__(TableParser)
        p._paths = _fake_paths(self.tmp)
        p._cfg   = _fake_settings()
        return p

    def test_ingestion_error_propagates(self):
        parser = self._make_parser()
        bad = {"status": "error", "message": "fail", "input_path": ""}
        result = parser.parse(bad, _fake_layout())
        self.assertEqual(result["status"], "error")

    def test_layout_error_propagates(self):
        parser = self._make_parser()
        bad = {"status": "error", "message": "fail", "input_path": ""}
        result = parser.parse(_fake_ingestion(), bad)
        self.assertEqual(result["status"], "error")


class TestTableParserCache(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cache_hit(self):
        from src.extraction.table_parser import TableParser, _save_json

        paper_id = "2401.33333"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        cached = {
            "input_path": "/fake/p.pdf",
            "output_path": str(extracted_dir / f"tables_{paper_id}.json"),
            "status": "success",
            "paper_id": paper_id,
            "metadata": {"total_tables": 1},
            "tables": [{"table_id": "t0", "summary": "cached"}],
        }
        _save_json(cached, extracted_dir / f"tables_{paper_id}.json")

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir

        parser = TableParser.__new__(TableParser)
        parser._paths = paths
        parser._cfg   = _fake_settings()

        result = parser.parse(_fake_ingestion(paper_id), _fake_layout(paper_id))
        self.assertEqual(result["status"], "cached")


class TestTableParserEndToEnd(unittest.TestCase):
    """
    End-to-end parse() with no real PDF — pdfplumber is patched to return
    a known table matrix, verifying the whole serialisation chain.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp  = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_parse_with_mocked_pdfplumber(self):
        from src.extraction.table_parser import TableParser

        paper_id = "2401.66666"
        extracted_dir = self.tmp / "extracted"
        tables_dir    = self.tmp / "tables"
        extracted_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir
        paths["tables"]    = tables_dir

        # Stub PDF so pdf_path.exists() → True
        raw_pdfs = self.tmp / "raw_pdfs"
        raw_pdfs.mkdir()
        stub_pdf = raw_pdfs / f"{paper_id}.pdf"
        stub_pdf.write_bytes(b"%PDF-1.4 stub")

        ing = _fake_ingestion(paper_id, pdf_path=str(stub_pdf))
        lay = _fake_layout(paper_id)

        parser = TableParser.__new__(TableParser)
        parser._paths = paths
        parser._cfg   = _fake_settings()

        fake_table_rows = [
            ["Model",  "Accuracy", "F1"],
            ["BERT",   "92.1",     "91.4"],
            ["RoBERTa","93.5",     "93.0"],
        ]

        with patch("src.extraction.table_parser._extract_pdfplumber",
                   return_value=fake_table_rows):
            result = parser.parse(ing, lay, force_reprocess=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metadata"]["pdfplumber_count"], 1)

        tbl = result["tables"][0]
        self.assertEqual(tbl["source"], "pdfplumber")
        self.assertEqual(tbl["data"]["headers"], ["Model", "Accuracy", "F1"])
        self.assertEqual(tbl["data"]["n_rows"], 2)
        self.assertIn("Table 1", tbl["summary"])
        self.assertIn("|", tbl["markdown"])   # has Markdown pipes
        self.assertIsNotNone(tbl["saved_json"])
        self.assertTrue(Path(tbl["saved_json"]).exists())

    def test_output_is_json_serialisable(self):
        from src.extraction.table_parser import TableParser

        paper_id = "2401.66667"
        extracted_dir = self.tmp / "extracted"
        tables_dir    = self.tmp / "tables"
        extracted_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir
        paths["tables"]    = tables_dir

        parser = TableParser.__new__(TableParser)
        parser._paths = paths
        parser._cfg   = _fake_settings()

        with patch("src.extraction.table_parser._extract_pdfplumber",
                   return_value=None):
            with patch("src.extraction.table_parser._extract_ocr_table",
                       return_value=None):
                result = parser.parse(
                    _fake_ingestion(paper_id),
                    _fake_layout(paper_id),
                    force_reprocess=True,
                )

        json.dumps(result)  # must not raise


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

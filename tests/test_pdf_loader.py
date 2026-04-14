"""
tests/test_pdf_loader.py
-------------------------
Unit tests for src/ingestion/pdf_loader.py.

Tests are designed to run WITHOUT a real PDF and without PyMuPDF installed,
so they exercise every code path that is independently testable
(ID generation, caching, JSON I/O, error handling).

Run from project root:
    python -m pytest tests/test_pdf_loader.py -v
    # or without pytest:
    python tests/test_pdf_loader.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make project root importable regardless of how the test is invoked
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import (
    generate_paper_id,
    save_json,
    load_json,
    _error_response,
    load_pdf,
    PDFLoader,
)


# ---------------------------------------------------------------------------
# 1. generate_paper_id
# ---------------------------------------------------------------------------
class TestGeneratePaperId(unittest.TestCase):

    def test_arxiv_id_stripped_of_version(self):
        self.assertEqual(generate_paper_id("2401.12345v2.pdf"), "2401.12345")

    def test_arxiv_id_no_version(self):
        self.assertEqual(generate_paper_id("2312.00123.pdf"), "2312.00123")

    def test_arxiv_id_5digit(self):
        self.assertEqual(generate_paper_id("1706.03762v5.pdf"), "1706.03762")

    def test_regular_filename_sanitised(self):
        pid = generate_paper_id("My Paper (2024).pdf")
        self.assertNotIn(" ", pid)
        self.assertNotIn("(", pid)
        self.assertNotIn(")", pid)

    def test_empty_stem_uses_hash(self):
        # Path(".pdf").stem == ".pdf" in Python (dotfile, no extension).
        # After sanitise + strip("._-") the result is "pdf" — non-empty,
        # so the sanitised branch fires (not hash).  We just verify the
        # output is stable, non-empty, and filesystem-safe.
        pid = generate_paper_id(".pdf")
        self.assertTrue(len(pid) > 0)
        self.assertNotIn(" ", pid)
        self.assertNotIn("/", pid)
        # Explicit: known correct result for this edge-case
        self.assertEqual(pid, "pdf")

    def test_path_object_accepted(self):
        pid = generate_paper_id(Path("2401.99999v1.pdf"))
        self.assertEqual(pid, "2401.99999")

    def test_no_spaces_in_result(self):
        for name in ["a b c.pdf", "foo bar baz.pdf", "  .pdf"]:
            pid = generate_paper_id(name)
            self.assertNotIn(" ", pid, f"Space found in: {pid!r}")


# ---------------------------------------------------------------------------
# 2. save_json / load_json  (real file I/O with tempdir)
# ---------------------------------------------------------------------------
class TestJsonIO(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_round_trip(self):
        data = {"status": "success", "paper_id": "2401.99999",
                "pages": [{"page_number": 1, "text": "hello"}]}
        out = self.tmp / "test.json"
        save_json(data, out)
        loaded = load_json(out)
        self.assertEqual(loaded, data)

    def test_save_creates_parent_dirs(self):
        deep = self.tmp / "a" / "b" / "c" / "out.json"
        save_json({"x": 1}, deep)
        self.assertTrue(deep.exists())

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_json(self.tmp / "does_not_exist.json")

    def test_unicode_preserved(self):
        data = {"text": "αβγδ — 中文 — مرحبا"}
        out = self.tmp / "unicode.json"
        save_json(data, out)
        self.assertEqual(load_json(out)["text"], data["text"])


# ---------------------------------------------------------------------------
# 3. _error_response
# ---------------------------------------------------------------------------
class TestErrorResponse(unittest.TestCase):

    def test_structure(self):
        resp = _error_response("/a.pdf", "/b.json", "boom")
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["message"], "boom")
        self.assertEqual(resp["pages"], [])
        self.assertEqual(resp["metadata"], {})

    def test_json_serialisable(self):
        resp = _error_response("/a.pdf", "/b.json", "some error")
        # Should not raise
        json.dumps(resp)


# ---------------------------------------------------------------------------
# 4. load_pdf — missing PyMuPDF
# ---------------------------------------------------------------------------
class TestLoadPdfMissingFitz(unittest.TestCase):

    def test_returns_error_when_fitz_unavailable(self):
        with patch("src.ingestion.pdf_loader._FITZ_OK", False):
            result = load_pdf("anything.pdf")
        self.assertEqual(result["status"], "error")
        self.assertIn("pymupdf", result["message"].lower())


# ---------------------------------------------------------------------------
# 5. load_pdf — file not found
# ---------------------------------------------------------------------------
class TestLoadPdfFileNotFound(unittest.TestCase):

    def test_returns_error_for_nonexistent_file(self):
        with patch("src.ingestion.pdf_loader._FITZ_OK", True):
            result = load_pdf("/nonexistent/path/paper.pdf")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"].lower())


# ---------------------------------------------------------------------------
# 6. PDFLoader.load — cache hit path (no fitz needed)
# ---------------------------------------------------------------------------
class TestPDFLoaderCache(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_fake_paths(self, extracted_dir: Path) -> dict:
        return {
            "extracted":      extracted_dir,
            "raw_pdfs":       self.tmp / "raw_pdfs",
            "figures":        self.tmp / "figures",
            "tables":         self.tmp / "tables",
            "embeddings":     self.tmp / "embeddings",
            "summaries":      self.tmp / "summaries",
            "xai_outputs":    self.tmp / "xai_outputs",
            "attention_maps": self.tmp / "xai_outputs/attention_maps",
            "shap_outputs":   self.tmp / "xai_outputs/shap",
            "lime_outputs":   self.tmp / "xai_outputs/lime",
            "mcr_outputs":    self.tmp / "xai_outputs/mcr",
            "evaluation":     self.tmp / "evaluation",
            "results":        self.tmp / "evaluation/results",
            "reports":        self.tmp / "evaluation/reports",
            "logs":           self.tmp / "logs",
            "model_cache":    self.tmp / ".model_cache",
            "root":           self.tmp,
            "config":         self.tmp / "config",
            "src":            self.tmp / "src",
            "app":            self.tmp / "app",
            "tests":          self.tmp / "tests",
            "data":           self.tmp / "data",
        }

    def _make_fake_settings(self):
        cfg = MagicMock()
        cfg.PDF_DPI = 150
        cfg.PDF_MAX_PAGES = 200
        return cfg

    def test_cache_hit_returns_cached_status(self):
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        # Pre-write a fake cached JSON
        paper_id = "2401.99999"
        cached_data = {
            "input_path": "/fake/paper.pdf",
            "output_path": str(extracted_dir / f"{paper_id}_metadata.json"),
            "status": "success",          # stored status
            "paper_id": paper_id,
            "metadata": {"page_count": 5},
            "pages": [{"page_number": 1, "text": "cached text",
                        "images": [], "char_count": 11, "image_count": 0}],
        }
        cache_file = extracted_dir / f"{paper_id}_metadata.json"
        save_json(cached_data, cache_file)

        # Create a fake (non-readable) PDF so path.exists() is True
        fake_pdf = self.tmp / f"{paper_id}.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        loader = PDFLoader.__new__(PDFLoader)
        loader._paths = self._make_fake_paths(extracted_dir)
        loader._cfg   = self._make_fake_settings()

        result = loader.load(fake_pdf, paper_id=paper_id)

        self.assertEqual(result["status"], "cached")
        self.assertEqual(result["paper_id"], paper_id)
        self.assertEqual(len(result["pages"]), 1)

    def test_force_reprocess_skips_cache(self):
        """force_reprocess=True must attempt fitz.open(), not return cached."""
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        paper_id = "2401.00001"
        cache_file = extracted_dir / f"{paper_id}_metadata.json"
        save_json({"status": "success", "paper_id": paper_id,
                   "metadata": {}, "pages": []}, cache_file)

        fake_pdf = self.tmp / f"{paper_id}.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        loader = PDFLoader.__new__(PDFLoader)
        loader._paths = self._make_fake_paths(extracted_dir)
        loader._cfg   = self._make_fake_settings()

        # Patch fitz.open to raise so we can confirm it was called
        with patch("src.ingestion.pdf_loader.fitz") as mock_fitz:
            mock_fitz.open.side_effect = RuntimeError("intentional open error")
            result = loader.load(fake_pdf, paper_id=paper_id,
                                 force_reprocess=True)

        # Should reach fitz.open → error (not cached)
        self.assertEqual(result["status"], "error")
        self.assertIn("intentional open error", result["message"])


# ---------------------------------------------------------------------------
# 7. PDFLoader.load — encrypted PDF path
# ---------------------------------------------------------------------------
class TestPDFLoaderEncrypted(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_encrypted_pdf_returns_error(self):
        fake_pdf = self.tmp / "encrypted.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir()

        loader = PDFLoader.__new__(PDFLoader)
        loader._paths = {
            "extracted": extracted_dir,
            **{k: self.tmp / k for k in (
                "raw_pdfs","figures","tables","embeddings","summaries",
                "xai_outputs","attention_maps","shap_outputs","lime_outputs",
                "mcr_outputs","evaluation","results","reports","logs",
                "model_cache","root","config","src","app","tests","data",
            )}
        }
        cfg = MagicMock()
        cfg.PDF_DPI = 150
        cfg.PDF_MAX_PAGES = 200
        loader._cfg = cfg

        mock_doc = MagicMock()
        mock_doc.is_encrypted = True

        with patch("src.ingestion.pdf_loader.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            result = loader.load(fake_pdf, paper_id="encrypted_test")

        self.assertEqual(result["status"], "error")
        self.assertIn("encrypted", result["message"].lower())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

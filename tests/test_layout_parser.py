"""
tests/test_layout_parser.py
----------------------------
Unit tests for src/layout/layout_parser.py.

All tests run WITHOUT a real PDF, without Detectron2, and without layoutparser.
They exercise every code path testable in isolation:
  - Coordinate helpers
  - Crop/save logic
  - Element ID generation
  - Cache hit / miss behaviour
  - Error propagation from upstream ingestion
  - PyMuPDF-native backend via a mocked fitz document
  - ML-backend failure → native fallback
  - parse_layout convenience wrapper

Run from project root:
    python -m pytest tests/test_layout_parser.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.layout_parser import (
    _make_element_id,
    _pdf_rect_to_img,
    _rects_overlap,
    _save_json,
    _load_json,
    _error_response,
    _crop_and_save,
    LayoutParser,
    parse_layout,
    _PyMuPDFNativeBackend,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _fake_ingestion(paper_id: str = "2401.99999", pages: int = 2) -> dict:
    """Build a minimal ingestion result dict."""
    return {
        "input_path": f"/fake/{paper_id}.pdf",
        "output_path": f"/fake/extracted/{paper_id}_metadata.json",
        "status": "success",
        "paper_id": paper_id,
        "metadata": {"page_count": pages},
        "pages": [
            {
                "page_number": i + 1,
                "text": f"Page {i+1} text.",
                "images": [],
                "char_count": 12,
                "image_count": 0,
            }
            for i in range(pages)
        ],
    }


def _fake_paths(tmp: Path) -> dict:
    """Build a minimal path map pointing into a temp directory."""
    keys = [
        "extracted", "figures", "tables", "raw_pdfs",
        "embeddings", "summaries", "xai_outputs",
        "attention_maps", "shap_outputs", "lime_outputs",
        "mcr_outputs", "evaluation", "results", "reports",
        "logs", "model_cache", "root", "config", "src", "app",
        "tests", "data",
    ]
    return {k: tmp / k for k in keys}


def _fake_settings(dpi: int = 72) -> MagicMock:
    cfg = MagicMock()
    cfg.PDF_DPI = dpi
    cfg.LAYOUT_SCORE_THRESHOLD = 0.80
    return cfg


# ---------------------------------------------------------------------------
# 1. Coordinate helpers
# ---------------------------------------------------------------------------

class TestCoordinateHelpers(unittest.TestCase):

    def test_pdf_rect_to_img_scale_1(self):
        """At scale=1 (72 dpi) result equals input coords."""
        import fitz
        rect = fitz.Rect(10, 20, 100, 200)
        result = _pdf_rect_to_img(rect, scale=1.0)
        self.assertAlmostEqual(result[0], 10.0)
        self.assertAlmostEqual(result[1], 20.0)
        self.assertAlmostEqual(result[2], 100.0)
        self.assertAlmostEqual(result[3], 200.0)

    def test_pdf_rect_to_img_scale_2(self):
        """At scale=2 all coords are doubled."""
        import fitz
        rect = fitz.Rect(10, 20, 50, 80)
        result = _pdf_rect_to_img(rect, scale=2.0)
        self.assertAlmostEqual(result[0], 20.0)
        self.assertAlmostEqual(result[1], 40.0)
        self.assertAlmostEqual(result[2], 100.0)
        self.assertAlmostEqual(result[3], 160.0)

    def test_rects_no_overlap(self):
        import fitz
        r1 = fitz.Rect(0, 0, 10, 10)
        r2 = fitz.Rect(20, 20, 30, 30)
        self.assertFalse(_rects_overlap(r1, r2))

    def test_rects_full_overlap(self):
        import fitz
        r1 = fitz.Rect(0, 0, 10, 10)
        r2 = fitz.Rect(0, 0, 10, 10)
        self.assertTrue(_rects_overlap(r1, r2, threshold=0.99))

    def test_rects_partial_overlap_below_threshold(self):
        import fitz
        r1 = fitz.Rect(0, 0, 10, 10)
        r2 = fitz.Rect(8, 0, 18, 10)  # 20% overlap of r1
        self.assertFalse(_rects_overlap(r1, r2, threshold=0.5))

    def test_rects_partial_overlap_above_threshold(self):
        import fitz
        r1 = fitz.Rect(0, 0, 10, 10)
        r2 = fitz.Rect(2, 0, 12, 10)  # 80% overlap of r1
        self.assertTrue(_rects_overlap(r1, r2, threshold=0.5))

    def test_rects_zero_area_r1(self):
        import fitz
        r1 = fitz.Rect(5, 5, 5, 5)   # degenerate rect
        r2 = fitz.Rect(0, 0, 10, 10)
        self.assertFalse(_rects_overlap(r1, r2))


# ---------------------------------------------------------------------------
# 2. Element ID generation
# ---------------------------------------------------------------------------

class TestMakeElementId(unittest.TestCase):

    def test_format(self):
        eid = _make_element_id("2401.99999", 3, "figure", 7)
        self.assertEqual(eid, "2401.99999_p0003_figure_007")

    def test_zero_padding(self):
        eid = _make_element_id("paper", 1, "text", 0)
        self.assertEqual(eid, "paper_p0001_text_000")

    def test_no_spaces(self):
        eid = _make_element_id("my_paper", 10, "table", 99)
        self.assertNotIn(" ", eid)


# ---------------------------------------------------------------------------
# 3. JSON I/O
# ---------------------------------------------------------------------------

class TestJsonIO(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        data = {"elements": [{"type": "figure", "bbox": [1, 2, 3, 4]}]}
        p = self.tmp / "layout.json"
        _save_json(data, p)
        loaded = _load_json(p)
        self.assertEqual(loaded["elements"][0]["type"], "figure")

    def test_creates_parent_dirs(self):
        deep = self.tmp / "a" / "b" / "layout.json"
        _save_json({"x": 1}, deep)
        self.assertTrue(deep.exists())


# ---------------------------------------------------------------------------
# 4. Error response
# ---------------------------------------------------------------------------

class TestErrorResponse(unittest.TestCase):

    def test_structure(self):
        r = _error_response("/in.pdf", "/out.json", "boom")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["elements"], [])
        self.assertEqual(r["backend"], "none")

    def test_json_serialisable(self):
        r = _error_response("/in.pdf", "/out.json", "boom")
        json.dumps(r)  # must not raise


# ---------------------------------------------------------------------------
# 5. _crop_and_save — without Pillow available / degenerate bbox
# ---------------------------------------------------------------------------

class TestCropAndSave(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_non_visual_type_returns_none(self):
        result = _crop_and_save(
            page_image_path=self.tmp / "page.png",
            bbox=[0, 0, 100, 100],
            element_type="text",
            paper_id="pid",
            page_number=1,
            element_idx=0,
            figures_dir=self.tmp / "figures",
            tables_dir=self.tmp / "tables",
        )
        self.assertIsNone(result)

    def test_missing_page_image_returns_none(self):
        result = _crop_and_save(
            page_image_path=self.tmp / "nonexistent_page.png",
            bbox=[0, 0, 100, 100],
            element_type="figure",
            paper_id="pid",
            page_number=1,
            element_idx=0,
            figures_dir=self.tmp / "figures",
            tables_dir=self.tmp / "tables",
        )
        self.assertIsNone(result)

    def test_valid_crop_saves_png(self):
        """Create a real small PNG, crop it, verify saved file exists."""
        from PIL import Image
        import numpy as np
        # 200x200 white image
        img = Image.fromarray(
            (255 * np.ones((200, 200, 3), dtype="uint8")), mode="RGB"
        )
        page_img_path = self.tmp / "page_0001.png"
        img.save(str(page_img_path))

        figures_dir = self.tmp / "figures"
        result = _crop_and_save(
            page_image_path=page_img_path,
            bbox=[10.0, 10.0, 150.0, 150.0],
            element_type="figure",
            paper_id="pid",
            page_number=1,
            element_idx=0,
            figures_dir=figures_dir,
            tables_dir=self.tmp / "tables",
        )
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

    def test_degenerate_bbox_returns_none(self):
        """A bbox where x1 <= x0 must not crash, must return None."""
        from PIL import Image
        import numpy as np
        img = Image.fromarray(
            (255 * np.ones((100, 100, 3), dtype="uint8")), mode="RGB"
        )
        page_img_path = self.tmp / "page_0001.png"
        img.save(str(page_img_path))

        result = _crop_and_save(
            page_image_path=page_img_path,
            bbox=[50.0, 50.0, 10.0, 80.0],  # x1 < x0 — degenerate
            element_type="figure",
            paper_id="pid",
            page_number=1,
            element_idx=0,
            figures_dir=self.tmp / "figures",
            tables_dir=self.tmp / "tables",
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 6. LayoutParser.parse — upstream error propagation
# ---------------------------------------------------------------------------

class TestLayoutParserUpstreamError(unittest.TestCase):

    def test_returns_error_when_ingestion_failed(self):
        failed_ingestion = {
            "status": "error",
            "message": "PDF not found",
            "input_path": "/bad.pdf",
            "paper_id": "",
        }
        parser = LayoutParser.__new__(LayoutParser)
        tmp = tempfile.mkdtemp()
        parser._paths   = _fake_paths(Path(tmp))
        parser._cfg     = _fake_settings()
        parser._lp_backend  = None
        parser._d2_backend  = None
        parser._pymupdf_backend = _PyMuPDFNativeBackend()

        result = parser.parse(failed_ingestion)
        self.assertEqual(result["status"], "error")
        self.assertIn("upstream", result["message"].lower())


# ---------------------------------------------------------------------------
# 7. LayoutParser.parse — cache hit
# ---------------------------------------------------------------------------

class TestLayoutParserCache(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cache_hit_returns_cached_status(self):
        paper_id = "2401.99999"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        cached_data = {
            "input_path":  f"/fake/{paper_id}.pdf",
            "output_path": str(extracted_dir / f"layout_{paper_id}.json"),
            "status":      "success",
            "paper_id":    paper_id,
            "backend":     "pymupdf_native",
            "metadata":    {"total_elements": 5, "element_counts": {}},
            "elements":    [{"element_id": "e1", "type": "figure"}],
        }
        cache_file = extracted_dir / f"layout_{paper_id}.json"
        _save_json(cached_data, cache_file)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir

        parser = LayoutParser.__new__(LayoutParser)
        parser._paths   = paths
        parser._cfg     = _fake_settings()
        parser._lp_backend  = None
        parser._d2_backend  = None
        parser._pymupdf_backend = _PyMuPDFNativeBackend()

        result = parser.parse(_fake_ingestion(paper_id))
        self.assertEqual(result["status"], "cached")
        self.assertEqual(len(result["elements"]), 1)


# ---------------------------------------------------------------------------
# 8. LayoutParser.parse — force_reprocess ignores cache
# ---------------------------------------------------------------------------

class TestLayoutParserForceReprocess(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_force_reprocess_ignores_cache(self):
        """Even if cache exists, force_reprocess=True must recompute."""
        paper_id = "2401.11111"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        cache_file = extracted_dir / f"layout_{paper_id}.json"
        _save_json({"status": "success", "paper_id": paper_id,
                    "backend": "old", "metadata": {}, "elements": []},
                   cache_file)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir

        parser = LayoutParser.__new__(LayoutParser)
        parser._paths = paths
        parser._cfg   = _fake_settings()
        parser._lp_backend  = None
        parser._d2_backend  = None
        parser._pymupdf_backend = _PyMuPDFNativeBackend()

        # Mock _detect_page to return empty list (no real PDF needed)
        with patch.object(parser, "_detect_page", return_value=[]):
            with patch("src.layout.layout_parser._FITZ_OK", False):
                result = parser.parse(
                    _fake_ingestion(paper_id), force_reprocess=True
                )

        self.assertEqual(result["status"], "success")
        # Must NOT have returned the old cached backend value
        self.assertNotEqual(result.get("backend"), "old")


# ---------------------------------------------------------------------------
# 9. LayoutParser._pick_backend
# ---------------------------------------------------------------------------

class TestPickBackend(unittest.TestCase):

    def _make_parser(self) -> LayoutParser:
        p = LayoutParser.__new__(LayoutParser)
        p._paths = {}
        p._cfg   = _fake_settings()
        p._lp_backend  = None
        p._d2_backend  = None
        p._pymupdf_backend = _PyMuPDFNativeBackend()
        return p

    def test_picks_layoutparser_when_available(self):
        p = self._make_parser()
        with patch("src.layout.layout_parser._LP_OK", True):
            self.assertEqual(p._pick_backend(), "layoutparser")

    def test_picks_detectron2_when_lp_unavailable(self):
        p = self._make_parser()
        with patch("src.layout.layout_parser._LP_OK", False), \
             patch("src.layout.layout_parser._D2_OK", True):
            self.assertEqual(p._pick_backend(), "detectron2")

    def test_falls_back_to_pymupdf_when_neither_available(self):
        p = self._make_parser()
        with patch("src.layout.layout_parser._LP_OK", False), \
             patch("src.layout.layout_parser._D2_OK", False):
            self.assertEqual(p._pick_backend(), "pymupdf_native")


# ---------------------------------------------------------------------------
# 10. LayoutParser.parse — end-to-end with mocked fitz doc
#     (tests the pymupdf_native backend without Detectron2)
# ---------------------------------------------------------------------------

class TestLayoutParserPyMuPDFBackend(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build_mock_fitz_page(self, page_num: int = 1):
        """
        Build a MagicMock that mimics a fitz.Page.

        Returns text blocks, no images, no tables.
        """
        page = MagicMock()

        # Page rect (A4-ish at 72 dpi = 595 x 842 pts)
        rect = MagicMock()
        rect.x0, rect.y0, rect.x1, rect.y1 = 0, 0, 595, 842
        rect.width, rect.height = 595, 842
        type(page).rect = PropertyMock(return_value=rect)

        # Two text blocks
        # Format: (x0, y0, x1, y1, text, block_no, block_type)
        page.get_text.return_value = [
            (50, 50, 300, 80,  "Introduction\n", 0, 0),
            (50, 100, 500, 200, "This paper proposes a new method.\n", 1, 0),
        ]

        # No embedded images
        page.get_images.return_value = []

        # find_tables raises (simulates old fitz or no tables)
        page.find_tables.side_effect = AttributeError("no find_tables")

        return page

    def test_pymupdf_native_elements_created(self):
        """
        Full parse() run using pymupdf_native backend with a mock fitz doc.
        Verifies element dicts are created and have required keys.
        """
        paper_id = "2401.77777"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)
        # Pages dir (pdf_loader renders page PNGs here; layout parser reads them)
        (extracted_dir / paper_id / "pages").mkdir(parents=True, exist_ok=True)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir
        paths["figures"]   = self.tmp / "figures"
        paths["tables"]    = self.tmp / "tables"

        cfg = _fake_settings(dpi=72)

        # Create a stub PDF so pdf_path.exists() returns True,
        # allowing the fitz.open() call to be reached (and intercepted).
        raw_pdfs_dir = self.tmp / "raw_pdfs"
        raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        stub_pdf = raw_pdfs_dir / f"{paper_id}.pdf"
        stub_pdf.write_bytes(b"%PDF-1.4 stub")

        # Build mock fitz document with 2 pages
        mock_doc = MagicMock()
        mock_doc.__getitem__ = lambda self_, idx: self._build_mock_fitz_page(idx + 1)

        parser = LayoutParser.__new__(LayoutParser)
        parser._paths   = paths
        parser._cfg     = cfg
        parser._lp_backend  = None
        parser._d2_backend  = None
        parser._pymupdf_backend = _PyMuPDFNativeBackend()

        # Build ingestion result pointing at the real stub PDF
        ingestion = _fake_ingestion(paper_id, pages=2)
        ingestion["input_path"] = str(stub_pdf)

        # Patch fitz.open to return our mock doc (avoids parsing the stub bytes)
        with patch("src.layout.layout_parser._FITZ_OK", True), \
             patch("src.layout.layout_parser._LP_OK", False), \
             patch("src.layout.layout_parser._D2_OK", False), \
             patch("src.layout.layout_parser.fitz") as mock_fitz:

            mock_fitz.open.return_value = mock_doc
            mock_fitz.Rect = __import__("fitz").Rect  # use real Rect for coords

            result = parser.parse(ingestion, force_reprocess=True)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["backend"], "pymupdf_native")
        self.assertGreaterEqual(len(result["elements"]), 1)

        # Every element must have required keys
        required_keys = {
            "element_id", "type", "page", "bbox", "bbox_pdf", "score", "saved_path"
        }
        for elem in result["elements"]:
            for key in required_keys:
                self.assertIn(key, elem, f"Missing key '{key}' in element: {elem}")

        # Output JSON must be written
        out_path = extracted_dir / f"layout_{paper_id}.json"
        self.assertTrue(out_path.exists())

    def test_output_is_json_serialisable(self):
        """The returned dict must be fully JSON-serialisable."""
        paper_id = "2401.88888"
        extracted_dir = self.tmp / "extracted"
        extracted_dir.mkdir(parents=True)

        paths = _fake_paths(self.tmp)
        paths["extracted"] = extracted_dir
        paths["figures"]   = self.tmp / "figures"
        paths["tables"]    = self.tmp / "tables"

        parser = LayoutParser.__new__(LayoutParser)
        parser._paths   = paths
        parser._cfg     = _fake_settings()
        parser._lp_backend  = None
        parser._d2_backend  = None
        parser._pymupdf_backend = _PyMuPDFNativeBackend()

        with patch.object(parser, "_detect_page", return_value=[
            {"type": "text",   "bbox": [1.0, 2.0, 100.0, 50.0],
             "bbox_pdf": [1.0, 2.0, 100.0, 50.0], "score": 1.0},
            {"type": "figure", "bbox": [10.0, 60.0, 200.0, 200.0],
             "bbox_pdf": [10.0, 60.0, 200.0, 200.0], "score": 0.9},
        ]):
            with patch("src.layout.layout_parser._FITZ_OK", False):
                result = parser.parse(_fake_ingestion(paper_id),
                                      force_reprocess=True)

        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# 11. parse_layout convenience wrapper
# ---------------------------------------------------------------------------

class TestParseLayoutWrapper(unittest.TestCase):

    def test_returns_error_on_upstream_failure(self):
        failed = {
            "status": "error",
            "message": "corrupt PDF",
            "input_path": "/bad.pdf",
            "paper_id": "",
        }
        with patch("src.layout.layout_parser.LayoutParser") as MockLP:
            instance = MockLP.return_value
            instance.parse.return_value = _error_response(
                "/bad.pdf", "", "upstream ingestion error: corrupt PDF"
            )
            result = parse_layout(failed)
        self.assertEqual(result["status"], "error")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

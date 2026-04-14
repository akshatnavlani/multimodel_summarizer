"""
src/extraction/table_parser.py
-------------------------------
Stage 3C — Table Parsing.

Responsibilities
----------------
1. For every ``table`` element in the layout result:
   a. Try pdfplumber to extract the raw cell matrix (native PDFs).
   b. Fall back to OCR on the saved image crop (scanned docs).
2. Serialise each table to:
   - ``data``    : list-of-rows JSON  (row 0 = header)
   - ``markdown``: pipe-delimited Markdown (used in Gemini prompt, Stage 7)
   - ``summary`` : natural-language description using a template
3. Save  data/extracted/tables_{paper_id}.json  and  data/tables/{paper_id}/*.json
4. Return the pipeline contract payload.

Natural-language template (Stage 7 compatible)
----------------------------------------------
The template produces statements that Gemini can cite as ``[Table N]``:

  "Table N compares {headers} across {n_rows} entries.
   Best value in column '{col}': {max_val} ({row_label}).
   Worst value: {min_val} ({row_label}).
   [Optional trend sentence if monotone column detected.]"

Pipeline contract
-----------------
    {
        "input_path":  str,
        "output_path": str,
        "status":      "success" | "cached" | "error",
        "paper_id":    str,
        "metadata":    {
            "total_tables":     int,
            "pdfplumber_count": int,
            "ocr_count":        int
        },
        "tables": [
            {
                "table_id":   str,          # "{paper_id}_table_{idx:03d}"
                "element_id": str,
                "page":       int,
                "source":     "pdfplumber" | "ocr" | "empty",
                "data": {
                    "headers":  list[str],
                    "rows":     list[list[str]],
                    "n_rows":   int,
                    "n_cols":   int
                },
                "markdown":   str,
                "summary":    str,
                "saved_json": str | null     # per-table JSON path
            }
        ]
    }

Dependencies
------------
    pip install pdfplumber pymupdf
    (ocr_engine handles paddleocr / tesseract)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

try:
    import fitz
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False


# ---------------------------------------------------------------------------
# Path / settings helpers
# ---------------------------------------------------------------------------

def _get_paths() -> Dict[str, Path]:
    from config.paths import get_project_paths
    return get_project_paths(create_dirs=True)


def _get_settings():
    from config.settings import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _error_response(input_path: str, output_path: str, msg: str) -> Dict[str, Any]:
    logger.error("[table] ERROR — %s", msg)
    return {
        "input_path":  input_path,
        "output_path": output_path,
        "status":      "error",
        "paper_id":    "",
        "metadata":    {},
        "tables":      [],
        "message":     msg,
    }


# ---------------------------------------------------------------------------
# Table data structures
# ---------------------------------------------------------------------------

def _empty_table_data() -> Dict[str, Any]:
    return {"headers": [], "rows": [], "n_rows": 0, "n_cols": 0}


def _build_table_data(raw_rows: List[List[Optional[str]]]) -> Dict[str, Any]:
    """
    Normalise a raw list-of-rows into a clean table data dict.

    - Row 0 is treated as the header if it has fewer numeric cells than
      subsequent rows; otherwise a synthetic numeric header is created.
    - None cells are replaced with empty strings.
    - Completely empty rows are dropped.
    """
    if not raw_rows:
        return _empty_table_data()

    # Normalise: convert None → ""
    cleaned: List[List[str]] = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in raw_rows
    ]
    # Drop fully empty rows
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return _empty_table_data()

    # Determine header: row 0 if it looks like labels (mostly non-numeric)
    def _numeric_ratio(row: List[str]) -> float:
        if not row:
            return 0.0
        numeric = sum(1 for c in row if re.match(r"^-?\d+\.?\d*$", c.replace(",", "")))
        return numeric / len(row)

    if len(cleaned) > 1 and _numeric_ratio(cleaned[0]) < 0.5:
        headers = cleaned[0]
        rows    = cleaned[1:]
    else:
        n_cols  = max(len(r) for r in cleaned)
        headers = [f"Col{i+1}" for i in range(n_cols)]
        rows    = cleaned

    # Pad short rows so all rows have the same width
    n_cols = max(len(headers), max((len(r) for r in rows), default=0))
    headers = headers + [""] * (n_cols - len(headers))
    rows    = [r + [""] * (n_cols - len(r)) for r in rows]

    return {
        "headers": headers,
        "rows":    rows,
        "n_rows":  len(rows),
        "n_cols":  n_cols,
    }


# ---------------------------------------------------------------------------
# Markdown serialiser
# ---------------------------------------------------------------------------

def _to_markdown(table_data: Dict[str, Any]) -> str:
    """
    Convert table data to pipe-delimited Markdown.

    Example:
        | Method | Acc  | F1   |
        |--------|------|------|
        | Ours   | 92.1 | 91.4 |
    """
    headers = table_data.get("headers", [])
    rows    = table_data.get("rows", [])
    if not headers and not rows:
        return ""

    def _fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [_fmt_row(headers), sep] + [_fmt_row(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Natural-language summary generator
# ---------------------------------------------------------------------------

def _generate_nl_summary(
    table_data: Dict[str, Any],
    table_label: str,
) -> str:
    """
    Produce a concise natural-language summary of the table suitable
    for the Gemini summarisation prompt.

    The template follows Stage 8 (LLM prompt) requirements:
    'The table compares {headers} across {methods}. Best result: {max_cell}.'
    """
    headers = table_data.get("headers", [])
    rows    = table_data.get("rows", [])
    n_rows  = table_data.get("n_rows", 0)

    if not headers or not rows:
        return f"{table_label} contains no extractable data."

    row_label_col = headers[0] if headers else "Entry"
    value_cols    = headers[1:] if len(headers) > 1 else []

    header_str = ", ".join(headers[:6])  # cap at 6 for readability
    parts = [
        f"{table_label} compares {header_str} across {n_rows} entries."
    ]

    # Find best / worst numeric values per value column
    for col_idx, col_name in enumerate(value_cols[:3], start=1):   # cap at 3
        numeric_cells: List[Tuple[float, str]] = []
        for row in rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx].replace(",", "").replace("%", "").strip()
            try:
                val    = float(cell)
                label  = row[0] if row else f"row{col_idx}"
                numeric_cells.append((val, label))
            except ValueError:
                continue

        if len(numeric_cells) < 2:
            continue

        best_val,  best_label  = max(numeric_cells, key=lambda x: x[0])
        worst_val, worst_label = min(numeric_cells, key=lambda x: x[0])
        parts.append(
            f"Best {col_name}: {best_val} ({best_label}); "
            f"Worst: {worst_val} ({worst_label})."
        )

        # Monotone trend detection
        vals = [v for v, _ in numeric_cells]
        if all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
            parts.append(f"{col_name} shows an increasing trend.")
        elif all(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
            parts.append(f"{col_name} shows a decreasing trend.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# pdfplumber extraction
# ---------------------------------------------------------------------------

def _extract_pdfplumber(
    pdf_path: Path,
    page_number: int,      # 1-based
    bbox_pdf: List[float],
) -> Optional[List[List[Optional[str]]]]:
    """
    Extract table cells using pdfplumber for a specific bounding box.

    Returns list-of-rows, or None on failure.
    """
    if not _PDFPLUMBER_OK:
        return None
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return None
            page = pdf.pages[page_number - 1]

            if bbox_pdf and len(bbox_pdf) == 4:
                # Crop to element bounding box before table extraction
                cropped = page.crop(tuple(bbox_pdf))
                tables  = cropped.extract_tables()
            else:
                tables  = page.extract_tables()

            if not tables:
                return None
            # Return the largest table found in the region
            return max(tables, key=lambda t: len(t) * len(t[0]) if t and t[0] else 0)
    except Exception as exc:
        logger.warning("[table] pdfplumber error page %d: %s", page_number, exc)
        return None


# ---------------------------------------------------------------------------
# OCR fallback for table images
# ---------------------------------------------------------------------------

def _extract_ocr_table(image_path: str) -> Optional[List[List[str]]]:
    """
    Last-resort table extraction: run OCR on the saved image crop and
    split by whitespace/alignment into a crude row-column structure.

    Quality is lower than pdfplumber but better than nothing for scanned docs.
    """
    try:
        from src.extraction.ocr_engine import run_ocr
        result = run_ocr(image_path)
        if result.get("status") != "success":
            return None
        text = result.get("text", "").strip()
        if not text:
            return None

        # Split into lines → each line is a row; cells split on 2+ spaces
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cells = re.split(r"\s{2,}", line)
            rows.append(cells)
        return rows if rows else None
    except Exception as exc:
        logger.warning("[table] OCR table extraction error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main TableParser class
# ---------------------------------------------------------------------------

class TableParser:
    """
    Parses table elements from the layout result.

    For each ``table`` element:
    1. pdfplumber (native PDF extraction)
    2. OCR on saved crop (scanned doc fallback)
    3. Serialises to JSON + Markdown + NL summary

    Usage
    -----
    ::

        from src.extraction.table_parser import TableParser

        parser = TableParser()
        result = parser.parse(ingestion_result, layout_result)
        for tbl in result["tables"]:
            print(tbl["table_id"], tbl["summary"])
    """

    def __init__(self) -> None:
        self._paths = _get_paths()
        self._cfg   = _get_settings()

    def parse(
        self,
        ingestion_result: Dict[str, Any],
        layout_result:    Dict[str, Any],
        force_reprocess:  bool = False,
    ) -> Dict[str, Any]:
        """
        Extract and serialise all table elements.

        Parameters
        ----------
        ingestion_result : dict — from pdf_loader.load()
        layout_result    : dict — from layout_parser.parse()
        force_reprocess  : bool

        Returns
        -------
        dict — pipeline contract payload
        """
        for stage, res in (("ingestion", ingestion_result), ("layout", layout_result)):
            if res.get("status") == "error":
                return _error_response(
                    res.get("input_path", ""),
                    "",
                    f"Upstream {stage} error: {res.get('message', 'unknown')}",
                )

        paper_id   = ingestion_result["paper_id"]
        input_path = ingestion_result["input_path"]
        output_json = self._paths["extracted"] / f"tables_{paper_id}.json"

        # ── Cache check ───────────────────────────────────────────────
        if not force_reprocess and output_json.exists():
            logger.info("[table] Cache hit — %s", output_json)
            try:
                cached = _load_json(output_json)
                cached["status"] = "cached"
                return cached
            except Exception as exc:
                logger.warning("[table] Cache corrupt (%s); reprocessing.", exc)

        logger.info("[table] Parsing tables for paper_id=%s", paper_id)

        pdf_path    = Path(input_path)
        tables_dir  = self._paths["tables"] / paper_id
        tables_dir.mkdir(parents=True, exist_ok=True)

        table_elements = [
            e for e in layout_result.get("elements", [])
            if e.get("type") == "table"
        ]

        tables: List[Dict[str, Any]] = []
        pdfplumber_count = 0
        ocr_count        = 0

        for idx, element in enumerate(table_elements):
            table_id  = f"{paper_id}_table_{idx:03d}"
            page_num  = element.get("page", 1)
            bbox_pdf  = element.get("bbox_pdf", [])
            saved_img = element.get("saved_path")   # crop from layout stage

            # Human-readable label for NL summary  (1-based)
            table_label = f"Table {idx + 1}"

            raw_rows: Optional[List[List]] = None
            source = "empty"

            # ── 1. pdfplumber ─────────────────────────────────────────
            if pdf_path.exists():
                raw_rows = _extract_pdfplumber(pdf_path, page_num, bbox_pdf)
                if raw_rows:
                    source = "pdfplumber"
                    pdfplumber_count += 1

            # ── 2. OCR fallback ───────────────────────────────────────
            if not raw_rows and saved_img and Path(saved_img).exists():
                raw_rows = _extract_ocr_table(saved_img)
                if raw_rows:
                    source = "ocr"
                    ocr_count += 1

            # ── Normalise & serialise ─────────────────────────────────
            table_data = _build_table_data(raw_rows or [])
            markdown   = _to_markdown(table_data)
            summary    = _generate_nl_summary(table_data, table_label)

            # Save per-table JSON  (used by downstream TAPAS)
            per_table_json = tables_dir / f"{table_id}.json"
            per_table_payload = {
                "table_id":  table_id,
                "element_id":element.get("element_id", ""),
                "page":      page_num,
                "source":    source,
                "data":      table_data,
                "markdown":  markdown,
                "summary":   summary,
            }
            _save_json(per_table_payload, per_table_json)

            tables.append({
                **per_table_payload,
                "saved_json": str(per_table_json),
            })

            logger.debug(
                "[table] %s — source=%s  rows=%d  cols=%d",
                table_id, source,
                table_data["n_rows"], table_data["n_cols"],
            )

        result: Dict[str, Any] = {
            "input_path":  input_path,
            "output_path": str(output_json),
            "status":      "success",
            "paper_id":    paper_id,
            "metadata": {
                "total_tables":     len(tables),
                "pdfplumber_count": pdfplumber_count,
                "ocr_count":        ocr_count,
            },
            "tables": tables,
        }

        _save_json(result, output_json)
        logger.info(
            "[table] Done — %d tables (pdfplumber=%d, ocr=%d) → %s",
            len(tables), pdfplumber_count, ocr_count, output_json,
        )
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def parse_tables(
    ingestion_result: Dict[str, Any],
    layout_result:    Dict[str, Any],
    force_reprocess:  bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: instantiate TableParser and call parse().

    Example
    -------
    ::

        from src.ingestion.pdf_loader   import load_pdf
        from src.layout.layout_parser   import parse_layout
        from src.extraction.table_parser import parse_tables

        ingestion = load_pdf("data/raw_pdfs/2401.12345.pdf")
        layout    = parse_layout(ingestion)
        tables    = parse_tables(ingestion, layout)
        for t in tables["tables"]:
            print(t["table_id"])
            print(t["summary"])
    """
    return TableParser().parse(ingestion_result, layout_result, force_reprocess)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json as _json
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None or not target.exists():
        print("Usage: python -m src.extraction.table_parser <paper.pdf>")
        print(f"  pdfplumber available: {_PDFPLUMBER_OK}")
        sys.exit(0)

    from src.ingestion.pdf_loader   import load_pdf
    from src.layout.layout_parser   import parse_layout

    ing    = load_pdf(target)
    layout = parse_layout(ing)
    result = parse_tables(ing, layout)

    summary = {k: v for k, v in result.items() if k != "tables"}
    summary["tables_preview"] = [
        {k: v for k, v in t.items() if k in ("table_id","page","source","summary")}
        for t in result["tables"][:5]
    ]
    print(_json.dumps(summary, indent=2, ensure_ascii=False))

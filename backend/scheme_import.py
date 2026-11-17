import logging
import re
import time
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer

from backend.config import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

DEFAULT_SCHEME_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
_ALLOWED_HOST = "www.tn.gov.in"
_REQUEST_TIMEOUT = (10, 30)
_USER_AGENT = "AgriculturalSchemesRAG/1.0 (local document importer)"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_EMPTY_VALUES = {"", "-", "na", "n/a", "not available"}


class SchemeImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SchemeLink:
    title: str
    url: str


@dataclass(frozen=True)
class SchemeRecord:
    title: str
    url: str
    fields: tuple[tuple[str, str], ...]


class _SchemeListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[SchemeLink] = []
        self._anchor: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._anchor = {"href": dict(attrs).get("href") or "", "text": ""}

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor is not None:
            href = self._anchor["href"].strip()
            title = " ".join(self._anchor["text"].split())
            if href and title:
                self.links.append(SchemeLink(title=title, url=href))
            self._anchor = None


class _SchemeDetailParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr" and self._row is None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None and self._cell is None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")
        elif tag == "a" and self._cell is not None:
            href = (dict(attrs).get("href") or "").strip()
            if href:
                self._cell.append(f" ({href[:2048]})")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._cell is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            self.rows.append(self._row)
            self._row = None


def _validate_official_url(url: str, *, detail: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise SchemeImportError(f"Refusing non-official scheme URL: {url}")
    expected_path = "/scheme_details.php" if detail else "/scheme_list.php"
    if parsed.path != expected_path:
        raise SchemeImportError(f"Unexpected scheme page path: {url}")
    if detail and not parse_qs(parsed.query).get("id"):
        raise SchemeImportError(f"Scheme detail URL is missing its id: {url}")


def _get_page(session: requests.Session, url: str, *, detail: bool = False) -> str:
    _validate_official_url(url, detail=detail)
    try:
        response = session.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
            allow_redirects=False,
        )
        if response.is_redirect:
            raise SchemeImportError(f"Refusing a redirect from official page: {url}")
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SchemeImportError(f"Could not retrieve {url}: {exc}") from exc

    _validate_official_url(response.url, detail=detail)
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        raise SchemeImportError(f"Expected an HTML scheme page at {url}")
    return response.text


def find_scheme_links(html_text: str, listing_url: str = DEFAULT_SCHEME_LIST_URL) -> list[SchemeLink]:
    _validate_official_url(listing_url)
    parser = _SchemeListParser()
    parser.feed(html_text)

    links: list[SchemeLink] = []
    seen_ids: set[str] = set()
    for link in parser.links:
        absolute_url = urljoin(listing_url, link.url)
        parsed = urlparse(absolute_url)
        if parsed.path != "/scheme_details.php":
            continue
        _validate_official_url(absolute_url, detail=True)
        scheme_id = parse_qs(parsed.query)["id"][0]
        if scheme_id in seen_ids:
            continue
        seen_ids.add(scheme_id)
        links.append(SchemeLink(title=link.title, url=absolute_url))
    if not links:
        raise SchemeImportError(
            "The listing page did not contain any scheme detail links"
        )
    return links


def parse_scheme_detail(
    html_text: str,
    link: SchemeLink,
) -> SchemeRecord:
    parser = _SchemeDetailParser()
    parser.feed(html_text)
    fields: list[tuple[str, str]] = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        label = row[0].rstrip(":").strip()
        value = " ".join(row[1].split())
        if not label or value.casefold() in _EMPTY_VALUES:
            continue
        fields.append((label, value))

    official_title = next(
        (
            value
            for label, value in fields
            if label.casefold() in {"scheme title/name", "scheme name", "scheme title"}
        ),
        link.title,
    )
    if not fields:
        raise SchemeImportError(f"No scheme detail fields found at {link.url}")
    return SchemeRecord(title=official_title, url=link.url, fields=tuple(fields))


def _pdf_font() -> str:
    candidates = [
        Path.home() / "AppData/Local/Microsoft/Windows/Fonts/Nirmala.ttf",
        Path("C:/Windows/Fonts/Nirmala.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                pdfmetrics.registerFont(TTFont("SchemeUnicode", str(candidate)))
                return "SchemeUnicode"
            except (OSError, ValueError):
                logger.warning("Could not register PDF font %s", candidate)
    return "Helvetica"


def _slug(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.casefold()).strip("-")
    return normalized[:70] or "scheme"


def _invariant_canvas(*args: Any, **kwargs: Any) -> Canvas:
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def write_scheme_pdf(record: SchemeRecord, output_path: Path) -> None:
    font_name = _pdf_font()
    text = " ".join([record.title, *(f"{label} {value}" for label, value in record.fields)])
    if font_name == "Helvetica" and not text.isascii():
        raise SchemeImportError(
            "Scheme content includes non-ASCII text, but no Unicode PDF font was "
            "found. Install Noto Sans Tamil or provide a system Nirmala UI font."
        )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SchemeTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=16,
        leading=20,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#163d2b"),
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "SchemeBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )
    source_style = ParagraphStyle(
        "SchemeSource",
        parent=body_style,
        fontSize=8,
        textColor=colors.HexColor("#444444"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=record.title,
        author="Government of Tamil Nadu scheme portal (locally generated PDF)",
    )
    story: list[Flowable] = [
        Paragraph(escape(record.title), title_style),
        Paragraph(
            "Scheme information transcribed from the Government of Tamil Nadu "
            "scheme portal. Verify current eligibility, funding, and dates with "
            "the relevant department.",
            body_style,
        ),
    ]
    story.extend(
        Paragraph(f"<b>{escape(label)}:</b> {escape(value)}", body_style)
        for label, value in record.fields
    )
    story.extend(
        [
            Spacer(1, 8),
            Paragraph(f"Official source: {escape(record.url)}", source_style),
        ]
    )
    document.build(story, canvasmaker=_invariant_canvas)


def import_tn_schemes(
    *,
    output_dir: Path,
    backend_url: str = DEFAULT_BACKEND_URL,
    source_url: str = DEFAULT_SCHEME_LIST_URL,
    limit: int | None = None,
    delay_seconds: float = 0.5,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    _validate_official_url(source_url)
    if not backend_url.startswith(("http://", "https://")):
        raise ValueError("backend_url must use http:// or https://")

    http = session or requests.Session()
    http.headers.update({"User-Agent": _USER_AGENT})
    listing_html = _get_page(http, source_url)
    scheme_links = find_scheme_links(listing_html, source_url)
    if limit is not None:
        scheme_links = scheme_links[:limit]
    logger.info("Found %d scheme detail pages", len(scheme_links))

    generated: list[Path] = []
    failures: list[str] = []
    for index, link in enumerate(scheme_links):
        try:
            if index:
                time.sleep(delay_seconds)
            detail_html = _get_page(http, link.url, detail=True)
            record = parse_scheme_detail(detail_html, link)
            scheme_id = parse_qs(urlparse(link.url).query)["id"][0]
            output_path = output_dir / f"{_slug(record.title)}-{_slug(scheme_id)}.pdf"
            if output_path.is_file():
                with output_path.open("rb") as existing_pdf:
                    if existing_pdf.read(5) != b"%PDF-":
                        raise SchemeImportError(
                            f"Existing output is not a PDF: {output_path}"
                        )
                logger.info("Reusing existing PDF %s", output_path)
            else:
                write_scheme_pdf(record, output_path)
                logger.info("Created PDF %s", output_path)
            generated.append(output_path)
        except (SchemeImportError, OSError, ValueError) as exc:
            logger.error("Could not import scheme %r: %s", link.title, exc)
            failures.append(f"{link.title}: {exc}")

    results: list[dict[str, Any]] = []
    for start in range(0, len(generated), 10):
        batch = generated[start : start + 10]
        files = []
        try:
            for path in batch:
                content = path.read_bytes()
                if len(content) > MAX_UPLOAD_BYTES:
                    raise SchemeImportError(
                        f"Generated PDF {path.name} exceeds the backend upload size limit"
                    )
                files.append(
                    ("files", (path.name, content, "application/pdf"))
                )
            response = http.post(
                f"{backend_url.rstrip('/')}/upload-pdf",
                files=files,
                timeout=max(_REQUEST_TIMEOUT[1], 600),
            )
            response.raise_for_status()
            payload = response.json()
            batch_results = payload.get("results")
            if not isinstance(batch_results, list):
                raise SchemeImportError("Backend upload response has no results list")
            results.extend(batch_results)
            for result in batch_results:
                logger.info(
                    "%s: %s",
                    result.get("filename", "PDF"),
                    result.get("message", result.get("status", "unknown")),
                )
        except (requests.RequestException, OSError, ValueError, SchemeImportError) as exc:
            failures.append(f"Upload batch starting at {batch[0].name}: {exc}")

    if failures:
        raise SchemeImportError(
            f"Imported {len(results)} PDFs with {len(failures)} failure(s): "
            + "; ".join(failures)
        )
    return results

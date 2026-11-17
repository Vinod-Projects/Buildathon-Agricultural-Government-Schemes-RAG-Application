import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from pypdf import PdfReader

from backend.scheme_import import (
    DEFAULT_SCHEME_LIST_URL,
    SchemeImportError,
    SchemeLink,
    find_scheme_links,
    import_tn_schemes,
    parse_scheme_detail,
    write_scheme_pdf,
)


LISTING_HTML = """
<html><body>
<a href="scheme_details.php?id=MTU2Ng==">Training to Farmers</a>
<a href="scheme_details.php?id=MTU2Ng==">Training to Farmers (duplicate link)</a>
<a href="https://example.com/other.pdf">Unrelated PDF</a>
</body></html>
"""

DETAIL_HTML = """
<html><body><table>
<tr><td><b>Scheme Title/Name:</b></td><td>Training to Farmers</td></tr>
<tr><td><b>Concerned Department:</b></td><td>Agriculture Department</td></tr>
<tr><td><b>Funding Pattern:</b></td><td>Rs. 300 per farmer for 2 days.</td></tr>
<tr><td><b>Eligibility criteria:</b></td><td>Farmers in listed districts.</td></tr>
<tr><td><b>Uploaded File:</b></td><td>-</td></tr>
</table></body></html>
"""


def make_response(url: str, content: str, content_type: str = "text/html") -> Mock:
    response = Mock()
    response.url = url
    response.text = content
    response.headers = {"content-type": content_type}
    response.status_code = 200
    response.is_redirect = False
    response.raise_for_status = Mock()
    return response


def test_listing_parser_keeps_only_unique_official_scheme_detail_pages() -> None:
    links = find_scheme_links(LISTING_HTML, DEFAULT_SCHEME_LIST_URL)

    assert links == [
        SchemeLink(
            title="Training to Farmers",
            url="https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
        )
    ]


def test_listing_parser_rejects_an_unexpected_source_host() -> None:
    with pytest.raises(SchemeImportError, match="non-official"):
        find_scheme_links(
            '<a href="https://example.com/scheme_details.php?id=1">Scheme</a>'
        )


def test_detail_parser_extracts_scheme_fields_and_skips_empty_attachment() -> None:
    record = parse_scheme_detail(
        DETAIL_HTML,
        SchemeLink(
            title="Listed title",
            url="https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
        ),
    )

    assert record.title == "Training to Farmers"
    assert ("Funding Pattern", "Rs. 300 per farmer for 2 days.") in record.fields
    assert ("Eligibility criteria", "Farmers in listed districts.") in record.fields
    assert all(label != "Uploaded File" for label, _ in record.fields)


def test_generated_pdf_contains_source_and_scheme_content(tmp_path: Path) -> None:
    record = parse_scheme_detail(
        DETAIL_HTML,
        SchemeLink(
            title="Training to Farmers",
            url="https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
        ),
    )
    output = tmp_path / "training.pdf"

    write_scheme_pdf(record, output)

    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    assert "Training to Farmers" in extracted
    assert "Funding Pattern" in extracted
    assert "Farmers in listed districts" in extracted
    assert "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==" in extracted


def test_generated_pdf_is_reproducible_for_duplicate_detection(tmp_path: Path) -> None:
    record = parse_scheme_detail(
        DETAIL_HTML,
        SchemeLink(
            title="Training to Farmers",
            url="https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
        ),
    )
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    write_scheme_pdf(record, first)
    write_scheme_pdf(record, second)

    assert first.read_bytes() == second.read_bytes()


def test_import_generates_pdfs_and_uploads_them_to_the_rag_api(
    tmp_path: Path,
) -> None:
    source_url = DEFAULT_SCHEME_LIST_URL
    detail_url = "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng=="
    session = Mock()
    session.headers = {}
    session.get.side_effect = [
        make_response(source_url, LISTING_HTML),
        make_response(detail_url, DETAIL_HTML),
    ]
    upload_response = make_response(
        "http://127.0.0.1:8000/upload-pdf",
        json.dumps(
            {
                "results": [
                    {
                        "filename": "training-to-farmers-mtu2ng.pdf",
                        "status": "processed",
                        "message": "PDF processed",
                    }
                ]
            }
        ),
        "application/json",
    )
    upload_response.json.return_value = {
        "results": [
            {
                "filename": "training-to-farmers-mtu2ng.pdf",
                "status": "processed",
                "message": "PDF processed",
            }
        ]
    }
    session.post.return_value = upload_response

    results = import_tn_schemes(
        output_dir=tmp_path,
        limit=1,
        delay_seconds=0,
        session=session,
    )

    assert results[0]["status"] == "processed"
    generated_pdfs = list(tmp_path.glob("*.pdf"))
    assert len(generated_pdfs) == 1
    assert session.post.call_count == 1
    uploaded_files = session.post.call_args.kwargs["files"]
    assert uploaded_files[0][0] == "files"
    assert uploaded_files[0][1][2] == "application/pdf"
    assert uploaded_files[0][1][1].startswith(b"%PDF-")


def test_import_reuses_existing_generated_pdf(tmp_path: Path) -> None:
    source_url = DEFAULT_SCHEME_LIST_URL
    detail_url = "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng=="
    output_path = tmp_path / "training-to-farmers-mtu2ng.pdf"
    record = parse_scheme_detail(
        DETAIL_HTML,
        SchemeLink(title="Training to Farmers", url=detail_url),
    )
    write_scheme_pdf(record, output_path)
    original_pdf = output_path.read_bytes()

    session = Mock()
    session.headers = {}
    session.get.side_effect = [
        make_response(source_url, LISTING_HTML),
        make_response(detail_url, DETAIL_HTML),
    ]
    upload_response = make_response(
        "http://127.0.0.1:8000/upload-pdf",
        '{"results": []}',
        "application/json",
    )
    upload_response.json.return_value = {"results": []}
    session.post.return_value = upload_response

    import_tn_schemes(
        output_dir=tmp_path,
        limit=1,
        delay_seconds=0,
        session=session,
    )

    uploaded_files = session.post.call_args.kwargs["files"]
    assert uploaded_files[0][1][1] == original_pdf
    assert output_path.read_bytes() == original_pdf


def test_import_rejects_redirects_to_non_official_host(tmp_path: Path) -> None:
    session = Mock()
    session.headers = {}
    redirect = make_response(
        "https://example.com/scheme_list.php?dep_id=Mg==",
        LISTING_HTML,
    )
    redirect.is_redirect = True
    session.get.return_value = redirect

    with pytest.raises(SchemeImportError, match="redirect"):
        import_tn_schemes(
            output_dir=tmp_path,
            limit=1,
            delay_seconds=0,
            session=session,
        )


def test_import_handles_backend_upload_failure(tmp_path: Path) -> None:
    source_url = DEFAULT_SCHEME_LIST_URL
    detail_url = "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng=="
    session = Mock()
    session.headers = {}
    session.get.side_effect = [
        make_response(source_url, LISTING_HTML),
        make_response(detail_url, DETAIL_HTML),
    ]
    session.post.side_effect = requests.ConnectionError("backend unavailable")

    with pytest.raises(SchemeImportError, match="Upload batch"):
        import_tn_schemes(
            output_dir=tmp_path,
            limit=1,
            delay_seconds=0,
            session=session,
        )

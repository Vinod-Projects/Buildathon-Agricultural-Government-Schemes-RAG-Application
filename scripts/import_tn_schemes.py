import argparse
import logging
import sys
from pathlib import Path

from backend.scheme_import import (
    DEFAULT_BACKEND_URL,
    DEFAULT_SCHEME_LIST_URL,
    SchemeImportError,
    import_tn_schemes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local PDFs from Tamil Nadu scheme detail pages and upload "
            "them to the running Agricultural Schemes RAG backend."
        )
    )
    parser.add_argument("--source-url", default=DEFAULT_SCHEME_LIST_URL)
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help="Running FastAPI URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "scheme_pdfs",
        help="Directory for generated PDFs (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Import only the first N listed schemes (useful for a test run)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between scheme detail requests (default: %(default)s)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        results = import_tn_schemes(
            output_dir=args.output_dir,
            backend_url=args.backend_url,
            source_url=args.source_url,
            limit=args.limit,
            delay_seconds=args.delay,
        )
    except (SchemeImportError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1

    processed = sum(item.get("status") == "processed" for item in results)
    duplicates = sum(item.get("status") == "duplicate" for item in results)
    failed = sum(item.get("status") == "failed" for item in results)
    print(
        f"Uploaded {len(results)} scheme PDFs: {processed} processed, "
        f"{duplicates} already indexed, {failed} failed."
    )
    print(f"Generated PDFs are in: {args.output_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

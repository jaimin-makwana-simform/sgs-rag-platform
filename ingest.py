"""CLI: ingest the seed SGS PDFs (and optionally custom uploads) into Azure AI Search.

Usage:
    python ingest.py            # ingest all configured dirs (seed docs + custom_docs)
    python ingest.py --custom   # ingest only the custom_docs folder
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_settings
from src.pdf_loader import discover_pdfs
from src.rag import ingest_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Azure AI Search.")
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Only ingest the custom_docs folder (tagged doc_source=custom).",
    )
    args = parser.parse_args()

    settings = get_settings()
    root = Path(__file__).parent
    custom_dir = (root / settings.custom_docs_dir).resolve()

    # Discover PDFs, then partition into custom vs seed so each is tagged correctly.
    if args.custom:
        pdfs = discover_pdfs([settings.custom_docs_dir], root=root)
    else:
        pdfs = discover_pdfs(settings.docs_dirs_list, root=root)

    if not pdfs:
        print("No PDFs found.")
        return

    custom_pdfs = [p for p in pdfs if p.parent == custom_dir]
    seed_pdfs = [p for p in pdfs if p.parent != custom_dir]

    print(f"Found {len(pdfs)} PDF(s). Ingesting into '{settings.azure_search_index_name}'...")
    for p in pdfs:
        tag = "custom" if p.parent == custom_dir else "seed"
        print(f"  - [{tag}] {p.name}")

    total_files = 0
    total_chunks = 0
    skipped: list[str] = []
    for group, source in ((seed_pdfs, "seed"), (custom_pdfs, "custom")):
        if not group:
            continue
        result = ingest_files(settings, group, doc_source=source)
        total_files += result.files_processed
        total_chunks += result.chunks_uploaded
        skipped.extend(result.skipped_files)

    print("\nDone.")
    print(f"  Files processed : {total_files}")
    print(f"  Chunks uploaded : {total_chunks}")
    if skipped:
        print(f"  Skipped (no text): {', '.join(skipped)}")


if __name__ == "__main__":
    main()

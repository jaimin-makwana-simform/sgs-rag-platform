"""Generate a ground truth evaluation dataset for the SGS RAG chatbot.

Approach (hybrid, human-in-the-loop):
  1. For each source PDF, Azure OpenAI drafts factual / definitional / multi-hop /
     procedural Q&A pairs grounded in that document, citing the page(s) used.
  2. Hand-curated seed questions (cross-document disambiguation, comparative, and
     unanswerable / out-of-scope) are merged in from eval/seed_questions.jsonl.
  3. The combined set is written to eval/ground_truth.jsonl for YOU TO REVIEW.

Every record supports both retrieval scoring (relevant_docs / relevant_pages) and
answer scoring (ground_truth_answer).

Usage:
    python -m eval.generate_ground_truth                 # default allocation (~45 + seeds)
    python -m eval.generate_ground_truth --out eval/gt.jsonl

Requires a configured .env (Azure OpenAI chat deployment).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import AzureOpenAI

from src.config import get_settings
from src.embeddings import build_client
from src.pdf_loader import discover_pdfs, extract_pages

ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = Path(__file__).resolve().parent / "seed_questions.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent / "ground_truth.jsonl"

# How many LLM-drafted questions to request per document (richer docs get more).
# Any discovered PDF not listed here uses DEFAULT_PER_DOC.
PER_DOC_ALLOCATION: dict[str, int] = {
    "SGS-Group-Policy-Anti-Corruption-and-Conflicts-of-Interest.pdf": 8,
    "SGS-Corporate-Social-Responsibilit-Policy.pdf": 5,
    "SGS-Ethical-Reporting-Policy-EN.pdf": 5,
    "SGS-Group-GEER-General-Conditions-of-Inspection-and-Testing-Services-EN.pdf": 5,
    "SGS-Legal-General-Conditions-of-Services.pdf": 4,
    "SGS-General-Conditions-of-Service-for-China-EN.pdf": 4,
    "SGS-Legal-General-Conditions-of-Services-India-IR-A4-EN-13-09-V3.pdf": 4,
    "SGS-General-Conditions-for-Customised-Audit-Services-EN.pdf": 4,
    "SGS-SCS-Disputes-and-Appeals-Policy-and-Process-EN.pdf": 3,
    "SGS-CRS-SAS-Disputes-and-Appeal-Policy-and-Process-A4-EN.pdf": 3,
}
DEFAULT_PER_DOC = 3

GEN_CATEGORIES = ["numeric_factual", "definitional", "multi_hop", "procedural"]

DRAFT_SYSTEM_PROMPT = (
    "You are an expert QA dataset author creating a GROUND TRUTH evaluation set for "
    "a retrieval-augmented chatbot over SGS corporate documents. You write precise, "
    "unambiguous questions whose answers are fully contained in the supplied document, "
    "and you give the exact reference answer plus the page number(s) that support it. "
    "Never invent facts not present in the text."
)

DRAFT_USER_TEMPLATE = """Document file name: {file_name}

The document text is provided below with page markers like [p.1], [p.2], etc.

Create exactly {n} high-quality question/answer pairs that a user might realistically
ask about THIS document. Requirements:
- Answers MUST be fully supported by the text below. Do not use outside knowledge.
- Prefer specific, checkable facts (numbers, dates, thresholds, definitions,
  obligations, process steps) over vague questions.
- Spread questions across these categories where the document supports them:
  {categories}.
- For each question, list the page number(s) whose text supports the answer.
- Make questions self-contained: mention the document/subject so they are not
  ambiguous when mixed with questions about similar documents.

Return ONLY valid JSON of the form:
{{"questions": [
  {{"question": "...",
    "ground_truth_answer": "...",
    "category": "one of {categories}",
    "difficulty": "easy|medium|hard",
    "relevant_pages": [1]}}
]}}

Document text:
{doc_text}
"""


def _page_tagged_text(pdf_path: Path) -> str:
    pages = extract_pages(pdf_path)
    return "\n\n".join(f"[p.{p.page_number}]\n{p.text}" for p in pages)


def _slug(file_name: str) -> str:
    base = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    base = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return base[:40]


def draft_for_document(
    client: AzureOpenAI,
    settings,
    pdf_path: Path,
    n: int,
) -> list[dict]:
    """Ask the chat model to draft n Q&A pairs for one document."""
    doc_text = _page_tagged_text(pdf_path)
    user_prompt = DRAFT_USER_TEMPLATE.format(
        file_name=pdf_path.name,
        n=n,
        categories=", ".join(GEN_CATEGORIES),
        doc_text=doc_text,
    )
    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        temperature=0.4,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    payload = json.loads(response.choices[0].message.content)
    items = payload.get("questions", [])

    slug = _slug(pdf_path.name)
    records: list[dict] = []
    for i, item in enumerate(items, 1):
        category = item.get("category", "numeric_factual")
        if category not in GEN_CATEGORIES:
            category = "numeric_factual"
        pages = item.get("relevant_pages") or []
        pages = [int(p) for p in pages if str(p).strip().isdigit()]
        records.append(
            {
                "id": f"{slug}-{category}-{i}",
                "question": item.get("question", "").strip(),
                "ground_truth_answer": item.get("ground_truth_answer", "").strip(),
                "relevant_docs": [pdf_path.name],
                "relevant_pages": pages,
                "category": category,
                "difficulty": item.get("difficulty", "medium"),
                "answerable": True,
                "notes": "LLM-drafted; review before use.",
            }
        )
    return records


def load_seeds() -> list[dict]:
    if not SEED_FILE.exists():
        return []
    seeds = []
    for line in SEED_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            seeds.append(json.loads(line))
    return seeds


def summarize(records: list[dict]) -> None:
    by_cat: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    answerable = 0
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        for d in r["relevant_docs"] or ["(none)"]:
            by_doc[d] = by_doc.get(d, 0) + 1
        answerable += 1 if r["answerable"] else 0

    print("\n=== Ground truth summary ===")
    print(f"Total questions : {len(records)}")
    print(f"Answerable      : {answerable}")
    print(f"Unanswerable    : {len(records) - answerable}")
    print("\nBy category:")
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:26s} {v}")
    print("\nBy document (relevant_docs):")
    for k, v in sorted(by_doc.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d}  {k}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground truth eval dataset.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path.")
    parser.add_argument(
        "--no-seeds", action="store_true", help="Skip merging seed_questions.jsonl."
    )
    args = parser.parse_args()

    settings = get_settings()
    client = build_client(settings)

    pdfs = discover_pdfs(["."], root=ROOT) + discover_pdfs(["General_Conditions"], root=ROOT)
    pdfs = sorted(set(pdfs))
    print(f"Discovered {len(pdfs)} document(s). Drafting Q&A with "
          f"'{settings.azure_openai_chat_deployment}'...\n")

    all_records: list[dict] = []
    for pdf in pdfs:
        n = PER_DOC_ALLOCATION.get(pdf.name, DEFAULT_PER_DOC)
        print(f"  drafting {n:2d} for {pdf.name}")
        try:
            all_records.extend(draft_for_document(client, settings, pdf, n))
        except Exception as exc:  # noqa: BLE001 - keep going on a single-doc failure
            print(f"    !! failed: {type(exc).__name__}: {exc}")

    if not args.no_seeds:
        seeds = load_seeds()
        print(f"\nMerging {len(seeds)} hand-curated seed question(s).")
        all_records.extend(seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summarize(all_records)
    print(f"\nWrote {len(all_records)} questions -> {out_path}")
    print("NEXT: review the file, fix any inaccurate answers/pages, then use it for eval.")


if __name__ == "__main__":
    main()

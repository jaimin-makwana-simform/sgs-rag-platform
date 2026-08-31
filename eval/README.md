# Ground Truth Dataset — SGS RAG Evaluation

A curated Q&A dataset for evaluating **both retrieval and answer quality** of the
SGS RAG chatbot.

## Files

| File | Purpose |
| --- | --- |
| `seed_questions.jsonl` | Hand-curated, fact-verified questions — the hard cases (cross-document disambiguation, comparative, and unanswerable/out-of-scope). Do not regenerate; edit by hand. |
| `generate_ground_truth.py` | LLM-drafts factual/definitional/multi-hop/procedural questions per document (with citations) and merges the seeds. |
| `ground_truth.jsonl` | The generated output you review and use for evaluation. Created by the generator. |

## Record schema (one JSON object per line)

```json
{
  "id": "seed-gc-phil-law-01",
  "question": "Under the SGS Philippines General Conditions..., which laws govern disputes?",
  "ground_truth_answer": "The substantive laws of the Philippines, Court of Makati; arbitration in Singapore.",
  "relevant_docs": ["SGS-Legal-General-Conditions-of-Services.pdf"],
  "relevant_pages": [3],
  "category": "cross_doc_disambiguation",
  "difficulty": "easy|medium|hard",
  "answerable": true,
  "notes": "why this question exists / caveats"
}
```

- **`relevant_docs` / `relevant_pages`** → score **retrieval** (recall@k, MRR, hit-rate, context precision). Empty for unanswerable questions.
- **`ground_truth_answer`** → score **answer quality** (correctness / faithfulness, e.g. LLM-as-judge). For unanswerable questions the expected behavior is a refusal ("not in the documents").

## Categories

| Category | What it tests |
| --- | --- |
| `numeric_factual` | Precise numbers/dates/thresholds (e.g. 1.5%/month interest, US$20,000 cap) |
| `definitional` | Concept definitions (e.g. "close relative", "PEP") |
| `multi_hop` | Answers spanning multiple sections |
| `procedural` | Process/steps (dispute→appeal, due-diligence request) |
| `cross_doc_disambiguation` | ⭐ Must retrieve the **right** near-duplicate document (governing law differs per GC variant) |
| `comparative` | ⭐ Combine facts from two documents (e.g. sample retention 2 vs 3 months) |
| `unanswerable` | ⭐ Out-of-scope; bot should decline instead of hallucinating |

The ⭐ categories are hand-seeded because LLMs draft them poorly.

## How to generate

Requires a configured `.env` (Azure OpenAI chat deployment).

```bash
# from the project root, inside the venv
python -m eval.generate_ground_truth
```

This writes `eval/ground_truth.jsonl` (~45 LLM-drafted + ~16 seeded ≈ 60 questions)
and prints a distribution summary by category and document.

## ⚠️ Review before trusting

The LLM-drafted rows are marked `"notes": "LLM-drafted; review before use."`.
**Read them and fix any inaccurate answers or page numbers** — a ground truth set is
only as good as its labels. The seed rows are already fact-verified against the PDFs.

## Verified facts used in the seed set (for reference)

| Document | Governing law | Arbitration seat | Sample retention |
| --- | --- | --- | --- |
| Philippines (`SGS-Legal-General-Conditions-of-Services.pdf`) | Philippines (Court of Makati) | Singapore | 2 months |
| China | Switzerland | Paris | 3 months |
| India | Switzerland | Paris | 3 months |
| Customised Audit | England | Paris | — |
| GEER (Inspection & Testing) | Germany | Company's registered office (courts) | — |

> Note: China and India share Swiss law + Paris seat, making them the hardest pair
> to disambiguate — deliberately included.

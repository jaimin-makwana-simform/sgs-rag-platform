"""Evaluation harness with a persisted Default-Foundry-IQ baseline.

Runs a pipeline over the ground-truth dataset, scores each answer with
``azure-ai-evaluation`` (Groundedness, Relevance, Retrieval, Response
Completeness, F1) plus a cheap retrieval recall@k, and aggregates the results.

The Default Foundry IQ result is treated as the **baseline**: it is computed once
and cached (keyed by a fingerprint of the dataset), so every Custom run can be
compared against it without recomputing. ``compare()`` produces per-metric deltas
and an overall verdict (which config wins on most primary metrics).
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import openai
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    ResponseCompletenessEvaluator,
    RetrievalEvaluator,
)

from .config import Settings
from .pipelines import get_pipeline
from .pipelines.base import Answer, Pipeline

# All metrics here are higher-is-better, so Custom − Baseline > 0 means Custom wins.
# LLM-judged metrics are on a 1-5 scale; f1 and recall@k are 0-1. Scales differ
# across metrics but are consistent across pipelines, which is what comparison needs.
PRIMARY_METRICS = ("groundedness", "relevance", "retrieval", "response_completeness")
CHEAP_METRICS = ("f1", "recall_at_k")
ALL_METRICS = PRIMARY_METRICS + CHEAP_METRICS

BASELINE_FILENAME = "baseline_foundry_iq.json"


@dataclass
class EvalReport:
    mode: str
    config: dict
    dataset_path: str
    dataset_fingerprint: str
    n_questions: int
    metrics: dict[str, float | None]  # metric -> mean over rows (None if not run)
    rows: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "EvalReport":
        return cls(**d)


@dataclass
class ComparisonReport:
    baseline: EvalReport
    custom: EvalReport
    deltas: dict[str, float | None]  # custom - baseline per metric
    winners: dict[str, str]  # metric -> "custom" | "baseline" | "tie" | "n/a"
    verdict: str  # overall summary


# --------------------------------------------------------------------------- #
# Dataset + fingerprint
# --------------------------------------------------------------------------- #
def _fingerprint(dataset_path: str | Path) -> str:
    data = Path(dataset_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def load_dataset(dataset_path: str | Path) -> list[dict]:
    rows = []
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Evaluator wiring
# --------------------------------------------------------------------------- #
def _model_config(settings: Settings) -> AzureOpenAIModelConfiguration:
    return AzureOpenAIModelConfiguration(
        azure_endpoint=settings.chat_endpoint,
        api_key=settings.chat_api_key,
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.chat_api_version,
    )


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, openai.RateLimitError) or "429" in text or "rate_limit" in text


def _with_retry(fn: Callable, *, retries: int, base_wait: float, what: str):
    """Call ``fn`` with linear backoff on rate-limit (429) errors.

    Tight quotas (e.g. gpt-5-1's 10K TPM) make bursts of judge/answer calls 429;
    waiting lets the per-minute budget refill instead of aborting the whole run.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == retries:
                raise
            wait = min(60.0, base_wait * (attempt + 1))
            print(f"  rate-limited on {what}; retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)


def _apply_judge_token_cap(settings: Settings) -> None:
    """Lower the eval SDK's oversized reasoning-model completion budget (60000) so a
    single judge call fits within tight per-minute quotas. Best-effort; SDK-internal."""
    try:
        from azure.ai.evaluation._legacy.prompty import _prompty as _eval_prompty

        _eval_prompty.DEFAULT_MAX_COMPLETION_TOKENS_REASONING_MODELS = (
            settings.eval_judge_max_completion_tokens
        )
    except Exception:  # noqa: BLE001, S110 - best-effort; use SDK default if it moves
        pass


def _is_reasoning_model(deployment: str) -> bool:
    """Whether the judge deployment is a reasoning model (gpt-5 / o-series).

    Those models' chat API rejects ``max_tokens`` (needs ``max_completion_tokens``)
    and ``temperature``/``top_p`` — the eval SDK adapts its request only when the
    evaluator is created with ``is_reasoning_model=True``, so we must detect and flag
    it or every LLM-judged metric fails with a 400 and scores come back ``None``.
    """
    name = deployment.lower()
    return name.startswith(("o1", "o3", "o4")) or "gpt-5" in name


def _build_evaluators(
    settings: Settings, metrics: tuple[str, ...]
) -> dict[str, tuple[object, Callable[[dict, Answer, str], dict], str]]:
    """Return metric -> (evaluator, input_builder(row, answer, context), result_key)."""
    mc = _model_config(settings)
    reasoning = _is_reasoning_model(settings.azure_openai_chat_deployment)
    if reasoning:
        _apply_judge_token_cap(settings)
    registry: dict[str, tuple[object, Callable, str]] = {}

    if "groundedness" in metrics:
        registry["groundedness"] = (
            GroundednessEvaluator(mc, is_reasoning_model=reasoning),
            lambda row, ans, ctx: {"query": row["question"], "response": ans.text, "context": ctx},
            "groundedness",
        )
    if "relevance" in metrics:
        registry["relevance"] = (
            RelevanceEvaluator(mc, is_reasoning_model=reasoning),
            lambda row, ans, ctx: {"query": row["question"], "response": ans.text},
            "relevance",
        )
    if "retrieval" in metrics:
        registry["retrieval"] = (
            RetrievalEvaluator(mc, is_reasoning_model=reasoning),
            lambda row, ans, ctx: {"query": row["question"], "context": ctx},
            "retrieval",
        )
    if "response_completeness" in metrics:
        registry["response_completeness"] = (
            ResponseCompletenessEvaluator(mc, is_reasoning_model=reasoning),
            lambda row, ans, ctx: {
                "response": ans.text,
                "ground_truth": row.get("ground_truth_answer", ""),
            },
            "response_completeness",
        )
    if "f1" in metrics:
        registry["f1"] = (
            F1ScoreEvaluator(),
            lambda row, ans, ctx: {
                "response": ans.text,
                "ground_truth": row.get("ground_truth_answer", ""),
            },
            "f1_score",
        )
    return registry


def _extract_score(result: dict, result_key: str) -> float | None:
    """Pull the numeric score from an evaluator's result dict."""
    for key in (result_key, f"{result_key}_score", f"gpt_{result_key}"):
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])
    # fall back to the first numeric value
    for v in result.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _recall_at_k(row: dict, answer: Answer) -> float | None:
    """Fraction of expected docs that appear in the retrieved sources.

    Only meaningful for answerable rows that declare ``relevant_docs``.
    """
    expected = {d.lower() for d in row.get("relevant_docs", []) if d}
    if not expected:
        return None
    retrieved = {s.source_file.lower() for s in answer.sources}
    hits = sum(1 for d in expected if any(d in r or r in d for r in retrieved))
    return hits / len(expected)


# --------------------------------------------------------------------------- #
# Core run
# --------------------------------------------------------------------------- #
def run_evaluation(
    pipeline: Pipeline,
    dataset_path: str | Path,
    settings: Settings,
    *,
    metrics: tuple[str, ...] = ALL_METRICS,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalReport:
    """Run ``pipeline`` over the dataset and score every answer.

    ``limit`` caps the number of questions (useful under tight model quota).
    ``progress(done, total)`` is called after each row if provided.
    """
    rows = load_dataset(dataset_path)
    if limit is not None:
        rows = rows[:limit]

    llm_metrics = tuple(m for m in metrics if m in PRIMARY_METRICS or m == "f1")
    evaluators = _build_evaluators(settings, llm_metrics)

    retry = {"retries": settings.eval_retry_max, "base_wait": settings.eval_retry_wait}

    per_row: list[dict] = []
    for i, row in enumerate(rows):
        question = row["question"]
        answer = _with_retry(
            functools.partial(pipeline.answer, question), what="answer", **retry
        )
        context = "\n\n".join(s.content for s in answer.sources)

        scores: dict[str, float | None] = {}
        for metric, (evaluator, build_inputs, result_key) in evaluators.items():
            try:
                result = _with_retry(
                    functools.partial(evaluator, **build_inputs(row, answer, context)),
                    what=metric, **retry,
                )
                scores[metric] = _extract_score(result, result_key)
            except Exception as exc:  # noqa: BLE001 - one metric failing shouldn't abort
                scores[metric] = None
                scores[f"{metric}_error"] = str(exc)[:200]

        if "recall_at_k" in metrics:
            scores["recall_at_k"] = _recall_at_k(row, answer)

        per_row.append(
            {
                "id": row.get("id"),
                "question": question,
                "answerable": row.get("answerable"),
                "response": answer.text,
                "n_sources": len(answer.sources),
                "scores": scores,
            }
        )
        if progress:
            progress(i + 1, len(rows))

    aggregates = _aggregate(per_row, metrics)
    return EvalReport(
        mode=pipeline.mode,
        config=pipeline.describe_config(),
        dataset_path=str(dataset_path),
        dataset_fingerprint=_fingerprint(dataset_path),
        n_questions=len(rows),
        metrics=aggregates,
        rows=per_row,
    )


def _aggregate(per_row: list[dict], metrics: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in metrics:
        vals = [
            r["scores"][metric]
            for r in per_row
            if r["scores"].get(metric) is not None
        ]
        out[metric] = round(sum(vals) / len(vals), 4) if vals else None
    return out


# --------------------------------------------------------------------------- #
# Persistence + baseline
# --------------------------------------------------------------------------- #
def _results_dir(settings: Settings) -> Path:
    d = Path(settings.eval_results_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_report(settings: Settings, report: EvalReport, filename: str) -> Path:
    path = _results_dir(settings) / filename
    path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return path


def load_cached_baseline(
    settings: Settings, dataset_path: str | Path
) -> EvalReport | None:
    """Return the cached Foundry IQ baseline if it matches the dataset, else None.

    A pure cache read — never calls the backend. Lets callers (e.g. the UI) decide
    whether a baseline run is needed without triggering one.
    """
    baseline_path = _results_dir(settings) / BASELINE_FILENAME
    if not baseline_path.exists():
        return None
    cached = EvalReport.from_json(json.loads(baseline_path.read_text()))
    if cached.dataset_fingerprint == _fingerprint(dataset_path):
        return cached
    return None


def get_or_compute_baseline(
    settings: Settings,
    dataset_path: str | Path,
    *,
    force: bool = False,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalReport:
    """Return the cached Foundry IQ baseline, recomputing it if stale or forced.

    The cache is keyed by the dataset fingerprint, so editing the dataset
    automatically invalidates a stale baseline.
    """
    if not force:
        cached = load_cached_baseline(settings, dataset_path)
        if cached is not None:
            return cached

    pipeline = get_pipeline(settings, "foundry_iq")
    report = run_evaluation(
        pipeline, dataset_path, settings, limit=limit, progress=progress
    )
    save_report(settings, report, BASELINE_FILENAME)
    return report


def compare(baseline: EvalReport, custom: EvalReport) -> ComparisonReport:
    """Diff a Custom run against the baseline, metric by metric."""
    deltas: dict[str, float | None] = {}
    winners: dict[str, str] = {}
    custom_wins = baseline_wins = 0

    for metric in ALL_METRICS:
        b = baseline.metrics.get(metric)
        c = custom.metrics.get(metric)
        if b is None or c is None:
            deltas[metric] = None
            winners[metric] = "n/a"
            continue
        delta = round(c - b, 4)
        deltas[metric] = delta
        if abs(delta) < 1e-6:
            winners[metric] = "tie"
        elif delta > 0:
            winners[metric] = "custom"
            if metric in PRIMARY_METRICS:
                custom_wins += 1
        else:
            winners[metric] = "baseline"
            if metric in PRIMARY_METRICS:
                baseline_wins += 1

    if custom_wins > baseline_wins:
        verdict = f"Custom wins ({custom_wins} vs {baseline_wins} primary metrics)"
    elif baseline_wins > custom_wins:
        verdict = f"Default Foundry IQ wins ({baseline_wins} vs {custom_wins} primary metrics)"
    else:
        verdict = f"Tie on primary metrics ({custom_wins} each)"

    return ComparisonReport(
        baseline=baseline, custom=custom, deltas=deltas, winners=winners, verdict=verdict
    )

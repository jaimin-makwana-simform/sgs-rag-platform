"""Pipeline abstraction shared by every retrieval/answering strategy.

Both the Custom RAG pipeline (local hybrid search) and the Default Foundry IQ
pipeline (Foundry Agent + Knowledge Base) implement the same small interface, so
the UI and the evaluation harness treat them interchangeably. Adding a new
strategy later means adding one `Pipeline` subclass — nothing above it changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..search_index import Retrieved


@dataclass
class Answer:
    """A grounded answer plus the sources it was grounded in."""

    text: str
    sources: list[Retrieved] = field(default_factory=list)


class Pipeline(ABC):
    """Common interface for a retrieval + answering strategy.

    ``mode`` is the stable identifier used in the UI, settings (``RAG_MODE``) and
    persisted evaluation results ("custom" | "foundry_iq").
    """

    mode: str = "base"

    @abstractmethod
    def answer(self, question: str) -> Answer:
        """Answer a question, returning the text and the sources used."""

    @abstractmethod
    def describe_config(self) -> dict:
        """Return a JSON-serializable snapshot of the effective configuration.

        Used to label evaluation runs so every result is self-describing.
        """

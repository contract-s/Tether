"""Tell a human that something needs (or deserves) their attention."""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from tether.hitl.store import ReviewRecord

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def notify(self, review: ReviewRecord) -> None: ...


def message(review: ReviewRecord, base_url: str) -> str:
    if review.status == "info":
        return f"[Tether] FYI on task {review.task_id}: {review.reason}"
    return (f"[Tether] Review needed ({review.level.replace('_', ' ')}): {review.reason}\n"
            f"{base_url}/reviews/{review.id}")


class LogNotifier:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url

    def notify(self, review: ReviewRecord) -> None:
        log.warning(message(review, self.base_url))


class WebhookNotifier:
    """Posts {"text": ...} to a webhook, e.g. a Slack incoming webhook."""

    def __init__(self, url: str, base_url: str = "http://localhost:8000",
                 client: httpx.Client | None = None) -> None:
        self.url = url
        self.base_url = base_url
        self._client = client or httpx.Client(timeout=10)

    def notify(self, review: ReviewRecord) -> None:
        try:
            self._client.post(self.url, json={"text": message(review, self.base_url)})
        except httpx.HTTPError as e:  # a failed notification must not break the task
            log.error("review notification failed: %s", e)

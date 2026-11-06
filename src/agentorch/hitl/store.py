"""Durable records of tasks and the review queue (Postgres in production, SQLite in tests).

The LangGraph checkpointer stores each run's full state; these tables are the queryable
index on top of it: which tasks exist, what status they're in, and which human reviews
are waiting, with the decisions people made.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import JSON, Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    task: Mapped[str] = mapped_column(Text)
    require_review: Mapped[bool] = mapped_column(Boolean, default=False)
    # queued | running | awaiting_review | done | failed | rejected
    status: Mapped[str] = mapped_column(String(32), index=True)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[float] = mapped_column(Float)


class ReviewRecord(Base):
    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)  # EscalationRequest.id
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    trigger: Mapped[str] = mapped_column(String(32), index=True)
    level: Mapped[str] = mapped_column(String(32))
    # pending (waiting on a human) | resolved | info (a notify-level message)
    status: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str] = mapped_column(Text)
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    decision: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    resolved_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class ReviewMessage(Base):
    """The reviewer's Q&A with the agents about a review."""

    __tablename__ = "review_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String(160), index=True)
    role: Mapped[str] = mapped_column(String(16))  # human | agent
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[float] = mapped_column(Float)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # One shared in-memory database across threads (tests).
        return create_engine(url, connect_args={"check_same_thread": False},
                             poolclass=StaticPool)
    return create_engine(url, pool_pre_ping=True)


class HitlStore:
    def __init__(self, engine: Engine) -> None:
        Base.metadata.create_all(engine)
        self._session = sessionmaker(engine, expire_on_commit=False)

    # --- tasks -----------------------------------------------------------------

    def create_task(
        self, task_id: str, user_id: str, task: str, require_review: bool
    ) -> TaskRecord:
        now = time.time()
        record = TaskRecord(id=task_id, user_id=user_id, task=task,
                            require_review=require_review, status="queued",
                            created_at=now, updated_at=now)
        with self._session.begin() as s:
            s.add(record)
        return record

    def update_task(self, task_id: str, **fields: Any) -> None:
        with self._session.begin() as s:
            record = s.get(TaskRecord, task_id)
            if record is None:
                raise KeyError(task_id)
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = time.time()

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._session() as s:
            return s.get(TaskRecord, task_id)

    def list_tasks(self, user_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        q = select(TaskRecord).order_by(TaskRecord.created_at.desc()).limit(limit)
        if user_id:
            q = q.where(TaskRecord.user_id == user_id)
        with self._session() as s:
            return list(s.scalars(q))

    # --- reviews ---------------------------------------------------------------

    def add_review(
        self, request: dict[str, Any], status: str = "pending"
    ) -> tuple[ReviewRecord, bool]:
        """Insert unless it already exists. Returns (record, created)."""
        with self._session.begin() as s:
            existing = s.get(ReviewRecord, request["id"])
            if existing is not None:
                return existing, False
            record = ReviewRecord(
                id=request["id"], task_id=request["task_id"], user_id=request["user_id"],
                trigger=request["trigger"], level=request["level"], status=status,
                reason=request["reason"], request=request, created_at=time.time(),
            )
            s.add(record)
            return record, True

    def get_review(self, review_id: str) -> ReviewRecord | None:
        with self._session() as s:
            return s.get(ReviewRecord, review_id)

    def list_reviews(
        self, status: str | None = None, task_id: str | None = None, limit: int = 100
    ) -> list[ReviewRecord]:
        q = select(ReviewRecord).order_by(ReviewRecord.created_at.desc()).limit(limit)
        if status:
            q = q.where(ReviewRecord.status == status)
        if task_id:
            q = q.where(ReviewRecord.task_id == task_id)
        with self._session() as s:
            return list(s.scalars(q))

    def resolve_review(self, review_id: str, decision: dict[str, Any]) -> ReviewRecord:
        """Record a decision. Raises ValueError if the review isn't pending (double submit)."""
        with self._session.begin() as s:
            record = s.get(ReviewRecord, review_id, with_for_update=True)
            if record is None:
                raise KeyError(review_id)
            if record.status != "pending":
                raise ValueError(f"review {review_id} is already {record.status}")
            record.status = "resolved"
            record.decision = decision
            record.reviewer = decision.get("reviewer")
            record.resolved_at = time.time()
            return record

    def add_message(
        self, review_id: str, role: str, content: str, author: str | None = None
    ) -> ReviewMessage:
        message = ReviewMessage(review_id=review_id, role=role, content=content,
                                author=author, created_at=time.time())
        with self._session.begin() as s:
            s.add(message)
        return message

    def messages(self, review_id: str) -> list[ReviewMessage]:
        q = (select(ReviewMessage).where(ReviewMessage.review_id == review_id)
             .order_by(ReviewMessage.id))
        with self._session() as s:
            return list(s.scalars(q))

    def past_decisions(
        self, user_id: str, trigger: str, level: str, exclude_id: str | None = None,
        limit: int = 5,
    ) -> list[ReviewRecord]:
        """Earlier resolved reviews of the same kind for this user (precedent for reviewers)."""
        q = (select(ReviewRecord)
             .where(ReviewRecord.user_id == user_id, ReviewRecord.trigger == trigger,
                    ReviewRecord.level == level, ReviewRecord.status == "resolved")
             .order_by(ReviewRecord.resolved_at.desc()).limit(limit))
        if exclude_id:
            q = q.where(ReviewRecord.id != exclude_id)
        with self._session() as s:
            return list(s.scalars(q))

"""Shared notes board: lets specialists working on the same task pass findings around.

These tools need to know which task they belong to, so they read the task's working
memory from the ToolContext the registry passes in.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentorch.tools.registry import ToolContext


class NotesWriteInput(BaseModel):
    text: str = Field(
        description="A finding other specialists on this task may need: a fact with its "
        "source, a dead end to avoid, or a decision. One note per finding."
    )


class NotesWriteOutput(BaseModel):
    total_notes: int


class NotesReadInput(BaseModel):
    pass


class Note(BaseModel):
    author: str
    text: str


class NotesReadOutput(BaseModel):
    notes: list[Note]


def write_note(args: NotesWriteInput, ctx: ToolContext) -> NotesWriteOutput:
    if ctx.working_memory is None:
        raise RuntimeError("no working memory for this task")
    total = ctx.working_memory.add_note(f"{ctx.caller.value}:{ctx.subtask_id}", args.text)
    return NotesWriteOutput(total_notes=total)


def read_notes(args: NotesReadInput, ctx: ToolContext) -> NotesReadOutput:
    if ctx.working_memory is None:
        raise RuntimeError("no working memory for this task")
    return NotesReadOutput(notes=[Note(author=n["author"], text=n["text"])
                                  for n in ctx.working_memory.notes()])

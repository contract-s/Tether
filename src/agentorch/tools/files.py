"""File read/write confined to the agent workspace directory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

MAX_READ_BYTES = 200_000


class FileReadInput(BaseModel):
    path: str = Field(description="Path relative to the workspace, e.g. 'notes/sources.md'")


class FileReadOutput(BaseModel):
    path: str
    content: str
    truncated: bool


class FileWriteInput(BaseModel):
    path: str = Field(description="Path relative to the workspace")
    content: str
    overwrite: bool = Field(description="Replace the file if it already exists")


class FileWriteOutput(BaseModel):
    path: str
    bytes_written: int


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"{relative!r} is outside the workspace")
        return target

    def read(self, args: FileReadInput) -> FileReadOutput:
        path = self.resolve(args.path)
        data = path.read_bytes()
        truncated = len(data) > MAX_READ_BYTES
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return FileReadOutput(path=args.path, content=text, truncated=truncated)

    def write(self, args: FileWriteInput) -> FileWriteOutput:
        path = self.resolve(args.path)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{args.path!r} exists; set overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = args.content.encode("utf-8")
        path.write_bytes(data)
        return FileWriteOutput(path=args.path, bytes_written=len(data))

"""Sandboxed Python execution in a throwaway Docker container.

The container has no network, a read-only root filesystem, capped memory/CPU/processes,
runs as an unprivileged user, and is killed after a timeout.
"""

from __future__ import annotations

import subprocess
import uuid

from pydantic import BaseModel, Field

MAX_OUTPUT_CHARS = 20_000


class CodeExecInput(BaseModel):
    code: str = Field(description="Python 3.11 source to run. Print results to stdout.")


class CodeExecOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class DockerSandbox:
    def __init__(self, image: str, timeout_seconds: int) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds

    def run(self, args: CodeExecInput) -> CodeExecOutput:
        name = f"tether-sandbox-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            self.image, "python", "-I", "-",
        ]
        try:
            proc = subprocess.run(
                cmd, input=args.code, capture_output=True, text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return CodeExecOutput(
                stdout=_clip(e.stdout), stderr=_clip(e.stderr), exit_code=-1, timed_out=True
            )
        return CodeExecOutput(
            stdout=_clip(proc.stdout), stderr=_clip(proc.stderr),
            exit_code=proc.returncode, timed_out=False,
        )


def _clip(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + "\n[truncated]"

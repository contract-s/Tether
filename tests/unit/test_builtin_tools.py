import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text

from agentorch.tools.code_exec import CodeExecInput, DockerSandbox
from agentorch.tools.db_query import DbQueryInput, ReadOnlyDatabase
from agentorch.tools.files import FileReadInput, FileWriteInput, Workspace
from agentorch.tools.http import HttpClient, HttpRequestInput, is_sensitive

# --- files ---------------------------------------------------------------

def test_workspace_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(FileWriteInput(path="notes/a.md", content="hello", overwrite=False))
    assert ws.read(FileReadInput(path="notes/a.md")).content == "hello"


def test_workspace_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(FileWriteInput(path="a.txt", content="1", overwrite=False))
    with pytest.raises(FileExistsError):
        ws.write(FileWriteInput(path="a.txt", content="2", overwrite=False))


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
def test_workspace_blocks_paths_outside_root(tmp_path: Path, path: str) -> None:
    with pytest.raises(PermissionError):
        Workspace(tmp_path / "ws").resolve(path)


# --- http ----------------------------------------------------------------

def test_http_get_uses_client_and_non_get_is_sensitive() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=f"{req.method} ok"))
    client = HttpClient(httpx.Client(transport=transport), allow_private=True)
    args = HttpRequestInput(method="GET", url="https://example.com", headers=[], body=None)
    assert client.request(args).body == "GET ok"
    assert not is_sensitive(args)
    assert is_sensitive(args.model_copy(update={"method": "POST"}))


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1", "file:///etc/passwd"])
def test_http_blocks_private_and_non_http_urls(url: str) -> None:
    with pytest.raises((PermissionError, ValueError)):
        HttpClient().request(HttpRequestInput(method="GET", url=url, headers=[], body=None))


# --- db ------------------------------------------------------------------

@pytest.fixture
def sqlite_db() -> ReadOnlyDatabase:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO t VALUES (1, 'a'), (2, 'b')"))
    return ReadOnlyDatabase(engine=engine)


def test_db_select(sqlite_db: ReadOnlyDatabase) -> None:
    out = sqlite_db.query(DbQueryInput(sql="SELECT id, name FROM t ORDER BY id"))
    assert out.columns == ["id", "name"] and out.rows == [[1, "a"], [2, "b"]]


@pytest.mark.parametrize("sql", ["DELETE FROM t", "SELECT 1; DROP TABLE t", "UPDATE t SET id=0"])
def test_db_rejects_writes(sqlite_db: ReadOnlyDatabase, sql: str) -> None:
    with pytest.raises(PermissionError):
        sqlite_db.query(DbQueryInput(sql=sql))


# --- code exec (needs Docker) -------------------------------------------

def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


docker = pytest.mark.skipif(not _docker_available(), reason="Docker is not running")


@docker
def test_sandbox_runs_code() -> None:
    out = DockerSandbox("python:3.11-slim", 60).run(CodeExecInput(code="print(sum(range(10)))"))
    assert out.exit_code == 0 and out.stdout.strip() == "45"


@docker
def test_sandbox_has_no_network() -> None:
    code = "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)"
    out = DockerSandbox("python:3.11-slim", 60).run(CodeExecInput(code=code))
    assert out.exit_code != 0


@docker
def test_sandbox_times_out() -> None:
    out = DockerSandbox("python:3.11-slim", 3).run(CodeExecInput(code="while True: pass"))
    assert out.timed_out

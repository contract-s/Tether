import pytest
from pydantic import BaseModel

from tether.agents.schemas import Specialist as S
from tether.tools.defaults import build_default_registry
from tether.tools.registry import (
    ApprovalRequiredError,
    RateLimit,
    RateLimitExceededError,
    Tool,
    ToolPermissionError,
    ToolRegistry,
    UnknownToolError,
    strict_schema,
)


class EchoIn(BaseModel):
    text: str
    loud: bool = False


class EchoOut(BaseModel):
    text: str


def echo(args: EchoIn) -> EchoOut:
    if args.text == "boom":
        raise RuntimeError("exploded")
    return EchoOut(text=args.text.upper() if args.loud else args.text)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_registry(**tool_kwargs) -> tuple[ToolRegistry, FakeClock]:
    clock = FakeClock()
    registry = ToolRegistry(clock=clock)
    registry.register(Tool(
        name="echo", description="Echo text", input_model=EchoIn, output_model=EchoOut,
        handler=echo, allowed_specialists=frozenset({S.RESEARCH}), **tool_kwargs,
    ))
    return registry, clock


def test_successful_call_is_logged() -> None:
    registry, _ = make_registry()
    record = registry.invoke("echo", S.RESEARCH, {"text": "hi", "loud": True})
    assert record.success and record.output == {"text": "HI"}
    assert registry.log.calls == [record]


def test_handler_exception_is_captured_not_raised() -> None:
    registry, _ = make_registry()
    record = registry.invoke("echo", S.RESEARCH, {"text": "boom"})
    assert not record.success
    assert "RuntimeError: exploded" in record.error


def test_invalid_input_is_captured() -> None:
    registry, _ = make_registry()
    record = registry.invoke("echo", S.RESEARCH, {"wrong": 1})
    assert not record.success and record.error.startswith("invalid input")


def test_unknown_tool_and_permission_checks() -> None:
    registry, _ = make_registry()
    with pytest.raises(UnknownToolError):
        registry.invoke("nope", S.RESEARCH, {})
    with pytest.raises(ToolPermissionError):
        registry.invoke("echo", S.WRITING, {"text": "hi"})


def test_rate_limit_uses_sliding_window() -> None:
    registry, clock = make_registry(rate_limit=RateLimit(calls=2, per_seconds=10))
    registry.invoke("echo", S.RESEARCH, {"text": "1"})
    registry.invoke("echo", S.RESEARCH, {"text": "2"})
    with pytest.raises(RateLimitExceededError):
        registry.invoke("echo", S.RESEARCH, {"text": "3"})
    assert registry.log.calls[-1].error.startswith("blocked")
    clock.now = 10.0
    assert registry.invoke("echo", S.RESEARCH, {"text": "4"}).success


def test_sensitive_calls_need_approval() -> None:
    registry, _ = make_registry(is_sensitive=lambda args: args.loud)
    assert registry.invoke("echo", S.RESEARCH, {"text": "quiet"}).success
    with pytest.raises(ApprovalRequiredError):
        registry.invoke("echo", S.RESEARCH, {"text": "x", "loud": True})
    record = registry.invoke("echo", S.RESEARCH, {"text": "x", "loud": True}, approved=True)
    assert record.success and record.sensitive


def test_strict_schema_closes_objects_and_requires_all_fields() -> None:
    class Inner(BaseModel):
        title: str  # a property literally named "title" must survive

    class Outer(BaseModel):
        items: list[Inner]
        flag: bool = False

    schema = strict_schema(Outer)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items", "flag"]
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False and "title" in inner["properties"]


def test_default_registry_access_map() -> None:
    registry = build_default_registry()
    names = lambda s: {t.name for t in registry.for_specialist(s)}  # noqa: E731
    shared = {"file_read", "notes_read", "notes_write"}
    assert names(S.RESEARCH) == shared | {"web_search", "http_request"}
    assert names(S.WRITING) == shared | {"file_write"}
    assert names(S.ANALYSIS) == shared | {"file_write", "code_exec", "db_query"}
    assert names(S.CODE) == shared | {"http_request", "file_write", "code_exec"}
    for definition in [d for s in S for d in registry.anthropic_tools(s)]:
        assert definition["strict"] is True
        assert definition["input_schema"]["additionalProperties"] is False


def test_strict_schema_rejects_free_form_maps() -> None:
    class WithMap(BaseModel):
        headers: dict[str, str]

    with pytest.raises(ValueError, match="free-form maps"):
        strict_schema(WithMap)

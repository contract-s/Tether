"""Outbound HTTP calls. Anything other than GET counts as sensitive (external side effects)."""

from __future__ import annotations

import ipaddress
import socket
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

MAX_BODY_CHARS = 50_000


class HttpRequestInput(BaseModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    url: str = Field(description="Absolute http(s) URL")
    headers: dict[str, str] = Field(description="Request headers; {} for none")
    body: str | None = Field(description="Request body for POST/PUT/PATCH, else null")


class HttpRequestOutput(BaseModel):
    status_code: int
    headers: dict[str, str]
    body: str
    truncated: bool


def is_sensitive(args: HttpRequestInput) -> bool:
    return args.method != "GET"


def _check_public_url(url: str) -> None:
    """Block requests to localhost and private networks (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"only absolute http(s) URLs are allowed, got {url!r}")
    for info in socket.getaddrinfo(parsed.hostname, None):
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global:
            raise PermissionError(f"{parsed.hostname} resolves to non-public address {addr}")


class HttpClient:
    def __init__(self, client: httpx.Client | None = None, allow_private: bool = False) -> None:
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=False)
        self._allow_private = allow_private

    def request(self, args: HttpRequestInput) -> HttpRequestOutput:
        if not self._allow_private:
            _check_public_url(args.url)
        resp = self._client.request(
            args.method, args.url, headers=args.headers, content=args.body
        )
        text = resp.text
        return HttpRequestOutput(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            body=text[:MAX_BODY_CHARS],
            truncated=len(text) > MAX_BODY_CHARS,
        )

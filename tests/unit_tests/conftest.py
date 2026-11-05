"""Shared fixtures for unit tests.

Unit tests never touch the network. Credentials are stubbed so the wrappers'
environment validation passes, and the Oxylabs clients are replaced with fakes.
"""

import json
from typing import Any, Dict, Iterator, List

import httpx
import pytest

from langchain_oxylabs import OxylabsSearchAPIWrapper
from langchain_oxylabs import web_api as web_api_module


@pytest.fixture(autouse=True)
def stub_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy credentials so `validate_environment` succeeds offline."""
    monkeypatch.setenv("OXYLABS_USERNAME", "test-user")
    monkeypatch.setenv("OXYLABS_PASSWORD", "test-password")
    monkeypatch.setenv("OXYLABS_WEB_API_KEY", "test-web-api-key")


class FakeSearchEngine:
    """Stands in for `RealtimeClient.google`, recording the params it receives."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    def scrape_search(self, query: str, **params: Any) -> Any:
        self.calls.append({"query": query, "params": params})
        return self._response

    @property
    def last_params(self) -> Dict[str, Any]:
        return self.calls[-1]["params"]


class FakeSERPResponse:
    """Mimics the Oxylabs SDK response object, which exposes `.raw`."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw


def make_raw(results: Any) -> Dict[str, Any]:
    """Build a well-formed Oxylabs raw payload wrapping `results`."""
    return {"results": [{"content": {"results": results}}]}


@pytest.fixture
def organic_results() -> Dict[str, Any]:
    return {
        "organic": [
            {
                "pos": 1,
                "title": "Etno Dvaras",
                "desc": "Traditional Lithuanian food.",
                "url": "https://example.com/etno",
                "pos_overall": 1,
            }
        ]
    }


@pytest.fixture
def wrapper_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """Build an `OxylabsSearchAPIWrapper` wired to a `FakeSearchEngine`."""
    created: List[FakeSearchEngine] = []

    def _factory(results: Any = None, **wrapper_kwargs: Any) -> Any:
        wrapper = OxylabsSearchAPIWrapper(**wrapper_kwargs)
        engine = FakeSearchEngine(FakeSERPResponse(make_raw(results or {})))
        # `search_engine` is a plain pydantic field; assign the fake in place.
        wrapper.search_engine = engine
        created.append(engine)
        return wrapper, engine

    yield _factory


class FakeWebAPI:
    """Answers Web API requests from a queue of (status, body) replies, in order.

    The last reply repeats once the queue runs out. Every request is recorded so
    tests can check what was sent.
    """

    def __init__(self) -> None:
        self.replies: List[Any] = []
        self.requests: List[httpx.Request] = []

    def reply(self, status: int, body: Any) -> "FakeWebAPI":
        self.replies.append((status, body))
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(body, Exception):
            raise body
        return httpx.Response(status, json=body)

    @property
    def last_json(self) -> Dict[str, Any]:
        return json.loads(self.requests[-1].content)

    @property
    def paths(self) -> List[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]


@pytest.fixture
def web_api(monkeypatch: pytest.MonkeyPatch) -> FakeWebAPI:
    """Route every httpx client the Web API wrapper opens to a FakeWebAPI."""
    fake = FakeWebAPI()
    transport = httpx.MockTransport(fake.handler)
    real_client, real_async_client = httpx.Client, httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.Client:
        return real_client(transport=transport, **kwargs)

    def async_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=transport, **kwargs)

    monkeypatch.setattr(web_api_module.httpx, "Client", client)
    monkeypatch.setattr(web_api_module.httpx, "AsyncClient", async_client)
    # Retries and polling would otherwise sleep for real.
    monkeypatch.setattr(web_api_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(web_api_module.random, "uniform", lambda *_: 0)
    return fake

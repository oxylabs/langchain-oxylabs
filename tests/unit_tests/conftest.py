"""Shared fixtures for unit tests.

Unit tests never touch the network. Credentials are stubbed so the wrapper's
environment validation passes, and the Oxylabs client is replaced with a fake.
"""

from typing import Any, Dict, Iterator, List

import pytest

from langchain_oxylabs import OxylabsSearchAPIWrapper


@pytest.fixture(autouse=True)
def stub_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy credentials so `validate_environment` succeeds offline."""
    monkeypatch.setenv("OXYLABS_USERNAME", "test-user")
    monkeypatch.setenv("OXYLABS_PASSWORD", "test-password")


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

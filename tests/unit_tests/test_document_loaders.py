"""Unit tests for `OxylabsLoader` -- no network access."""

import json
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

from langchain_oxylabs import OxylabsLoader


class FakeRealtimeAPI:
    """Stands in for `oxylabs.internal.api.RealtimeAPI`."""

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def get_response(self, params: Dict[str, Any], config: Dict[str, Any]) -> Any:
        self.calls.append({"params": params, "config": config})
        return self._responses.pop(0)


def make_response(content: Any, job: Any = None) -> Dict[str, Any]:
    return {
        "results": [{"content": content}],
        "job": job if job is not None else {"url": "https://example.com/1"},
    }


def build_loader(responses: List[Any], **loader_kwargs: Any) -> Any:
    loader = OxylabsLoader(**loader_kwargs)
    fake = FakeRealtimeAPI(responses)
    loader._oxylabs_api = fake
    return loader, fake


class TestValidation:
    def test_requires_urls_or_queries(self) -> None:
        with pytest.raises(ValueError, match="Either `urls` or `queries`"):
            OxylabsLoader(params={"markdown": True})

    def test_accepts_urls(self) -> None:
        loader = OxylabsLoader(urls=["https://example.com"], params={})
        assert loader is not None

    def test_accepts_queries(self) -> None:
        loader = OxylabsLoader(queries=["gaming headset"], params={})
        assert loader is not None


class TestContentExtraction:
    def test_string_content_passes_through(self) -> None:
        loader, _ = build_loader(
            [make_response("# Markdown page")],
            urls=["https://example.com/1"],
            params={"markdown": True},
        )
        docs = list(loader.lazy_load())
        assert docs[0].page_content == "# Markdown page"

    def test_dict_content_is_json_encoded(self) -> None:
        payload = {"results": {"organic": [{"pos": 1}]}}
        loader, _ = build_loader(
            [make_response(payload)],
            queries=["gaming headset"],
            params={"parse": True},
        )
        docs = list(loader.lazy_load())
        assert json.loads(docs[0].page_content) == payload

    def test_unexpected_content_type_raises(self) -> None:
        loader, _ = build_loader(
            [make_response(12345)], urls=["https://example.com/1"], params={}
        )
        with pytest.raises(RuntimeError, match="Response Validation Error"):
            list(loader.lazy_load())

    def test_missing_results_raises(self) -> None:
        loader, _ = build_loader(
            [{"job": {}}], urls=["https://example.com/1"], params={}
        )
        with pytest.raises(RuntimeError, match="Response Validation Error"):
            list(loader.lazy_load())


class TestMetadata:
    def test_extracts_known_attributes(self) -> None:
        job = {
            "url": "https://example.com/1",
            "query": "headset",
            "created_at": "2026-01-01 00:00:00",
        }
        loader, _ = build_loader(
            [make_response("body", job=job)],
            urls=["https://example.com/1"],
            params={},
        )
        docs = list(loader.lazy_load())
        assert docs[0].metadata == job

    def test_omits_absent_and_empty_attributes(self) -> None:
        loader, _ = build_loader(
            [make_response("body", job={"url": "https://example.com/1", "query": ""})],
            urls=["https://example.com/1"],
            params={},
        )
        docs = list(loader.lazy_load())
        assert docs[0].metadata == {"url": "https://example.com/1"}


class TestRequestBuilding:
    def test_one_request_per_url_with_params_merged(self) -> None:
        loader, fake = build_loader(
            [make_response("a"), make_response("b")],
            urls=["https://example.com/1", "https://example.com/2"],
            params={"markdown": True},
        )
        list(loader.lazy_load())
        assert [c["params"]["url"] for c in fake.calls] == [
            "https://example.com/1",
            "https://example.com/2",
        ]
        assert all(c["params"]["markdown"] is True for c in fake.calls)

    def test_one_request_per_query(self) -> None:
        loader, fake = build_loader(
            [make_response("a"), make_response("b")],
            queries=["headset", "mouse"],
            params={"source": "amazon_search"},
        )
        list(loader.lazy_load())
        assert [c["params"]["query"] for c in fake.calls] == ["headset", "mouse"]
        assert all(c["params"]["source"] == "amazon_search" for c in fake.calls)

    def test_request_timeout_is_passed_as_config(self) -> None:
        loader, fake = build_loader(
            [make_response("a")],
            urls=["https://example.com/1"],
            params={},
            request_timeout=42,
        )
        list(loader.lazy_load())
        assert fake.calls[0]["config"] == {"request_timeout": 42}

    def test_default_request_timeout(self) -> None:
        loader, fake = build_loader(
            [make_response("a")], urls=["https://example.com/1"], params={}
        )
        list(loader.lazy_load())
        assert fake.calls[0]["config"]["request_timeout"] == (
            OxylabsLoader.DEFAULT_REQUEST_TIMEOUT
        )


class TestLaziness:
    def test_lazy_load_is_lazy(self) -> None:
        """Documents are fetched one at a time, not buffered up front."""
        loader, fake = build_loader(
            [make_response("a"), make_response("b")],
            urls=["https://example.com/1", "https://example.com/2"],
            params={},
        )
        iterator = loader.lazy_load()
        assert len(fake.calls) == 0
        next(iterator)
        assert len(fake.calls) == 1

    def test_load_returns_all_documents(self) -> None:
        loader, _ = build_loader(
            [make_response("a"), make_response("b")],
            urls=["https://example.com/1", "https://example.com/2"],
            params={},
        )
        docs = loader.load()
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

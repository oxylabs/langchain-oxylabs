"""Integration tests for the Oxylabs Web API tools and loader (need a Web API key)."""

import json
from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_oxylabs import OxylabsWebLoader, OxylabsWebScrape, OxylabsWebSearch

SANDBOX_PRODUCT = "https://sandbox.oxylabs.io/products/1"


class TestOxylabsWebSearchIntegration(ToolsIntegrationTests):
    """Test OxylabsWebSearch with LangChain's integration tests."""

    @property
    def tool_constructor(self) -> Type[OxylabsWebSearch]:
        return OxylabsWebSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "python programming language", "location": "GB"}


class TestOxylabsWebScrapeIntegration(ToolsIntegrationTests):
    """Test OxylabsWebScrape with LangChain's integration tests."""

    @property
    def tool_constructor(self) -> Type[OxylabsWebScrape]:
        return OxylabsWebScrape

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"url": SANDBOX_PRODUCT}


def test_web_search_returns_ranked_results() -> None:
    output = json.loads(
        OxylabsWebSearch().invoke(
            {"query": "python programming language", "max_results": 3}
        )
    )
    assert 0 < len(output["results"]) <= 3
    assert all(result["url"].startswith("http") for result in output["results"])


def test_web_scrape_returns_markdown() -> None:
    output = OxylabsWebScrape().invoke({"url": SANDBOX_PRODUCT})
    assert isinstance(output, str)
    assert len(output) > 500, f"suspiciously short page: {output[:200]!r}"


def test_web_scrape_json_prompt_returns_fields() -> None:
    output = OxylabsWebScrape().invoke(
        {"url": SANDBOX_PRODUCT, "json_prompt": "the game title and its price"}
    )
    parsed = json.loads(output)
    assert isinstance(parsed, dict) and parsed, f"no fields extracted: {output!r}"


def test_web_loader_loads_documents() -> None:
    docs = OxylabsWebLoader([SANDBOX_PRODUCT]).load()
    assert len(docs) == 1
    assert docs[0].metadata["source"] == SANDBOX_PRODUCT
    assert docs[0].metadata["status_code"] == 200
    assert len(docs[0].page_content) > 500

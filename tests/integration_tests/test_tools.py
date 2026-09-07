"""Integration test for OxylabsSearchRun and OxylabsSearchResults."""

import json
from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_oxylabs import (
    OxylabsSearchAPIWrapper,
    OxylabsSearchResults,
    OxylabsSearchRun,
)
from tests.integration_tests.assertions import (
    assert_formatted_output,
    assert_results_payload,
)


class TestOxylabsSearchRunIntegration(ToolsIntegrationTests):
    """Test OxylabsSearchRun tool with LangChain's integration tests."""

    @property
    def tool_constructor(self) -> Type[OxylabsSearchRun]:
        return OxylabsSearchRun

    @property
    def tool_constructor_params(self) -> dict:
        return {"wrapper": OxylabsSearchAPIWrapper()}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "What's the weather like today?",
            "geo_location": "United Kingdom",
        }


class TestOxylabsSearchResultsIntegration(ToolsIntegrationTests):
    """Test OxylabsSearchResults tool with LangChain's integration tests."""

    @property
    def tool_constructor(self) -> Type[OxylabsSearchResults]:
        return OxylabsSearchResults

    @property
    def tool_constructor_params(self) -> dict:
        return {"wrapper": OxylabsSearchAPIWrapper()}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "What's the weather like today?",
            "geo_location": "United Kingdom",
        }


def test_oxylabs_search_call() -> None:
    """Test a simple call to Oxylabs Search API."""
    oxylabs_search_tool = OxylabsSearchRun(wrapper=OxylabsSearchAPIWrapper())

    output = oxylabs_search_tool.invoke({"query": "Python programming language"})

    assert oxylabs_search_tool.name == "oxylabs_search"
    assert_formatted_output(output)


def test_oxylabs_search_results_call() -> None:
    """Test a simple results call to Oxylabs Search API."""
    oxylabs_search_tool = OxylabsSearchResults(wrapper=OxylabsSearchAPIWrapper())

    output = oxylabs_search_tool.invoke({"query": "Python programming language"})

    assert oxylabs_search_tool.name == "oxylabs_search_results"
    assert isinstance(output, str)
    assert_results_payload(json.loads(output))

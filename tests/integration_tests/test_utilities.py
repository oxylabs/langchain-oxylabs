"""Integration test for OxylabsSearchAPIWrapper."""

from langchain_oxylabs import OxylabsSearchAPIWrapper
from tests.integration_tests.assertions import (
    assert_formatted_output,
    assert_results_payload,
)

QUERY = "Python programming language"


def test_call() -> None:
    """A live search returns formatted, parsed text."""
    chain = OxylabsSearchAPIWrapper()
    assert_formatted_output(chain.run(QUERY))


async def test_async_call() -> None:
    """The async path returns formatted, parsed text."""
    chain = OxylabsSearchAPIWrapper()
    assert_formatted_output(await chain.arun(QUERY))


def test_results() -> None:
    """A live search returns structured result pages."""
    chain = OxylabsSearchAPIWrapper()
    assert_results_payload(chain.results(QUERY))


async def test_async_results() -> None:
    """The async path returns structured result pages."""
    chain = OxylabsSearchAPIWrapper()
    assert_results_payload(await chain.aresults(QUERY))


def test_geo_location_is_applied() -> None:
    """A per-call geo_location reaches the API and is reflected back.

    Regression cover for geo_location being silently discarded before it left
    the client.
    """
    chain = OxylabsSearchAPIWrapper()
    results = chain.results("coffee shops", geo_location="Vilnius,Lithuania")
    assert_results_payload(results)

    reported = [
        (page.get("search_information") or {}).get("geo_location")
        for page in results
        if page.get("search_information")
    ]
    # Google does not always include a search_information block, so only
    # assert when the API actually told us where it searched from.
    if any(value for value in reported):
        assert any(
            value and "Vilnius" in value for value in reported
        ), f"geo_location was not honored; API reported {reported}"

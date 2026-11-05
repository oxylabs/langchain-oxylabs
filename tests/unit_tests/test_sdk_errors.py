"""Unit tests for reporting requests the `oxylabs` SDK fails silently -- no network.

The SDK catches a failed request, logs the reason, and returns nothing. These
fakes do the same, so the tests exercise the real log-capturing path.
"""

import asyncio
import logging
import re
import threading
from typing import Any, Dict

import pytest

from langchain_oxylabs import (
    OxylabsAPIError,
    OxylabsLoader,
    OxylabsSearchAPIWrapper,
    OxylabsSearchResults,
    OxylabsSearchRun,
)
from langchain_oxylabs.utilities import SDK_LOGGER, raise_sdk_errors
from tests.unit_tests.conftest import FakeSERPResponse

sdk_logger = logging.getLogger(SDK_LOGGER)

UNAUTHORIZED = (
    "HTTP error occurred: 401 Client Error: Unauthorized for url:"
    " https://realtime.oxylabs.io/v1/queries"
)


class FailingSearchEngine:
    """Logs an error and returns an empty response, like the SDK on a 401."""

    def __init__(self, message: str = UNAUTHORIZED) -> None:
        self.message = message

    def scrape_search(self, query: str, **params: Any) -> Any:
        sdk_logger.error(self.message)
        return FakeSERPResponse({})


class FailingRealtimeAPI:
    def get_response(self, params: Dict[str, Any], config: Dict[str, Any]) -> Any:
        sdk_logger.error(UNAUTHORIZED)
        return None


def failing_wrapper(message: str = UNAUTHORIZED) -> OxylabsSearchAPIWrapper:
    wrapper = OxylabsSearchAPIWrapper()
    wrapper.search_engine = FailingSearchEngine(message)
    return wrapper


class TestRaiseSdkErrors:
    def test_passes_through_when_nothing_is_logged(self) -> None:
        with raise_sdk_errors():
            pass

    def test_ignores_warnings(self) -> None:
        with raise_sdk_errors():
            sdk_logger.warning("not a failure")

    @pytest.mark.parametrize(
        "logged, expected",
        [
            (UNAUTHORIZED, "Authentication failed (401)"),
            ("HTTP error occurred: 403 Client Error: Forbidden", "Access denied (403)"),
            ("HTTP error occurred: 429 Client Error", "Rate limited (429)"),
            ("Timeout error. The request has timed out.", "timed out"),
            ("Error occurred: connection reset", "connection reset"),
        ],
    )
    def test_names_the_cause(self, logged: str, expected: str) -> None:
        with pytest.raises(OxylabsAPIError, match=re.escape(expected)):
            with raise_sdk_errors():
                sdk_logger.error(logged)

    def test_ignores_errors_from_other_threads(self) -> None:
        """Concurrent calls must not see each other's failures."""
        with raise_sdk_errors():
            other = threading.Thread(target=sdk_logger.error, args=(UNAUTHORIZED,))
            other.start()
            other.join()

    def test_removes_its_handler(self) -> None:
        before = list(sdk_logger.handlers)
        with pytest.raises(OxylabsAPIError):
            with raise_sdk_errors():
                sdk_logger.error(UNAUTHORIZED)
        assert sdk_logger.handlers == before


class TestSearchWrapper:
    """Regression: a 401 used to come back as an empty result set."""

    def test_run_raises(self) -> None:
        with pytest.raises(OxylabsAPIError, match="OXYLABS_USERNAME"):
            failing_wrapper().run("q")

    def test_results_raises(self) -> None:
        with pytest.raises(OxylabsAPIError, match="401"):
            failing_wrapper().results("q")

    def test_async_results_raise(self) -> None:
        with pytest.raises(OxylabsAPIError, match="401"):
            asyncio.run(failing_wrapper().aresults("q"))


class TestSearchTools:
    def test_run_tool_returns_the_error_to_the_agent(self) -> None:
        out = OxylabsSearchRun(wrapper=failing_wrapper()).invoke({"query": "q"})
        assert out.startswith("Authentication failed (401)")

    async def test_results_tool_returns_the_error_to_the_agent(self) -> None:
        out = await OxylabsSearchResults(wrapper=failing_wrapper()).ainvoke(
            {"query": "q"}
        )
        assert out.startswith("Authentication failed (401)")


class TestLoader:
    def test_failed_request_names_the_cause(self) -> None:
        """Regression: this used to raise "'NoneType' object is not subscriptable"."""
        loader = OxylabsLoader(urls=["https://example.com"], params={})
        loader._oxylabs_api = FailingRealtimeAPI()
        with pytest.raises(OxylabsAPIError, match="Authentication failed"):
            loader.load()

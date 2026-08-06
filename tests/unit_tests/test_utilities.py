"""Unit tests for `OxylabsSearchAPIWrapper` -- no network access."""

from typing import Any

import pytest

from langchain_oxylabs import OxylabsSearchAPIWrapper
from langchain_oxylabs.utilities import (
    BINARY_CONTENT_REPLACEMENT,
    _get_default_params,
    get_sdk_type,
)


def test_get_sdk_type_identifies_langchain() -> None:
    sdk_type = get_sdk_type()
    assert sdk_type.startswith("oxylabs-sdk-langchain/")


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OXYLABS_USERNAME", raising=False)
    monkeypatch.delenv("OXYLABS_PASSWORD", raising=False)
    with pytest.raises(Exception):
        OxylabsSearchAPIWrapper()


def test_unsupported_source_raises() -> None:
    with pytest.raises(NotImplementedError, match="is not supported"):
        OxylabsSearchAPIWrapper(params={"source": "bing_search"})


class TestGetParams:
    """`get_params` decides what actually reaches the Oxylabs API."""

    def test_empty_defaults_are_omitted(self) -> None:
        wrapper = OxylabsSearchAPIWrapper()
        params = wrapper.get_params()
        # `locale`/`geo_location` default to "" and must not be sent.
        assert "locale" not in params
        assert "geo_location" not in params
        assert params["source"] == "google_search"

    def test_per_call_geo_location_is_applied(self) -> None:
        """Regression: an invocation-time geo_location used to be discarded.

        `geo_location` defaults to "" and was filtered out of the outgoing
        params, and overrides were only applied to keys already present -- so
        an LLM-supplied geo_location never reached the API.
        """
        wrapper = OxylabsSearchAPIWrapper()
        params = wrapper.get_params(geo_location="Vilnius,Lithuania")
        assert params["geo_location"] == "Vilnius,Lithuania"

    def test_per_call_override_beats_configured_value(self) -> None:
        wrapper = OxylabsSearchAPIWrapper(
            params={"geo_location": "California,United States"}
        )
        params = wrapper.get_params(geo_location="Tokyo,Japan")
        assert params["geo_location"] == "Tokyo,Japan"

    def test_configured_value_survives_when_not_overridden(self) -> None:
        """Regression: the tool's schema default used to clobber this."""
        wrapper = OxylabsSearchAPIWrapper(params={"geo_location": "Vilnius,Lithuania"})
        params = wrapper.get_params()
        assert params["geo_location"] == "Vilnius,Lithuania"

    def test_wrapper_only_params_are_not_forwarded(self) -> None:
        wrapper = OxylabsSearchAPIWrapper(params={"result_categories": ["organic"]})
        params = wrapper.get_params(result_categories=["organic"])
        assert "result_categories" not in params

    def test_falsy_override_does_not_clear_configured_value(self) -> None:
        wrapper = OxylabsSearchAPIWrapper(params={"geo_location": "Vilnius,Lithuania"})
        params = wrapper.get_params(geo_location=None)
        assert params["geo_location"] == "Vilnius,Lithuania"


class TestValidateResponseCategories:
    def test_keeps_known_and_drops_unknown(self) -> None:
        validated = OxylabsSearchAPIWrapper.validate_response_categories(
            ["organic", "knowledge_graph", "not_a_category", "local_information"]
        )
        assert validated == ["knowledge_graph", "local_information"]

    def test_preserves_caller_order(self) -> None:
        validated = OxylabsSearchAPIWrapper.validate_response_categories(
            ["local_information", "knowledge_graph"]
        )
        assert validated == ["local_information", "knowledge_graph"]


class TestValidateResponse:
    def test_unpacks_nested_results(self, wrapper_factory: Any) -> None:
        wrapper, _ = wrapper_factory({"organic": [{"pos": 1}]})
        results = wrapper.results("query")
        assert results == [{"organic": [{"pos": 1}]}]

    def test_unparsed_content_returns_empty(self, wrapper_factory: Any) -> None:
        """`results()` swallows validation errors and returns []."""
        wrapper, engine = wrapper_factory()
        engine._response.raw = {"results": [{"content": "raw html string"}]}
        assert wrapper.results("query") == []

    def test_no_results_key_returns_empty(self, wrapper_factory: Any) -> None:
        wrapper, engine = wrapper_factory()
        engine._response.raw = {}
        assert wrapper.results("query") == []


class TestRunFormatting:
    def test_run_renders_organic_results_as_text(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        text = wrapper.run("restaurants in Vilnius")
        assert "ORGANIC RESULTS ITEMS:" in text
        assert "Etno Dvaras" in text

    def test_excluded_attributes_are_omitted(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        """`pos_overall` is excluded by default to save tokens."""
        wrapper, _ = wrapper_factory(organic_results)
        text = wrapper.run("query")
        assert "POS_OVERALL" not in text

    def test_no_results_yields_placeholder(self, wrapper_factory: Any) -> None:
        wrapper, _ = wrapper_factory({})
        assert wrapper.run("query") == "No good search result found"

    def test_binary_image_data_redacted_by_default(self, wrapper_factory: Any) -> None:
        wrapper, _ = wrapper_factory(
            {"organic": [{"title": "Pic", "image_data": "AAAAbase64AAAA"}]}
        )
        text = wrapper.run("query")
        assert "AAAAbase64AAAA" not in text
        assert BINARY_CONTENT_REPLACEMENT in text

    def test_binary_image_data_kept_when_requested(self, wrapper_factory: Any) -> None:
        wrapper, _ = wrapper_factory(
            {"organic": [{"title": "Pic", "image_data": "AAAAbase64AAAA"}]},
            include_binary_image_data=True,
        )
        assert "AAAAbase64AAAA" in wrapper.run("query")

    def test_result_categories_filter_output(self, wrapper_factory: Any) -> None:
        payload = {
            "organic": [{"title": "An organic hit"}],
            "local_pack": {"items": [{"title": "A local hit"}]},
        }
        wrapper, _ = wrapper_factory(
            payload, params={"result_categories": ["local_information"]}
        )
        text = wrapper.run("query")
        assert "A local hit" in text
        assert "An organic hit" not in text

    def test_recursion_depth_is_respected(self, wrapper_factory: Any) -> None:
        deep = {"organic": [{"a": {"b": {"c": {"d": "buried-value"}}}}]}
        wrapper, _ = wrapper_factory(deep, parsing_recursion_depth=2)
        assert "buried-value" not in wrapper.run("query")


class TestQueryForwarding:
    def test_query_and_params_reach_the_client(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, engine = wrapper_factory(organic_results)
        wrapper.run("restaurants in Vilnius", geo_location="Vilnius,Lithuania")
        assert engine.calls[-1]["query"] == "restaurants in Vilnius"
        assert engine.last_params["geo_location"] == "Vilnius,Lithuania"

    async def test_arun_forwards_geo_location(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, engine = wrapper_factory(organic_results)
        await wrapper.arun("query", geo_location="Tokyo,Japan")
        assert engine.last_params["geo_location"] == "Tokyo,Japan"


def test_default_params_shape() -> None:
    defaults = _get_default_params()
    assert defaults["source"] == "google_search"
    assert defaults["parse"] is True

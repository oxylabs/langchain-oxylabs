"""Unit tests for the Oxylabs search tools -- no network access."""

import json
from typing import Any

from langchain_core.messages import ToolMessage

from langchain_oxylabs import OxylabsSearchResults, OxylabsSearchRun
from langchain_oxylabs.tools import OxylabsSearchQueryInput


class TestArgsSchema:
    def test_exposes_query_and_geo_location(self, wrapper_factory: Any) -> None:
        wrapper, _ = wrapper_factory()
        tool = OxylabsSearchRun(wrapper=wrapper)
        assert set(tool.args) == {"query", "geo_location"}

    def test_geo_location_has_no_hardcoded_default(self) -> None:
        """Regression: it used to default to "California,United States".

        A non-None default meant `_run` always forwarded that value, silently
        overriding a `geo_location` configured on the wrapper.
        """
        assert OxylabsSearchQueryInput.model_fields["geo_location"].default is None

    def test_query_is_required(self) -> None:
        assert OxylabsSearchQueryInput.model_fields["query"].is_required()

    def test_tool_names_are_stable(self, wrapper_factory: Any) -> None:
        """These names are part of the public surface agents bind against."""
        wrapper, _ = wrapper_factory()
        assert OxylabsSearchRun(wrapper=wrapper).name == "oxylabs_search"
        assert OxylabsSearchResults(wrapper=wrapper).name == "oxylabs_search_results"


class TestOxylabsSearchRun:
    def test_returns_formatted_text(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        out = OxylabsSearchRun(wrapper=wrapper).invoke({"query": "restaurants"})
        assert isinstance(out, str)
        assert "Etno Dvaras" in out

    def test_geo_location_reaches_the_api(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        """Regression: end-to-end path for an LLM-supplied geo_location."""
        wrapper, engine = wrapper_factory(organic_results)
        OxylabsSearchRun(wrapper=wrapper).invoke(
            {"query": "restaurants", "geo_location": "Vilnius,Lithuania"}
        )
        assert engine.last_params["geo_location"] == "Vilnius,Lithuania"

    def test_omitted_geo_location_is_not_sent(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        """Regression: no geo_location must mean none sent, not California."""
        wrapper, engine = wrapper_factory(organic_results)
        OxylabsSearchRun(wrapper=wrapper).invoke({"query": "restaurants"})
        assert "geo_location" not in engine.last_params

    def test_configured_geo_location_is_preserved(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        """Regression: the schema default used to clobber wrapper config."""
        wrapper, engine = wrapper_factory(
            organic_results, params={"geo_location": "Vilnius,Lithuania"}
        )
        OxylabsSearchRun(wrapper=wrapper).invoke({"query": "restaurants"})
        assert engine.last_params["geo_location"] == "Vilnius,Lithuania"

    def test_tool_call_returns_tool_message(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        tool = OxylabsSearchRun(wrapper=wrapper)
        result = tool.invoke(
            {
                "args": {"query": "restaurants"},
                "id": "call_1",
                "name": "oxylabs_search",
                "type": "tool_call",
            }
        )
        assert isinstance(result, ToolMessage)
        assert "Etno Dvaras" in str(result.content)

    def test_constructor_kwargs_are_forwarded(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, engine = wrapper_factory(organic_results)
        tool = OxylabsSearchRun(wrapper=wrapper, kwargs={"domain": "lt"})
        tool.invoke({"query": "restaurants"})
        assert engine.last_params["domain"] == "lt"

    async def test_async_invoke(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        out = await OxylabsSearchRun(wrapper=wrapper).ainvoke({"query": "q"})
        assert "Etno Dvaras" in out


class TestOxylabsSearchResults:
    def test_returns_json_string(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        out = OxylabsSearchResults(wrapper=wrapper).invoke({"query": "restaurants"})
        parsed = json.loads(out)
        assert parsed == [organic_results]

    def test_geo_location_reaches_the_api(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, engine = wrapper_factory(organic_results)
        OxylabsSearchResults(wrapper=wrapper).invoke(
            {"query": "q", "geo_location": "Tokyo,Japan"}
        )
        assert engine.last_params["geo_location"] == "Tokyo,Japan"

    def test_empty_results_serialize_to_empty_list(self, wrapper_factory: Any) -> None:
        wrapper, engine = wrapper_factory()
        engine._response.raw = {"results": [{"content": "unparsed"}]}
        out = OxylabsSearchResults(wrapper=wrapper).invoke({"query": "q"})
        assert json.loads(out) == []

    async def test_async_invoke_returns_json(
        self, wrapper_factory: Any, organic_results: Any
    ) -> None:
        wrapper, _ = wrapper_factory(organic_results)
        out = await OxylabsSearchResults(wrapper=wrapper).ainvoke({"query": "q"})
        assert json.loads(out) == [organic_results]

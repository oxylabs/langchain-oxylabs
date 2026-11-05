"""Unit tests for the Oxylabs Web API wrapper, tools and loader -- no network access."""

import json
from typing import Any, Dict

import httpx
import pytest
from langchain_core.messages import ToolMessage
from pydantic import SecretStr

from langchain_oxylabs import (
    OxylabsWebAPIError,
    OxylabsWebAPIWrapper,
    OxylabsWebLoader,
    OxylabsWebScrape,
    OxylabsWebSearch,
)
from tests.unit_tests.conftest import FakeWebAPI


def search_body() -> Dict[str, Any]:
    return {
        "state": "done",
        "results": [
            {
                "title": "Etno Dvaras",
                "outline": ["Traditional Lithuanian food."],
                "url": "https://example.com/etno",
                "metadata": {"position": 1},
            }
        ],
        "related_questions": [
            {"question": "Where to eat in Vilnius?", "title": None, "snippet": None}
        ],
        "related_searches": [{"query": "vilnius restaurants"}],
        "metadata": {"request_id": "1"},
    }


def scrape_body(key: str = "markdown", value: Any = "# Page\n" + "text " * 200) -> Any:
    return {
        "state": "done",
        "results": [
            {
                key: value,
                "metadata": {
                    "url": "https://example.com/",
                    "statuses": {"http_request": {"code": 200, "tag": "HTTP_OK"}},
                },
            }
        ],
        "metadata": {"request_id": "42"},
    }


class TestWrapper:
    def test_key_comes_from_the_environment(self) -> None:
        wrapper = OxylabsWebAPIWrapper()
        assert wrapper.oxylabs_web_api_key.get_secret_value() == "test-web-api-key"

    def test_missing_key_says_which_variable_to_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OXYLABS_WEB_API_KEY")
        with pytest.raises(ValueError, match="OXYLABS_WEB_API_KEY"):
            OxylabsWebAPIWrapper()

    def test_surrounding_whitespace_is_stripped(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        OxylabsWebAPIWrapper(oxylabs_web_api_key=SecretStr("  key-from-file\n")).search(
            "q"
        )
        assert web_api.requests[-1].headers["authorization"] == "Bearer key-from-file"

    def test_env_key_is_stripped_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OXYLABS_WEB_API_KEY", "env-key\n")
        key = OxylabsWebAPIWrapper().oxylabs_web_api_key.get_secret_value()
        assert key == "env-key"

    def test_whitespace_inside_the_key_is_rejected_without_quoting_it(self) -> None:
        with pytest.raises(ValueError, match="whitespace") as error:
            OxylabsWebAPIWrapper(oxylabs_web_api_key=SecretStr("secret\nkey"))
        assert "secret" not in str(error.value)

    def test_rejected_header_never_echoes_the_key(self, web_api: FakeWebAPI) -> None:
        web_api.reply(0, httpx.LocalProtocolError("Illegal header value b'Bearer k'"))
        with pytest.raises(OxylabsWebAPIError, match="headers were rejected") as error:
            OxylabsWebAPIWrapper().search("q")
        assert "Bearer" not in str(error.value)
        assert len(web_api.requests) == 1

    def test_key_is_not_in_repr(self) -> None:
        assert "test-web-api-key" not in repr(OxylabsWebAPIWrapper())

    def test_sends_bearer_key_and_sdk_header(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        OxylabsWebAPIWrapper().search("q")
        headers = web_api.requests[-1].headers
        assert headers["authorization"] == "Bearer test-web-api-key"
        assert headers["x-oxylabs-sdk"].startswith("oxylabs-sdk-langchain/")

    def test_search_payload(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        OxylabsWebAPIWrapper().search("q", max_results=3, location="de")
        assert web_api.paths == ["POST /v1/search"]
        assert web_api.last_json == {"query": "q", "max_results": 3, "location": "DE"}

    def test_omitted_location_is_not_sent(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        OxylabsWebAPIWrapper().search("q")
        assert "location" not in web_api.last_json

    def test_empty_query_is_rejected_before_calling(self, web_api: FakeWebAPI) -> None:
        with pytest.raises(OxylabsWebAPIError, match="query"):
            OxylabsWebAPIWrapper().search("  ")
        assert web_api.requests == []

    def test_scrape_payload(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body())
        OxylabsWebAPIWrapper().scrape(
            "https://example.com", location="us", device="mobile", run_js=True
        )
        assert web_api.last_json == {
            "url": "https://example.com",
            "output": ["markdown"],
            "location": "US",
            "device": "mobile",
            "run_js": True,
        }

    def test_json_prompt_asks_for_the_ai_parser(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body("json", {"title": "x"}))
        OxylabsWebAPIWrapper().scrape("https://example.com", json_prompt="the title")
        assert web_api.last_json["output"] == ["json"]
        assert web_api.last_json["json"] == {"prompt": "the title"}

    def test_relative_url_is_rejected(self, web_api: FakeWebAPI) -> None:
        with pytest.raises(OxylabsWebAPIError, match="absolute"):
            OxylabsWebAPIWrapper().scrape("example.com")

    def test_retries_transient_errors(self, web_api: FakeWebAPI) -> None:
        web_api.reply(503, {"detail": "busy"}).reply(200, search_body())
        assert OxylabsWebAPIWrapper().search("q")["results"]
        assert len(web_api.requests) == 2

    def test_retries_connection_errors(self, web_api: FakeWebAPI) -> None:
        web_api.reply(0, httpx.ConnectError("down")).reply(200, search_body())
        assert OxylabsWebAPIWrapper().search("q")["results"]

    def test_gives_up_after_max_retries(self, web_api: FakeWebAPI) -> None:
        web_api.reply(502, {"detail": "bad gateway"})
        with pytest.raises(OxylabsWebAPIError, match="502"):
            OxylabsWebAPIWrapper(max_retries=1).search("q")
        assert len(web_api.requests) == 2

    def test_faulted_request_is_not_retried(self, web_api: FakeWebAPI) -> None:
        web_api.reply(
            500,
            {
                "state": "faulted",
                "title": "REQUEST_FAILED_AFTER_TOO_MANY_RETRIES",
                "detail": "Try again.",
                "metadata": {"request_id": "99"},
            },
        )
        with pytest.raises(OxylabsWebAPIError, match="request_id 99"):
            OxylabsWebAPIWrapper().scrape("https://example.com")
        assert len(web_api.requests) == 1

    def test_spent_quota_is_not_retried(self, web_api: FakeWebAPI) -> None:
        web_api.reply(429, {"title": "QUOTA_EXCEEDED"})
        with pytest.raises(OxylabsWebAPIError, match="Quota exceeded"):
            OxylabsWebAPIWrapper().search("q")
        assert len(web_api.requests) == 1

    def test_bad_key(self, web_api: FakeWebAPI) -> None:
        web_api.reply(401, {})
        with pytest.raises(OxylabsWebAPIError, match="Authentication failed"):
            OxylabsWebAPIWrapper().search("q")

    def test_validation_errors_name_the_field(self, web_api: FakeWebAPI) -> None:
        web_api.reply(
            400,
            {
                "title": "VALIDATION_ERROR",
                "errors": [{"pointer": "#/location", "detail": "bad pattern"}],
            },
        )
        with pytest.raises(OxylabsWebAPIError, match="#/location: bad pattern"):
            OxylabsWebAPIWrapper().search("q")

    def test_timed_out_scrape_is_queued_and_polled(self, web_api: FakeWebAPI) -> None:
        web_api.reply(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.reply(202, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.reply(200, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.reply(200, scrape_body())
        response = OxylabsWebAPIWrapper().scrape("https://example.com", run_js=True)
        assert response["state"] == "done"
        assert web_api.paths == [
            "POST /v1/scrape",
            "POST /v1/async/scrape",
            "GET /v1/async/scrape/7",
            "GET /v1/async/scrape/7",
        ]

    def test_polling_gives_up_after_max_wait(self, web_api: FakeWebAPI) -> None:
        web_api.reply(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.reply(200, {"state": "pending", "metadata": {"request_id": "7"}})
        with pytest.raises(OxylabsWebAPIError, match="still running"):
            OxylabsWebAPIWrapper(max_wait=0).scrape("https://example.com")

    async def test_async_search(self, web_api: FakeWebAPI) -> None:
        web_api.reply(500, {}).reply(200, search_body())
        response = await OxylabsWebAPIWrapper().asearch("q", location="lt")
        assert response["results"][0]["title"] == "Etno Dvaras"
        assert web_api.last_json["location"] == "LT"

    async def test_async_scrape_queues_on_timeout(self, web_api: FakeWebAPI) -> None:
        web_api.reply(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.reply(202, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.reply(200, scrape_body())
        response = await OxylabsWebAPIWrapper(poll_interval=0).ascrape(
            "https://example.com"
        )
        assert response["state"] == "done"
        assert web_api.paths[-1] == "GET /v1/async/scrape/7"


class TestOxylabsWebSearch:
    def test_tool_name_and_args_are_stable(self) -> None:
        """These names are part of the public surface agents bind against."""
        tool = OxylabsWebSearch()
        assert tool.name == "oxylabs_web_search"
        assert set(tool.args) == {"query", "max_results", "location"}

    def test_returns_compact_json(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        out = json.loads(OxylabsWebSearch().invoke({"query": "restaurants"}))
        assert out == {
            "results": [
                {
                    "title": "Etno Dvaras",
                    "url": "https://example.com/etno",
                    "outline": ["Traditional Lithuanian food."],
                }
            ],
            "related_questions": ["Where to eat in Vilnius?"],
        }

    def test_api_errors_reach_the_agent_as_text(self, web_api: FakeWebAPI) -> None:
        web_api.reply(401, {})
        out = OxylabsWebSearch().invoke({"query": "q"})
        assert "Authentication failed" in out

    def test_place_name_location_is_rejected_as_text(self, web_api: FakeWebAPI) -> None:
        out = OxylabsWebSearch().invoke({"query": "q", "location": "Germany"})
        assert isinstance(out, str)
        assert web_api.requests == []

    def test_tool_call_returns_tool_message(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        result = OxylabsWebSearch().invoke(
            {
                "args": {"query": "restaurants", "location": "LT"},
                "id": "call_1",
                "name": "oxylabs_web_search",
                "type": "tool_call",
            }
        )
        assert isinstance(result, ToolMessage)
        assert "Etno Dvaras" in str(result.content)

    async def test_async_invoke(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, search_body())
        out = await OxylabsWebSearch().ainvoke({"query": "q"})
        assert "Etno Dvaras" in out


class TestOxylabsWebScrape:
    def test_tool_name_and_args_are_stable(self) -> None:
        tool = OxylabsWebScrape()
        assert tool.name == "oxylabs_web_scrape"
        assert set(tool.args) == {
            "url",
            "output",
            "location",
            "device",
            "run_js",
            "json_prompt",
        }

    def test_returns_markdown(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body())
        out = OxylabsWebScrape().invoke({"url": "https://example.com"})
        assert out.startswith("# Page")
        assert "[" not in out

    def test_html_output(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body("html", "<html>" + "x" * 600 + "</html>"))
        out = OxylabsWebScrape().invoke(
            {"url": "https://example.com", "output": "html"}
        )
        assert web_api.last_json["output"] == ["html"]
        assert out.startswith("<html>")

    def test_json_prompt_returns_parsed_fields(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body("json", {"price": "91,99 €"}))
        out = OxylabsWebScrape().invoke(
            {"url": "https://example.com", "json_prompt": "the price"}
        )
        assert json.loads(out) == {"price": "91,99 €"}

    def test_null_json_key_does_not_hide_the_page(self, web_api: FakeWebAPI) -> None:
        body = scrape_body()
        body["results"][0]["json"] = None
        web_api.reply(200, body)
        out = OxylabsWebScrape().invoke({"url": "https://example.com"})
        assert out.startswith("# Page")

    def test_long_pages_are_truncated(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="a" * 1000))
        out = OxylabsWebScrape(max_content_chars=100).invoke(
            {"url": "https://example.com"}
        )
        assert out.startswith("a" * 100 + "\n\n[Truncated: showing 100 of 1000")

    def test_truncation_can_be_disabled(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="a" * 60_000))
        out = OxylabsWebScrape(max_content_chars=None).invoke(
            {"url": "https://example.com"}
        )
        assert out == "a" * 60_000

    def test_thin_page_suggests_run_js(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="Loading..."))
        out = OxylabsWebScrape().invoke({"url": "https://example.com"})
        assert "run_js" in out

    def test_no_run_js_hint_when_already_rendered(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="Loading..."))
        out = OxylabsWebScrape().invoke({"url": "https://example.com", "run_js": True})
        assert out == "Loading..."

    def test_api_errors_reach_the_agent_as_text(self, web_api: FakeWebAPI) -> None:
        web_api.reply(400, {"errors": [{"pointer": "#/url", "detail": "bad"}]})
        out = OxylabsWebScrape().invoke({"url": "https://example.com"})
        assert "#/url: bad" in out

    async def test_async_invoke(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body())
        out = await OxylabsWebScrape().ainvoke({"url": "https://example.com"})
        assert out.startswith("# Page")


class TestOxylabsWebLoader:
    def test_one_document_per_url(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="first")).reply(
            200, scrape_body(value="second")
        )
        docs = OxylabsWebLoader(["https://a.example", "https://b.example"]).load()
        assert [d.page_content for d in docs] == ["first", "second"]
        assert docs[0].metadata == {
            "source": "https://a.example",
            "final_url": "https://example.com/",
            "status_code": 200,
            "request_id": "42",
        }

    def test_options_reach_the_api(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body("html", "<p>x</p>"))
        docs = OxylabsWebLoader(
            ["https://a.example"], output="html", location="de", run_js=True
        ).load()
        assert docs[0].page_content == "<p>x</p>"
        assert web_api.last_json == {
            "url": "https://a.example",
            "output": ["html"],
            "location": "DE",
            "run_js": True,
        }

    def test_explicit_key_is_used(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body())
        OxylabsWebLoader(["https://a.example"], oxylabs_web_api_key="other").load()
        assert web_api.requests[-1].headers["authorization"] == "Bearer other"

    def test_empty_url_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="urls"):
            OxylabsWebLoader([])

    async def test_async_load(self, web_api: FakeWebAPI) -> None:
        web_api.reply(200, scrape_body(value="first"))
        docs = await OxylabsWebLoader(["https://a.example"]).aload()
        assert docs[0].page_content == "first"

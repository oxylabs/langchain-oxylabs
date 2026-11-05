"""Client for the Oxylabs Web API (webapi.oxylabs.io)."""

import asyncio
import random
import time
from typing import Any, Dict, List, Literal, Optional

import httpx
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_oxylabs.utilities import get_sdk_type

DEFAULT_BASE_URL = "https://webapi.oxylabs.io"

# Transient statuses worth another attempt. A 429 that says the quota is spent, a
# faulted request and a scrape timeout are verdicts and are never retried.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TIMEOUT_TITLE = "REQUEST_FAILED_SCRAPE_TIMEOUT"
QUOTA_TITLE = "QUOTA_EXCEEDED"

ScrapeFormat = Literal["markdown", "html"]
Device = Literal["desktop", "mobile"]


class OxylabsWebAPIError(Exception):
    """An Oxylabs Web API call failed. The message says why and what to do."""


class _ScrapeTimeout(OxylabsWebAPIError):
    """The synchronous scrape ran out of time; the request can be queued instead."""


def _json_body(resp: httpx.Response) -> Optional[Dict[str, Any]]:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _detail(body: Optional[Dict[str, Any]], resp: httpx.Response) -> str:
    """The useful part of an error body: per-field errors first, then `detail`."""
    if body is None:
        return resp.text[:500]
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(
            ": ".join(str(part) for part in (e.get("pointer"), e.get("detail")) if part)
            for e in errors
            if isinstance(e, dict)
        )
    return str(body.get("detail") or body.get("message") or body)[:500]


def _retryable(resp: httpx.Response, body: Optional[Dict[str, Any]]) -> bool:
    if resp.status_code not in RETRY_STATUS or body is None:
        return resp.status_code in RETRY_STATUS
    return body.get("title") not in (TIMEOUT_TITLE, QUOTA_TITLE) and (
        body.get("state") != "faulted"
    )


def _check(resp: httpx.Response, path: str) -> Dict[str, Any]:
    """Return the response body, or raise an error that says what went wrong."""
    body = _json_body(resp)
    title = (body or {}).get("title")
    if resp.status_code == 401:
        raise OxylabsWebAPIError(
            "Authentication failed (401). Check the Oxylabs Web API key: it comes from"
            " a Web API instance in the Oxylabs dashboard (dashboard.oxylabs.io)."
        )
    if title == QUOTA_TITLE:
        raise OxylabsWebAPIError(
            "Quota exceeded (429): the plan's quota is spent. Check the plan in the"
            " Oxylabs dashboard; retrying will not help."
        )
    if resp.status_code == 429:
        raise OxylabsWebAPIError(
            "Rate limited (429) and still limited after retries. Lower concurrency,"
            " or check the remaining quota in the Oxylabs dashboard."
        )
    if title == TIMEOUT_TITLE:
        raise _ScrapeTimeout(f"{path} timed out: {_detail(body, resp)}")
    if body is not None and body.get("state") == "faulted":
        request_id = (body.get("metadata") or {}).get("request_id")
        reason = ": ".join(str(p) for p in (title, body.get("detail")) if p)
        raise OxylabsWebAPIError(
            f'{path} returned {resp.status_code} with state "faulted"'
            f" ({reason or 'no reason given'}). If it keeps failing, quote"
            f" request_id {request_id or 'unknown'} to Oxylabs support."
        )
    if resp.status_code >= 400:
        raise OxylabsWebAPIError(
            f"{path} returned {resp.status_code}: {_detail(body, resp)}"
        )
    if body is None:
        raise OxylabsWebAPIError(f"{path} returned a non-JSON body: {resp.text[:200]}")
    return body


def _request_id(body: Dict[str, Any]) -> str:
    request_id = (body.get("metadata") or {}).get("request_id")
    if not request_id:
        raise OxylabsWebAPIError(f"The queued request has no request_id: {body}")
    return str(request_id)


class OxylabsWebAPIWrapper(BaseModel):
    """Calls the Oxylabs Web API `/v1/search` and `/v1/scrape` endpoints.

    Setup:
        Create a Web API instance in the Oxylabs dashboard (dashboard.oxylabs.io)
        and set its key as ``OXYLABS_WEB_API_KEY``. Keys from other Oxylabs
        products don't work here.

        .. code-block:: bash

            pip install -U langchain-oxylabs
            export OXYLABS_WEB_API_KEY="your-web-api-key"

    Example:
        .. code-block:: python

            from langchain_oxylabs import OxylabsWebAPIWrapper

            web_api = OxylabsWebAPIWrapper()
            web_api.search("best hiking boots", max_results=5, location="DE")
            web_api.scrape("https://example.com")
    """

    oxylabs_web_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "OXYLABS_WEB_API_KEY",
            error_message=(
                "Oxylabs Web API key is missing. Set the `OXYLABS_WEB_API_KEY`"
                " environment variable or pass `oxylabs_web_api_key`."
            ),
        )
    )
    base_url: str = DEFAULT_BASE_URL
    request_timeout: float = Field(
        default=200, description="Seconds to wait for one HTTP response."
    )
    max_retries: int = Field(
        default=2, description="Extra attempts for 429, 5xx and connection errors."
    )
    poll_interval: float = Field(
        default=5, description="Seconds between checks on a queued scrape."
    )
    max_wait: float = Field(
        default=600,
        description="Seconds to wait for a queued scrape before giving up.",
    )

    model_config = ConfigDict(extra="forbid")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.oxylabs_web_api_key.get_secret_value()}",
            "x-oxylabs-sdk": get_sdk_type(),
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _backoff(self, attempt: int) -> float:
        # Full jitter, so concurrent callers that hit the same 429 don't retry in step.
        return random.uniform(0, 2 ** (attempt - 1))

    def _request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        last_error: Optional[OxylabsWebAPIError] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self._backoff(attempt))
            try:
                with httpx.Client(timeout=self.request_timeout) as client:
                    resp = client.request(
                        method, self._url(path), json=payload, headers=self._headers()
                    )
            except httpx.TimeoutException as exc:
                raise OxylabsWebAPIError(f"Request to {path} timed out.") from exc
            except httpx.HTTPError as exc:
                last_error = OxylabsWebAPIError(
                    f"Could not reach the Oxylabs Web API at {self._url(path)}: {exc}"
                )
                continue
            if attempt < self.max_retries and _retryable(resp, _json_body(resp)):
                continue
            return _check(resp, path)
        raise last_error or OxylabsWebAPIError(f"{path} failed after retries.")

    async def _arequest(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        last_error: Optional[OxylabsWebAPIError] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                await asyncio.sleep(self._backoff(attempt))
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                    resp = await client.request(
                        method, self._url(path), json=payload, headers=self._headers()
                    )
            except httpx.TimeoutException as exc:
                raise OxylabsWebAPIError(f"Request to {path} timed out.") from exc
            except httpx.HTTPError as exc:
                last_error = OxylabsWebAPIError(
                    f"Could not reach the Oxylabs Web API at {self._url(path)}: {exc}"
                )
                continue
            if attempt < self.max_retries and _retryable(resp, _json_body(resp)):
                continue
            return _check(resp, path)
        raise last_error or OxylabsWebAPIError(f"{path} failed after retries.")

    @staticmethod
    def _search_payload(
        query: str, max_results: int, location: Optional[str]
    ) -> Dict[str, Any]:
        if not query.strip():
            raise OxylabsWebAPIError("`query` must not be empty.")
        payload: Dict[str, Any] = {"query": query, "max_results": max_results}
        if location:
            payload["location"] = location.strip().upper()
        return payload

    @staticmethod
    def _scrape_payload(
        url: str,
        output: ScrapeFormat,
        location: Optional[str],
        device: Optional[Device],
        run_js: bool,
        json_prompt: Optional[str],
        json_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not url.startswith(("http://", "https://")):
            raise OxylabsWebAPIError(
                f"`url` must be an absolute http(s) URL, got: {url!r}"
            )
        payload: Dict[str, Any] = {"url": url, "output": [output]}
        if json_prompt or json_schema:
            # The AI parser: returns the named fields under `json` instead of the page.
            payload["output"] = ["json"]
            payload["json"] = {
                k: v for k, v in (("prompt", json_prompt), ("schema", json_schema)) if v
            }
        if location:
            payload["location"] = location.strip().upper()
        if device:
            payload["device"] = device
        if run_js:
            payload["run_js"] = True
        return payload

    def search(
        self, query: str, max_results: int = 10, location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search the web. Returns the API's response: `results`, each with
        `title`, `url` and `outline` (snippets), plus `related_questions` and
        `related_searches`.

        Args:
            query: What to search for.
            max_results: Maximum number of results, 1 to 20.
            location: ISO 3166-1 alpha-2 country code to rank results for, e.g. "DE".
        """
        payload = self._search_payload(query, max_results, location)
        return self._request("POST", "/v1/search", payload)

    async def asearch(
        self, query: str, max_results: int = 10, location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of `search`."""
        payload = self._search_payload(query, max_results, location)
        return await self._arequest("POST", "/v1/search", payload)

    def scrape(
        self,
        url: str,
        output: ScrapeFormat = "markdown",
        location: Optional[str] = None,
        device: Optional[Device] = None,
        run_js: bool = False,
        json_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch one page. Returns the API's response: `results`, each with the
        page under the key named by `output` (or `json` with `json_prompt` or
        `json_schema`) and its `metadata`.

        A scrape that outlives the synchronous call is queued and polled until it
        finishes or `max_wait` runs out.

        Args:
            url: Absolute http(s) URL.
            output: "markdown" (default) or "html".
            location: Two-letter country code to fetch the page from.
            device: "desktop" (the API's default) or "mobile".
            run_js: Render the page's JavaScript. Slower; use for pages that
                render client-side.
            json_prompt: Fields to extract, in plain words, using the AI parser.
            json_schema: JSON Schema the AI parser's output must follow.
        """
        payload = self._scrape_payload(
            url, output, location, device, run_js, json_prompt, json_schema
        )
        try:
            return self._request("POST", "/v1/scrape", payload)
        except _ScrapeTimeout:
            queued = self._request("POST", "/v1/async/scrape", payload)
            return self._poll(_request_id(queued))

    async def ascrape(
        self,
        url: str,
        output: ScrapeFormat = "markdown",
        location: Optional[str] = None,
        device: Optional[Device] = None,
        run_js: bool = False,
        json_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of `scrape`."""
        payload = self._scrape_payload(
            url, output, location, device, run_js, json_prompt, json_schema
        )
        try:
            return await self._arequest("POST", "/v1/scrape", payload)
        except _ScrapeTimeout:
            queued = await self._arequest("POST", "/v1/async/scrape", payload)
            return await self._apoll(_request_id(queued))

    def _poll(self, request_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.max_wait
        while True:
            body = self._request("GET", f"/v1/async/scrape/{request_id}")
            if body.get("state") != "pending":
                return body
            if time.monotonic() >= deadline:
                raise OxylabsWebAPIError(_still_pending(request_id, self.max_wait))
            time.sleep(self.poll_interval)

    async def _apoll(self, request_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.max_wait
        while True:
            body = await self._arequest("GET", f"/v1/async/scrape/{request_id}")
            if body.get("state") != "pending":
                return body
            if time.monotonic() >= deadline:
                raise OxylabsWebAPIError(_still_pending(request_id, self.max_wait))
            await asyncio.sleep(self.poll_interval)


def _still_pending(request_id: str, max_wait: float) -> str:
    return (
        f"Scrape {request_id} was still running after {max_wait:.0f}s."
        " Raise `max_wait` on OxylabsWebAPIWrapper for slow pages."
    )


def scrape_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The `results` of a scrape response, skipping anything that isn't a result."""
    results = response.get("results")
    return [r for r in results or [] if isinstance(r, dict)]

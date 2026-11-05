"""Tools for the Oxylabs Search API and the Oxylabs Web API."""

import json
from typing import Any, Dict, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_oxylabs.utilities import OxylabsSearchAPIWrapper
from langchain_oxylabs.web_api import (
    Device,
    OxylabsWebAPIError,
    OxylabsWebAPIWrapper,
    ScrapeFormat,
    scrape_results,
)


class OxylabsSearchQueryInput(BaseModel):
    """Input for the OxylabsSearch tool."""

    query: str = Field(description="query to retrieve on Oxylabs Search API")
    geo_location: Optional[str] = Field(
        default=None,
        description="Geographic location for the search;"
        " set this if location-specific information is requested."
        " Examples: `Vilnius,Lithuania`, `California,United States`, `DE`.",
    )


class OxylabsSearchRun(BaseTool):
    """Oxylabs Search Run tool.

    Setup:
        Install ``langchain-oxylabs``,
        and set environment variables ``OXYLABS_USERNAME`` and ``OXYLABS_PASSWORD``.

        .. code-block:: bash

            pip install langchain-oxylabs
            export OXYLABS_USERNAME="your-oxylabs-username"
            export OXYLABS_PASSWORD="your-oxylabs-password"

    Instantiation:
        .. code-block:: python

            from langchain_oxylabs import OxylabsSearchRun, OxylabsSearchAPIWrapper

            oxylabs_wrapper = OxylabsSearchAPIWrapper()
            tool = OxylabsSearchRun(wrapper=oxylabs_wrapper)

    Invocation with args:
        .. code-block:: python

            tool_.invoke({"query": "Visit restaurants in Vilnius."})

    Invocation with ToolCall:

        .. code-block:: python

            tool = OxylabsSearchRun(wrapper=oxylabs_wrapper, kwargs={"result_categories": ["local_information", "combined_search_result"]})
            model_generated_tool_call = {"args": {"query": "Visit restaurants in Vilnius.", "geo_location": "Vilnius,Lithuania"}, "id": "1", "name": "oxylabs_search", "type": "tool_call",}
            tool.invoke(model_generated_tool_call)

        .. code-block:: python

            ToolMessage(
                ('  LOCAL PACK: \n'
                 '  ITEMS ITEMS: \n'
                 '    ITEMS-ITEM-1: \n'
                 '    CID: 13950149882539119249\n'
                 '    POS: 1\n'
                 '    TITLE: Etno Dvaras\n'
                 '    RATING: 4.5\n'
                 '    ADDRESS: Lithuanian\n'
                 '    ITEMS-ITEM-2: \n'
                 '    CID: 711702509070991018\n'
                 '    POS: 2\n'
                 '    TITLE: Lokys\n'
                 '    RATING: 4.5\n'
                 '    ADDRESS: Lithuanian\n'
                 '    ITEMS-ITEM-3: \n'
                 '    CID: 7630589473191639738\n'
                 '    POS: 3\n'
                 '    TITLE: Senoji trobelė\n'
                 '    RATING: 4.4\n'
                 '    ADDRESS: Lithuanian\n'
                 '\n'
                 '  ORGANIC RESULTS ITEMS: \n'
                 '    ORGANIC-ITEM-1: \n'
                 '    POS: 1\n'
                 '    URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-Vilnius_Vilnius_County.html\n'
                 '    DESC: Some of the best restaurants in Vilnius for families with children '
                 'include: Momo grill Vilnius · Jurgis ir Drakonas Ogmios · RoseHip Vegan\xa0'
                 '...\n'
                 '    TITLE: THE 10 BEST Restaurants in Vilnius (Updated November ...\n'
                 '    SITELINKS: \n'
                 '      SITELINKS: \n'
                 '      INLINE ITEMS: \n'
                 '        INLINE-ITEM-1: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-zfp58-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Vilnius Dinner Restaurants\n'
                 '        INLINE-ITEM-2: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-zfp2-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Vilnius Breakfast Restaurants\n'
                 '        INLINE-ITEM-3: \n'
                 '        URL: '
                 'https://www.tripadvisor.com/Restaurants-g274951-c8-Vilnius_Vilnius_County.html\n'
                 '        TITLE: Cafés in Vilnius\n'
                 '    URL_SHOWN: https://www.tripadvisor.com› ... › Vilnius\n'
                 '    FAVICON_TEXT: Tripadvisor\n'
                 '    ORGANIC-ITEM-2: \n'
                 '    POS: 2\n'
                 '    URL: '
                 'https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania\n'
                 "    DESC: Jun 24, 2024 — Lithuania's capital has established itself as an "
                 "affordable culinary hotspot as four of the city's restaurants awarded "
                 'Michelin stars.\n'
                 '    TITLE: Star quality: a foodie guide to Vilnius\n'
                 '    URL_SHOWN: https://theweek.com› Culture & Life › Food & Drink\n'
                 '    FAVICON_TEXT: The Week\n'
                 '    ...
            )

    """  # noqa: E501

    name: str = "oxylabs_search"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a compiled, formatted summary of query results. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = {**self.kwargs}
        if geo_location:
            kwargs_["geo_location"] = geo_location

        return self.wrapper.run(query, **kwargs_)

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = {**self.kwargs}
        if geo_location:
            kwargs_["geo_location"] = geo_location

        return await self.wrapper.arun(query, **kwargs_)


class OxylabsSearchResults(BaseTool):
    """Oxylabs Search Results tool.

    Setup:
        Install ``langchain-oxylabs``,
        and set environment variables ``OXYLABS_USERNAME`` and ``OXYLABS_PASSWORD``.

        .. code-block:: bash

            pip install langchain-oxylabs
            export OXYLABS_USERNAME="your-oxylabs-username"
            export OXYLABS_PASSWORD="your-oxylabs-password"

    Instantiation:
        .. code-block:: python

            from langchain_oxylabs import OxylabsSearchRun, OxylabsSearchAPIWrapper

            oxylabs_wrapper = OxylabsSearchAPIWrapper()
            tool = OxylabsSearchResults(wrapper=oxylabs_wrapper)

    Invocation with args:
        .. code-block:: python

            response_results = tool.invoke({"query": "Visit restaurants in Vilnius."})
            response_results = json.loads(response_results)
            for item in response_results:
                print(item)

    Invocation with ToolCall:

        .. code-block:: python

            model_generated_tool_call = {"args": {"query": "Visit restaurants in Vilnius.", "geo_location": "Vilnius,Lithuania"}, "id": "1", "name": "oxylabs_search", "type": "tool_call",}
            tool.invoke(model_generated_tool_call)

        .. code-block:: python

            ToolMessage('[{"paid": [], "organic": [{"pos": 1, "url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-Vilnius_Vilnius_County.html", '
             '"desc": "Some of the best restaurants in Vilnius for families with children '
             'include: Momo grill Vilnius \\u00b7 Jurgis ir Drakonas Ogmios \\u00b7 '
             'RoseHip Vegan\\u00a0...", "title": "THE 10 BEST Restaurants in Vilnius '
             '(Updated November ...", "sitelinks": {"inline": [{"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-zfp58-Vilnius_Vilnius_County.html", '
             '"title": "Vilnius Dinner Restaurants"}, {"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-zfp2-Vilnius_Vilnius_County.html", '
             '"title": "Vilnius Breakfast Restaurants"}, {"url": '
             '"https://www.tripadvisor.com/Restaurants-g274951-c8-Vilnius_Vilnius_County.html", '
             '"title": "Caf\\u00e9s in Vilnius"}]}, "url_shown": '
             '"https://www.tripadvisor.com\\u203a ... \\u203a Vilnius", "pos_overall": 1, '
             '"favicon_text": "Tripadvisor"}, {"pos": 2, "url": '
             '"https://www.amsterdamfoodie.nl/2022/foodie-guide-to-vilnius-lithuania/", '
             '"desc": "Dec 2, 2022 \\u2014 Vilnius takes the top spot for most creative '
             'dining with the fabulous restaurant Amandus. Modern gastronomy, with a nod '
             'to Lithuanian tradition, goes hand in\\u00a0...", "title": "A Foodie\'s '
             'Guide to Vilnius, Lithuania", "url_shown": '
             '"https://www.amsterdamfoodie.nl\\u203a blog", "pos_overall": 2, '
             '"favicon_text": "Amsterdam Foodie"}, {"pos": 3, "url": '
             '"https://www.reddit.com/r/Vilnius/comments/uudd1p/good_restaurants_street_food_in_vilnius/", '
             '"desc": "- Alin\\u0117 Lei\\u010diai - also traditional peasant food of '
             'history but in a more upscale environment, and more expensive, table service '
             'and more diverse\\u00a0...", "title": "Good restaurants & street food in '
             'Vilnius?", "url_shown": "10+ comments  \\u00b7  2 years ago", "pos_overall": '
             '3, "favicon_text": "Reddit\\u00a0\\u00b7\\u00a0r/Vilnius"}, {"pos": 4, '
             '"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania", '
             '"desc": "Jun 24, 2024 \\u2014 Lithuania\'s capital has established itself as '
             "an affordable culinary hotspot as four of the city's restaurants awarded "
             'Michelin stars.", "title": "Star quality: a foodie guide to Vilnius", '
             '"sitelinks": {"inline": [{"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=Nineteen18,-With%20its%20%60%60sleek%20and", '
             '"title": "Nineteen18"}, {"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=D%C5%BEiaugsmas,-D%C5%BEiaugsmas", '
             '"title": "D\\u017eiaugsmas"}, {"url": '
             '"https://theweek.com/culture-life/food-drink/foodie-guide-to-vilnius-lithuania#:~:text=Gaspar%27s", '
             '"title": "Gaspar\'s"}]}, "url_shown": "https://theweek.com\\u203a Culture & '
             'Life \\u203a Food & Drink", "pos_overall": 4, "favicon_text": "The Week"}], '
             '"local_pack": {"items": [{"cid": "13950149882539119249", "pos": 1, "title": '
             '"Etno Dvaras", "rating": 4.5, "address": "Lithuanian", "rating_count": 0}, '
             '{"cid": "7630589473191639738", "pos": 2, "title": "Senoji trobel\\u0117", '
             '"rating": 4.4, "address": "Lithuanian", "rating_count": 0}, {"cid": '
             '"6362402554387935672", "pos": 3, "title": "Ertlio Namas", "rating": 4.7, '
             '"address": "\\u0160v. Jono g. 7", "subtitle": "Lithuanian", "rating_count": '
             '0}], "pos_overall": 5}, "related_searches": {"pos_overall": 6, '
             '"related_searches": ["Best restaurants in Vilnius old town", "Unique places '
             'to eat in Vilnius", "Best visit restaurants in vilnius", "Best Lithuanian '
             'restaurants in Vilnius", "Restaurants Vilnius old Town", "Vilnius '
             'restaurants Michelin", "Fine dining Vilnius", "Tripadvisor Vilnius '
             'Restaurants"]}, "search_information": {"query": "Visit restaurants in '
             'Vilnius.", "geo_location": "United States", "showing_results_for": "Visit '
             'restaurants in Vilnius.", "total_results_count": 1260000}, '
             '"total_results_count": 1260000}]'
         )
    """  # noqa: E501

    name: str = "oxylabs_search_results"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a JSON array of response page objects. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    model_config = ConfigDict(
        extra="allow",
    )

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = {**self.kwargs}
        if geo_location:
            kwargs_["geo_location"] = geo_location

        return json.dumps(self.wrapper.results(query, **kwargs_))

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = {**self.kwargs}
        if geo_location:
            kwargs_["geo_location"] = geo_location

        return json.dumps(await self.wrapper.aresults(query, **kwargs_))


# Below this many characters, a page fetched without JavaScript is likely an empty
# shell that renders client-side.
THIN_CONTENT_CHARS = 500


class OxylabsWebSearchInput(BaseModel):
    """Input for the OxylabsWebSearch tool."""

    query: str = Field(
        description="What to search for, as a natural search phrase. Keyword-shaped"
        " queries beat sentence-shaped ones. One question per search.",
        min_length=1,
        max_length=2048,
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return. The response may contain"
        " fewer.",
        ge=1,
        le=20,
    )
    location: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code to rank results for, e.g. `DE`,"
        " `US`, `LT`. Pass it whenever the answer is geographic. A place name such"
        " as `Germany` is rejected.",
        pattern="^[A-Za-z]{2}$",
    )


class OxylabsWebSearch(BaseTool):
    """Oxylabs Web API search tool.

    Setup:
        Install ``langchain-oxylabs`` and set ``OXYLABS_WEB_API_KEY`` to the key of
        a Web API instance from the Oxylabs dashboard.

        .. code-block:: bash

            pip install -U langchain-oxylabs
            export OXYLABS_WEB_API_KEY="your-web-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_oxylabs import OxylabsWebSearch

            tool = OxylabsWebSearch()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "best hiking boots", "location": "DE"})

        .. code-block:: python

            '{"results": [{"title": "...", "url": "https://...", "outline": ["..."]},'
            ' ...], "related_questions": ["..."]}'

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke(
                {
                    "args": {"query": "best hiking boots", "location": "DE"},
                    "id": "1",
                    "name": "oxylabs_web_search",
                    "type": "tool_call",
                }
            )
    """

    name: str = "oxylabs_web_search"
    description: str = (
        "Search the live web and return ranked organic results. Use for anything"
        " where being out of date makes the answer wrong: current events, news,"
        " prices, availability, versions, rankings, or any fact past your knowledge"
        " cutoff. Results are ranked as someone in the `location` country would see"
        " them. Returns titles, URLs and short snippets (`outline`), not page"
        " content: read the pages worth reading with `oxylabs_web_scrape`."
    )
    wrapper: OxylabsWebAPIWrapper = Field(default_factory=OxylabsWebAPIWrapper)
    args_schema: Type[BaseModel] = OxylabsWebSearchInput
    handle_tool_error: bool = True
    handle_validation_error: bool = True

    @staticmethod
    def _format(response: Dict[str, Any]) -> str:
        results = [
            {key: item[key] for key in ("title", "url", "outline") if key in item}
            for item in response.get("results") or []
            if isinstance(item, dict)
        ]
        questions = [
            item["question"]
            for item in response.get("related_questions") or []
            if isinstance(item, dict) and item.get("question")
        ]
        out: Dict[str, Any] = {"results": results}
        if questions:
            out["related_questions"] = questions
        return json.dumps(out, ensure_ascii=False)

    def _run(
        self,
        query: str,
        max_results: int = 10,
        location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            return self._format(self.wrapper.search(query, max_results, location))
        except OxylabsWebAPIError as exc:
            raise ToolException(str(exc)) from exc

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        location: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            response = await self.wrapper.asearch(query, max_results, location)
        except OxylabsWebAPIError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response)


class OxylabsWebScrapeInput(BaseModel):
    """Input for the OxylabsWebScrape tool."""

    url: str = Field(description="Absolute http(s) URL of the page to read.")
    output: ScrapeFormat = Field(
        default="markdown",
        description="`markdown` to read the page: far fewer tokens, structure intact."
        " `html` only when you need the markup itself. Ignored when `json_prompt` is"
        " set.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Two-letter country code to fetch the page from. Use it when the"
        " page varies by country: pricing, availability, language.",
        pattern="^[A-Za-z]{2}$",
    )
    device: Optional[Device] = Field(
        default=None,
        description="Viewport to fetch as. The API's default is desktop.",
    )
    run_js: bool = Field(
        default=False,
        description="Execute the page's JavaScript. Needed for pages that render"
        " client-side and come back empty otherwise. Slower: try without it first.",
    )
    json_prompt: Optional[str] = Field(
        default=None,
        description="Fields to pull out as JSON, in plain words: name them and say"
        " what shape you want, e.g. `the product title, price and a list of image"
        " links`. Leave unset to read the page yourself.",
    )


class OxylabsWebScrape(BaseTool):
    """Oxylabs Web API scrape tool.

    Setup:
        Install ``langchain-oxylabs`` and set ``OXYLABS_WEB_API_KEY`` to the key of
        a Web API instance from the Oxylabs dashboard.

        .. code-block:: bash

            pip install -U langchain-oxylabs
            export OXYLABS_WEB_API_KEY="your-web-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_oxylabs import OxylabsWebScrape

            tool = OxylabsWebScrape()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"url": "https://sandbox.oxylabs.io/products/1"})

        .. code-block:: python

            tool.invoke(
                {
                    "url": "https://sandbox.oxylabs.io/products/1",
                    "json_prompt": "the game title, price and platforms",
                }
            )

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke(
                {
                    "args": {"url": "https://sandbox.oxylabs.io/products/1"},
                    "id": "1",
                    "name": "oxylabs_web_scrape",
                    "type": "tool_call",
                }
            )
    """

    name: str = "oxylabs_web_scrape"
    description: str = (
        "Fetch and read a single URL, including JavaScript-heavy and bot-protected"
        " pages that an ordinary fetch can't open. Returns the page as Markdown by"
        " default. If the page comes back nearly empty, call again with"
        " `run_js` set to true. Set `json_prompt` to get named fields as JSON"
        " instead of the whole page."
    )
    wrapper: OxylabsWebAPIWrapper = Field(default_factory=OxylabsWebAPIWrapper)
    args_schema: Type[BaseModel] = OxylabsWebScrapeInput
    handle_tool_error: bool = True
    handle_validation_error: bool = True
    max_content_chars: Optional[int] = Field(
        default=50_000,
        description="Truncate page content to this many characters, so one large"
        " page doesn't fill the model's context. None disables truncation.",
    )

    def _format(self, response: Dict[str, Any], output: str, run_js: bool) -> str:
        results = scrape_results(response)
        if not results:
            raise ToolException(f"The scrape returned no results: {response}")

        if "json" in results[0]:
            parsed = [r.get("json") for r in results]
            return json.dumps(parsed[0] if len(parsed) == 1 else parsed)

        text = "\n\n".join(str(r.get(output) or "") for r in results)
        notes = []
        limit = self.max_content_chars
        if limit is not None and len(text) > limit:
            notes.append(
                f"[Truncated: showing {limit} of {len(text)} characters. Use"
                " `json_prompt` to pull out only the fields you need.]"
            )
            text = text[:limit]
        elif not run_js and len(text.strip()) < THIN_CONTENT_CHARS:
            notes.append(
                "[Very little content. If the page renders client-side, call again"
                " with `run_js` set to true.]"
            )
        return "\n\n".join([text, *notes]) if notes else text

    def _run(
        self,
        url: str,
        output: ScrapeFormat = "markdown",
        location: Optional[str] = None,
        device: Optional[Device] = None,
        run_js: bool = False,
        json_prompt: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            response = self.wrapper.scrape(
                url, output, location, device, run_js, json_prompt
            )
        except OxylabsWebAPIError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response, output, run_js)

    async def _arun(
        self,
        url: str,
        output: ScrapeFormat = "markdown",
        location: Optional[str] = None,
        device: Optional[Device] = None,
        run_js: bool = False,
        json_prompt: Optional[str] = None,
        # run_manager is added to enable tracing
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            response = await self.wrapper.ascrape(
                url, output, location, device, run_js, json_prompt
            )
        except OxylabsWebAPIError as exc:
            raise ToolException(str(exc)) from exc
        return self._format(response, output, run_js)

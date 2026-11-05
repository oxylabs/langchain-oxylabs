# langchain-oxylabs

LangChain integration for the Oxylabs **Web API** and the Oxylabs
[**Web Scraper API**](https://oxylabs.io/products/scraper-api). Gives LangChain
agents live web search and page reading, and bulk-scrapes pages into LangChain
`Document` objects for retrieval-augmented generation.

[![PyPI version](https://img.shields.io/pypi/v/langchain-oxylabs?style=flat-square)](https://pypi.org/project/langchain-oxylabs/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-oxylabs?style=flat-square)](https://pypi.org/project/langchain-oxylabs/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-oxylabs?style=flat-square)](https://pypi.org/project/langchain-oxylabs/)
[![License](https://img.shields.io/pypi/l/langchain-oxylabs?style=flat-square)](https://github.com/oxylabs/langchain-oxylabs/blob/main/LICENSE)

[![](https://dcbadge.limes.pink/api/server/Pds3gBmKMH?style=for-the-badge&theme=discord)](https://discord.gg/Pds3gBmKMH) [![YouTube](https://img.shields.io/badge/YouTube-Oxylabs-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@oxylabs)

## Why use this

A language model's knowledge is fixed at training time and it cannot browse the
web on its own. This package closes that gap in two ways:

- **As an agent tool** — the model decides when to search and for what, including
  which geographic location to search from, and gets back live results.
- **As a document loader** — you bulk-scrape URLs or search queries ahead of
  time, then chunk, embed, and retrieve them in a RAG pipeline.

## Requirements

- Python 3.10 or newer
- Works with both `langchain-core` 0.3.x and LangChain v1 (`langchain-core` 1.x)

## Installation

```bash
pip install -U langchain-oxylabs
```

## Which Oxylabs product

The package covers two Oxylabs products. They use different credentials, so use
the classes for the product you have.

| Product | Credentials | Classes |
| :--- | :--- | :--- |
| **Web API** | One API key: `OXYLABS_WEB_API_KEY` | `OxylabsWebSearch`, `OxylabsWebScrape`, `OxylabsWebLoader`, `OxylabsWebAPIWrapper` |
| **Web Scraper API** | Username and password: `OXYLABS_USERNAME`, `OXYLABS_PASSWORD` | `OxylabsSearchRun`, `OxylabsSearchResults`, `OxylabsLoader`, `OxylabsSearchAPIWrapper` |

## Web API

### Credentials

Create a Web API instance in the
[Oxylabs dashboard](https://dashboard.oxylabs.io/) and copy its API key. Keys
from other Oxylabs products don't work with the Web API.

```bash
export OXYLABS_WEB_API_KEY="your-web-api-key"
```

The key can also be passed directly:

```python
from langchain_oxylabs import OxylabsWebAPIWrapper, OxylabsWebSearch

tool = OxylabsWebSearch(
    wrapper=OxylabsWebAPIWrapper(oxylabs_web_api_key="your-web-api-key")
)
```

### What to use when

| Class | Returns | Use it for |
| :--- | :--- | :--- |
| `OxylabsWebSearch` | JSON string: titles, URLs, snippets | Letting an agent search the live web. |
| `OxylabsWebScrape` | Page as Markdown (or HTML), or JSON fields | Letting an agent read a URL, including JavaScript-heavy and bot-protected pages. |
| `OxylabsWebLoader` | `Document` objects | Loading a list of URLs for RAG indexing. |
| `OxylabsWebAPIWrapper` | Raw API responses | Calling the API from your own code, or configuring timeouts and retries. |

### Search and scrape tools

```python
from langchain_oxylabs import OxylabsWebScrape, OxylabsWebSearch

search = OxylabsWebSearch()
print(search.invoke({"query": "best hiking boots", "max_results": 5, "location": "DE"}))
# {"results": [{"title": "...", "url": "https://...", "outline": ["..."]}, ...],
#  "related_questions": ["..."]}

scrape = OxylabsWebScrape()
print(scrape.invoke({"url": "https://sandbox.oxylabs.io/products/1"}))
```

`location` is a two-letter country code (`DE`, `US`, `LT`). Search results are
ranked as someone in that country would see them, and pages are fetched from
there.

Scrape options:

| Argument | Default | Purpose |
| :--- | :--- | :--- |
| `output` | `markdown` | `markdown` or `html` |
| `location` | unset | Country to fetch the page from |
| `device` | desktop | `desktop` or `mobile` |
| `run_js` | `False` | Render the page's JavaScript. Slower; use it when a page comes back nearly empty. |
| `json_prompt` | unset | Fields to extract, in plain words. Returns them as JSON instead of the page. |

```python
scrape.invoke(
    {
        "url": "https://sandbox.oxylabs.io/products/1",
        "json_prompt": "the game title, price and platforms",
    }
)
# '{"game_title": "The Legend of Zelda: Ocarina of Time", "price": "91,99 €", ...}'
```

Pages longer than 50,000 characters are truncated so one page doesn't fill the
model's context. Change it with `OxylabsWebScrape(max_content_chars=...)`, or
pass `None` to turn it off.

API errors (a bad key, a spent quota, an invalid argument) are returned to the
agent as the tool's output rather than raised, so the agent can correct itself.
Calling `OxylabsWebAPIWrapper` directly raises `OxylabsWebAPIError` instead.

Both tools support `ainvoke` for async use.

### Using with an agent

```python
from langchain.agents import create_agent
from langchain_oxylabs import OxylabsWebScrape, OxylabsWebSearch

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[OxylabsWebSearch(), OxylabsWebScrape()],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What do hiking boots cost in Germany?"}]}
)
print(result["messages"][-1].content)
```

The model searches, then reads the pages worth reading, and picks `location`
from the question.

### Document loader

```python
from langchain_oxylabs import OxylabsWebLoader

loader = OxylabsWebLoader(
    urls=[
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ],
    location="DE",
)

for document in loader.lazy_load():
    print(document.metadata)  # source, final_url, status_code, request_id
    print(document.page_content[:250])
```

`OxylabsWebLoader` takes the same `output`, `location`, `device` and `run_js`
options as the scrape tool. `aload()` and `alazy_load()` are available for async
code.

### Configuring the wrapper

| Option | Default | Purpose |
| :--- | :--- | :--- |
| `request_timeout` | `200` | Seconds to wait for one HTTP response |
| `max_retries` | `2` | Extra attempts after a 429, 5xx or connection error |
| `max_wait` | `600` | Seconds to wait for a slow scrape |
| `poll_interval` | `5` | Seconds between checks on a slow scrape |

A scrape that takes longer than the API's synchronous limit (often with
`run_js`) is queued automatically and checked every `poll_interval` seconds until
it finishes or `max_wait` runs out.

## Web Scraper API

### Credentials

Sign up for a free trial or purchase the product in the
[Oxylabs dashboard](https://dashboard.oxylabs.io/en/registration) to create your
API user credentials, then set:

```bash
export OXYLABS_USERNAME="your-oxylabs-username"
export OXYLABS_PASSWORD="your-oxylabs-password"
```

Credentials can also be passed directly:

```python
OxylabsSearchAPIWrapper(
    oxylabs_username="your-oxylabs-username",
    oxylabs_password="your-oxylabs-password",
)
```

### What to use when

| Class | Returns | Use it for |
| :--- | :--- | :--- |
| `OxylabsSearchRun` | Formatted plain text | Giving an agent readable search results. The flattened text costs fewer tokens than raw JSON and models parse it reliably. |
| `OxylabsSearchResults` | JSON string | When *your code*, not the model, consumes the results. |
| `OxylabsLoader` | `Document` objects | Bulk scraping for RAG indexing. |
| `OxylabsSearchAPIWrapper` | — | Shared client and configuration for both search tools. |

### Search tools

```python
import json
from langchain_oxylabs import (
    OxylabsSearchAPIWrapper,
    OxylabsSearchResults,
    OxylabsSearchRun,
)

oxylabs_wrapper = OxylabsSearchAPIWrapper()

# Formatted text, for a model to read
run_tool = OxylabsSearchRun(wrapper=oxylabs_wrapper)
print(run_tool.invoke({"query": "Visit restaurants in Vilnius."}))

# JSON, for your own code to consume
results_tool = OxylabsSearchResults(wrapper=oxylabs_wrapper)
for result in json.loads(results_tool.invoke({"query": "Visit restaurants in Paris."})):
    for key, value in result.items():
        print(f"{key}: {value}")
```

Both tools also support `ainvoke` for async use.

> **Note:** the search tools query Google (`google_search`). Other Oxylabs
> sources such as `amazon_search` are available through `OxylabsLoader`.

#### Geo-targeted results

`geo_location` scopes results to a location — the main reason to use Oxylabs for
search data. Supply it per call, or configure a default on the wrapper:

```python
# Per call. An agent can set this itself from the user's question.
run_tool.invoke({"query": "best coffee", "geo_location": "Vilnius,Lithuania"})

# Or as the default for every search through this wrapper
oxylabs_wrapper = OxylabsSearchAPIWrapper(
    params={"geo_location": "Vilnius,Lithuania"}
)
```

A per-call value takes precedence over the wrapper default. If neither is set,
Oxylabs applies its own default.

#### Configuring the wrapper

`OxylabsSearchAPIWrapper(params={...})` accepts
[Oxylabs Web Scraper API parameters](https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/generic-target#additional).
Frequently useful ones:

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `geo_location` | unset | Location to search from, e.g. `Vilnius,Lithuania` |
| `domain` | `com` | Google domain, e.g. `lt` for google.lt |
| `pages` | `1` | Number of result pages to retrieve |
| `limit` | `5` | Results per page |
| `user_agent_type` | `desktop` | `desktop` or `mobile` |
| `locale` | unset | Interface language |

The wrapper itself also accepts:

| Option | Default | Purpose |
| :--- | :--- | :--- |
| `result_categories` | all | Restrict which result sections are returned (see below) |
| `include_binary_image_data` | `False` | Keep base64 image data instead of redacting it. Redaction keeps token counts down. |
| `parsing_recursion_depth` | `5` | How deep to walk nested results when formatting text |

Valid `result_categories` values: `knowledge_graph`, `combined_search_result`,
`product_information`, `local_information`, `search_information`. Restricting
them trims irrelevant sections and reduces tokens:

```python
# Only local results -- useful for "restaurants near X" style questions
wrapper = OxylabsSearchAPIWrapper(
    params={"result_categories": ["local_information"]}
)
```

### Using with an agent

```python
from langchain.agents import create_agent
from langchain_oxylabs import OxylabsSearchAPIWrapper, OxylabsSearchRun

tool = OxylabsSearchRun(wrapper=OxylabsSearchAPIWrapper())
agent = create_agent(model="anthropic:claude-sonnet-5", tools=[tool])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find good coffee in Vilnius"}]}
)
print(result["messages"][-1].content)
```

The model chooses when to search and which `geo_location` to use, based on the
question it is answering.

### Document loader

`OxylabsLoader` scrapes into LangChain
[`Document`](https://python.langchain.com/docs/concepts/documents/) objects for
RAG pipelines. Pass either `urls` or `queries`, plus any Oxylabs API parameters.

```python
from langchain_oxylabs import OxylabsLoader

# Scrape specific URLs as markdown
loader = OxylabsLoader(
    urls=[
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ],
    params={"markdown": True},
)

for document in loader.lazy_load():
    print(document.metadata)          # url (or query) and created_at
    print(document.page_content[:250])
```

```python
# Or scrape parsed search results for a list of queries
loader = OxylabsLoader(
    queries=["gaming headset", "gaming chair"],
    params={
        "source": "amazon_search",
        "parse": True,
        "geo_location": "DE",
        "currency": "EUR",
        "pages": 3,
    },
)

documents = loader.load()
```

Each document carries `url` (or `query`) and `created_at` in its metadata, so
retrieved facts can be traced back to their source and freshness.

`lazy_load()` issues one request per URL or query and yields documents as they
arrive; prefer it to `load()` for large batches.

## Troubleshooting

### Web API

**`Oxylabs Web API key is missing`** — `OXYLABS_WEB_API_KEY` is unset. Set it,
or pass `oxylabs_web_api_key` to `OxylabsWebAPIWrapper`.

**`Authentication failed (401)`** — the key is wrong, or it belongs to another
Oxylabs product. Copy the key of a Web API instance from the dashboard.

**`String should match pattern '^[A-Za-z]{2}$'`** — `location` must be a
two-letter country code such as `DE`, not a place name.

**A page comes back nearly empty** — it renders client-side. Call the scrape
again with `run_js=True`.

**`... with state "faulted"`** — the API could not fetch the page after its own
retries. Faulted requests are not charged. If it keeps happening, quote the
`request_id` from the message to Oxylabs support.

### Web Scraper API

**`RuntimeError: Please set up environment variables`** — `OXYLABS_USERNAME` and
`OXYLABS_PASSWORD` are unset, or contain a stray newline or space from being
copied. Pass them explicitly to `OxylabsSearchAPIWrapper` to rule the
environment out.

**`NotImplementedError: Source ... is not supported`** — the search tools only
support `google_search`. Use `OxylabsLoader` for other Oxylabs sources.

**Empty results / `"No good search result found"`** — the API occasionally
returns a response the wrapper cannot parse, and this is currently reported as an
empty result rather than an error. Retrying usually succeeds.

**Slow calls** — scraping is inherently slower than a normal HTTP request; single
calls can take tens of seconds. `request_timeout` (default 165s) is configurable.

## Development

```bash
poetry install --with test,lint,typing

make test               # unit tests, no network access
make lint               # ruff + mypy
make integration_tests  # live API calls, requires credentials for both products
```

## Links

- [Oxylabs Web Scraper API documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api)
- [Release notes](https://github.com/oxylabs/langchain-oxylabs/releases)
- [LangChain tools concept guide](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain agents guide](https://docs.langchain.com/oss/python/langchain/agents)

## License

MIT — see [LICENSE](LICENSE).

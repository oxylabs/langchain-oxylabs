# langchain-oxylabs

LangChain integration for the [Oxylabs Web Scraper API](https://oxylabs.io/products/scraper-api).
Gives LangChain agents live web search, and bulk-scrapes pages into LangChain
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

## Credentials

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

## What to use when

| Class | Returns | Use it for |
| :--- | :--- | :--- |
| `OxylabsSearchRun` | Formatted plain text | Giving an agent readable search results. The flattened text costs fewer tokens than raw JSON and models parse it reliably. |
| `OxylabsSearchResults` | JSON string | When *your code*, not the model, consumes the results. |
| `OxylabsLoader` | `Document` objects | Bulk scraping for RAG indexing. |
| `OxylabsSearchAPIWrapper` | — | Shared client and configuration for both search tools. |

## Search tools

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

### Geo-targeted results

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

### Configuring the wrapper

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

## Using with an agent

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

## Document loader

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
make integration_tests  # live API calls, requires credentials
```

## Links

- [Oxylabs Web Scraper API documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api)
- [Release notes](https://github.com/oxylabs/langchain-oxylabs/releases)
- [LangChain tools concept guide](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain agents guide](https://docs.langchain.com/oss/python/langchain/agents)

## License

MIT — see [LICENSE](LICENSE).

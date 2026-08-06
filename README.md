# langchain-oxylabs

This package contains the LangChain integration with Oxylabs, providing tools to scrape Google search results 
with Oxylabs Web Scraper API using LangChain's framework.

[![](https://dcbadge.limes.pink/api/server/Pds3gBmKMH?style=for-the-badge&theme=discord)](https://discord.gg/Pds3gBmKMH) [![YouTube](https://img.shields.io/badge/YouTube-Oxylabs-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@oxylabs)

## Installation

```bash
pip install -U langchain-oxylabs
```

## Credentials
Create your API user credentials: Sign up for a free trial or purchase the product
in the [Oxylabs dashboard](https://dashboard.oxylabs.io/en/registration)
to create your API user credentials.

Configure your Oxylabs credentials by setting the following environment variables:
- `OXYLABS_USERNAME` - Your Oxylabs API username
- `OXYLABS_PASSWORD` - Your Oxylabs API password

## Requirements

- Python 3.10 or newer
- Works with both `langchain-core` 0.3.x and LangChain v1 (`langchain-core` 1.x)

## Usage
`langchain_oxylabs` package provides the following classes:
- `OxylabsSearchRun` - A tool that returns scraped Google search results in a formatted text
- `OxylabsSearchResults` - A tool that returns scraped Google search results in a JSON format
- `OxylabsSearchAPIWrapper` - An API wrapper for initializing Oxylabs API
- `OxylabsLoader` - A document loader that scrapes URLs or search queries into LangChain `Document` objects

### Search tools

Here is an example usage of these classes:

```python
import json
from langchain_oxylabs import OxylabsSearchRun, OxylabsSearchResults, OxylabsSearchAPIWrapper

# Initialize the API wrapper
oxylabs_wrapper = OxylabsSearchAPIWrapper()

# Initialize the search run tool
run_tool = OxylabsSearchRun(wrapper=oxylabs_wrapper)

# Invoke the tool and print results
results_text = run_tool.invoke({"query": "Visit restaurants in Vilnius."})
print(results_text)

# Initialize the search results tool
results_tool = OxylabsSearchResults(wrapper=oxylabs_wrapper)

# Invoke the tool and print results
response_results = results_tool.invoke({"query": "Visit restaurants in Paris."})
response_results = json.loads(response_results)
for result in response_results:
    for key, value in result.items():
        print(f"{key}: {value}")
```

#### Geo-targeted results

Pass `geo_location` to scope results to a location. It can be supplied per call
(an agent may set it from the user's question), or configured once on the wrapper
as a default for every search:

```python
# Per call -- takes precedence
run_tool.invoke({"query": "best coffee", "geo_location": "Vilnius,Lithuania"})

# Or as a default for this wrapper
oxylabs_wrapper = OxylabsSearchAPIWrapper(
    params={"geo_location": "Vilnius,Lithuania"}
)
```

### Document loader

`OxylabsLoader` scrapes content into LangChain `Document` objects, which makes it
convenient for retrieval-augmented generation. Pass either `urls` or `queries`,
plus any [Oxylabs Web Scraper API parameters](https://developers.oxylabs.io/scraper-apis/web-scraper-api).

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
    print(document.metadata)
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

`lazy_load()` issues one request per URL or query and yields documents as they
arrive, so it is preferable to `load()` for large batches.

## Using with an agent

The search tools work with LangChain's agents. `create_agent` decides when to
search and with what `geo_location`:

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

## Development

```bash
poetry install --with test,lint,typing

make test               # unit tests, no network access
make lint               # ruff + mypy
make integration_tests  # live API calls, requires credentials
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.


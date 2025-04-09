# langchain-oxylabs

This package contains the LangChain integration with Oxylabs, providing tools to scrape Google search results 
with Oxylabs Web Scraper API using LangChain's framework.

[![](https://dcbadge.vercel.app/api/server/eWsVUJrnG5)](https://discord.gg/Pds3gBmKMH)

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

## Usage
`langchain_oxylabs` package provides the following classes:
- `OxylabsSearchRun` - A tool that returns scraped Google search results in a formatted text
- `OxylabsSearchResults` - A tool that returns scraped Google search results in a JSON format
- `OxylabsSearchAPIWrapper` - An API wrapper for initializing Oxylabs API

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

## License
This project is licensed under the MIT License - see the LICENSE file for details.


import json
import os
from typing import Any, Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_oxylabs.utilities import get_sdk_type


class OxylabsLoader(BaseLoader):
    """
    Oxylabs document loader integration.

    Setup:
        Install ``oxylabs``, ``langchain_community`` and set ``OXYLABS_USERNAME``, ``OXYLABS_PASSWORD`` environment variables.

        .. code-block:: bash

            pip install -U oxylabs langchain_community
            export OXYLABS_USERNAME=OXYLABS_USERNAME
            export OXYLABS_PASSWORD=OXYLABS_PASSWORD

    Usage example:
        .. code-block:: python
            from langchain_community.document_loaders import OxylabsLoader

            loader = OxylabsLoader(
                urls=[
                    "https://sandbox.oxylabs.io/products/1",
                    "https://sandbox.oxylabs.io/products/2",
                ],
                params={"markdown": True},
            )

            documents = loader.lazy_load()

            print(documents[0].page_content[:250])

            # [![](data:image/svg+xml...)![logo](data:image/gif;base64...)![logo](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FnavLogo.a8764883.png&w=750&q=75)](/)
            #
            # Game platforms:
            #
            # * **All**
            #
            # * [Nintendo platform](/products/category/nintendo)
            #
            # + wii
            # + wii-u
            # + nintendo-64
            # + switch
            # + gamecube
            # + game-boy-advance
            # + 3ds
            # + ds

    Advanced examples:
        .. code-block:: python
            loader = OxylabsLoader(
                queries=["gaming headset", "gaming chair", "computer mouse"],
                params={"source": "amazon_search", "parse": True, "geo_location": "DE", "currency": "EUR", "pages": 3},
            )

        .. code-block:: python
            loader = OxylabsLoader(
                queries=["europe gdp per capita", "us gdp per capita"],
                params={"source": "google_search", "parse": True, "geo_location": "Paris, France", "user_agent_type": "mobile"},
            )
    """  # noqa: E501

    DEFAULT_REQUEST_TIMEOUT: int = 165

    def __init__(
        self,
        params: dict[str, Any],
        urls: list[str] | None = None,
        queries: list[str] | None = None,
        *,
        oxylabs_username: Optional[str] = None,
        oxylabs_password: Optional[str] = None,
        request_timeout: int | None = DEFAULT_REQUEST_TIMEOUT,
    ):
        """Oxylabs document loader integration.

        Args:
            params: Oxylabs API parameters as described [here](https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/generic-target#additional).
            oxylabs_username: Oxylabs API username.
            oxylabs_password: Oxylabs API password.
        """

        if urls is None and queries is None:
            raise ValueError("Either `urls` or `queries` must be provided.")

        try:
            from oxylabs.internal.api import (  # type: ignore[import-untyped]
                APICredentials,
                RealtimeAPI,
            )
        except ImportError:
            raise ImportError(
                "`oxylabs` package not found, please run `pip install oxylabs`"
            )

        oxylabs_username = oxylabs_username or os.getenv("OXYLABS_USERNAME")
        oxylabs_password = oxylabs_password or os.getenv("OXYLABS_PASSWORD")
        credentials = APICredentials(oxylabs_username, oxylabs_password)
        self._oxylabs_api = RealtimeAPI(credentials, sdk_type=get_sdk_type())

        self._urls = urls
        self._queries = queries
        self._params = params

        self._config = {"request_timeout": request_timeout}

    @staticmethod
    def _get_content_from_response(response: Any) -> str:
        try:
            result_page = response["results"][0]

            result_page = dict(result_page)
            content = result_page["content"]

            if isinstance(content, dict):
                return json.dumps(content)
            elif isinstance(content, str):
                return content
            else:
                raise ValueError(
                    "Unexpected content type: {}: {}".format(
                        type(content), str(content)[:100]
                    )
                )

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Response Validation Error: {exc!s}") from exc

    @staticmethod
    def _get_metadata_from_response(response: Any) -> dict[str, Any]:
        metadata = {}

        attributes = ["url", "query", "created_at"]
        for attr in attributes:
            if response["job"].get(attr):
                metadata[attr] = response["job"][attr]

        return metadata

    def lazy_load(self) -> Iterator[Document]:
        """
        Load data from Oxylabs API into the list of Documents.
        """
        if self._urls is not None:
            params_list = [{"url": url, **self._params} for url in self._urls]

        elif self._queries is not None:
            params_list = [{"query": query, **self._params} for query in self._queries]
        else:
            raise ValueError("Either `urls` or `queries` must be provided.")

        for params in params_list:
            response = self._oxylabs_api.get_response(params, self._config)

            content = self._get_content_from_response(response)
            metadata = self._get_metadata_from_response(response)

            yield Document(page_content=content, metadata=metadata)

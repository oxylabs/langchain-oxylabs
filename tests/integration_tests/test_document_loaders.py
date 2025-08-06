"""Test OxylabsLoader."""

import json
import os
from typing import Any

import pytest

from langchain_oxylabs import OxylabsLoader


@pytest.mark.skipif(
    os.getenv("OXYLABS_USERNAME") is None or os.getenv("OXYLABS_PASSWORD") is None,
    reason="Oxylabs username and password are required",
)
@pytest.mark.parametrize(
    "loader_args",
    [
        {
            "urls": [
                "https://sandbox.oxylabs.io/products/1",
                "https://sandbox.oxylabs.io/products/2",
            ],
            "params": {"parse": True},
        },
        {
            "queries": ["gaming headsets", "gaming chairs"],
            "params": {"source": "amazon_search", "parse": True},
        },
        {
            "queries": ["iPhone 14", "iPhone 15", "iPhone 16"],
            "params": {"source": "google_search", "parse": True},
        },
    ],
)
def test_oxylabs_loader_lazy_load_parsed(
    loader_args: dict[str, Any],
) -> None:
    loader = OxylabsLoader(**loader_args)

    for document in loader.lazy_load():
        metadata = document.metadata

        if "urls" in loader_args:
            assert metadata["url"] in loader_args["urls"]
        if "queries" in loader_args:
            assert metadata["query"] in loader_args["queries"]

        assert metadata["created_at"] is not None

        assert isinstance(json.loads(document.page_content), dict)


@pytest.mark.skipif(
    os.getenv("OXYLABS_USERNAME") is None or os.getenv("OXYLABS_PASSWORD") is None,
    reason="Oxylabs username and password are required",
)
@pytest.mark.parametrize(
    ("loader_args", "expected_content"),
    [
        (
            {
                "urls": ["https://sandbox.oxylabs.io/products/1"],
                "params": {"markdown": True},
            },
            "Zelda",
        ),
        (
            {
                "queries": ["gaming headsets"],
                "params": {"source": "amazon_search", "markdown": True},
            },
            "headset",
        ),
        (
            {
                "queries": ["iPhone 14"],
                "params": {"source": "google_search", "markdown": True},
            },
            "iPhone",
        ),
    ],
)
def test_oxylabs_loader_lazy_load_markdown(
    loader_args: dict[str, Any],
    expected_content: str,
) -> None:
    loader = OxylabsLoader(**loader_args)

    for document in loader.lazy_load():
        metadata = document.metadata

        if "urls" in loader_args:
            assert metadata["url"] in loader_args["urls"]
        if "queries" in loader_args:
            assert metadata["query"] in loader_args["queries"]

        assert metadata["created_at"] is not None

        assert expected_content in document.page_content

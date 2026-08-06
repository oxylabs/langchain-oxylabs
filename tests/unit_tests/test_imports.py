"""Guard the public surface of the package."""

import langchain_oxylabs

EXPECTED_ALL = [
    "OxylabsSearchRun",
    "OxylabsSearchResults",
    "OxylabsSearchAPIWrapper",
    "OxylabsLoader",
    "__version__",
]


def test_all_is_stable() -> None:
    assert sorted(langchain_oxylabs.__all__) == sorted(EXPECTED_ALL)


def test_every_export_is_importable() -> None:
    for name in langchain_oxylabs.__all__:
        assert getattr(langchain_oxylabs, name) is not None


def test_version_is_populated() -> None:
    assert langchain_oxylabs.__version__

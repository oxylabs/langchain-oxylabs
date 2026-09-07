"""Shared assertions for the live integration tests.

These deliberately check the *shape* of Oxylabs responses rather than their
text. Asserting on scraped wording (an exact Google snippet, say) fails at
random as search results change, which makes the suite useless as a signal.
"""

from typing import Any

# Returned by `_process_response` when nothing could be parsed out of the
# response. Asserting against it turns these tests into a health check: an
# Oxylabs-side fault (for example a 613 "faulted after too many retries")
# surfaces as this placeholder rather than as an error.
NO_RESULTS_PLACEHOLDER = "No good search result found"

# Section headers emitted by the text formatter. At least one must appear in a
# real result set.
FORMATTER_MARKERS = ("ITEMS:", "RESULTS", "INFORMATION")

# Keys that appear at the top level of a parsed Google SERP. At least one must
# be present; which ones vary by query and over time.
SERP_KEYS = {
    "organic",
    "paid",
    "local_pack",
    "knowledge",
    "related_searches",
    "related_questions",
    "search_information",
    "top_stories",
    "popular_products",
    "pla",
}

FAULT_HINT = (
    "This usually means an Oxylabs API-side fault rather than a problem with"
    " this package. Check the result `status_code` in the raw response."
)


def assert_formatted_output(output: Any) -> None:
    """Check formatted search text looks like a real, parsed result set."""
    assert isinstance(output, str)
    assert (
        output != NO_RESULTS_PLACEHOLDER
    ), f"Oxylabs returned no parseable results. {FAULT_HINT}"
    assert len(output) > 200, f"suspiciously short output: {output[:200]!r}"
    assert any(
        marker in output for marker in FORMATTER_MARKERS
    ), f"no formatter section headers found in output: {output[:200]!r}"


def assert_results_payload(results: Any) -> None:
    """Check the structured result pages look like real SERP data."""
    assert isinstance(results, list)
    assert results, f"Oxylabs returned no result pages. {FAULT_HINT}"
    assert all(isinstance(page, dict) for page in results)
    assert any(
        SERP_KEYS & set(page) for page in results
    ), f"no recognized SERP keys in any page: {[sorted(p)[:5] for p in results]}"

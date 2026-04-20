from __future__ import annotations

import re

from nvd_claude_proxy.util.anthropic_headers import (
    new_request_id,
    standard_response_headers,
)


def test_request_id_shape():
    rid = new_request_id()
    assert re.fullmatch(r"req_[a-f0-9]{20}", rid)


def test_standard_headers_include_required_keys():
    h = standard_response_headers("req_test")
    # Every Anthropic SDK I know reads at least these keys.
    required = {
        "anthropic-request-id",
        "request-id",
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
    }
    assert required.issubset(h.keys())
    assert h["anthropic-request-id"] == "req_test"
    assert h["request-id"] == "req_test"


def test_ratelimit_reset_is_iso8601():
    h = standard_response_headers("req_test")
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        h["anthropic-ratelimit-requests-reset"],
    )

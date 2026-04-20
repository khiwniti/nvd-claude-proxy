from __future__ import annotations

from nvd_claude_proxy.util.tokens import approximate_tokens


def test_simple_body():
    body = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }
    n = approximate_tokens(body)
    assert 3 < n < 50


def test_skips_base64_image_data():
    big = "A" * 100_000
    body = {
        "model": "claude-opus-4-7",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": big,
                        },
                    },
                ],
            }
        ],
    }
    n = approximate_tokens(body)
    # Must NOT be proportional to the 100k-byte base64 blob.
    assert n < 500

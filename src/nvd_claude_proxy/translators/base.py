from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseTransformer(abc.ABC):
    """Abstract base class for modular translators/transformers.

    Each transformer can hook into the request (Anthropic -> NVIDIA),
    response (NVIDIA -> Anthropic), or stream chunk (NVIDIA -> Anthropic)
    to perform custom logic, sanitization, or feature injection.
    """

    def name(self) -> str:
        return self.__class__.__name__

    def transform_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the outgoing request payload.

        This is called *after* initial translation from Anthropic to OpenAI,
        but before it is sent to NVIDIA NIM.
        """
        return payload

    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the incoming non-streaming response.

        This is called *after* initial translation from OpenAI to Anthropic,
        but before it is returned to the client.
        """
        return response

    def transform_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a single streaming chunk (Anthropic-formatted).

        Returning None will drop the chunk from the stream.
        """
        return chunk


class TransformerChain:
    """A collection of transformers applied sequentially."""

    def __init__(self, transformers: List[BaseTransformer] | None = None) -> None:
        self.transformers = transformers or []

    def transform_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for transformer in self.transformers:
            payload = transformer.transform_request(payload)
        return payload

    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        for transformer in self.transformers:
            response = transformer.transform_response(response)
        return response

    def transform_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for transformer in self.transformers:
            chunk = transformer.transform_stream_chunk(chunk)
            if chunk is None:
                return None
        return chunk

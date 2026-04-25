from __future__ import annotations

import orjson
from dataclasses import dataclass


def encode_sse(event: str, data: dict) -> bytes:
    """Anthropic SSE frame: `event: X\ndata: {json}\n\n`."""
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


@dataclass
class SSEEvent:
    event: str
    data: str
    id: str | None = None
    retry: int | None = None


class SSEDecoder:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._event = "message"
        self._data: list[str] = []
        self._id: str | None = None
        self._retry: int | None = None

    def decode(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer.extend(chunk)
        events = []
        while True:
            idx1 = self._buffer.find(b"\n\n")
            idx2 = self._buffer.find(b"\r\n\r\n")
            if idx1 == -1 and idx2 == -1:
                break

            if idx1 != -1 and (idx2 == -1 or idx1 < idx2):
                idx = idx1
                sep_len = 2
            else:
                idx = idx2
                sep_len = 4

            raw_event = bytes(self._buffer[:idx])
            del self._buffer[: idx + sep_len]

            lines = raw_event.split(b"\n")
            lines = [line.removesuffix(b"\r") for line in lines]

            for line in lines:
                if not line:
                    continue
                if line.startswith(b":"):
                    continue

                if b":" in line:
                    field, value = line.split(b":", 1)
                    if value.startswith(b" "):
                        value = value[1:]
                else:
                    field = line
                    value = b""

                field_str = field.decode("utf-8", errors="replace")
                val_str = value.decode("utf-8", errors="replace")

                if field_str == "event":
                    self._event = val_str
                elif field_str == "data":
                    self._data.append(val_str)
                elif field_str == "id":
                    if b"\0" not in value:
                        self._id = val_str
                elif field_str == "retry":
                    try:
                        self._retry = int(val_str)
                    except ValueError:
                        pass

            if not self._data:
                self._event = "message"
                continue

            data_str = "\n".join(self._data)
            events.append(
                SSEEvent(event=self._event, data=data_str, id=self._id, retry=self._retry)
            )

            self._event = "message"
            self._data = []

        return events

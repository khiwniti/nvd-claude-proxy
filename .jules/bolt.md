## 2026-04-23 - Fast JSON Serialization
**Learning:** `orjson` is available as a dependency and is significantly faster than the standard library `json` module, making it ideal for high-performance loops like token estimation that walk through deep request bodies.
**Action:** Replace `json.dumps()` with `orjson.dumps()` when converting primitives to strings in hot paths like `util/tokens.py`.

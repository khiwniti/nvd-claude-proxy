from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Set

import yaml


@dataclass(slots=True)
class ServerToolSpec:
    family: str
    versions: List[int]
    beta: str
    schema_path: str | None = None
    _cached_schema: dict[str, Any] | None = None

    def matches(self, t_type: str) -> bool:
        """Return True if t_type is a version of this tool family (e.g. web_search_20250305)."""
        for v in self.versions:
            if t_type == f"{self.family}_{v}":
                return True
        return False

    @property
    def schema(self) -> dict[str, Any]:
        if self._cached_schema is not None:
            return self._cached_schema
        if not self.schema_path:
            return {"type": "object", "properties": {}}
        
        path = Path(self.schema_path)
        if not path.exists():
            # Fallback relative to project root
            pkg_root = Path(__file__).parent.parent.parent
            path = pkg_root / self.schema_path
            
        if path.exists():
            import json
            try:
                self._cached_schema = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._cached_schema = {"type": "object", "properties": {}}
        else:
            self._cached_schema = {"type": "object", "properties": {}}
            
        return self._cached_schema


class ServerToolRegistry:
    def __init__(self, specs: List[ServerToolSpec]) -> None:
        self.specs = specs
        self._all_types: Set[str] = set()
        for spec in specs:
            for v in spec.versions:
                self._all_types.add(f"{spec.family}_{v}")

    def is_server_tool(self, t_type: str) -> bool:
        return t_type in self._all_types

    def get_spec(self, t_type: str) -> ServerToolSpec | None:
        for spec in self.specs:
            if spec.matches(t_type):
                return spec
        return None


def load_server_tool_registry(path: str | Path | None = None) -> ServerToolRegistry:
    if path is None:
        path = Path("config/server_tools.yaml")
    
    if not Path(path).exists():
        return ServerToolRegistry([])
        
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    specs = [
        ServerToolSpec(
            family=d["family"],
            versions=d["versions"],
            beta=d["beta"],
            schema_path=d.get("schema")
        )
        for d in data
    ]
    return ServerToolRegistry(specs)

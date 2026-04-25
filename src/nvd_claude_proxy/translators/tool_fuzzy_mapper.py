"""Fuzzy tool name mapping to correct common model hallucinations.

Ported from claude-code-router's tool repair strategies.
"""

from __future__ import annotations

import difflib
from typing import Any


class FuzzyToolMapper:
    """Corrects common model hallucinations of tool names."""

    # Key is hallucinated name (lowercase), Value is the Claude Code tool name.
    STRICT_MAPPINGS = {
        "read": "view_file",
        "read_file": "view_file",
        "view": "view_file",
        "cat": "view_file",
        "write": "write_to_file",
        "write_file": "write_to_file",
        "create_file": "write_to_file",
        "list": "list_dir",
        "ls": "list_dir",
        "dir": "list_dir",
        "search": "grep_search",
        "grep": "grep_search",
        "bash": "run_command",
        "shell": "run_command",
        "run": "run_command",
        "edit": "edit_file",
        "replace": "replace_file_content",
        "think": "thinking",
    }

    # Common argument key hallucinations.
    # Key is tool name (or '*' for all), Value is a map of hallucinated -> real.
    STRICT_ARG_MAPPINGS = {
        "view_file": {
            "path": "AbsolutePath",
            "filepath": "AbsolutePath",
            "filename": "AbsolutePath",
            "file_path": "AbsolutePath",
        },
        "write_to_file": {
            "path": "TargetFile",
            "filepath": "TargetFile",
            "filename": "TargetFile",
            "file_path": "TargetFile",
            "text": "CodeContent",
            "content": "CodeContent",
            "file_content": "CodeContent",
        },
        "list_dir": {
            "path": "DirectoryPath",
            "directory": "DirectoryPath",
        },
        "run_command": {
            "command": "CommandLine",
            "cmd": "CommandLine",
        },
        "grep_search": {
            "query": "Query",
            "path": "SearchPath",
            "pattern": "Query",
        },
    }

    def __init__(self, valid_names: set[str]) -> None:
        self.valid_names = valid_names
        # Lowercase version for easier lookups
        self._lower_to_valid = {n.lower(): n for n in valid_names}

    def map_name(self, name: str) -> str | None:
        """Attempt to map a potentially hallucinated name to a valid one.

        Returns the valid name if a match is found, else None.
        """
        if name in self.valid_names:
            return name

        lower_name = name.lower()

        # 1. Exact case-insensitive match
        if lower_name in self._lower_to_valid:
            return self._lower_to_valid[lower_name]

        # 2. Strict alias mapping
        if lower_name in self.STRICT_MAPPINGS:
            alias = self.STRICT_MAPPINGS[lower_name]
            if alias in self.valid_names:
                return alias

        # 3. Fuzzy matching (difflib)
        matches = difflib.get_close_matches(name, self.valid_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        # 4. Prefix/Suffix matching
        for valid in self.valid_names:
            v_lower = valid.lower()
            if lower_name in v_lower or v_lower in lower_name:
                # Only return if it's a "safe" substring match (length check)
                if len(lower_name) > 3 and len(v_lower) > 3:
                    return valid

        return None

    def map_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Attempt to map hallucinated argument keys to valid ones.

        Example: read_file(path="/foo") -> read_file(file_path="/foo")
        """
        if tool_name not in self.STRICT_ARG_MAPPINGS:
            return arguments

        mapping = self.STRICT_ARG_MAPPINGS[tool_name]
        new_args = {}
        for k, v in arguments.items():
            if k in mapping:
                new_args[mapping[k]] = v
            else:
                new_args[k] = v
        return new_args

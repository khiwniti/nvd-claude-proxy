from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure `src/` is importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Test-time defaults.
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-test-key")
os.environ.setdefault("MODEL_CONFIG_PATH", str(ROOT / "config" / "models.yaml"))


@pytest.fixture
def model_registry():
    from nvd_claude_proxy.config.models import load_model_registry

    return load_model_registry(os.environ["MODEL_CONFIG_PATH"])

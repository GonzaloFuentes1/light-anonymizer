"""The pure logic of the review viewer (anonymizer/ui/review-core.js), tested with node --test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_review_core():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests" / "ui" / "review-core.test.js")],
        capture_output=True, text=True, timeout=120, check=False, cwd=ROOT,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr

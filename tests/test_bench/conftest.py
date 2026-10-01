from pathlib import Path

import pytest

from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx(tmp_path: Path) -> Context:
    """Generation context in a temporary folder, using the repository's face cache without downloading."""
    return Context(
        root=tmp_path / "generated",
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )

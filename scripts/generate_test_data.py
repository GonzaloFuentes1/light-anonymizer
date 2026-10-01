"""Generates the fictitious test dataset in test_data/generated/.

Usage (from the repository root):
    uv run python scripts/generate_test_data.py [--seed 33] [--only pdf_text images] [--no-downloads]

It is a shortcut to ``python -m test_bench.generate``; see that module for the options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_bench.generate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

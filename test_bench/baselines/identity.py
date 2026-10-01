"""``identity`` baseline: copies every file untouched.

Every element and every sensitive metadata entry must show up as a leak. If the evaluator misses
one, it is blind to that case. It never reports errors, not even for files that should be rejected.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from test_bench.schema import FileEntry, FileResult, Manifest


def process(file_entry: FileEntry, manifest: Manifest, folder: Path, details: dict[str, Any]) -> FileResult:
    source = Path(manifest.root) / file_entry.path
    dest = folder / file_entry.path
    if not source.exists():
        # Nothing to copy; still no error is reported (identity rejects nothing).
        details.setdefault("missing_input", []).append(file_entry.path)
        return FileResult(input=file_entry.path, output=None, pages_processed=0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return FileResult(input=file_entry.path, output=file_entry.path, pages_processed=len(file_entry.pages))

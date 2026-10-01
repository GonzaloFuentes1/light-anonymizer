"""Locks shared by the engines and the audit report.

PyMuPDF is not thread-safe: every call into it goes through ``PDF_LOCK`` (one page at a time),
so two analyses, the page renders and the audit report can interleave without crashing.
"""

from __future__ import annotations

import threading

PDF_LOCK = threading.RLock()

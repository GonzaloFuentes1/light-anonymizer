"""Shared test setup."""

import pytest

from anonymizer.engine import ocr


@pytest.fixture(autouse=True)
def fresh_ocr_cache():
    """Each test reads its images anew: the session's OCR cache must not carry readings from one
    test (or from a test's stand-in OCR engine) into another."""
    ocr.clear_cache()
    yield
    ocr.clear_cache()

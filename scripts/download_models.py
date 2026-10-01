"""Downloads the models for development and verifies their SHA-256 (the packaged app bundles them).

Usage (from the repository root):
    uv run python scripts/download_models.py

The OCR models (PP-OCRv6) ship inside the ``rapidocr`` package; only YuNet is downloaded here.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/26cc381e4d/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
}
DESTINATION = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    DESTINATION.mkdir(exist_ok=True)
    failures = 0
    for name, (url, sha) in MODELS.items():
        path = DESTINATION / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            print(f"descargando {name} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "anonymizer-dev/0.1"})
            with urllib.request.urlopen(request, timeout=120) as r:
                path.write_bytes(r.read())
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == sha
        print(f"{name}: {'SHA-256 correcto' if ok else 'SHA-256 NO COINCIDE'}")
        failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

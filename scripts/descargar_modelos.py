"""Descarga los modelos para desarrollo y verifica su SHA-256 (la aplicación empaquetada los trae incluidos).

Uso (desde la raíz del repositorio):
    uv run python scripts/descargar_modelos.py

Los modelos de OCR (PP-OCRv6) vienen dentro del paquete ``rapidocr``; aquí solo se descarga YuNet.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELOS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/26cc381e4d/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
}
DESTINO = Path(__file__).resolve().parents[1] / "modelos"


def main() -> int:
    DESTINO.mkdir(exist_ok=True)
    fallas = 0
    for nombre, (url, sha) in MODELOS.items():
        ruta = DESTINO / nombre
        if not ruta.exists() or hashlib.sha256(ruta.read_bytes()).hexdigest() != sha:
            print(f"descargando {nombre} ...")
            solicitud = urllib.request.Request(url, headers={"User-Agent": "anonimizador-desarrollo/0.1"})
            with urllib.request.urlopen(solicitud, timeout=120) as r:
                ruta.write_bytes(r.read())
        ok = hashlib.sha256(ruta.read_bytes()).hexdigest() == sha
        print(f"{nombre}: {'SHA-256 correcto' if ok else 'SHA-256 NO COINCIDE'}")
        fallas += not ok
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())

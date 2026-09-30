from pathlib import Path

import pytest

from banco_pruebas.contexto import Contexto
from banco_pruebas.ficticios import Ficticios
from banco_pruebas.rostros import ProveedorRostros

RAIZ_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx(tmp_path: Path) -> Contexto:
    """Contexto de generación en una carpeta temporal, usando la caché de rostros del repositorio sin descargar."""
    return Contexto(
        raiz=tmp_path / "generado",
        semilla=33,
        fict=Ficticios(33),
        rostros=ProveedorRostros(RAIZ_REPO / "datos_prueba" / "cache" / "rostros", 33, permitir_descarga=False),
    )

"""Láminas de antes y después: el archivo original junto a la salida de cada sistema.

Uso:
    uv run python -m banco_pruebas.comparar [--sistemas oraculo cuaderno] [--salida resultados/comparaciones] [--id ID ...]

Sin ``--id``, toma un archivo por categoría. Cada lámina muestra la primera página (o la página
indicada con ``ID:pagina``) del original y de la salida de cada sistema en ``resultados/<sistema>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from banco_pruebas.esquema import Manifiesto
from banco_pruebas.lienzo import fuente
from banco_pruebas.visualizar import paginas_visibles

TITULOS = {
    "original": "Original",
    "oraculo": "Oráculo (con la respuesta)",
    "cuaderno": "Cuaderno actual",
    "identidad": "Identidad (sin cambios)",
    "prototipo": "Prototipo (detección real)",
}
POR_DEFECTO = [
    "pdft_cuaderno",
    "pdft_honorarios_01",
    "pdfm_redaccion_falsa",
    "pdfe_informe_300dpi",
    "img_rot_nota_45",
    "img_exif_orientacion_6",
    "ced_frente_foto_perspectiva",
    "pant_chat_movil",
    "rost_grupo_compuesto_01",
]
ALTO = 900


def _pagina(ruta: Path, formato: str, indice: int) -> Image.Image | None:
    if not ruta.exists():
        return None
    try:
        paginas = paginas_visibles(ruta, formato)
    except Exception:  # noqa: BLE001 - la salida puede no abrirse
        return None
    return paginas[min(indice, len(paginas) - 1)] if paginas else None


def _aviso(texto: str, ancho: int) -> Image.Image:
    img = Image.new("RGB", (ancho, ALTO), (236, 236, 236))
    ImageDraw.Draw(img).multiline_text((24, ALTO // 2 - 40), texto, font=fuente("sans", 22), fill=(90, 90, 90))
    return img


def lamina(man: Manifiesto, archivo_id: str, pagina: int, sistemas: list[str], raiz_resultados: Path) -> Image.Image:
    a = next(x for x in man.archivos if x.id == archivo_id)
    original = _pagina(Path(man.raiz) / a.ruta, a.formato, pagina)
    assert original is not None, f"no se pudo abrir {a.ruta}"
    escala = ALTO / original.height
    ancho = max(1, int(original.width * escala))
    columnas = [("original", original.resize((ancho, ALTO)))]
    for s in sistemas:
        img = _pagina(raiz_resultados / s / "archivos" / a.ruta, a.formato, pagina)
        if img is None:
            columnas.append((s, _aviso("Sin salida:\neste sistema no\nprocesa el archivo", ancho)))
        else:
            columnas.append((s, img.resize((ancho, ALTO))))
    margen, cabecera = 16, 56
    hoja = Image.new("RGB", (len(columnas) * (ancho + margen) + margen, ALTO + cabecera + margen), "white")
    d = ImageDraw.Draw(hoja)
    for i, (nombre, img) in enumerate(columnas):
        x = margen + i * (ancho + margen)
        d.text((x, 14), TITULOS.get(nombre, nombre), font=fuente("sans_negrita", 20), fill=(20, 20, 20))
        hoja.paste(img, (x, cabecera))
        d.rectangle((x - 1, cabecera - 1, x + ancho, cabecera + ALTO), outline=(180, 180, 180))
    return hoja


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Láminas de antes y después.")
    ap.add_argument("--manifiesto", type=Path, default=Path("datos_prueba/generado/manifiesto.json"))
    ap.add_argument("--resultados", type=Path, default=Path("resultados/detalle"))
    ap.add_argument("--sistemas", nargs="+", default=["prototipo", "cuaderno"])
    ap.add_argument("--salida", type=Path, default=Path("resultados/ejemplos"))
    ap.add_argument("--id", nargs="*", help="ID o ID:pagina (sin esto, uno por categoría)")
    args = ap.parse_args(argv)
    man = Manifiesto.cargar(args.manifiesto)
    args.salida.mkdir(parents=True, exist_ok=True)
    for pedido in args.id or POR_DEFECTO:
        archivo_id, _, pag = pedido.partition(":")
        img = lamina(man, archivo_id, int(pag or 0), args.sistemas, args.resultados)
        destino = args.salida / f"{archivo_id}{'_p' + pag if pag else ''}.jpg"
        img.save(destino, quality=88)
        print(destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

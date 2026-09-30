"""PDF con datos personales escondidos en metadatos y estructuras no visibles.

Cada estructura lleva un canario propio (un valor único de una persona distinta), de modo que
si aparece en la salida se sabe exactamente qué estructura no se limpió:

1. ``metadatos_completos.pdf``: diccionario Info, paquete XMP, anotaciones (nota, texto libre,
   resaltado con ventana emergente), archivo incrustado, anotación de archivo adjunto, capa
   opcional apagada, campo de formulario, marcador y JavaScript de documento.
2. ``revision_incremental.pdf``: una línea con un RUT se borra en una segunda revisión guardada
   de forma incremental; la revisión anterior sigue en los bytes del archivo.
3. ``redaccion_falsa.pdf``: la fuga clásica de "censura por transparencia": rectángulos negros
   dibujados sobre texto que sigue siendo extraíble, una anotación de redacción sin aplicar y
   una anotación cuadrada negra sobre un teléfono.
4. ``solo_permisos.pdf``: cifrado solo con contraseña de propietario (restringe copiar e
   imprimir, pero abre sin contraseña). Debe procesarse.

Las coordenadas están en el espacio de página sin rotar de PyMuPDF (el de ``search_for``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf

from banco_pruebas.contexto import Contexto
from banco_pruebas.esquema import Archivo, Elemento, Metadato, Pagina, Poligono
from banco_pruebas.ficticios import (
    FORMATOS_CORREO,
    FORMATOS_RUT,
    FORMATOS_TELEFONO,
    FRASES_ADMINISTRATIVAS,
    Ficticios,
    Persona,
    sin_tildes,
)
from banco_pruebas.lienzo import ruta_fuente

MODULO = "pdf_metadatos"
CATEGORIA = "pdf_metadatos"
PREFIJO = "pdfm"

ANCHO, ALTO = 612.0, 792.0  # tamaño carta, el habitual en documentos públicos chilenos
MARGEN = 64.0
INTERLINEA = 17.0

# Contraseña de propietario del archivo con solo permisos (no es un dato personal).
CONTRASENA_PROPIETARIO = "propietario-ficticio-2026"

# Alias de fuente en PyMuPDF -> nombre en ``lienzo.FUENTES``. "helv" es la Helvetica base-14,
# que se usa cuando conviene que el texto quede como cadena literal en el flujo de contenido.
ALIAS_FUENTES = {
    "sans": "dvs",
    "sans_negrita": "dvsb",
    "serif": "dvse",
    "serif_negrita": "dvseb",
}

FECHA_CREACION = "D:20260312091500-03'00'"
FECHA_MODIFICACION = "D:20260318164210-03'00'"


@lru_cache(maxsize=16)
def _fuente(nombre: str) -> pymupdf.Font:
    if nombre == "helv":
        return pymupdf.Font("helv")
    return pymupdf.Font(fontfile=str(ruta_fuente(nombre)))


def quad_a_poligono(q: pymupdf.Quad) -> Poligono:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def rect_a_poligono(r: pymupdf.Rect) -> Poligono:
    return [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]


def nivel_persona(p: Persona) -> str:
    return "base" if p.en_lista else "fuera_de_alcance"


def literal_pdf(texto: str) -> str:
    """Cadena literal PDF (entre paréntesis) con los caracteres especiales escapados."""
    return "(" + texto.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def helv_a_literal(doc: pymupdf.Document, pagina: pymupdf.Page) -> None:
    """Reescribe el texto en Helvetica de la página como cadenas literales ``(…) Tj``.

    PyMuPDF escribe el texto base-14 en hexadecimal; como literal Latin-1, el valor queda
    legible tal cual en el flujo descomprimido (así la búsqueda en bytes lo encuentra, igual
    que en un PDF producido por muchas herramientas).
    """
    # PyMuPDF escribe "/helv 10 Tf <colores> [<hex>]TJ"; se conserva todo lo que va entre Tf y TJ.
    patron = re.compile(rb"(/helv [\d.]+ Tf[^\[(]*?)\[<([0-9a-fA-F]+)>\]TJ")

    def reemplazo(m: re.Match[bytes]) -> bytes:
        texto = bytes.fromhex(m.group(2).decode()).decode("latin-1")
        return m.group(1) + literal_pdf(texto).encode("latin-1") + b" Tj"

    for xref in pagina.get_contents():
        flujo = doc.xref_stream(xref)
        nuevo = patron.sub(reemplazo, flujo)
        if nuevo != flujo:
            doc.update_stream(xref, nuevo)


def fijar_fechas_adjuntos(doc: pymupdf.Document) -> None:
    """Reemplaza la fecha actual que PyMuPDF pone en ``/Params`` de cada ``/EmbeddedFile`` por una fija.

    Sin esto los bytes del PDF cambian en cada generación (y la zona horaria de la máquina queda
    escrita en el archivo).
    """
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type") != ("name", "/EmbeddedFile"):
            continue
        if doc.xref_get_key(xref, "Params")[0] != "dict":
            continue
        doc.xref_set_key(xref, "Params/CreationDate", literal_pdf(FECHA_CREACION))
        doc.xref_set_key(xref, "Params/ModDate", literal_pdf(FECHA_CREACION))


class HojaPdf:
    """Escribe texto en una página PyMuPDF y registra cada trozo como Elemento con su cuadrilátero.

    El cuadrilátero es el que entrega ``page.search_for(..., quads=True)`` acotado a la caja
    tipográfica del trozo recién escrito; para texto invisible (capa opcional apagada), que
    ``search_for`` no ve, se usa la caja tipográfica calculada con las métricas de la fuente
    (la misma convención: ascendente y descendente de la fuente por el cuerpo).
    """

    def __init__(self, pagina: pymupdf.Page, indice: int = 0) -> None:
        self.pagina = pagina
        self.indice = indice
        self.elementos: list[Elemento] = []
        for nombre, alias in ALIAS_FUENTES.items():
            pagina.insert_font(fontname=alias, fontfile=str(ruta_fuente(nombre)))

    def ancho(self, texto: str, tam: float = 10.0, fuente: str = "sans") -> float:
        return _fuente(fuente).text_length(texto, tam)

    def caja(self, x: float, y: float, texto: str, tam: float, fuente: str) -> pymupdf.Rect:
        f = _fuente(fuente)
        return pymupdf.Rect(x, y - f.ascender * tam, x + f.text_length(texto, tam), y - f.descender * tam)

    def buscar(self, texto: str, clip: pymupdf.Rect) -> Poligono:
        """Cuadrilátero único de ``texto`` dentro de ``clip`` según PyMuPDF (falla si no hay exactamente uno)."""
        encontrados = self.pagina.search_for(texto, clip=clip, quads=True)
        if len(encontrados) != 1:
            raise RuntimeError(f"se esperaba una aparición de {texto!r} en {clip}, hay {len(encontrados)}")
        return quad_a_poligono(encontrados[0])

    def escribir(
        self,
        x: float,
        y: float,
        texto: str,
        *,
        tam: float = 10.0,
        fuente: str = "sans",
        tipo: str = "texto",
        nivel: str = "base",
        capa: str = "texto",
        etiquetas: dict[str, Any] | None = None,
        color: tuple[float, float, float] = (0.1, 0.1, 0.1),
        oc: int = 0,
        registrar: bool = True,
    ) -> float:
        """Escribe ``texto`` con la línea base en (x, y); devuelve la x donde termina."""
        alias = "helv" if fuente == "helv" else ALIAS_FUENTES[fuente]
        self.pagina.insert_text((x, y), texto, fontname=alias, fontsize=tam, color=color, oc=oc)
        caja = self.caja(x, y, texto, tam, fuente)
        if registrar:
            if capa == "oculto":
                poligono = rect_a_poligono(caja)
            else:
                poligono = self.buscar(texto, caja + (-2, -2, 2, 2))
            self.elementos.append(
                Elemento(
                    tipo=tipo,
                    pagina=self.indice,
                    poligono=poligono,
                    valor=texto,
                    nivel=nivel,
                    capa=capa,
                    etiquetas={"fuente": fuente, "tam_pt": tam, **(etiquetas or {})},
                )
            )
        return caja.x1

    def campo(
        self,
        x: float,
        y: float,
        etiqueta: str,
        valor: str,
        *,
        tipo: str,
        x_valor: float | None = None,
        tam: float = 10.0,
        **kwargs: Any,
    ) -> float:
        """Etiqueta neutra ("RUT:") seguida del dato; la etiqueta se registra como ``texto``."""
        fin = self.escribir(x, y, etiqueta, tam=tam, fuente=kwargs.pop("fuente_etiqueta", "sans_negrita"))
        inicio = x_valor if x_valor is not None else fin + self.ancho(" ", tam)
        return self.escribir(inicio, y, valor, tam=tam, tipo=tipo, **kwargs)

    def dato(
        self,
        x: float,
        y: float,
        p: Persona,
        clase: str,
        formato: str = "",
        *,
        tam: float = 10.0,
        nivel: str | None = None,
        etiquetas: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float:
        """Escribe un dato de la persona con su tipo, nivel y etiquetas según las reglas del manifiesto."""
        texto, tipo, nivel_def, etq = valor_persona(p, clase, formato)
        return self.escribir(
            x, y, texto, tam=tam, tipo=tipo, nivel=nivel or nivel_def, etiquetas={**etq, **(etiquetas or {})}, **kwargs
        )


def valor_persona(p: Persona, clase: str, formato: str = "") -> tuple[str, str, str, dict[str, Any]]:
    """(texto, tipo, nivel, etiquetas) de un dato de la persona.

    ``clase``: nombre, direccion, rut, correo, movil, fijo.
    """
    match clase:
        case "nombre":
            return p.nombre_completo, "nombre", nivel_persona(p), {"en_lista": p.en_lista, "variante": "completo"}
        case "direccion":
            return p.direccion, "direccion", nivel_persona(p), {"en_lista": p.en_lista}
        case "rut":
            formato = formato or "puntos"
            etq = {"formato": formato, "dv_valido": p.rut_dv_valido}
            return p.rut(formato), "rut", FORMATOS_RUT[formato], etq
        case "correo":
            formato = formato or "punto"
            return p.correo(formato), "correo", FORMATOS_CORREO[formato], {"formato": formato}
        case "movil" | "fijo":
            tel = p.telefono if clase == "movil" else p.telefono_fijo
            formato = formato or {"movil": "movil_internacional", "santiago": "santiago_internacional"}.get(
                tel.clase, "regional_parentesis"
            )
            etq = {"formato": formato, "clase": tel.clase}
            return tel.formatear(formato), "telefono", FORMATOS_TELEFONO[formato], etq
    raise ValueError(clase)


def formato_fijo(p: Persona, preferido: str) -> str:
    """Formato de teléfono fijo compatible con la clase del fijo de la persona."""
    if p.telefono_fijo.clase == "santiago":
        return {"internacional": "santiago_internacional", "parentesis": "santiago_parentesis"}.get(
            preferido, "santiago_nacional"
        )
    return {"internacional": "regional_internacional", "parentesis": "regional_parentesis"}.get(
        preferido, "regional_guion"
    )


def paginas_de(ruta: Path) -> list[Pagina]:
    """Páginas en el espacio sin rotar de PyMuPDF (ancho y alto de la caja de recorte)."""
    with pymupdf.open(ruta) as d:
        return [
            Pagina(indice=i, ancho=p.cropbox.width, alto=p.cropbox.height, unidad="pt", rotacion=p.rotation)
            for i, p in enumerate(d)
        ]


def _encabezado(hoja: HojaPdf, titulo: str, subtitulo: str, folio: str, fecha: str) -> float:
    """Encabezado institucional neutro; devuelve la y de la siguiente línea."""
    y = 72.0
    hoja.escribir(MARGEN, y, "GOBIERNO REGIONAL DE LA REGIÓN FICTICIA", tam=9, fuente="sans_negrita")
    y += 13
    hoja.escribir(MARGEN, y, "División de Presupuesto e Inversión Regional", tam=9)
    y += 30
    hoja.escribir(MARGEN, y, titulo, tam=14, fuente="serif_negrita")
    y += 18
    hoja.escribir(MARGEN, y, subtitulo, tam=10.5, fuente="serif")
    y += 24
    hoja.campo(MARGEN, y, "Folio:", folio, tipo="texto")
    fin = hoja.escribir(360, y, "Fecha:", fuente="sans_negrita")
    hoja.escribir(fin + hoja.ancho(" "), y, fecha, etiquetas={"senuelo": "fecha"})
    hoja.pagina.draw_line((MARGEN, y + 9), (ANCHO - MARGEN, y + 9), color=(0.3, 0.3, 0.3), width=0.6)
    return y + 30


def _parrafos(hoja: HojaPdf, y: float, frases: list[str], tam: float = 10.0) -> float:
    for frase in frases:
        hoja.escribir(MARGEN, y, frase, tam=tam, fuente="serif")
        y += INTERLINEA
    return y


def _frases(rng: Any, n: int) -> list[str]:
    indices = rng.choice(len(FRASES_ADMINISTRATIVAS), size=n, replace=False)
    return [FRASES_ADMINISTRATIVAS[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# 1. Metadatos completos
# ---------------------------------------------------------------------------


def _xmp(creador: str, rut: str) -> str:
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:gore="http://ns.ficticio.cl/gore/1.0/">\n'
        "   <dc:creator><rdf:Seq><rdf:li>" + creador + "</rdf:li></rdf:Seq></dc:creator>\n"
        '   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Informe de actividades</rdf:li></rdf:Alt></dc:title>\n'
        "   <xmp:CreatorTool>Microsoft® Word para Microsoft 365</xmp:CreatorTool>\n"
        "   <xmp:CreateDate>2026-03-12T09:15:00-03:00</xmp:CreateDate>\n"
        "   <gore:rutFuncionario>" + rut + "</gore:rutFuncionario>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


def _metadatos_completos(ctx: Contexto, f: Ficticios, rng: Any) -> Archivo:
    ruta_rel = "pdf_metadatos/metadatos_completos.pdf"
    doc = pymupdf.open()
    pag = doc.new_page(width=ANCHO, height=ALTO)
    hoja = HojaPdf(pag)

    # -- contenido visible: bloque pequeño con datos personales -----------------
    titular = f.persona()
    contraparte = f.persona(en_lista=False)
    y = _encabezado(
        hoja, "Informe de actividades", "Convenio de prestación de servicios a honorarios", f.folio(), f.fecha()
    )
    hoja.escribir(MARGEN, y, "Antecedentes del prestador", tam=11, fuente="sans_negrita")
    y += 20
    xv = MARGEN + 90
    for etiqueta, clase, formato in (
        ("Nombre:", "nombre", ""),
        ("RUT:", "rut", "puntos"),
        ("Correo:", "correo", "punto"),
        ("Teléfono:", "movil", "movil_nacional"),
        ("Domicilio:", "direccion", ""),
    ):
        hoja.escribir(MARGEN, y, etiqueta, fuente="sans_negrita")
        fin = hoja.dato(xv, y, titular, clase, formato)
        if clase == "direccion":
            hoja.escribir(fin, y, f", {titular.comuna}")
        y += INTERLINEA
    hoja.campo(MARGEN, y, "Monto bruto:", f.monto(), tipo="texto", x_valor=xv, etiquetas={"senuelo": "monto"})
    y += 28

    hoja.escribir(MARGEN, y, "Resumen de actividades", tam=11, fuente="sans_negrita")
    y += 20
    frases = _frases(rng, 4)
    y_resaltado = y + INTERLINEA  # la segunda línea lleva el resaltado con nota emergente
    y = _parrafos(hoja, y, frases)
    y += 8
    fin = hoja.escribir(MARGEN, y, "Contraparte técnica:", fuente="sans_negrita")
    hoja.dato(fin + hoja.ancho(" "), y, contraparte, "nombre")
    y += 30

    # -- campo de formulario con un correo (visible y en /V) ----------------------
    p_form = f.persona()
    correo_form = p_form.correo("inicial")
    hoja.escribir(MARGEN, y, "Correo de notificación:", fuente="sans_negrita")
    caja_widget = pymupdf.Rect(MARGEN + 140, y - 12, MARGEN + 380, y + 5)
    widget = pymupdf.Widget()
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.field_name = "correo_notificacion"
    widget.field_label = "Correo de notificación"
    widget.field_value = correo_form
    widget.rect = caja_widget
    widget.text_font = "Helv"
    widget.text_fontsize = 10
    widget.border_color = (0.4, 0.4, 0.4)
    widget.border_width = 0.6
    pag.add_widget(widget)
    hoja.elementos.append(
        Elemento(
            tipo="correo",
            pagina=0,
            poligono=hoja.buscar(correo_form, caja_widget + (-2, -2, 2, 2)),
            valor=correo_form,
            nivel=FORMATOS_CORREO["inicial"],
            capa="texto",
            etiquetas={"formato": "inicial", "estructura": "formulario"},
        )
    )
    y += 34

    # -- anotación de texto libre con un correo (visible) -------------------------
    p_libre = f.persona()
    correo_libre = p_libre.correo("guion_bajo")
    texto_libre = f"Enviar observaciones a {correo_libre}"
    caja_libre = pymupdf.Rect(MARGEN, y, MARGEN + 330, y + 22)
    libre = pag.add_freetext_annot(
        caja_libre, texto_libre, fontsize=9, fontname="helv", text_color=(0.1, 0.1, 0.5), fill_color=(1, 1, 0.85)
    )
    libre.set_info(title="Revisor", subject="Observación")
    libre.update()
    prefijo_libre = "Enviar observaciones a"
    hoja.elementos.append(
        Elemento(
            tipo="texto",
            pagina=0,
            poligono=hoja.buscar(prefijo_libre, caja_libre + (-2, -4, 2, 2)),
            valor=prefijo_libre,
            capa="texto",
            etiquetas={"fuente": "helv", "tam_pt": 9, "estructura": "anotacion_freetext"},
        )
    )
    hoja.elementos.append(
        Elemento(
            tipo="correo",
            pagina=0,
            poligono=hoja.buscar(correo_libre, caja_libre + (-2, -4, 2, 2)),
            valor=correo_libre,
            nivel=FORMATOS_CORREO["guion_bajo"],
            capa="texto",
            etiquetas={"formato": "guion_bajo", "estructura": "anotacion_freetext"},
        )
    )
    y += 40

    # -- capa opcional "Notas internas", apagada por defecto ----------------------
    p_ocg = f.persona()
    rut_ocg = p_ocg.rut("sin_puntos")
    ocg = doc.add_ocg("Notas internas", on=False)
    fin = hoja.escribir(
        MARGEN,
        y,
        "Nota interna: verificar RUT",
        fuente="helv",
        oc=ocg,
        capa="oculto",
        etiquetas={"estructura": "ocg_apagada"},
    )
    hoja.escribir(
        fin + hoja.ancho(" ", fuente="helv"),
        y,
        rut_ocg,
        fuente="helv",
        oc=ocg,
        tipo="rut",
        capa="oculto",
        nivel=FORMATOS_RUT["sin_puntos"],
        etiquetas={"formato": "sin_puntos", "dv_valido": p_ocg.rut_dv_valido, "estructura": "ocg_apagada"},
    )
    helv_a_literal(doc, pag)

    # -- pie neutro ---------------------------------------------------------------
    hoja.escribir(MARGEN, ALTO - 48, "Documento de prueba generado con datos ficticios.", tam=8, color=(0.4, 0.4, 0.4))

    # -- anotaciones con contenido no visible -------------------------------------
    p_nota = f.persona()
    rut_nota = p_nota.rut("puntos")
    nota = pag.add_text_annot((ANCHO - 44, 70), f"Revisar RUT {rut_nota} de {p_nota.nombre_completo}", icon="Note")
    nota.set_info(title="Jefatura", subject="Revisión")
    nota.update()

    p_resaltado = f.persona()
    fono_resaltado = p_resaltado.telefono.formatear("movil_internacional")
    quads = pag.search_for(frases[1], quads=True)
    resaltado = pag.add_highlight_annot(quads)
    resaltado.set_info(content=f"Confirmar con el encargado al {fono_resaltado}", title="Revisor")
    resaltado.set_popup(pymupdf.Rect(ANCHO - 230, y_resaltado - 10, ANCHO - 30, y_resaltado + 70))
    resaltado.update()

    # -- adjuntos --------------------------------------------------------------------
    p_adjunto = f.persona()
    rut_adjunto = p_adjunto.rut("puntos")
    contenido = (
        f"Nombre: {p_adjunto.nombre_completo}\nRUT: {rut_adjunto}\nCorreo: {p_adjunto.correo('punto')}\n"
    ).encode()
    doc.embfile_add(
        "datos_contacto.txt", contenido, filename="datos_contacto.txt", desc="Datos de contacto del prestador"
    )

    p_anexo = f.persona()
    rut_anexo = p_anexo.rut("sin_puntos")
    anexo = (
        f"Anexo de antecedentes\n{p_anexo.nombre_completo}\nRUT {rut_anexo}\n"
        f"Fono {p_anexo.telefono.formatear('movil_compacto')}\n"
    ).encode()
    adjunto = pag.add_file_annot(
        (ANCHO - 44, ALTO - 64), anexo, "anexo_antecedentes.txt", desc="Anexo de antecedentes", icon="PushPin"
    )
    adjunto.update()

    # -- JavaScript de documento ---------------------------------------------------
    p_js = f.persona()
    fono_js = p_js.telefono.formatear("movil_internacional")
    js = f'var contacto = "{fono_js}"; app.alert("Consultas sobre este informe al " + contacto);'
    xref_js = doc.get_new_xref()
    doc.update_object(xref_js, f"<< /S /JavaScript /JS {literal_pdf(js)} >>")
    doc.xref_set_key(doc.pdf_catalog(), "Names/JavaScript", f"<< /Names [(contacto) {xref_js} 0 R] >>")

    # -- marcadores -----------------------------------------------------------------
    p_marcador = f.persona()
    marcador = f"Antecedentes de {p_marcador.nombre_completo}"
    doc.set_toc([[1, "Informe de actividades", 1], [2, marcador, 1], [2, "Resumen de actividades", 1]])

    # -- diccionario Info y XMP ------------------------------------------------------
    p_autor, p_titulo, p_asunto, p_clave, p_creador = (f.persona() for _ in range(5))
    rut_titulo = p_titulo.rut("puntos")
    correo_asunto = p_asunto.correo("punto")
    fono_clave = p_clave.telefono_fijo.formatear(formato_fijo(p_clave, "parentesis"))
    archivo_word = f"informe_{sin_tildes(p_creador.apellido_p).lower()}_{sin_tildes(p_creador.apellido_m).lower()}"
    archivo_word = archivo_word.replace("ñ", "n") + ".docx"
    doc.set_metadata(
        {
            "author": p_autor.nombre_completo,
            "title": f"Informe de honorarios RUT {rut_titulo}",
            "subject": f"Contacto: {correo_asunto}",
            "keywords": f"honorarios; convenio; {fono_clave}",
            "creator": f"Microsoft Word - {archivo_word}",
            "producer": "Microsoft® Word para Microsoft 365",
            "creationDate": FECHA_CREACION,
            "modDate": FECHA_MODIFICACION,
        }
    )
    p_xmp_creador, p_xmp_rut = f.persona(), f.persona()
    rut_xmp = p_xmp_rut.rut("sin_puntos")
    doc.set_xml_metadata(_xmp(p_xmp_creador.nombre_completo, rut_xmp))

    fijar_fechas_adjuntos(doc)
    doc.subset_fonts()
    destino = ctx.ruta(ruta_rel)
    doc.save(destino, garbage=3, deflate=True, no_new_id=True)
    doc.close()

    metadatos = [
        Metadato("pdf.info.author", p_autor.nombre_completo, {"tipo_dato": "nombre"}),
        Metadato("pdf.info.title", rut_titulo, {"tipo_dato": "rut", "formato": "puntos"}),
        Metadato("pdf.info.subject", correo_asunto, {"tipo_dato": "correo", "formato": "punto"}),
        Metadato("pdf.info.keywords", fono_clave, {"tipo_dato": "telefono", "clase": p_clave.telefono_fijo.clase}),
        Metadato("pdf.info.creator", archivo_word, {"tipo_dato": "nombre", "nota": "apellidos en el nombre del .docx"}),
        Metadato("pdf.xmp", p_xmp_creador.nombre_completo, {"campo": "dc:creator", "tipo_dato": "nombre"}),
        Metadato("pdf.xmp", rut_xmp, {"campo": "gore:rutFuncionario", "tipo_dato": "rut", "formato": "sin_puntos"}),
        Metadato("pdf.anotacion", rut_nota, {"anotacion": "Text", "tipo_dato": "rut", "tambien": "nombre"}),
        Metadato("pdf.anotacion", correo_libre, {"anotacion": "FreeText", "tipo_dato": "correo", "visible": True}),
        Metadato("pdf.anotacion", fono_resaltado, {"anotacion": "Highlight", "tipo_dato": "telefono"}),
        Metadato(
            "pdf.adjunto",
            rut_adjunto,
            {
                "forma": "embfile",
                "nombre_archivo": "datos_contacto.txt",
                "tipo_dato": "rut",
                "tambien": "nombre,correo",
            },
        ),
        Metadato(
            "pdf.adjunto",
            rut_anexo,
            {"forma": "FileAttachment", "nombre_archivo": "anexo_antecedentes.txt", "tipo_dato": "rut"},
        ),
        Metadato("pdf.ocg", rut_ocg, {"capa": "Notas internas", "tipo_dato": "rut", "encendida": False}),
        Metadato("pdf.formulario", correo_form, {"campo": "correo_notificacion", "tipo_dato": "correo"}),
        Metadato("pdf.marcador", p_marcador.nombre_completo, {"titulo": marcador, "tipo_dato": "nombre"}),
        Metadato("pdf.javascript", fono_js, {"tipo_dato": "telefono", "ubicacion": "Names/JavaScript"}),
    ]
    return Archivo(
        id=f"{PREFIJO}_metadatos_completos",
        ruta=ruta_rel,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion=(
            "Informe de una página con un bloque de datos visible y un canario distinto en cada estructura: "
            "Info, XMP, anotaciones, adjuntos, capa opcional apagada, formulario, marcador y JavaScript."
        ),
        paginas=paginas_de(destino),
        elementos=hoja.elementos,
        metadatos_sensibles=metadatos,
        etiquetas={"estructuras": sorted({m.donde for m in metadatos})},
    )


# ---------------------------------------------------------------------------
# 2. Revisión incremental
# ---------------------------------------------------------------------------


def _revision_incremental(ctx: Contexto, f: Ficticios, rng: Any) -> Archivo:
    ruta_rel = "pdf_metadatos/revision_incremental.pdf"
    destino = ctx.ruta(ruta_rel)
    doc = pymupdf.open()
    pag = doc.new_page(width=ANCHO, height=ALTO)
    hoja = HojaPdf(pag)

    frases = _frases(rng, 4)
    solicitante = f.persona(dv_valido=False)
    funcionario = f.persona()
    reclamante = f.persona()
    y = _encabezado(hoja, "Resolución exenta", "Responde solicitud de acceso a la información", f.folio(), f.fecha())
    y = _parrafos(hoja, y, frases[:2])
    y += 10
    xv = MARGEN + 100
    hoja.escribir(MARGEN, y, "Solicitante:", fuente="sans_negrita")
    hoja.dato(xv, y, solicitante, "nombre")
    y += INTERLINEA
    hoja.escribir(MARGEN, y, "RUT:", fuente="sans_negrita")
    hoja.dato(xv, y, solicitante, "rut", "espacios_guion")
    y += INTERLINEA
    hoja.escribir(MARGEN, y, "Correo:", fuente="sans_negrita")
    hoja.dato(xv, y, solicitante, "correo", "con_anio")
    y += INTERLINEA * 2

    # Línea que se elimina en la segunda revisión (queda solo en la revisión anterior).
    rut_borrado = reclamante.rut("puntos")
    linea_borrada = f"Reclamante: RUT {rut_borrado} (solicita reserva de identidad)"
    y_borrada = y
    hoja.escribir(MARGEN, y, linea_borrada, fuente="helv", registrar=False)
    caja_borrada = hoja.caja(MARGEN, y, linea_borrada, 10.0, "helv")
    y += INTERLINEA * 2

    hoja.escribir(MARGEN, y, "Funcionario responsable:", fuente="sans_negrita")
    hoja.dato(MARGEN + 150, y, funcionario, "nombre")
    y += INTERLINEA
    hoja.escribir(MARGEN, y, "Teléfono:", fuente="sans_negrita")
    hoja.dato(MARGEN + 150, y, funcionario, "fijo", formato_fijo(funcionario, "internacional"))
    y += INTERLINEA * 2
    y = _parrafos(hoja, y, frases[2:])
    helv_a_literal(doc, pag)
    doc.subset_fonts()
    doc.set_metadata({"producer": "Sistema de gestión documental", "creationDate": FECHA_CREACION})
    doc.save(destino, garbage=3, deflate=True, no_new_id=True)
    doc.close()

    # Segunda revisión: se borra la línea con una redacción y se guarda de forma incremental.
    doc = pymupdf.open(destino)
    pag = doc[0]
    pag.add_redact_annot(caja_borrada + (-1, -1, 1, 1), fill=(1, 1, 1))
    pag.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    nota = "Versión 2: se retiró un dato a solicitud de la persona interesada."
    pag.insert_text((MARGEN, y_borrada), nota, fontname="helv", fontsize=9, color=(0.35, 0.35, 0.35))
    helv_a_literal(doc, pag)
    doc.set_metadata(
        {"producer": "Sistema de gestión documental", "creationDate": FECHA_CREACION, "modDate": FECHA_MODIFICACION}
    )
    # Equivale a ``saveIncr()``, pero sin regenerar el /ID del trailer (determinismo).
    doc.save(destino, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP, no_new_id=True)
    doc.close()

    # La nota de la versión 2 se registra ahora, contra la página ya guardada.
    elementos = hoja.elementos
    with pymupdf.open(destino) as d:
        quads = d[0].search_for(nota, quads=True)
        if len(quads) != 1:
            raise RuntimeError("no se encontró la nota de la versión 2")
        elementos.append(
            Elemento(
                tipo="texto",
                pagina=0,
                poligono=quad_a_poligono(quads[0]),
                valor=nota,
                capa="texto",
                etiquetas={"fuente": "helv", "tam_pt": 9, "revision": 2},
            )
        )
    return Archivo(
        id=f"{PREFIJO}_revision_incremental",
        ruta=ruta_rel,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion=(
            "Resolución guardada en dos revisiones: la segunda (guardado incremental) borra una línea con un RUT, "
            "que sigue recuperable en los bytes de la revisión anterior. La revisión vigente tiene otros datos visibles."
        ),
        paginas=paginas_de(destino),
        elementos=elementos,
        metadatos_sensibles=[
            Metadato(
                "pdf.revision_anterior",
                rut_borrado,
                {"tipo_dato": "rut", "formato": "puntos", "revision": 1, "linea": linea_borrada},
            )
        ],
        etiquetas={"revisiones": 2},
    )


# ---------------------------------------------------------------------------
# 3. Redacción falsa (rectángulos negros sobre texto extraíble)
# ---------------------------------------------------------------------------


def _redaccion_falsa(ctx: Contexto, f: Ficticios, rng: Any) -> Archivo:
    ruta_rel = "pdf_metadatos/redaccion_falsa.pdf"
    doc = pymupdf.open()
    pag = doc.new_page(width=ANCHO, height=ALTO)
    hoja = HojaPdf(pag)
    frases = _frases(rng, 5)
    denunciante = f.persona()
    testigo = f.persona()
    y = _encabezado(hoja, "Acta de denuncia", "Versión pública (censurada)", f.folio(), f.fecha())
    y = _parrafos(hoja, y, frases[:2])
    y += 10
    xv = MARGEN + 90

    def poligono_ultimo() -> pymupdf.Rect:
        return pymupdf.Rect(*hoja.elementos[-1].poligono[0], *hoja.elementos[-1].poligono[2])

    # 1 y 2: rectángulos negros dibujados en el contenido, encima del texto.
    hoja.escribir(MARGEN, y, "Denunciante:", fuente="sans_negrita")
    hoja.dato(xv, y, denunciante, "nombre", etiquetas={"tapado": "rectangulo_dibujado"})
    pag.draw_rect(poligono_ultimo() + (-1.5, -1, 1.5, 1), color=None, fill=(0, 0, 0), overlay=True)
    y += INTERLINEA
    hoja.escribir(MARGEN, y, "RUT:", fuente="sans_negrita")
    hoja.dato(xv, y, denunciante, "rut", "sin_puntos", etiquetas={"tapado": "rectangulo_dibujado"})
    pag.draw_rect(poligono_ultimo() + (-1.5, -1, 1.5, 1), color=None, fill=(0, 0, 0), overlay=True)
    y += INTERLINEA
    # 3: anotación de redacción marcada pero no aplicada.
    hoja.escribir(MARGEN, y, "Correo:", fuente="sans_negrita")
    hoja.dato(xv, y, denunciante, "correo", "punto", etiquetas={"tapado": "redact_sin_aplicar"})
    pag.add_redact_annot(poligono_ultimo() + (-1.5, -1, 1.5, 1), fill=(0, 0, 0))
    y += INTERLINEA
    # 4: anotación cuadrada con relleno negro sobre un teléfono.
    hoja.escribir(MARGEN, y, "Teléfono:", fuente="sans_negrita")
    hoja.dato(xv, y, denunciante, "movil", "movil_parentesis", etiquetas={"tapado": "anotacion_cuadrada"})
    cuadro = pag.add_rect_annot(poligono_ultimo() + (-1.5, -1, 1.5, 1))
    cuadro.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
    cuadro.set_border(width=0.5)
    cuadro.update()
    y += INTERLINEA * 2

    # Un dato visible sin tapar, para que el archivo tenga también el caso normal.
    hoja.escribir(MARGEN, y, "Testigo:", fuente="sans_negrita")
    fin = hoja.dato(xv, y, testigo, "nombre")
    hoja.escribir(fin, y, ",")
    y += INTERLINEA
    hoja.campo(MARGEN, y, "Monto reclamado:", f.monto(), tipo="texto", etiquetas={"senuelo": "monto"})
    y += INTERLINEA * 2
    _parrafos(hoja, y, frases[2:])

    doc.subset_fonts()
    destino = ctx.ruta(ruta_rel)
    doc.save(destino, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return Archivo(
        id=f"{PREFIJO}_redaccion_falsa",
        ruta=ruta_rel,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion=(
            "Censura aparente: rectángulos negros dibujados sobre un nombre y un RUT, una redacción sin aplicar "
            "sobre un correo y una anotación cuadrada negra sobre un teléfono; el texto sigue siendo extraíble."
        ),
        paginas=paginas_de(destino),
        elementos=hoja.elementos,
        etiquetas={"censura_aparente": True},
    )


# ---------------------------------------------------------------------------
# 4. Solo contraseña de propietario (permisos)
# ---------------------------------------------------------------------------


def _solo_permisos(ctx: Contexto, f: Ficticios, rng: Any) -> Archivo:
    ruta_rel = "pdf_metadatos/solo_permisos.pdf"
    doc = pymupdf.open()
    pag = doc.new_page(width=ANCHO, height=ALTO)
    hoja = HojaPdf(pag)
    frases = _frases(rng, 4)
    beneficiaria = f.persona()
    aval = f.persona(en_lista=False, siete_digitos=True)
    y = _encabezado(
        hoja, "Certificado de beneficio", "Subsidio regional de apoyo al emprendimiento", f.folio(), f.fecha()
    )
    y = _parrafos(hoja, y, frases[:2])
    y += 10
    xv = MARGEN + 100
    for etiqueta, clase, formato in (
        ("Titular:", "nombre", ""),
        ("RUT:", "rut", "sin_puntos"),
        ("Correo:", "correo", "mayusculas"),
        ("Teléfono fijo:", "fijo", formato_fijo(beneficiaria, "parentesis")),
        ("Celular:", "movil", "movil_guiones"),
        ("Domicilio:", "direccion", ""),
    ):
        hoja.escribir(MARGEN, y, etiqueta, fuente="sans_negrita")
        hoja.dato(xv, y, beneficiaria, clase, formato)
        y += INTERLINEA
    hoja.campo(MARGEN, y, "Monto asignado:", f.monto(), tipo="texto", x_valor=xv, etiquetas={"senuelo": "monto"})
    y += INTERLINEA * 2
    hoja.escribir(MARGEN, y, "Aval:", fuente="sans_negrita")
    fin = hoja.dato(xv, y, aval, "nombre")
    fin = hoja.escribir(fin, y, ", RUT")
    hoja.dato(fin + hoja.ancho(" "), y, aval, "rut", "puntos")
    y += INTERLINEA
    hoja.escribir(MARGEN, y, "Domicilio del aval:", fuente="sans_negrita")
    hoja.dato(MARGEN + 120, y, aval, "direccion")
    y += INTERLINEA * 2
    _parrafos(hoja, y, frases[2:])

    doc.subset_fonts()
    destino = ctx.ruta(ruta_rel)
    permisos = pymupdf.PDF_PERM_ACCESSIBILITY  # sin imprimir, copiar, modificar ni anotar
    doc.save(
        destino,
        garbage=3,
        deflate=True,
        no_new_id=True,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw=CONTRASENA_PROPIETARIO,
        user_pw="",
        permissions=permisos,
    )
    doc.close()
    return Archivo(
        id=f"{PREFIJO}_solo_permisos",
        ruta=ruta_rel,
        formato="pdf",
        categoria=CATEGORIA,
        descripcion=(
            "Certificado cifrado (AES-256) solo con contraseña de propietario: abre sin contraseña pero restringe "
            "copiar e imprimir. Debe procesarse, no rechazarse."
        ),
        paginas=paginas_de(destino),
        elementos=hoja.elementos,
        esperado="procesar",
        etiquetas={"cifrado": "AES-256", "contrasena_usuario": "", "permisos": permisos},
    )


def generar(ctx: Contexto) -> list[Archivo]:
    f = ctx.ficticios(MODULO)
    rng = ctx.rng(MODULO)
    return [
        _metadatos_completos(ctx, f, rng),
        _revision_incremental(ctx, f, rng),
        _redaccion_falsa(ctx, f, rng),
        _solo_permisos(ctx, f, rng),
    ]

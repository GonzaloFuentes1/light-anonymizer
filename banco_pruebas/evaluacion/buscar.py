"""Búsqueda de valores (canarios) y de sus fragmentos críticos en texto extraído y en bytes.

Cada valor del manifiesto se convierte en una o más *agujas*: la forma normalizada del valor
completo y los fragmentos críticos definidos en ``docs/metricas.md``:

- RUT: cuerpo sin dígito verificador (si tiene al menos 7 dígitos).
- Teléfono: últimos 7 dígitos.
- Correo: parte local más ``@`` (o, en formatos sin ``@``, todo lo que precede al dominio).
- Nombre: apellidos unidos (lo que va antes de la coma, o las dos últimas palabras).
- Dirección: calle y número (hasta el primer número).

Cada aguja pertenece a una *clase* que dice cómo se normaliza el texto donde se busca (el
*pajar*):

- ``digitos``: mayúsculas y sin separadores entre dígitos (``12.345.678-5`` -> ``123456785``).
  Solo se quitan separadores *entre* dígitos, para no unir números de un binario cualquiera.
- ``compacto``: sin espacios y en minúsculas (correos y URL).
- ``general``: ``esquema.normalizar`` (sin tildes, minúsculas, espacios colapsados).

Los bytes se convierten en texto con cuatro codificaciones (UTF-8, Latin-1, UTF-16 BE y
UTF-16 LE) extrayendo solo las corridas imprimibles (como ``strings``), de modo que buscar en
un archivo de varios MB no dispare coincidencias al azar dentro de datos comprimidos.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from banco_pruebas.esquema import normalizar

CLASES = ("digitos", "compacto", "general")

# Separadores que pueden ir entre los dígitos de un RUT o teléfono. Incluye guiones Unicode y
# los caracteres de control C1 (0x80-0x9f), que es como se ven el guion largo de WinAnsi y de
# PDFDocEncoding al decodificar como Latin-1.
_GUIONES = "".join(chr(c) for c in range(0x2010, 0x2016)) + chr(0x2212)  # guiones Unicode y signo menos
_SEPARADORES_DIGITOS = r"[\s.,\-" + _GUIONES + r"_()+/\x80-\x9f]"
_UNIR_DIGITOS = re.compile(rf"(?<=[0-9K]){_SEPARADORES_DIGITOS}{{1,3}}(?=[0-9K])")
_ESPACIOS = re.compile(r"\s+")


@dataclass(frozen=True)
class Aguja:
    """Una cadena normalizada que no debe aparecer en la salida."""

    forma: str
    clase: str  # digitos | compacto | general
    parte: str  # "valor" o el nombre del fragmento crítico
    # Expresión regular alternativa para valores con letras no ASCII: cada una se acepta como
    # 0 a 2 caracteres cualesquiera, porque hay escritores que las cambian por "?", las omiten
    # o las dejan como mojibake UTF-8 leído en Latin-1 ("MarÃ + guion blando + a").
    patron: str | None = None
    # Formas tal como se escribieron (en mayúsculas), para agujas de dígitos buscadas en fuentes
    # binarias, donde unir dígitos separados por espacios daría coincidencias al azar.
    literales: tuple[str, ...] = ()

    def en(self, pajar: str) -> bool:
        if self.forma in pajar:
            return True
        return self.patron is not None and _regex(self.patron).search(pajar) is not None


# Marcas combinantes (tildes, diéresis...): las que quita ``esquema.normalizar`` en texto latino.
_RANGOS_COMBINANTES = ((0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F))
_COMBINANTES = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _RANGOS_COMBINANTES) + "]")


def normalizar_general(texto: str) -> str:
    """Igual que ``esquema.normalizar`` para nombres y texto, pero sin bucle en Python (pajares de MB)."""
    t = _COMBINANTES.sub("", unicodedata.normalize("NFKD", texto)).casefold()
    return " ".join(t.split())


def normalizar_pajar(texto: str, clase: str) -> str:
    """Normaliza un texto donde se buscará (pajar) según la clase de aguja."""
    if clase == "digitos":
        return _UNIR_DIGITOS.sub("", texto.upper())
    if clase == "compacto":
        return _ESPACIOS.sub("", texto).casefold()
    return normalizar_general(texto)


_SIN_DV = re.compile(r"[\s\-" + _GUIONES + r"]*[0-9K]\s*$")


def _ultimos_digitos(texto: str, n: int) -> str:
    """Sufijo de ``texto`` (tal como está escrito) que contiene sus últimos ``n`` dígitos."""
    vistos = 0
    for i in range(len(texto) - 1, -1, -1):
        if texto[i].isdigit():
            vistos += 1
            if vistos == n:
                return texto[i:]
    return texto


def _digitos(valor: str) -> str:
    return "".join(c for c in valor.upper() if c.isdigit() or c == "K")


@lru_cache(maxsize=4096)
def _regex(patron: str) -> re.Pattern[str]:
    return re.compile(patron)


def _aguja_general(texto: str, parte: str) -> Aguja:
    forma = normalizar("texto", texto)
    patron = None
    if not texto.isascii():
        limpio = " ".join(unicodedata.normalize("NFC", texto).split()).casefold()
        patron = "".join(re.escape(c) if c.isascii() else ".{0,2}" for c in limpio)
    return Aguja(forma, "general", parte, patron)


def _es_inicial(palabra: str) -> bool:
    limpia = palabra.strip(".,;")
    return len(limpia) <= 2 or palabra.endswith(".")


def apellidos(valor: str) -> str | None:
    """Apellidos de un nombre escrito en cualquiera de las variantes habituales."""
    if "," in valor:
        candidato = valor.split(",")[0]
    else:
        palabras = valor.split()
        if len(palabras) < 3:
            return None
        candidato = " ".join(palabras[-2:])
    palabras = candidato.split()
    if len(palabras) < 2 or any(_es_inicial(p) for p in palabras):
        return None
    return candidato


def calle_numero(valor: str) -> str | None:
    """Calle y número de una dirección (hasta el primer número inclusive)."""
    m = re.match(r"^(\D*?\S*\d+)", valor)
    if not m or not re.search(r"[^\W\d]", m.group(1)):
        return None
    return m.group(1)


def agujas(tipo: str, valor: str | None) -> list[Aguja]:
    """Agujas (valor completo y fragmentos críticos) de un elemento del manifiesto."""
    if not valor or tipo in ("rostro", "firma"):
        return []
    salida: list[Aguja] = []
    if tipo == "rut":
        d = _digitos(normalizar("rut", valor))
        escrito = valor.upper().strip()
        if d:
            salida.append(Aguja(d, "digitos", "valor", literales=(escrito,)))
        if len(d) - 1 >= 7:
            cuerpo = _SIN_DV.sub("", escrito)
            salida.append(Aguja(d[:-1], "digitos", "cuerpo_rut", literales=(cuerpo,)))
    elif tipo == "telefono":
        d = "".join(c for c in valor if c.isdigit())
        escrito = valor.upper().strip()
        if d:
            salida.append(Aguja(d, "digitos", "valor", literales=(escrito,)))
        if len(d) > 7:
            salida.append(Aguja(d[-7:], "digitos", "ultimos7", literales=(_ultimos_digitos(escrito, 7),)))
    elif tipo in ("correo", "url", "qr"):
        c = normalizar_pajar(valor, "compacto")
        salida.append(Aguja(c, "compacto", "valor"))
        if tipo == "correo":
            if "@" in c:
                salida.append(Aguja(c.split("@")[0] + "@", "compacto", "local@"))
            else:
                m = re.match(r"^(.+?)[a-z0-9-]+(?:\.[a-z0-9-]+)+$", c)
                if m:
                    salida.append(Aguja(m.group(1), "compacto", "local@"))
    else:
        salida.append(_aguja_general(valor, "valor"))
        fragmento = apellidos(valor) if tipo == "nombre" else calle_numero(valor) if tipo == "direccion" else None
        if fragmento:
            parte = "apellidos" if tipo == "nombre" else "calle_numero"
            salida.append(_aguja_general(fragmento, parte))
    # sin agujas vacías ni repetidas
    vistas: set[tuple[str, str]] = set()
    unicas = []
    for a in salida:
        if a.forma and (a.forma, a.clase) not in vistas:
            vistas.add((a.forma, a.clase))
            unicas.append(a)
    return unicas


def agujas_metadato(valor: str | None, tipo: str | None = None) -> list[Aguja]:
    """Agujas de un canario de metadatos: el valor completo en las tres normalizaciones."""
    if not valor:
        return []
    if tipo and tipo not in ("texto",):
        base = agujas(tipo, valor)
        if base:
            return base
    salida = [
        _aguja_general(valor, "valor"),
        Aguja(normalizar_pajar(valor, "compacto"), "compacto", "valor"),
    ]
    d = _digitos(valor)
    if len(d) >= 7 and len(d) >= 0.6 * len(valor.replace(" ", "")):
        salida.append(Aguja(d, "digitos", "valor", literales=(valor.upper().strip(),)))
    return [a for a in salida if a.forma]


# ---------------------------------------------------------------------------
# Pajar: textos donde buscar, agrupados por origen, normalizados una sola vez por clase.
# ---------------------------------------------------------------------------


@dataclass
class Pajar:
    """Colección de textos de un archivo, agrupados por origen (``texto_pymupdf``, ``bytes``...)."""

    grupos: dict[str, list[str]] = field(default_factory=dict)
    binarios: set[str] = field(default_factory=set)
    _cache: dict[tuple[str, str], str] = field(default_factory=dict)

    def agregar(self, origen: str, texto: str | Iterable[str], binario: bool = False) -> None:
        """Agrega textos a un origen. ``binario``: corridas sacadas de bytes crudos (flujos de
        contenido, datos de imagen...), llenas de números separados por espacios; ahí los dígitos
        no se unen y las agujas de dígitos se buscan tal como se escribieron."""
        textos = [texto] if isinstance(texto, str) else list(texto)
        textos = [t for t in textos if t]
        if binario:
            self.binarios.add(origen)
        if textos:
            self.grupos.setdefault(origen, []).extend(textos)
            for clase in CLASES:
                self._cache.pop((origen, clase), None)

    def _normalizado(self, origen: str, clase: str) -> str:
        clave = (origen, clase)
        if clave not in self._cache:
            # \x00 separa textos distintos: no es separador de dígitos ni espacio
            if clase == "digitos" and origen in self.binarios:
                self._cache[clave] = "\n\x00\n".join(t.upper() for t in self.grupos[origen])
            else:
                self._cache[clave] = "\n\x00\n".join(normalizar_pajar(t, clase) for t in self.grupos[origen])
        return self._cache[clave]

    def buscar(self, agujas_: Iterable[Aguja], origenes: Iterable[str] | None = None) -> list[tuple[Aguja, str]]:
        """Devuelve los pares (aguja, origen) encontrados."""
        encontrados = []
        lista = list(agujas_)
        for origen in origenes if origenes is not None else list(self.grupos):
            if origen not in self.grupos:
                continue
            binario = origen in self.binarios
            for a in lista:
                pajar = self._normalizado(origen, a.clase)
                if binario and a.clase == "digitos":
                    hallado = any(x and x in pajar for x in a.literales)
                else:
                    hallado = a.en(pajar)
                if hallado:
                    encontrados.append((a, origen))
        return encontrados

    def contiene(self, agujas_: Iterable[Aguja], origenes: Iterable[str] | None = None) -> bool:
        return bool(self.buscar(agujas_, origenes))


# ---------------------------------------------------------------------------
# Bytes -> textos en varias codificaciones
# ---------------------------------------------------------------------------

_RUN_LATIN1 = re.compile(rb"[\t\n\r\x20-\x7e\x80-\xff]{4,}")
_RUN_UTF8 = re.compile(rb"(?:[\t\n\r\x20-\x7e]|[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}){4,}")
# UTF-16: Latin-1 imprimible más los guiones U+2010..U+2015 (guion largo del RUT)
_RUN_UTF16LE = re.compile(rb"(?:[\t\n\r\x20-\x7e\xa0-\xff]\x00|[\x10-\x15]\x20){4,}")
_RUN_UTF16BE = re.compile(rb"(?:\x00[\t\n\r\x20-\x7e\xa0-\xff]|\x20[\x10-\x15]){4,}")


def textos_de_bytes(datos: bytes) -> list[str]:
    """Corridas imprimibles de ``datos`` decodificadas como UTF-8, Latin-1, UTF-16 BE y UTF-16 LE."""
    if not datos:
        return []
    salida = []
    latin = [m.group().decode("latin-1") for m in _RUN_LATIN1.finditer(datos)]
    if latin:
        salida.append("\n".join(latin))
    if re.search(rb"[\x80-\xff]", datos):
        utf8 = [m.group().decode("utf-8", "ignore") for m in _RUN_UTF8.finditer(datos)]
        if utf8:
            salida.append("\n".join(utf8))
    if b"\x00" in datos:
        le = [m.group().decode("utf-16-le", "ignore") for m in _RUN_UTF16LE.finditer(datos)]
        be = [m.group().decode("utf-16-be", "ignore") for m in _RUN_UTF16BE.finditer(datos)]
        salida.extend("\n".join(x) for x in (le, be) if x)
    return salida


def decodificar_cadena(datos: bytes) -> list[str]:
    """Decodifica una cadena PDF o un campo de metadatos: BOM UTF-16/UTF-8, luego UTF-8, luego Latin-1."""
    if datos.startswith(b"\xfe\xff"):
        return [datos[2:].decode("utf-16-be", "ignore")]
    if datos.startswith(b"\xff\xfe"):
        return [datos[2:].decode("utf-16-le", "ignore")]
    if datos.startswith(b"\xef\xbb\xbf"):
        return [datos[3:].decode("utf-8", "ignore")]
    salida = []
    try:
        salida.append(datos.decode("utf-8"))
    except UnicodeDecodeError:
        salida.append(datos.decode("latin-1"))
    if b"\x00" in datos and len(datos) >= 4:
        salida.append(datos.decode("utf-16-be", "ignore"))
        salida.append(datos.decode("utf-16-le", "ignore"))
    return salida

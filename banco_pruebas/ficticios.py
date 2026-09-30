"""Datos personales ficticios, deterministas a partir de una semilla y sin repeticiones.

Cada valor (RUT, teléfono, correo) es único en todo el conjunto: funciona como canario, de
modo que si aparece en un archivo de salida se sabe exactamente de qué elemento vino.
Los dominios de correo y sitios son inventados; los nombres combinan nombres y apellidos
frecuentes en Chile, al azar.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field

NOMBRES_F = [
    "Ana María",
    "María José",
    "Camila",
    "Valentina",
    "Francisca",
    "Catalina",
    "Fernanda",
    "Sofía",
    "Javiera",
    "Constanza",
    "Daniela",
    "Isidora",
    "Antonia",
    "Josefa",
    "Paula",
    "Macarena",
    "Carolina",
    "Verónica",
    "Marcela",
    "Patricia",
    "Rosa",
    "Ximena",
    "Loreto",
    "Solange",
]
NOMBRES_M = [
    "José Tomás",
    "Ignacio",
    "Benjamín",
    "Martín",
    "Matías",
    "Joaquín",
    "Vicente",
    "Cristóbal",
    "Sebastián",
    "Nicolás",
    "Felipe",
    "Diego",
    "Rodrigo",
    "Andrés",
    "Gonzalo",
    "Héctor",
    "Luis Alberto",
    "Juan Pablo",
    "Patricio",
    "Mauricio",
    "Ramón",
    "Óscar",
    "Agustín",
    "Álvaro",
]
APELLIDOS = [
    "Rojas",
    "Peña",
    "González",
    "Muñoz",
    "Díaz",
    "Pérez",
    "Soto",
    "Contreras",
    "Silva",
    "Martínez",
    "Sepúlveda",
    "Morales",
    "Rodríguez",
    "López",
    "Fuentealba",
    "Araya",
    "Espinoza",
    "Valenzuela",
    "Castillo",
    "Tapia",
    "Reyes",
    "Gutiérrez",
    "Castro",
    "Pizarro",
    "Álvarez",
    "Vásquez",
    "Sánchez",
    "Fernández",
    "Ramírez",
    "Carrasco",
    "Gómez",
    "Cortés",
    "Herrera",
    "Núñez",
    "Jara",
    "Vergara",
    "Rivera",
    "Figueroa",
    "Riquelme",
    "Miranda",
    "Bravo",
    "Vera",
    "Molina",
    "Vega",
    "Campos",
    "Sandoval",
    "Orellana",
    "Cárdenas",
    "Olivares",
    "Alarcón",
    "Gallardo",
    "Garrido",
    "Salazar",
    "Henríquez",
    "Saavedra",
    "Navarro",
    "Aguilera",
    "Parra",
    "Aravena",
    "Vidal",
    "Cáceres",
    "Quiroz",
    "Ñancupil",
    "Huenchullán",
    "Catrileo",
    "Quilapán",
    "Painemal",
    "Millaqueo",
    "Peñailillo",
    "Ibáñez",
]
CALLES = [
    "Pasaje Los Alerces",
    "Avenida Los Canelos",
    "Calle Los Arrayanes",
    "Pasaje El Boldo",
    "Calle Las Quilas",
    "Avenida Los Coihues",
    "Pasaje Los Maitenes",
    "Calle El Radal",
    "Avenida Las Araucarias",
    "Pasaje Los Ulmos",
    "Calle Los Notros",
    "Pasaje El Litre",
    "Avenida Los Espinos",
    "Calle Las Lumas",
    "Pasaje Los Peumos",
]
COMUNAS = [
    "Concepción",
    "Talcahuano",
    "Chillán",
    "Los Ángeles",
    "Temuco",
    "Valdivia",
    "Puerto Montt",
    "Rancagua",
    "Talca",
    "La Serena",
    "Antofagasta",
    "Iquique",
    "Arica",
    "Coyhaique",
    "Punta Arenas",
    "Copiapó",
]
DOMINIOS = [
    "ejemplo.cl",
    "goreficticio.cl",
    "muniejemplo.cl",
    "correo.ejemplo.cl",
    "ficticio.gob.cl",
    "consultora-ficticia.cl",
    "mail.ejemplo.com",
    "servicioficticio.cl",
]
# Códigos de área regionales (7 dígitos de abonado). Santiago es 2 con 8 dígitos.
CODIGOS_AREA = {
    "32": "Valparaíso",
    "33": "Quillota",
    "34": "San Felipe",
    "35": "San Antonio",
    "41": "Concepción",
    "42": "Chillán",
    "43": "Los Ángeles",
    "45": "Temuco",
    "51": "La Serena",
    "52": "Copiapó",
    "53": "Ovalle",
    "55": "Antofagasta",
    "57": "Iquique",
    "58": "Arica",
    "61": "Punta Arenas",
    "63": "Valdivia",
    "64": "Osorno",
    "65": "Puerto Montt",
    "67": "Coyhaique",
    "71": "Talca",
    "72": "Rancagua",
    "73": "Linares",
    "75": "Curicó",
}

# Formatos de escritura. Nivel "base": deben censurarse siempre. "estres": variantes raras o antiguas.
FORMATOS_RUT = {
    "puntos": "base",  # 12.345.678-5
    "sin_puntos": "base",  # 12345678-5
    "espacios_guion": "base",  # 12.345.678 - 5
    "guion_largo": "base",  # 12.345.678–5 (autocorrección de procesadores de texto)
    "sin_guion": "estres",  # 123456785
    "comas": "estres",  # 12,345,678-5
    "espacios_internos": "estres",  # 12 345 678-5
}
FORMATOS_TELEFONO = {
    # móvil: 9 + 8 dígitos
    "movil_internacional": "base",  # +56 9 8123 4567
    "movil_compacto": "base",  # +56981234567
    "movil_nacional": "base",  # 9 8123 4567
    "movil_bloque": "base",  # 98123 4567
    "movil_parentesis": "base",  # (+56) 9 8123 4567
    "movil_guiones": "base",  # +56-9-8123-4567
    # Santiago: 2 + 8 dígitos
    "santiago_internacional": "base",  # +56 2 2123 4567
    "santiago_nacional": "base",  # 22 123 4567
    "santiago_parentesis": "base",  # (2) 2123 4567
    # regional: código de área de 2 dígitos + 7 dígitos
    "regional_internacional": "base",  # +56 41 221 3456
    "regional_parentesis": "base",  # (41) 221 3456
    "regional_compacto": "base",  # (41) 2213456
    "regional_guion": "base",  # 41-2213456
    # formas antiguas (antes de 2012-2016)
    "antiguo_movil_09": "estres",  # 09-8123 4567
    "antiguo_regional_0": "estres",  # (041) 2213456
    "antiguo_movil_8": "estres",  # 8123 4567 (8 dígitos sin el 9)
}
FORMATOS_CORREO = {
    "punto": "base",  # ana.rojas@ejemplo.cl
    "inicial": "base",  # arojas@ejemplo.cl
    "con_anio": "base",  # ana.rojas87@ejemplo.cl
    "guion_bajo": "base",  # ana_rojas@ejemplo.cl
    "mayusculas": "base",  # ANA.ROJAS@EJEMPLO.CL
    "mas": "base",  # ana.rojas+honorarios@ejemplo.cl
    "subdominio": "base",  # ana-rojas@correo.ejemplo.cl
    "arroba_texto": "estres",  # ana.rojas [arroba] ejemplo.cl
    "at": "estres",  # ana.rojas(at)ejemplo.cl
    "espacios": "estres",  # ana.rojas @ ejemplo.cl
}


def sin_tildes(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in t if not unicodedata.combining(c))


def digito_verificador(cuerpo: int) -> str:
    """Módulo 11, igual que ``rut_valido`` del cuaderno."""
    suma, factor = 0, 2
    for d in reversed(str(cuerpo)):
        suma += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - (suma % 11)
    return {11: "0", 10: "K"}.get(resto, str(resto))


def formatear_rut(cuerpo: int, dv: str, formato: str = "puntos") -> str:
    con_puntos = f"{cuerpo:,}".replace(",", ".")
    match formato:
        case "puntos":
            return f"{con_puntos}-{dv}"
        case "sin_puntos":
            return f"{cuerpo}-{dv}"
        case "espacios_guion":
            return f"{con_puntos} - {dv}"
        case "guion_largo":
            return f"{con_puntos}–{dv}"
        case "sin_guion":
            return f"{cuerpo}{dv}"
        case "comas":
            return f"{cuerpo:,}-{dv}"
        case "espacios_internos":
            return f"{cuerpo:,}".replace(",", " ") + f"-{dv}"
    raise ValueError(formato)


@dataclass
class Telefono:
    clase: str  # movil, santiago, regional
    nacional: str  # 9 dígitos, p. ej. 981234567 o 412213456
    area: str  # "9", "2" o el código de área regional

    def formatear(self, formato: str) -> str:
        n = self.nacional
        if self.clase == "movil":
            a, b = n[1:5], n[5:]
            return {
                "movil_internacional": f"+56 9 {a} {b}",
                "movil_compacto": f"+569{a}{b}",
                "movil_nacional": f"9 {a} {b}",
                "movil_bloque": f"9{a} {b}",
                "movil_parentesis": f"(+56) 9 {a} {b}",
                "movil_guiones": f"+56-9-{a}-{b}",
                "antiguo_movil_09": f"09-{a} {b}",
                "antiguo_movil_8": f"{a} {b}",
            }[formato]
        if self.clase == "santiago":
            a, b = n[1:5], n[5:]
            return {
                "santiago_internacional": f"+56 2 {a} {b}",
                "santiago_nacional": f"2{a[0]} {a[1:]} {b}",
                "santiago_parentesis": f"(2) {a} {b}",
            }[formato]
        area, abonado = n[:2], n[2:]
        return {
            "regional_internacional": f"+56 {area} {abonado[:3]} {abonado[3:]}",
            "regional_parentesis": f"({area}) {abonado[:3]} {abonado[3:]}",
            "regional_compacto": f"({area}) {abonado}",
            "regional_guion": f"{area}-{abonado}",
            "antiguo_regional_0": f"(0{area}) {abonado}",
        }[formato]


FORMATOS_POR_CLASE = {
    "movil": [f for f in FORMATOS_TELEFONO if f.startswith(("movil", "antiguo_movil"))],
    "santiago": [f for f in FORMATOS_TELEFONO if f.startswith("santiago")],
    "regional": [f for f in FORMATOS_TELEFONO if f.startswith(("regional", "antiguo_regional"))],
}


@dataclass
class Persona:
    nombres: str
    apellido_p: str
    apellido_m: str
    sexo: str  # F / M
    rut_cuerpo: int
    rut_dv: str
    rut_dv_valido: bool
    telefono: Telefono
    telefono_fijo: Telefono
    dominio: str
    direccion: str
    comuna: str
    nacimiento: str  # dd-mm-aaaa
    en_lista: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def nombre_completo(self) -> str:
        return f"{self.nombres} {self.apellido_p} {self.apellido_m}"

    def rut(self, formato: str = "puntos") -> str:
        return formatear_rut(self.rut_cuerpo, self.rut_dv, formato)

    def correo(self, formato: str = "punto") -> str:
        n = sin_tildes(self.nombres.split()[0]).lower().replace("ñ", "n")
        a = sin_tildes(self.apellido_p).lower().replace("ñ", "n")
        d = self.dominio
        return {
            "punto": f"{n}.{a}@{d}",
            "inicial": f"{n[0]}{a}@{d}",
            "con_anio": f"{n}.{a}{self.nacimiento[-2:]}@{d}",
            "guion_bajo": f"{n}_{a}@{d}",
            "mayusculas": f"{n}.{a}@{d}".upper(),
            "mas": f"{n}.{a}+honorarios@{d}",
            "subdominio": f"{n}-{a}@correo.{d}" if not d.startswith("correo.") else f"{n}-{a}@{d}",
            "arroba_texto": f"{n}.{a} [arroba] {d}",
            "at": f"{n}.{a}(at){d}",
            "espacios": f"{n}.{a} @ {d}",
        }[formato]


class Ficticios:
    """Fábrica de personas y valores ficticios únicos.

    ``derivar(nombre)`` entrega una fábrica con su propio azar (sembrado por nombre) que comparte
    el registro de valores usados: así cada generador produce siempre los mismos datos aunque
    se ejecute solo, y ningún valor se repite en todo el conjunto.
    """

    def __init__(self, semilla: int) -> None:
        self.semilla = semilla
        self.rng = random.Random(semilla)
        self._ruts: set[int] = set()
        self._telefonos: set[str] = set()
        self._correos: set[tuple[str, str]] = set()
        self._nombres: set[str] = set()
        self.personas: list[Persona] = []

    def derivar(self, nombre: str) -> Ficticios:
        hijo = Ficticios.__new__(Ficticios)
        hijo.semilla = self.semilla
        hijo.rng = random.Random(f"{self.semilla}:{nombre}")
        hijo._ruts, hijo._telefonos = self._ruts, self._telefonos
        hijo._correos, hijo._nombres = self._correos, self._nombres
        hijo.personas = self.personas
        return hijo

    def reservar(
        self,
        ruts: tuple[int, ...] = (),
        telefonos: tuple[str, ...] = (),
        correos: tuple[tuple[str, str], ...] = (),
        nombres: tuple[str, ...] = (),
    ) -> None:
        """Marca como usados valores escritos a mano (por ejemplo, los del cuaderno)."""
        self._ruts.update(ruts)
        self._telefonos.update(telefonos)
        self._correos.update(correos)
        self._nombres.update(nombres)

    # -- valores sueltos -----------------------------------------------------

    def rut(self, dv_valido: bool = True, siete_digitos: bool = False) -> tuple[int, str]:
        while True:
            cuerpo = (
                self.rng.randint(1_000_000, 9_999_999) if siete_digitos else self.rng.randint(10_000_000, 26_999_999)
            )
            if cuerpo not in self._ruts:
                break
        self._ruts.add(cuerpo)
        dv = digito_verificador(cuerpo)
        if not dv_valido:
            dv = self.rng.choice([c for c in "0123456789K" if c != dv])
        return cuerpo, dv

    def rut_con_k(self) -> tuple[int, str]:
        """Un RUT válido cuyo dígito verificador es K."""
        while True:
            cuerpo = self.rng.randint(10_000_000, 26_999_999)
            if cuerpo not in self._ruts and digito_verificador(cuerpo) == "K":
                self._ruts.add(cuerpo)
                return cuerpo, "K"

    def telefono(self, clase: str = "movil") -> Telefono:
        while True:
            if clase == "movil":
                nacional, area = "9" + "".join(self.rng.choice("0123456789") for _ in range(8)), "9"
            elif clase == "santiago":
                nacional, area = "22" + "".join(self.rng.choice("0123456789") for _ in range(7)), "2"
            else:
                area = self.rng.choice(sorted(CODIGOS_AREA))
                nacional = area + self.rng.choice("2345") + "".join(self.rng.choice("0123456789") for _ in range(6))
            # los últimos 7 dígitos también deben ser únicos (se buscan como fuga parcial)
            if nacional not in self._telefonos and not any(t[-7:] == nacional[-7:] for t in self._telefonos):
                self._telefonos.add(nacional)
                return Telefono(clase, nacional, area)

    # -- personas ------------------------------------------------------------

    def persona(self, dv_valido: bool = True, en_lista: bool = True, siete_digitos: bool = False) -> Persona:
        while True:
            sexo = self.rng.choice("FM")
            nombres = self.rng.choice(NOMBRES_F if sexo == "F" else NOMBRES_M)
            ap, am = self.rng.sample(APELLIDOS, 2)
            completo = f"{nombres} {ap} {am}"
            clave_correo = (sin_tildes(nombres.split()[0]).lower(), sin_tildes(ap).lower())
            if completo not in self._nombres and clave_correo not in self._correos:
                break
        self._nombres.add(completo)
        self._correos.add(clave_correo)
        cuerpo, dv = self.rut(dv_valido=dv_valido, siete_digitos=siete_digitos)
        clase_fijo = self.rng.choice(["santiago", "regional", "regional"])
        p = Persona(
            nombres=nombres,
            apellido_p=ap,
            apellido_m=am,
            sexo=sexo,
            rut_cuerpo=cuerpo,
            rut_dv=dv,
            rut_dv_valido=dv_valido,
            telefono=self.telefono("movil"),
            telefono_fijo=self.telefono(clase_fijo),
            dominio=self.rng.choice(DOMINIOS),
            direccion=f"{self.rng.choice(CALLES)} {self.rng.randint(100, 3999)}"
            + (f", depto {self.rng.randint(11, 1204)}" if self.rng.random() < 0.4 else ""),
            comuna=self.rng.choice(COMUNAS),
            nacimiento=f"{self.rng.randint(1, 28):02d}-{self.rng.randint(1, 12):02d}-{self.rng.randint(1950, 2002)}",
            en_lista=en_lista,
        )
        self.personas.append(p)
        return p

    def lista_nombres(self) -> list[str]:
        """Lista que se le entrega al motor: nombres (y direcciones) de las personas marcadas ``en_lista``."""
        salida = []
        for p in self.personas:
            if p.en_lista:
                salida.append(p.nombre_completo)
                salida.append(p.direccion)
        return salida

    # -- texto neutro (señuelos: no son datos personales) --------------------

    def monto(self) -> str:
        return f"$ {self.rng.randint(150, 4800) * 1000:,}".replace(",", ".")

    def fecha(self) -> str:
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                 "septiembre", "octubre", "noviembre", "diciembre"]  # fmt: skip
        return f"{self.rng.randint(1, 28)} de {self.rng.choice(meses)} de 2026"

    def folio(self) -> str:
        return f"{self.rng.randint(100, 9999)}/{self.rng.choice(['2025', '2026'])}"


FRASES_ADMINISTRATIVAS = [
    "Producto 1: Informe de avance del programa de fomento productivo.",
    "Producto 2: Sistematización de talleres con organizaciones comunitarias.",
    "Se adjunta el respaldo de las actividades realizadas durante el período.",
    "La contraparte técnica valida el cumplimiento de los productos comprometidos.",
    "El pago se efectuará previa recepción conforme del informe mensual.",
    "Objetivo: apoyar la ejecución del plan regional de ordenamiento territorial.",
    "Se realizaron tres reuniones de coordinación con los equipos municipales.",
    "La información se publica conforme a la Ley N° 20.285 sobre acceso a la información pública.",
    "Actividad: levantamiento de información en terreno en cuatro comunas de la región.",
    "Observaciones: sin observaciones por parte de la contraparte técnica.",
    "Resultados: se cumplió el 100% de las metas comprometidas para el mes.",
    "Fuente de financiamiento: Fondo Nacional de Desarrollo Regional.",
]

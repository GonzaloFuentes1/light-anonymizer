"""Fictitious personal data, deterministic from a seed and without repetitions.

Every value (RUT, phone, email) is unique across the whole dataset: it works as a canary, so
that if it shows up in an output file one knows exactly which element it came from.
Email domains and sites are made up; the names randomly combine given names and surnames
common in Chile.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field

FEMALE_NAMES = [
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
MALE_NAMES = [
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
SURNAMES = [
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
STREETS = [
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
COMMUNES = [
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
DOMAINS = [
    "ejemplo.cl",
    "goreficticio.cl",
    "muniejemplo.cl",
    "correo.ejemplo.cl",
    "ficticio.gob.cl",
    "consultora-ficticia.cl",
    "mail.ejemplo.com",
    "servicioficticio.cl",
]
# Regional area codes (7 subscriber digits). Santiago is 2 with 8 digits.
AREA_CODES = {
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

# Writing formats. Level "base": must always be redacted. "stress": rare or old variants.
RUT_FORMATS = {
    "dots": "base",  # 12.345.678-5
    "no_dots": "base",  # 12345678-5
    "spaced_hyphen": "base",  # 12.345.678 - 5
    "en_dash": "base",  # 12.345.678–5 (word processor autocorrection)
    "no_hyphen": "stress",  # 123456785
    "commas": "stress",  # 12,345,678-5
    "inner_spaces": "stress",  # 12 345 678-5
}
PHONE_FORMATS = {
    # mobile: 9 + 8 digits
    "mobile_international": "base",  # +56 9 8123 4567
    "mobile_compact": "base",  # +56981234567
    "mobile_national": "base",  # 9 8123 4567
    "mobile_block": "base",  # 98123 4567
    "mobile_parentheses": "base",  # (+56) 9 8123 4567
    "mobile_hyphens": "base",  # +56-9-8123-4567
    # Santiago: 2 + 8 digits
    "santiago_international": "base",  # +56 2 2123 4567
    "santiago_national": "base",  # 22 123 4567
    "santiago_parentheses": "base",  # (2) 2123 4567
    # regional: 2-digit area code + 7 digits
    "regional_international": "base",  # +56 41 221 3456
    "regional_parentheses": "base",  # (41) 221 3456
    "regional_compact": "base",  # (41) 2213456
    "regional_hyphen": "base",  # 41-2213456
    # old forms (before 2012-2016)
    "old_mobile_09": "stress",  # 09-8123 4567
    "old_regional_0": "stress",  # (041) 2213456
    "old_mobile_8": "stress",  # 8123 4567 (8 digits without the 9)
}
EMAIL_FORMATS = {
    "dot": "base",  # ana.rojas@ejemplo.cl
    "initial": "base",  # arojas@ejemplo.cl
    "with_year": "base",  # ana.rojas87@ejemplo.cl
    "underscore": "base",  # ana_rojas@ejemplo.cl
    "uppercase": "base",  # ANA.ROJAS@EJEMPLO.CL
    "plus": "base",  # ana.rojas+honorarios@ejemplo.cl
    "subdomain": "base",  # ana-rojas@correo.ejemplo.cl
    "spelled_arroba": "stress",  # ana.rojas [arroba] ejemplo.cl
    "at": "stress",  # ana.rojas(at)ejemplo.cl
    "spaces": "stress",  # ana.rojas @ ejemplo.cl
}


def strip_accents(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c))


def check_digit(body: int) -> str:
    """Modulo 11, the same as the notebook's ``rut_valido``."""
    total, factor = 0, 2
    for d in reversed(str(body)):
        total += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    rest = 11 - (total % 11)
    return {11: "0", 10: "K"}.get(rest, str(rest))


def format_rut(body: int, dv: str, format: str = "dots") -> str:
    with_dots = f"{body:,}".replace(",", ".")
    match format:
        case "dots":
            return f"{with_dots}-{dv}"
        case "no_dots":
            return f"{body}-{dv}"
        case "spaced_hyphen":
            return f"{with_dots} - {dv}"
        case "en_dash":
            return f"{with_dots}–{dv}"
        case "no_hyphen":
            return f"{body}{dv}"
        case "commas":
            return f"{body:,}-{dv}"
        case "inner_spaces":
            return f"{body:,}".replace(",", " ") + f"-{dv}"
    raise ValueError(format)


@dataclass
class Phone:
    kind: str  # mobile, santiago, regional
    national: str  # 9 digits, e.g. 981234567 or 412213456
    area: str  # "9", "2" or the regional area code

    def format(self, format: str) -> str:
        n = self.national
        if self.kind == "mobile":
            a, b = n[1:5], n[5:]
            return {
                "mobile_international": f"+56 9 {a} {b}",
                "mobile_compact": f"+569{a}{b}",
                "mobile_national": f"9 {a} {b}",
                "mobile_block": f"9{a} {b}",
                "mobile_parentheses": f"(+56) 9 {a} {b}",
                "mobile_hyphens": f"+56-9-{a}-{b}",
                "old_mobile_09": f"09-{a} {b}",
                "old_mobile_8": f"{a} {b}",
            }[format]
        if self.kind == "santiago":
            a, b = n[1:5], n[5:]
            return {
                "santiago_international": f"+56 2 {a} {b}",
                "santiago_national": f"2{a[0]} {a[1:]} {b}",
                "santiago_parentheses": f"(2) {a} {b}",
            }[format]
        area, subscriber = n[:2], n[2:]
        return {
            "regional_international": f"+56 {area} {subscriber[:3]} {subscriber[3:]}",
            "regional_parentheses": f"({area}) {subscriber[:3]} {subscriber[3:]}",
            "regional_compact": f"({area}) {subscriber}",
            "regional_hyphen": f"{area}-{subscriber}",
            "old_regional_0": f"(0{area}) {subscriber}",
        }[format]


PHONE_FORMATS_BY_KIND = {
    "mobile": [f for f in PHONE_FORMATS if f.startswith(("mobile", "old_mobile"))],
    "santiago": [f for f in PHONE_FORMATS if f.startswith("santiago")],
    "regional": [f for f in PHONE_FORMATS if f.startswith(("regional", "old_regional"))],
}


@dataclass
class Person:
    given_names: str
    paternal_surname: str
    maternal_surname: str
    sex: str  # F / M
    rut_body: int
    rut_dv: str
    rut_dv_valid: bool
    phone: Phone
    landline: Phone
    domain: str
    address: str
    commune: str
    birth_date: str  # dd-mm-yyyy
    in_list: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.given_names} {self.paternal_surname} {self.maternal_surname}"

    def rut(self, format: str = "dots") -> str:
        return format_rut(self.rut_body, self.rut_dv, format)

    def email(self, format: str = "dot") -> str:
        n = strip_accents(self.given_names.split()[0]).lower().replace("ñ", "n")
        a = strip_accents(self.paternal_surname).lower().replace("ñ", "n")
        d = self.domain
        return {
            "dot": f"{n}.{a}@{d}",
            "initial": f"{n[0]}{a}@{d}",
            "with_year": f"{n}.{a}{self.birth_date[-2:]}@{d}",
            "underscore": f"{n}_{a}@{d}",
            "uppercase": f"{n}.{a}@{d}".upper(),
            "plus": f"{n}.{a}+honorarios@{d}",
            "subdomain": f"{n}-{a}@correo.{d}" if not d.startswith("correo.") else f"{n}-{a}@{d}",
            "spelled_arroba": f"{n}.{a} [arroba] {d}",
            "at": f"{n}.{a}(at){d}",
            "spaces": f"{n}.{a} @ {d}",
        }[format]


class FakeData:
    """Factory of unique fictitious people and values.

    ``derive(name)`` returns a factory with its own randomness (seeded by name) that shares the
    registry of used values: this way each generator always produces the same data even when
    run alone, and no value repeats anywhere in the dataset.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self._ruts: set[int] = set()
        self._phones: set[str] = set()
        self._emails: set[tuple[str, str]] = set()
        self._names: set[str] = set()
        self.people: list[Person] = []

    def derive(self, name: str) -> FakeData:
        child = FakeData.__new__(FakeData)
        child.seed = self.seed
        child.rng = random.Random(f"{self.seed}:{name}")
        child._ruts, child._phones = self._ruts, self._phones
        child._emails, child._names = self._emails, self._names
        child.people = self.people
        return child

    def reserve(
        self,
        ruts: tuple[int, ...] = (),
        phones: tuple[str, ...] = (),
        emails: tuple[tuple[str, str], ...] = (),
        names: tuple[str, ...] = (),
    ) -> None:
        """Marks hand-written values as used (for example, the notebook's)."""
        self._ruts.update(ruts)
        self._phones.update(phones)
        self._emails.update(emails)
        self._names.update(names)

    # -- standalone values ---------------------------------------------------

    def rut(self, dv_valid: bool = True, seven_digits: bool = False) -> tuple[int, str]:
        while True:
            body = self.rng.randint(1_000_000, 9_999_999) if seven_digits else self.rng.randint(10_000_000, 26_999_999)
            if body not in self._ruts:
                break
        self._ruts.add(body)
        dv = check_digit(body)
        if not dv_valid:
            dv = self.rng.choice([c for c in "0123456789K" if c != dv])
        return body, dv

    def rut_with_k(self) -> tuple[int, str]:
        """A valid RUT whose check digit is K."""
        while True:
            body = self.rng.randint(10_000_000, 26_999_999)
            if body not in self._ruts and check_digit(body) == "K":
                self._ruts.add(body)
                return body, "K"

    def phone(self, kind: str = "mobile") -> Phone:
        while True:
            if kind == "mobile":
                national, area = "9" + "".join(self.rng.choice("0123456789") for _ in range(8)), "9"
            elif kind == "santiago":
                national, area = "22" + "".join(self.rng.choice("0123456789") for _ in range(7)), "2"
            else:
                area = self.rng.choice(sorted(AREA_CODES))
                national = area + self.rng.choice("2345") + "".join(self.rng.choice("0123456789") for _ in range(6))
            # the last 7 digits must also be unique (they are searched for as a partial leak)
            if national not in self._phones and not any(t[-7:] == national[-7:] for t in self._phones):
                self._phones.add(national)
                return Phone(kind, national, area)

    # -- people --------------------------------------------------------------

    def person(self, dv_valid: bool = True, in_list: bool = True, seven_digits: bool = False) -> Person:
        while True:
            sex = self.rng.choice("FM")
            given_names = self.rng.choice(FEMALE_NAMES if sex == "F" else MALE_NAMES)
            ps, ms = self.rng.sample(SURNAMES, 2)
            full = f"{given_names} {ps} {ms}"
            email_key = (strip_accents(given_names.split()[0]).lower(), strip_accents(ps).lower())
            if full not in self._names and email_key not in self._emails:
                break
        self._names.add(full)
        self._emails.add(email_key)
        body, dv = self.rut(dv_valid=dv_valid, seven_digits=seven_digits)
        landline_kind = self.rng.choice(["santiago", "regional", "regional"])
        p = Person(
            given_names=given_names,
            paternal_surname=ps,
            maternal_surname=ms,
            sex=sex,
            rut_body=body,
            rut_dv=dv,
            rut_dv_valid=dv_valid,
            phone=self.phone("mobile"),
            landline=self.phone(landline_kind),
            domain=self.rng.choice(DOMAINS),
            address=f"{self.rng.choice(STREETS)} {self.rng.randint(100, 3999)}"
            + (f", depto {self.rng.randint(11, 1204)}" if self.rng.random() < 0.4 else ""),
            commune=self.rng.choice(COMMUNES),
            birth_date=f"{self.rng.randint(1, 28):02d}-{self.rng.randint(1, 12):02d}-{self.rng.randint(1950, 2002)}",
            in_list=in_list,
        )
        self.people.append(p)
        return p

    def name_list(self) -> list[str]:
        """List handed to the engine: names (and addresses) of the people marked ``in_list``."""
        out = []
        for p in self.people:
            if p.in_list:
                out.append(p.full_name)
                out.append(p.address)
        return out

    # -- neutral text (decoys: they are not personal data) -------------------

    def amount(self) -> str:
        return f"$ {self.rng.randint(150, 4800) * 1000:,}".replace(",", ".")

    def date(self) -> str:
        months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                  "septiembre", "octubre", "noviembre", "diciembre"]  # fmt: skip
        return f"{self.rng.randint(1, 28)} de {self.rng.choice(months)} de 2026"

    def folio(self) -> str:
        return f"{self.rng.randint(100, 9999)}/{self.rng.choice(['2025', '2026'])}"


ADMINISTRATIVE_PHRASES = [
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

import re

import pytest

from banco_pruebas.ficticios import (
    FORMATOS_POR_CLASE,
    FORMATOS_RUT,
    Ficticios,
    digito_verificador,
    formatear_rut,
)


def test_digito_verificador_coincide_con_el_cuaderno():
    # En el cuaderno, 9.876.543-3 es válido y 15.782.334-9 no lo es.
    assert digito_verificador(9876543) == "3"
    assert digito_verificador(15782334) != "9"


@pytest.mark.parametrize("formato", sorted(FORMATOS_RUT))
def test_formatos_rut_conservan_los_digitos(formato):
    texto = formatear_rut(12345678, "5", formato)
    assert re.sub(r"\D", "", texto) == "123456785"


def test_valores_unicos_y_deterministas():
    a, b = Ficticios(33), Ficticios(33)
    pa = [a.persona() for _ in range(200)]
    pb = [b.persona() for _ in range(200)]
    assert [p.rut() for p in pa] == [p.rut() for p in pb]
    assert len({p.rut_cuerpo for p in pa}) == 200
    assert len({p.telefono.nacional for p in pa}) == 200
    assert len({p.correo() for p in pa}) == 200


def test_rut_dv_invalido_realmente_invalido():
    f = Ficticios(1)
    for _ in range(50):
        cuerpo, dv = f.rut(dv_valido=False)
        assert digito_verificador(cuerpo) != dv


@pytest.mark.parametrize("clase", sorted(FORMATOS_POR_CLASE))
def test_formatos_telefono(clase):
    t = Ficticios(2).telefono(clase)
    assert len(t.nacional) == 9
    for formato in FORMATOS_POR_CLASE[clase]:
        digitos = re.sub(r"\D", "", t.formatear(formato))
        assert t.nacional[-7:] in digitos

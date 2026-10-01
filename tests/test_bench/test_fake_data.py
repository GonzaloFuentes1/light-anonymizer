import re

import pytest

from test_bench.fake_data import (
    PHONE_FORMATS_BY_KIND,
    RUT_FORMATS,
    FakeData,
    check_digit,
    format_rut,
)


def test_check_digit_matches_the_notebook():
    # In the notebook, 9.876.543-3 is valid and 15.782.334-9 is not.
    assert check_digit(9876543) == "3"
    assert check_digit(15782334) != "9"


@pytest.mark.parametrize("format", sorted(RUT_FORMATS))
def test_rut_formats_keep_the_digits(format):
    text = format_rut(12345678, "5", format)
    assert re.sub(r"\D", "", text) == "123456785"


def test_values_unique_and_deterministic():
    a, b = FakeData(33), FakeData(33)
    pa = [a.person() for _ in range(200)]
    pb = [b.person() for _ in range(200)]
    assert [p.rut() for p in pa] == [p.rut() for p in pb]
    assert len({p.rut_body for p in pa}) == 200
    assert len({p.phone.national for p in pa}) == 200
    assert len({p.email() for p in pa}) == 200


def test_invalid_rut_dv_is_really_invalid():
    f = FakeData(1)
    for _ in range(50):
        body, dv = f.rut(dv_valid=False)
        assert check_digit(body) != dv


@pytest.mark.parametrize("kind", sorted(PHONE_FORMATS_BY_KIND))
def test_phone_formats(kind):
    t = FakeData(2).phone(kind)
    assert len(t.national) == 9
    for format in PHONE_FORMATS_BY_KIND[kind]:
        digits = re.sub(r"\D", "", t.format(format))
        assert t.national[-7:] in digits

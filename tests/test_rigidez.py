"""Pruebas de la cota tipo Kohn del giro de holonomía (holonomy.rigidez)."""
import numpy as np
import pytest

from holonomy import xxz
from holonomy.rigidez import ring_kohn_bound, twist_response


def test_kohn_igualdad_sin_giro():
    # Sin giro, el testigo girado es el óptimo: la cota es v.
    v, cota = ring_kohn_bound(6, 0.7, 0.7)
    assert cota == pytest.approx(v, abs=1e-9)


@pytest.mark.parametrize("n", [5, 6, 7])
@pytest.mark.parametrize("T0, T1", [(0.0, np.pi), (np.pi, 0.0), (np.pi / 3, 2 * np.pi / 3)])
def test_kohn_cota_anillo(n, T0, T1):
    v, cota = ring_kohn_bound(n, T0, T1)
    assert v <= cota + 1e-9


def test_kohn_regresion_n6():
    # data/rigidez_anillos.csv: v(6, pi) y la cota del testigo óptimo de Theta = 0.
    v, cota = ring_kohn_bound(6, 0.0, np.pi)
    assert v == pytest.approx(0.56168526375, abs=1e-7)
    assert cota == pytest.approx(0.5692220, abs=1e-7)


def test_giro_2pi_es_gauge():
    # Un flujo total de 2 pi es un gauge: la respuesta al giro es nula.
    assert abs(twist_response(xxz.eps_full, 6, 0.8, 0.3, 2 * np.pi)) < 1e-10
    # Giro nulo = identidad (evalúa eps dos veces).
    assert abs(twist_response(xxz.eps_full, 6, 0.8, 0.3, 0.0)) < 1e-14


def test_fundamental_ferro():
    # n = 4, Delta = -0.95: eps_full incluye el estado polarizado (E = n Delta/4).
    assert xxz.eps_full(4, -0.95, 0.0) >= 0.95 / 4 - 1e-12
    # Con Phi = 0 el fundamental sigue en el sector central (0,2542 > 0,2375);
    # con Phi = pi es el ferromagnético, fuera del sector de magnetización
    # mínima, que solo daría 0,1580.
    assert xxz.eps_full(4, -0.95, np.pi) == pytest.approx(0.95 / 4, abs=1e-12)
    central = -xxz.ground_energy(4, -0.95, np.pi, sectors=[2]) / 4
    assert central < 0.95 / 4 - 0.05

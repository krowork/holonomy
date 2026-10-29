"""Pruebas rápidas del paquete (pytest, menos de un minuto)."""
import numpy as np
import pytest

from holonomy import bethe, dual, sdp, xxz
from holonomy.paulis import Rz


def f4(theta):
    return 2 / (3 + np.sin(theta / 2))


@pytest.mark.parametrize("theta", [0.0, np.pi / 3, np.pi])
def test_anillo4_primal(theta):
    assert sdp.primal(4, [Rz(theta / 4)] * 4) == pytest.approx(f4(theta), abs=1e-7)


@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0, np.pi])
def test_anillo4_dual(theta):
    assert dual.v_cusp(xxz.eps, 4, theta)[0] == pytest.approx(f4(theta), abs=1e-12)


@pytest.mark.parametrize("Theta", [0.0, 1.0, np.pi])
def test_triangulo(Theta):
    assert dual.v_free(xxz.eps, 3, Theta)[0] == pytest.approx(1 / 3, abs=1e-12)


def test_n5_forma_cerrada():
    assert dual.v_smooth(xxz.eps, 5, np.pi)[0] == pytest.approx((5 + 4 * np.sqrt(5)) / 25, abs=1e-12)


def test_primal_u1_igual_al_dual():
    v_primal = sdp.primal_u1(5, np.pi / 2)
    assert v_primal == pytest.approx(dual.v_free(xxz.eps, 5, np.pi / 2)[0], abs=1e-6)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 10, 11])
@pytest.mark.parametrize("Delta", [0.4, 1.0, 2.5])
@pytest.mark.parametrize("Phi", [0.0, 1.3, np.pi])
def test_bethe_igual_a_diagonalizacion(n, Delta, Phi):
    assert bethe.eps(n, Delta, Phi) == pytest.approx(xxz.eps(n, Delta, Phi), abs=1e-13)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_ferromagnetico_incluido(n):
    # cerca de Delta = -1 el estado fundamental puede ser el ferromagnético
    for Phi in (0.0, np.pi):
        assert xxz.eps(n, -0.9, Phi) == pytest.approx(xxz.eps_full(n, -0.9, Phi), abs=1e-13)


@pytest.mark.parametrize("n,psi,D", [(6, 0.018587, 0.820), (7, 0.017261, 1.354)])
def test_borde_ventana(n, psi, D):
    p, De = dual.window_edge(bethe.eps, n, slope=bethe.cusp_slope)
    assert round(p / np.pi, 6) == psi
    assert round(De, 3) == D


def test_pendiente_cuspide_implicita_igual_a_diferencias():
    for n in (6, 7):
        assert bethe.cusp_slope(n, 1.0) == pytest.approx(dual.cusp_slope_fd(xxz.eps, n, 1.0), abs=1e-9)


@pytest.mark.parametrize("n", [101, 1025])
def test_bethe_preciso_cerca_de_isotropo(n):
    # Regresión: con cosh 2l - cos g escrito como 2 - sech^2 l (1 + cos g) se
    # perdían seis cifras por rapidez cerca de Delta = 1, y el dual quedaba por
    # debajo del valor exacto de Heisenberg.
    e1 = bethe.eps(n, 1.0, 0.0)
    for h in (1e-6, 1e-8):
        up, down = bethe.eps(n, 1 + h, 0.0) - e1, bethe.eps(n, 1 - h, 0.0) - e1
        assert up == pytest.approx(-down, rel=1e-4)
    v_heis = 4 * e1 / 3
    assert dual.v_cusp(bethe.eps, n, 0.0, xatol=1e-12)[0] >= v_heis - 1e-15

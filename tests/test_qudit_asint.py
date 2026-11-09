"""Tests de holonomy/qudit_asint.py (asintótica de qudits; tarea 8)."""
import warnings

import numpy as np
import pytest

from holonomy import qudit
from holonomy import qudit_asint as qa

warnings.filterwarnings("ignore", message="Solution may be inaccurate")
PI = np.pi


def test_d2_recupera_anillos():
    """d = 2: κ_2 = 1 − 3/(8(ln2 − ¼)) y ±π²/6 con las dos paridades."""
    c = qa.constantes(2)
    assert c["kappa"] == pytest.approx(1 - 3 / (8 * (np.log(2) - 0.25)), abs=1e-14)
    assert c["v_inf"] == pytest.approx((4 * np.log(2) - 1) / 3, abs=1e-14)
    tpi = np.array([PI / 2, -PI / 2])
    assert qa.coef_cero(2, 0, tpi) == pytest.approx(PI ** 2 / 6, abs=1e-7)
    assert qa.coef_cero(2, 1, tpi) == pytest.approx(-PI ** 2 / 6, abs=1e-7)


def test_d3_formas_cerradas():
    """d = 3: 2π²/9 y π²/6 (n ≡ 0); −π²/9 y −π²/18 con campo (n ≢ 0)."""
    C = 2 * PI / 3 * np.array([1.0, 0.0, -1.0])
    M = np.array([PI, 0.0, -PI])
    assert qa.coef_cero(3, 0, C) == pytest.approx(2 * PI ** 2 / 9, abs=1e-7)
    assert qa.coef_cero(3, 0, M) == pytest.approx(PI ** 2 / 6, abs=1e-7)
    assert qa.coef_cero(3, 1, C) == pytest.approx(-PI ** 2 / 9, abs=1e-7)
    assert qa.coef_cero(3, 2, M) == pytest.approx(-PI ** 2 / 18, abs=1e-7)
    assert qa.coef_cero(3, 1, M, campo=False) == pytest.approx(-PI ** 2 / 6, abs=1e-7)


def test_torcido_cero_y_gauge():
    """Giro nulo = Σ P; giro de 2π en un color y desplazamiento común: mismo E_0."""
    A = qa.uls_torcido(3, 5, (2, 2, 1), np.zeros(3)).toarray()
    assert np.array_equal(A, qudit.uls_sector_matrix(3, 5, (2, 2, 1)).toarray())
    s = np.array([0.3, -1.1, 0.8])
    e = qa.energia_min(3, 5, s)
    assert qa.energia_min(3, 5, s + np.array([0, 2 * PI, 0])) == pytest.approx(e, abs=1e-10)
    assert qa.energia_min(3, 5, s + 0.7) == pytest.approx(e, abs=1e-10)


def test_familia_frente_a_ring_witness():
    """λ_max(W) del testigo −P^s con campo h (D_ab −= h_a + h_b) y su N."""
    rng = np.random.default_rng(3)
    n = 4
    s, t, h = (qa.proj(rng.normal(size=3)) for _ in range(3))
    x = qudit.su_d_witness(3, s, n)
    for a in range(3):
        for b in range(a, 3):
            x[qudit.param_index(3, "D", a, b)] -= h[a] + h[b]
    E = qa.energias_torcidas(3, n, s)
    lam = n / 3 - min(E[c] + 2 * h @ qa.carga(c) for c in E)
    assert qudit.lam_max_sectors(3, n, x)[0] == pytest.approx(lam, abs=1e-10)
    assert qudit.ring_norm(3, n, x, t) == pytest.approx(qa.norma_iso(3, n, t, s), abs=1e-12)


def test_campo_lp_mezcla_equilibrada():
    """Tres sectores de cargas q_a con energías 0, 1, 2: G = 1 (mezcla uniforme)."""
    E = {(2, 1, 1): 0.0, (1, 2, 1): 1.0, (1, 1, 2): 2.0}
    G, h, act = qa.campo_lp(E)
    assert G == pytest.approx(1.0, abs=1e-9)
    assert sorted(act) == sorted(E)
    assert abs(h.sum()) < 1e-12


def test_campo_m_n7():
    """n = 7 en M: el campo equilibra los tres sectores (3,2,2), (2,3,2), (2,2,3)
    y baja la cota por debajo de la isótropa; v ≥ v de la tarea 7 (0,7548928)."""
    M = np.array([PI, 0.0, -PI])
    v, G, h, act = qa.v_familia(3, 7, M, M)
    vi = qa.v_familia(3, 7, M, M, campo=False)[0]
    assert sorted(act) == [(2, 2, 3), (2, 3, 2), (3, 2, 2)]
    assert vi - v > 0.02
    assert v >= 0.7548928 - 1e-7

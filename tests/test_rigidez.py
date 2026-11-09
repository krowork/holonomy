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


# --- Escaleras: testigo óptimo girado (tarea 3) ---------------------------------

CLARABEL = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
Y_GEN = np.array([0.3, -0.2, 0.5, 0.1, 0.4, -0.3, 0.7, 0.2])


def test_ladder_giro_nulo():
    from holonomy.rigidez import ladder_twist_y
    assert np.array_equal(ladder_twist_y(Y_GEN, 4, 0.0), Y_GEN)


def test_ladder_giro_2pi():
    # Un giro total de 2 pi en cada pata es un gauge: el mismo lambda_max.
    from holonomy import ladder
    from holonomy.rigidez import ladder_twist_y
    D = ladder.LadderDual(4)
    lam = D.lam_max(Y_GEN)
    assert D.lam_max(ladder_twist_y(Y_GEN, 4, 2 * np.pi)) == pytest.approx(lam, abs=1e-10)
    assert D.lam_max(ladder_twist_y(Y_GEN, 4, -2 * np.pi)) == pytest.approx(lam, abs=1e-10)
    # Y un giro intermedio sí cambia el espectro (el test no es trivial).
    assert abs(D.lam_max(ladder_twist_y(Y_GEN, 4, np.pi)) - lam) > 1e-3


@pytest.mark.parametrize("f", [0.0, 0.6, np.pi])
@pytest.mark.parametrize("psi, Psi", [(0.0, np.pi / 4), (0.3, np.pi), (1.0, -2.2)])
def test_ladder_giro_normalizacion(f, psi, Psi):
    # Girar las patas Psi/L lo mismo que los datos (abelian_phases suma psi/L
    # a las dos patas) conserva N.
    from holonomy import ladder
    from holonomy.rigidez import ladder_twist_y
    for L, D in ((3, ladder.LadderDual(3)), (5, ladder.LadderDual(5, dense_max=4, half=True))):
        ph0 = ladder.abelian_phases(L, f, periodic=True, psi=psi)
        ph1 = ladder.abelian_phases(L, f, periodic=True, psi=psi + Psi)
        assert D.norm(ladder_twist_y(Y_GEN, L, Psi), ph1) == pytest.approx(
            D.norm(Y_GEN, ph0), abs=1e-12)


def test_ladder_kohn_L4():
    # Testigo óptimo en f = pi, psi = 0, girado a cada psi: su lambda_max es
    # cota superior de v(psi) (data/ladder_limit_L4.csv; esos v son cotas
    # superiores a < 1e-8 de la inferior), con igualdad sin giro.
    import csv
    from holonomy import ladder
    from holonomy.report import DATA
    from holonomy.rigidez import ladder_twist_y
    L = 4
    D = ladder.LadderDual(L)
    v0, y, st, _ = D.v_dual_cg(ladder.abelian_phases(L, np.pi, periodic=True), **CLARABEL)
    assert st == "optimal"
    rows = [{k: float(r[k]) for k in ("psi", "v_max")}
            for r in csv.DictReader(open(DATA / "ladder_limit_L4.csv"))]
    assert len(rows) == 5
    assert v0 == pytest.approx(rows[0]["v_max"], abs=1e-8)
    for r in rows:
        cota = D.lam_max(ladder_twist_y(y, L, r["psi"]))
        assert cota >= r["v_max"] - 1e-8
    assert D.lam_max(ladder_twist_y(y, L, 0.0)) == pytest.approx(v0, abs=1e-12)


def test_witness_gap_bloques():
    # El gap es lambda_max menos el primer autovalor distinto de todos los
    # bloques: igual con bloques densos completos (todos los k) que con
    # dispersos (Lanczos) y solo k <= L; y coincide con el espectro completo.
    from holonomy import ladder
    from holonomy.rigidez import witness_gap
    Dc = ladder.LadderDual(4)
    Dh = ladder.LadderDual(4, dense_max=4, half=True)
    w = np.sort(np.concatenate([np.linalg.eigvalsh(
        sum(c * red[key] for key, c in zip(Dc.KEYS, Y_GEN))) for red in Dc.blocks]))
    lam = w[-1]
    ref = lam - w[w < lam - 1e-8][-1]
    assert witness_gap(Dc, Y_GEN) == pytest.approx(ref, abs=1e-10)
    assert witness_gap(Dh, Y_GEN) == pytest.approx(ref, abs=1e-10)
    g, mult = witness_gap(Dc, Y_GEN, detalle=True)
    assert sum(m for _, m in mult) == int(np.sum(w > lam - 1e-8))


def test_witness_gap_degenerado():
    # Óptimo de f = pi, psi = 0, L = 4: lambda_max degenerado entre los bloques
    # (4, 0) y (4, 2) (columna bloques de ladder_limit_L4.csv); el multiplete
    # superior los contiene y el gap es positivo.
    from holonomy import ladder
    from holonomy.rigidez import witness_gap
    D = ladder.LadderDual(4)
    _, y, st, _ = D.v_dual_cg(ladder.abelian_phases(4, np.pi, periodic=True), **CLARABEL)
    assert st == "optimal"
    g, mult = witness_gap(D, y, tol=1e-6, detalle=True)
    labs = {lab for lab, _ in mult}
    assert {(4, 0), (4, 2)} <= labs and g > 1e-3


def test_lam_max_aleatorio():
    # Lanczos con arranque aleatorio y más vectores = lam_max (y eigh denso).
    from holonomy import ladder
    from holonomy.rigidez import lam_max_aleatorio
    Dc = ladder.LadderDual(5)
    Dh = ladder.LadderDual(5, dense_max=4, half=True)
    lam = Dc.lam_max(Y_GEN)
    assert lam_max_aleatorio(Dh, Y_GEN, seed=1) == pytest.approx(lam, abs=1e-10)
    assert lam_max_aleatorio(Dh, Y_GEN, seed=2) == pytest.approx(lam, abs=1e-10)

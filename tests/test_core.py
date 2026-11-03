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


def test_escalera_flujos_y_primal_u1():
    from holonomy import ladder
    ph = ladder.abelian_phases(4, 0.7, periodic=True, psi=0.3)
    f, legs = ladder.plaquette_fluxes(4, ph, periodic=True)
    assert np.allclose(f, 0.7) and np.allclose(legs, (2 * 0.7 + 0.3, -2 * 0.7 + 0.3))
    # Escalera abierta 2x2 = anillo de 4 (Teorema 1).
    v = sdp.primal_u1_graph(4, ladder.ladder_edges(2), ladder.abelian_phases(2, 1.0))
    assert v == pytest.approx(f4(1.0), abs=1e-7)


def test_escalera_dual_por_generacion_de_columnas():
    # v_dual_cg (el SDP por generación de columnas, para L >= 6) = SDP completo.
    from holonomy import ladder
    D = ladder.LadderDual(4)
    ph = ladder.abelian_phases(4, 0.5 * np.pi, periodic=True, psi=0.3 * np.pi)
    opts = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    vs, _, _ = D.v_dual_sdp(ph, **opts)
    vc, _, st, gap = D.v_dual_cg(ph, **opts)
    assert st == "optimal" and gap < 1e-8
    assert vc == pytest.approx(vs, abs=1e-8)
    # Con bloques dispersos y Lanczos (dense_max pequeño), el mismo valor.
    Ds = ladder.LadderDual(4, dense_max=4)
    vd, _, st, gap = Ds.v_dual_cg(ph, **opts)
    assert st == "optimal" and gap < 1e-8
    assert vd == pytest.approx(vs, abs=1e-8)
    # Solo los bloques k <= L (los k > L tienen el mismo espectro): el mismo valor.
    vh, _, st, gap = ladder.LadderDual(4, dense_max=4, half=True).v_dual_cg(ph, **opts)
    assert st == "optimal" and gap < 1e-8
    assert vh == pytest.approx(vs, abs=1e-8)


def test_escalera_top_eig_lanczos():
    # top_eig con una matriz dispersa grande (pasa por eigsh) = eigh densa.
    import scipy.sparse as sp
    from holonomy import ladder
    A = sp.random(300, 300, density=0.02, random_state=1) + 1j * sp.random(300, 300, density=0.02, random_state=2)
    W = (A + A.conj().T).tocsr()
    w, V = ladder.LadderDual.top_eig(W, 5)
    we, Ve = np.linalg.eigh(W.toarray())
    assert w == pytest.approx(we[-5:], abs=1e-9)
    assert np.allclose(W @ V, V * w, atol=1e-8)


def test_escalera_antiunitaria_theta():
    # Proposición de bloques: Theta = P R_x(pi)^n T conmuta con W(y), conserva S^z y la
    # traslación (antiunitaria con P T_r P^-1 = T_r^-1) y cumple Theta^2 = 1; T empareja
    # (k, q) con (2L-k, -q). En la 2x4 completa (2^8 estados).
    from functools import reduce
    from holonomy import ladder
    L, n = 4, 8
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)

    def op(d):
        return reduce(np.kron, [d.get(i, np.eye(2)) for i in range(n)])

    def perm_matrix(perm):
        P = np.zeros((2 ** n, 2 ** n))
        for b in range(2 ** n):
            bits = [(b >> (n - 1 - s)) & 1 for s in range(n)]
            nb = [0] * n
            for s in range(n):
                nb[perm[s]] = bits[s]
            P[int("".join(map(str, nb)), 2), b] = 1
        return P

    edges, kinds = ladder.ladder_edges(L, True), ladder.edge_kinds(L, True)
    rng = np.random.default_rng(3)
    ab, bb, cb, at, bt, ct, ar, cr = rng.normal(size=8)
    co = {"b": (ab, bb, cb), "t": (at, bt, ct), "r": (ar, 0.0, cr)}
    W = 0
    for (i, j), kd in zip(edges, kinds):
        a, b, c = co[kd]
        W = W + a * (op({i: X, j: X}) + op({i: Y, j: Y})) + b * (op({i: X, j: Y}) - op({i: Y, j: X})) \
            + c * op({i: Z, j: Z})
    P = perm_matrix([(s // L) * L + (-(s % L)) % L for s in range(n)])
    Tr = perm_matrix([(s // L) * L + ((s % L) + 1) % L for s in range(n)])
    Ty = reduce(np.kron, [1j * Y] * n)
    U = P @ reduce(np.kron, [1j * X] * n) @ Ty   # Theta = U K
    Sz = sum(op({i: Z}) for i in range(n))
    assert np.abs(Ty @ W.conj() @ Ty.conj().T - W).max() < 1e-12      # T conmuta con W(y)
    assert np.abs(U @ W.conj() @ U.conj().T - W).max() < 1e-12        # Theta conmuta con W(y)
    assert np.abs(U @ U.conj() - np.eye(2 ** n)).max() < 1e-12        # Theta^2 = 1
    assert np.abs(U @ Sz.conj() @ U.conj().T - Sz).max() < 1e-12      # conserva S^z
    assert np.abs(U @ Tr.conj() @ U.conj().T - Tr.T).max() < 1e-12    # P T_r P^-1 = T_r^-1
    # Espectros de los bloques (k, q) y (2L-k, -q) iguales con el W(y) reducido.
    D = ladder.LadderDual(L)
    lab = {l: m for m, l in enumerate(D.labels)}
    y = (ab, bb, cb, at, bt, ct, ar, cr)
    for (k, q), m in lab.items():
        m2 = lab[(n - k, (-q) % L)]
        W1 = sum(c * D.blocks[m][key] for key, c in zip(D.KEYS, y))
        W2 = sum(c * D.blocks[m2][key] for key, c in zip(D.KEYS, y))
        assert np.linalg.eigvalsh(W1) == pytest.approx(np.linalg.eigvalsh(W2), abs=1e-12)


def test_escalera_regresion_periodic():
    # Valores calculados con ladder.py antes de extraer la maquinaria a
    # holonomy/periodic.py (tarea 5): la extracción no cambia nada.
    from holonomy import ladder
    opts = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    ph = ladder.abelian_phases(4, 0.5 * np.pi, periodic=True, psi=0.3 * np.pi)
    v, _, _ = ladder.LadderDual(4).v_dual_sdp(ph, **opts)
    assert v == pytest.approx(0.4348406811438309, abs=1e-10)
    y = (0.3, -0.2, 0.5, 0.1, 0.4, -0.3, 0.7, 0.2)
    assert ladder.LadderDual(5).lam_max(y) == pytest.approx(8.689812610957985, abs=1e-10)
    assert ladder.LadderDual(5, dense_max=4, half=True).lam_max(y) == pytest.approx(
        8.689812610957985, abs=1e-10)

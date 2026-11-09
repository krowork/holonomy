"""Pruebas del pegado de estados de Werner de qudits en anillos (holonomy.qudit).

d = 2 es el problema de los qubits (holonomia_werner.tex): se comprueba el
marco, el anillo de 4 (2/(3+sin(θ/2))), la tabla tab:valores (n = 5, 6) y que
el testigo de qudits de d = 2 es el de la Proposición 4. En d = 3, el dual por
sectores de color y momento frente al SDP primal.
"""
import resource
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import unitary_group

from holonomy import qudit
from holonomy.paulis import PAULI, pauli_string, werner_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ANGLES, CLARABEL_OPTS, TABLE, TABLE_TOL  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:Initializing a Constant with a nested list"),
    pytest.mark.filterwarnings("ignore:Solution may be inaccurate"),
]


def _su(d, rng):
    """Unitario de Haar de U(d) llevado a SU(d)."""
    U = unitary_group.rvs(d, random_state=rng)
    return U / np.linalg.det(U) ** (1 / d)


def _adjunta(U):
    """R(U)_ab = tr(σ_a U σ_b U†)/2: U σ_b U† = Σ_a R_ab σ_a."""
    return np.array([[np.real(np.trace(PAULI[a] @ U @ PAULI[b] @ U.conj().T)) / 2
                      for b in range(3)] for a in range(3)])


# Con las tolerancias de 1e-10 Clarabel suele terminar en «casi resuelto» (lo
# mismo que holonomy.sdp.primal con los qubits); los valores se comprueban aparte.
OK = ("optimal", "optimal_inaccurate")


def _uniforme(d, n, thetas):
    return [np.diag(np.exp(1j * np.asarray(thetas) / n))] * n


# --- Marco -----------------------------------------------------------------

def test_werner_d2_es_qubit():
    for v in (0.0, 0.3, 2 / 3, 1.0):
        assert np.allclose(qudit.werner(2, v), werner_state(v, np.eye(3)), atol=1e-14)
    # (1⊗U) singlete (1⊗U)† con U ∈ SU(2) no trivial es el ρ_e(v) de T = −R(U)^T,
    # es decir, R_e = R(U)^T = R(U†) en el convenio de anillos (no R(U): el
    # factor U actúa sobre el segundo qubit, ⟨σ_a ⊗ U†σ_b U⟩).
    rng = np.random.default_rng(1)
    for _ in range(3):
        U = _su(2, rng)
        R = _adjunta(U)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) > 0
        assert not np.allclose(R, np.eye(3), atol=1e-2)
        assert np.allclose(qudit.edge_state(2, 0.7, U), werner_state(0.7, R.T), atol=1e-13)
        assert not np.allclose(qudit.edge_state(2, 0.7, U), werner_state(0.7, R), atol=1e-3)


def test_werner_qudit_propiedades():
    d, v = 3, 0.4
    rho = qudit.werner(d, v)
    assert np.isclose(np.trace(rho).real, 1)
    assert np.allclose(rho, rho.conj().T)
    # Marginales maximalmente mezcladas y tr(ρ P) = 1/d − v(1 + 1/d) con P el intercambio.
    r4 = rho.reshape(d, d, d, d)
    assert np.allclose(np.einsum("abcb->ac", r4), np.eye(d) / d)
    assert np.allclose(np.einsum("abad->bd", r4), np.eye(d) / d)
    P = np.eye(d * d)[[b * d + a for a in range(d) for b in range(d)]]
    assert np.isclose(np.trace(rho @ P).real, 1 / d - v * (1 + 1 / d))
    # v = 1 es el proyector antisimétrico normalizado (rango binom(d, 2)).
    assert np.linalg.matrix_rank(qudit.werner(d, 1.0), tol=1e-10) == 3


def test_holonomia_gauge():
    rng = np.random.default_rng(7)
    d, n = 3, 4
    Us = [_su(d, rng) for _ in range(n)]
    Vs = [_su(d, rng) for _ in range(n)]
    Ug = [Vs[(i + 1) % n] @ Us[i] @ Vs[i].conj().T for i in range(n)]
    assert np.allclose(qudit.ring_holonomy(Us), qudit.ring_holonomy(Ug), atol=1e-12)
    th = np.array([0.4, -1.1, 0.7])
    assert np.allclose(qudit.ring_holonomy(_uniforme(d, n, th)), np.sort(th), atol=1e-12)


def test_gauge_holonomia():
    # d = 3, n = 3: el primal no cambia con un gauge aleatorio, y coincide con el
    # gauge uniforme de la misma holonomía.
    rng = np.random.default_rng(11)
    d, n = 3, 3
    Us = [_su(d, rng) for _ in range(n)]
    Vs = [_su(d, rng) for _ in range(n)]
    Ug = [Vs[(i + 1) % n] @ Us[i] @ Vs[i].conj().T for i in range(n)]
    v0, s0 = qudit.primal_ring(d, n, Us, **CLARABEL_OPTS)
    v1, s1 = qudit.primal_ring(d, n, Ug, **CLARABEL_OPTS)
    th = qudit.ring_holonomy(Us)
    v2, _ = qudit.primal_ring(d, n, _uniforme(d, n, th), **CLARABEL_OPTS)
    assert s0 in OK and s1 in OK
    assert v1 == pytest.approx(v0, abs=1e-7)
    assert v2 == pytest.approx(v0, abs=1e-7)


# --- Regresión d = 2 --------------------------------------------------------

def test_testigo_d2_es_prop4():
    # Con x = testigo α A + β B + c C de la Prop. 4 (A = −Σ(XX+YY), B = Σ(XY−YX),
    # C = −Σ ZZ), ring_witness es esa matriz y N = n(2α cos φ + 2β sin φ + c),
    # φ = Θ/n, con thetas = (Θ/2, −Θ/2).
    n, Theta = 5, 1.3
    al, be, c = 0.37, -0.21, 0.55
    x = qudit.qubit_witness_params(al, be, c)
    A = sum(-(pauli_string(n, ((i, 0), ((i + 1) % n, 0))) + pauli_string(n, ((i, 1), ((i + 1) % n, 1))))
            for i in range(n))
    B = sum(pauli_string(n, ((i, 0), ((i + 1) % n, 1))) - pauli_string(n, ((i, 1), ((i + 1) % n, 0)))
            for i in range(n))
    C = sum(-pauli_string(n, ((i, 2), ((i + 1) % n, 2))) for i in range(n))
    W = qudit.ring_witness(2, n, x).toarray()
    assert np.allclose(W, al * A + be * B + c * C, atol=1e-13)
    phi = Theta / n
    N = qudit.ring_norm(2, n, x, (Theta / 2, -Theta / 2))
    assert N == pytest.approx(n * (2 * al * np.cos(phi) + 2 * be * np.sin(phi) + c), abs=1e-13)


def test_norma_lineal_en_v():
    # Todo ρ factible cumple tr(ρ W) = Σ_e tr(ρ_e(v) Y) − c0 = v N(x): N es
    # lineal en v, sin término independiente (W lleva restada la traza).
    rng = np.random.default_rng(3)
    d, n = 3, 4
    th = np.array([0.9, -0.2, -0.7])
    x = rng.normal(size=qudit.n_params(d))
    Y = qudit.edge_term(d, x)
    U = np.diag(np.exp(1j * th / n))
    c0 = n * np.trace(Y).real / d**2
    for v in (0.0, 0.35, 1.0):
        val = n * np.trace(qudit.edge_state(d, v, U) @ Y).real - c0
        assert val == pytest.approx(v * qudit.ring_norm(d, n, x, th), abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, np.pi / 2, np.pi])
def test_regresion_d2_anillo4(theta):
    exact = 2 / (3 + np.sin(theta / 2))
    th = (theta / 2, -theta / 2)
    vp, st = qudit.primal_ring(2, 4, _uniforme(2, 4, th), **CLARABEL_OPTS)
    vd, _ = qudit.v_dual_ring(2, 4, th)
    assert st in OK
    assert vp == pytest.approx(exact, abs=1e-7)
    assert vd == pytest.approx(exact, abs=1e-7)


@pytest.mark.parametrize("n", [5, 6])
def test_regresion_d2_tabla(n):
    # n = 5 con el primal completo; n = 6 con el reducido por el toro (U(1) en
    # qubits): el completo de 64 estados tarda ~26 s y 3,6 GB por punto.
    for Theta, ref in zip(ANGLES, TABLE[n]):
        th = (Theta / 2, -Theta / 2)
        if n == 5:
            vp, _ = qudit.primal_ring(2, n, _uniforme(2, n, th), **CLARABEL_OPTS)
        else:
            vp, _ = qudit.primal_ring_toro(2, n, th, **CLARABEL_OPTS)
        vd, _ = qudit.v_dual_ring(2, n, th)
        assert abs(vp - ref) <= TABLE_TOL, (n, Theta, vp, ref)
        assert abs(vd - ref) <= TABLE_TOL, (n, Theta, vd, ref)
        assert vd == pytest.approx(vp, abs=1e-6)


@pytest.mark.slow  # ~26 s y 3,6 GB de pico
def test_regresion_d2_n6_completo():
    th = (np.pi / 4, -np.pi / 4)
    vp, st = qudit.primal_ring(2, 6, _uniforme(2, 6, th), **CLARABEL_OPTS)
    vt, _ = qudit.primal_ring_toro(2, 6, th, **CLARABEL_OPTS)
    assert st in OK
    assert abs(vp - TABLE[6][2]) <= TABLE_TOL
    assert vt == pytest.approx(vp, abs=1e-7)


# --- d = 3 ------------------------------------------------------------------

HOL3 = [np.array([1.0, 0.3, -1.3]), np.array([2.5, -0.4, -2.1])]


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("th", HOL3, ids=["hol1", "hol2"])
def test_dual_primal_d3(n, th):
    # n = 3 con el primal completo (27 estados); n = 4 con el reducido por el
    # toro (el completo de 81 estados pide 9 GB: test_primal_toro_d3_n4, slow).
    if n == 3:
        vp, st = qudit.primal_ring(3, n, _uniforme(3, n, th), **CLARABEL_OPTS)
    else:
        vp, st = qudit.primal_ring_toro(3, n, th, **CLARABEL_OPTS)
    vd, x = qudit.v_dual_ring(3, n, th)
    assert st in OK
    # Dualidad débil, con el ruido del primal (Clarabel deja ρ con autovalores
    # de −1e-8 y el v primal puede pasar ~3e-8 de la cota dual rigurosa).
    assert vd >= vp - 1e-7
    assert vd == pytest.approx(vp, abs=1e-6)
    # El valor devuelto es λ_max/N del testigo devuelto, con λ_max densa.
    lam = np.linalg.eigvalsh(qudit.ring_witness(3, n, x).toarray())[-1]
    assert vd == pytest.approx(lam / qudit.ring_norm(3, n, x, th), abs=1e-12)


def test_cortes_horquilla():
    sup, inf, x, st = qudit.v_dual_ring_cg(3, 4, HOL3[0])
    assert st == "optimal"
    assert abs(sup - inf) < 1e-7
    assert qudit.ring_norm(3, 4, x, HOL3[0]) == pytest.approx(1.0, abs=1e-12)


def test_nelder_mead_d2():
    # Nelder–Mead (sin certificado) da una cota superior; en d = 2, n = 4 llega
    # al valor exacto.
    th = (np.pi / 4, -np.pi / 4)
    vd, x = qudit.v_dual_ring(2, 4, th, metodo="nm", starts=2)
    exact = 2 / (3 + np.sin(np.pi / 4))
    assert vd >= exact - 1e-12
    assert vd == pytest.approx(exact, abs=1e-8)


def test_primal_toro_d3_n3():
    # El primal reducido por el toro diagonal coincide con el completo (n = 3).
    for th in HOL3:
        vp, _ = qudit.primal_ring(3, 3, _uniforme(3, 3, th), **CLARABEL_OPTS)
        vt, st = qudit.primal_ring_toro(3, 3, th, **CLARABEL_OPTS)
        assert st in OK
        assert vt == pytest.approx(vp, abs=1e-7)


def _limite_kb():
    lim = resource.getrlimit(resource.RLIMIT_AS)[0]
    return None if lim == resource.RLIM_INFINITY else lim // 1024


@pytest.mark.slow  # primal completo de 81 estados: 81 s y 9,1 GB de pico
@pytest.mark.skipif(_limite_kb() is not None and _limite_kb() < 14000000,
                    reason="necesita ulimit -v >= 14000000 (Clarabel aborta el proceso si no cabe)")
def test_primal_toro_d3_n4():
    th = HOL3[0]
    vp, _ = qudit.primal_ring(3, 4, _uniforme(3, 4, th), **CLARABEL_OPTS)
    vt, st = qudit.primal_ring_toro(3, 4, th, **CLARABEL_OPTS)
    assert st in OK
    assert vt == pytest.approx(vp, abs=1e-7)


@pytest.mark.parametrize("th", HOL3, ids=["hol1", "hol2"])
def test_dual_primal_d3_n5(th):
    # n = 5 (243 estados): el primal completo no cabe; se contrasta con el reducido.
    vt, st = qudit.primal_ring_toro(3, 5, th, **CLARABEL_OPTS)
    vd, _ = qudit.v_dual_ring(3, 5, th)
    assert st in OK
    assert vd >= vt - 1e-7
    assert vd == pytest.approx(vt, abs=1e-6)


def test_sectores_de_color():
    rng = np.random.default_rng(5)
    d, n = 3, 4
    for _ in range(3):
        x = rng.normal(size=qudit.n_params(d))
        lam = qudit.lam_max_sectors(d, n, x)[0]
        dense = np.linalg.eigvalsh(qudit.ring_witness(d, n, x).toarray())[-1]
        assert lam == pytest.approx(dense, abs=1e-10)
    # Un testigo cuyo máximo está en un sector no balanceado: el máximo se toma
    # sobre todos los sectores, no solo el central.
    x = np.zeros(qudit.n_params(d))
    x[qudit.param_index(d, "D", 0, 0)] = 1.0
    lam, counts, k = qudit.lam_max_sectors(d, n, x)
    dense = np.linalg.eigvalsh(qudit.ring_witness(d, n, x).toarray())[-1]
    assert lam == pytest.approx(dense, abs=1e-12)
    assert counts == (4, 0, 0)
    # Caso real: en el testigo óptimo de d = 3, n = 4, θ = 0 (v = 5/8), λ_max se
    # alcanza también en sectores con un color vacío, como (0, 2, 2).
    v, x = qudit.v_dual_ring(d, n, np.zeros(d))
    assert v == pytest.approx(5 / 8, abs=1e-9)
    labels, _, ops = qudit.sector_blocks(d, n)
    c0 = n * qudit._trace_coeffs(d) @ x / d**2
    lam = qudit.lam_max_sectors(d, n, x)[0]
    top = {lab[0] for lab, O in zip(labels, ops)
           if np.linalg.eigvalsh(np.tensordot(x, O, axes=1))[-1] - c0 > lam - 1e-7}
    assert (0, 2, 2) in top


def test_sectores_cubren_el_espacio():
    for d, n in ((2, 5), (3, 4), (4, 3)):
        dims = [b.shape[1] for b in qudit.sector_blocks(d, n)[1]]
        assert sum(dims) == d**n


def test_testigo_su_d():
    # El testigo SU(d) alineado con los datos, −Σ P_{i,i+1} girado, da en θ = 0
    # la cota d/(d+1)(1/d − e_P(n)), e_P = E_0(Σ P)/n.
    d, n = 3, 4
    x = qudit.su_d_witness(d, np.zeros(d))
    W = qudit.ring_witness(d, n, x).toarray()
    SP = np.zeros((d**n, d**n))
    for s in range(d**n):
        c = list(np.unravel_index(s, (d,) * n))
        for i in range(n):
            j = (i + 1) % n
            t = c.copy()
            t[i], t[j] = t[j], t[i]
            SP[np.ravel_multi_index(t, (d,) * n), s] += 1
    eP = np.linalg.eigvalsh(SP)[0] / n
    bound = np.linalg.eigvalsh(W)[-1] / qudit.ring_norm(d, n, x, np.zeros(d))
    assert bound == pytest.approx(d / (d + 1) * (1 / d - eP), abs=1e-12)


# --- Ronda 1 de revisión ----------------------------------------------------

def _embebe(d, n, bloques):
    """ρ de d^n × d^n a partir de bloques (números de color, códigos, ρ_b)."""
    R = np.zeros((d**n, d**n), dtype=complex)
    for _, states, M in bloques:
        R[np.ix_(states, states)] = M
    return R


def _marginal(d, n, R, i, j):
    vec = qudit._marginal_map(d, n, i, j) @ R.reshape(-1, order="F")
    return vec.reshape((d * d, d * d), order="F")


def test_toro_restricciones_son_el_marginal():
    # Las expresiones de las restricciones de primal_ring_toro, evaluadas en un ρ
    # aleatorio diagonal por bloques, son las entradas del marginal calculado con
    # _marginal_map (en particular ⟨ab|marg|ba⟩, no su conjugado).
    rng = np.random.default_rng(13)
    d, n = 3, 3
    bloques = []
    for counts, states in qudit.color_sectors(d, n):
        m = len(states)
        A = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
        bloques.append((counts, states, A @ A.conj().T))
    R = _embebe(d, n, bloques)
    val = {}
    for (_, _, M), ops in zip(bloques, qudit._toro_ops(d, n)):
        for key, S in ops.items():
            val[key] = val.get(key, 0) + np.sum(S * M)
    assert val["tr"] == pytest.approx(np.trace(R), abs=1e-12)
    n_x = 0
    for i in range(n):
        j = (i + 1) % n
        mg = _marginal(d, n, R, i, j)
        site = (qudit._site_map(d, n, i) @ R.reshape(-1, order="F")).reshape((d, d), order="F")
        for a in range(d - 1):
            assert val[("s", i, a)] == pytest.approx(site[a, a], abs=1e-12)
            for b in range(d - 1):
                assert val[("p", i, a, b)] == pytest.approx(mg[a * d + b, a * d + b], abs=1e-12)
        for a in range(d):
            for b in range(a + 1, d):
                ent = mg[a * d + b, b * d + a]
                assert abs(ent.imag) > 1e-3  # el test distingue la entrada de su conjugada
                assert val[("x", i, a, b)] == pytest.approx(ent, abs=1e-12)
                n_x += 1
    assert n_x == n * 3


def test_toro_estado_tiene_los_marginales():
    # Datos sin la simetría θ ↦ −θ (hol1): el estado devuelto tiene en cada arista
    # el marginal ρ_e(v) de θ, no el de −θ; con el de −θ fallaría por ~1e-2.
    d, n = 3, 3
    th = HOL3[0]
    v, st, bloques, diag = qudit.primal_ring_toro(d, n, th, return_state=True, **CLARABEL_OPTS)
    assert st in OK
    assert diag["lam_min"] > -1e-6 and diag["residuo"] < 1e-6
    R = _embebe(d, n, bloques)
    U = np.diag(np.exp(1j * th / n))
    for i in range(n):
        mg = _marginal(d, n, R, i, (i + 1) % n)
        assert np.allclose(mg, qudit.edge_state(d, v, U), atol=1e-6)
        assert not np.allclose(mg, qudit.edge_state(d, v, U.conj()), atol=1e-3)
    # El primal completo con return_state: mismo v y mismos marginales.
    v2, st2, R2, diag2 = qudit.primal_ring(d, n, _uniforme(d, n, th), return_state=True,
                                           **CLARABEL_OPTS)
    assert v2 == pytest.approx(v, abs=1e-7)
    assert diag2["lam_min"] > -1e-6 and diag2["residuo"] < 1e-6
    for i in range(n):
        assert np.allclose(_marginal(d, n, R2, i, (i + 1) % n), qudit.edge_state(d, v2, U), atol=1e-6)


def test_holonomia_rama_pi():
    # Autofase a 1e-12 de π: np.angle la da cerca de +π o de −π; las autofases
    # no son canónicas, pero e^{iθ_a} sí, y las dos ramas dan el mismo v.
    rng = np.random.default_rng(17)
    d, n = 3, 4
    lam = np.array([np.pi - 1e-12, 0.4, -(np.pi - 1e-12) - 0.4])
    V = _su(d, rng)
    H = V @ np.diag(np.exp(1j * lam)) @ V.conj().T
    Us = [_su(d, rng) for _ in range(n - 1)]
    P = np.eye(d, dtype=complex)
    for U in Us:
        P = U @ P
    Us.append(H @ P.conj().T)  # producto U_{n−1} ⋯ U_0 = H
    th = qudit.ring_holonomy(Us)
    on_circle = lambda t: np.sort_complex(np.round(np.exp(1j * np.asarray(t)), 9))
    assert np.allclose(on_circle(th), on_circle(lam), atol=1e-9)
    # Ramas distintas de la autofase cercana a π: datos gauge-equivalentes.
    t1 = np.array([np.pi - 1e-9, 0.4, -np.pi + 1e-9 - 0.4])
    t2 = t1 - np.array([2 * np.pi, 0, 0])
    v1, _ = qudit.v_dual_ring(d, n, t1)
    v2, _ = qudit.v_dual_ring(d, n, t2)
    assert v2 == pytest.approx(v1, abs=1e-8)
    p1, _ = qudit.primal_ring_toro(d, n, t1, **CLARABEL_OPTS)
    p2, _ = qudit.primal_ring_toro(d, n, t2, **CLARABEL_OPTS)
    assert p2 == pytest.approx(p1, abs=1e-7)

"""Igualdad v(psi) = v(0) en la cadena cerrada de diamantes sin flujo de triángulo.

Artículo (~/fisica/holonomia.tex): sección «Rigidez nula» de los diamantes
(Proposición «prop:diamantes» y Teorema «thm:igualdad»).

Enunciado. Cadena cerrada de k diamantes, f = 0, gauge simétrico (fase
s = psi/(2k) en las cuatro aristas exteriores de cada celda, 0 en la central).
Para todo k >= 2 y todo psi, v(psi) = v(0), es decir 1/3 (k par) y 1/3 - c/k
(k impar).

Cota inferior, v(psi) >= v(0) (estado primal girado por fragmentos).
 (a) Sea W = -H(a*, b*) el testigo del punto A. Si el sector sin singletes (el
     anillo de k espines 1/2 y k espines 1) está por encima del fundamental de
     H(a*, b*) en psi = 0, todo óptimo primal de psi = 0 tiene soporte en el
     fundamental E, que está en los sectores con alguna celda singlete.
 (b) Se promedia el óptimo sobre los intercambios u_m <-> w_m y sobre la
     rotación global alrededor de z (simetrías de los datos en psi = 0 que
     conmutan con H): sigue siendo óptimo, conmuta con cada T_m^2 y se parte en
     sumandos rho_sigma, uno por patrón sigma de celdas singlete/triplete.
 (c) En rho_sigma, con la celda m singlete, las marginales de las aristas que
     tocan u_m o w_m son I/2 (x) (diagonal) o el singlete: no cambian al rotar
     alrededor de z sus extremos. El gauge g con g_j - g_i = s en las demás
     aristas exteriores, 0 en las centrales, mete toda la vuelta psi en la celda
     m. La mezcla de los sumandos girados tiene las marginales de los datos de
     psi con el mismo v.
Cota superior, v(psi) <= v(0): k par, el techo 1/3; k >= 5 impar, la
Proposición prop:diamantes; k = 3, el exceso del anillo girado frente a 3c en
una malla, más una constante de Lipschitz en psi (este guion).

Comprueba:
  1. exceso del anillo en psi = 0 y a* para k = 2, 3, 4 (> 0; > 3c en k = 3);
  2. k = 3: exceso X(psi) = 4a* eps(psi) + 6 - 12a* del anillo girado en una
     malla de 61 puntos de [0, pi]; |dX/dpsi| <= 2 sqrt(2) a* (norma exacta del
     enlace derivado, sqrt(2)/2, por 2k enlaces y ds/dpsi = 1/(2k)); el mínimo
     de la malla menos la constante por media separación supera 3c;
  3. de punta a punta, k = 2, 3, 4: un estado en E con las marginales de v(0)
     (SDP de mínimos cuadrados; existe por (a)), los promedios, los sectores (el anillo tiene peso nulo), el
     gauge por sector y las marginales en psi = pi/3, 2pi/3, pi, 5pi/3 frente a
     los datos; control negativo: el estado sin girar no es factible en psi.

Escribe data/diamond_igualdad.csv.

Uso: python scripts/diamond_igualdad.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import itertools

import cvxpy as cp
import numpy as np
import sympy

from diamond_punto_a import diamond_H, eps_ring
from rigidez_diamantes import ring_twisted, spin_ops
from holonomy.diamond import chain_edges
from holonomy.report import write_csv

PSIS = (np.pi / 3, 2 * np.pi / 3, np.pi, 5 * np.pi / 3)
MALLA = np.linspace(0.0, np.pi, 61)
GS_TOL = 1e-8


def constantes():
    x = sympy.symbols("x")
    e = float(min(sympy.Poly(2 * x**3 + 8 * x**2 + 3 * x - 4, x).nroots(n=30), key=lambda z: sympy.re(z)))
    a_s = 1 / (2 - e)
    return a_s, 1 - 4 * a_s, 4 * a_s - 2 / 3


def enlace_derivado():
    """Norma de d/ds del enlace girado S^z T^z + (1/2)(e^{is} S^+ T^- + h.c.) en s = 0:
    (i/2)(S^+ T^- - h.c.), espín 1/2 por espín 1."""
    S, T = spin_ops(0.5), spin_ops(1)
    A = np.kron((S[0] + 1j * S[1]).toarray(), (T[0] - 1j * T[1]).toarray())
    return np.abs(np.linalg.eigvalsh(0.5j * (A - A.conj().T))).max()


def reducida(M, n, i, j):
    """Marginal de dos qubits (i, j) de rho = M M^dagger, M de forma (2^n, r)."""
    t = np.moveaxis(M.reshape((2,) * n + (M.shape[1],)), (i, j), (0, 1)).reshape(4, -1)
    return t @ t.conj().T


def objetivo(v, phi):
    """rho_e(v) = I/4 + v tau_e(phi), tau_e = (1 (x) U(phi)) (|s><s| - I/4) (...)^dagger."""
    s = np.array([0, 1, -1, 0]) / np.sqrt(2)
    U = np.kron(np.eye(2), np.diag(np.exp(1j * phi * np.array([1, -1]) / 2)))
    return np.eye(4) / 4 + v * (U @ (np.outer(s, s) - np.eye(4) / 4) @ U.conj().T)


def zs(n):
    """z_i = +-1 de cada estado de la base (sitio 0 el más significativo, bit 0 = arriba)."""
    idx = np.arange(2**n)
    return np.array([1 - 2 * ((idx >> (n - 1 - i)) & 1) for i in range(n)])


def gauge(k, m, s):
    """Ángulos g_i: g_j - g_i = s en las aristas exteriores fuera de la celda m y en
    (j_m, u_m), (j_m, w_m); g_u = g_w en cada celda; la vuelta queda en la celda m."""
    g = np.zeros(3 * k)
    j = 3 * ((m + 1) % k)
    for t in range(1, k + 1):
        c = (m + t) % k
        g[3 * c + 1] = g[3 * c + 2] = g[3 * c] + s
        if t < k:
            g[3 * ((c + 1) % k)] = g[3 * c] + 2 * s
    assert j == 3 * ((m + 1) % k) and g[j] == 0.0
    return g


def raiz(Y):
    """M con M M^T = Y (parte positiva de Y simétrica)."""
    y, Q = np.linalg.eigh((Y + Y.T) / 2)
    keep = y > 1e-13
    return Q[:, keep] * np.sqrt(y[keep])


def max_marg_err(M, k, v, phis):
    n = 3 * k
    return max(np.abs(reducida(M, n, i, j) - objetivo(v, ph)).max()
               for (i, j), ph in zip(chain_edges(k, periodic=True), phis))


def de_punta_a_punta(chk, k, a_s, b_s, c, rows):
    n = 3 * k
    v0 = 1 / 3 if k % 2 == 0 else 1 / 3 - c / k
    H = diamond_H(k, a_s / k, b_s / k).toarray()
    w, V = np.linalg.eigh(H)
    P = V[:, w < w[0] + GS_TOL]
    d = P.shape[1]
    chk.info(f"k = {k}: lambda_min = {w[0]:.12f}, dim E = {d}, siguiente nivel {w[d] - w[0]:.3e} por encima")

    # Óptimo primal con soporte en E (real: la conjugación es simetría en psi = 0). En E el
    # conjunto factible con v = v(0) no tiene interior y Clarabel falla con las igualdades;
    # se busca por mínimos cuadrados de las marginales, que valen 0 si existe (con las
    # tolerancias de 1e-10 Clarabel también falla, porque el óptimo es 0; la precisión
    # real es el error de marginales que se comprueba después).
    X = cp.Variable((d, d), symmetric=True)
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]

    def en_E(ops):
        O = np.ones((1, 1))
        for q in range(n):
            O = np.kron(O, ops.get(q, np.eye(2)))
        return (P.T @ O @ P).real
    res = []
    for (i, j) in chain_edges(k, periodic=True):
        for a in range(3):
            for b in range(3):
                res.append((en_E({i: paulis[a], j: paulis[b]}), -v0 if a == b else 0.0))
    for i in range(n):
        for a in range(3):
            res.append((en_E({i: paulis[a]}), 0.0))
    # Las filas son muy redundantes y con ellas Clarabel falla: se sustituyen por una base
    # ortonormal (en el producto de Frobenius) de su espacio, por SVD. La parte de los
    # objetivos fuera de ese espacio no depende de X y tiene que ser nula.
    A = np.array([G.reshape(-1) for G, _ in res])
    tv = np.array([t for _, t in res])
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((sv > 1e-9 * sv[0]).sum())
    fuera = np.abs(tv - U[:, :r] @ (U[:, :r].T @ tv)).max()
    chk.check(fuera < 1e-12, f"k = {k}: los objetivos están en el espacio de las filas (rango {r}, resto {fuera:.1e})")
    res = [cp.trace(X @ Vt[q].reshape(d, d).T) - (U[:, q] @ tv) / sv[q] for q in range(r)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.hstack(res))), [X >> 0, cp.trace(X) == 1])
    prob.solve(solver="CLARABEL")
    chk.info(f"k = {k}: Clarabel {prob.status}")
    Xv = X.value
    err_E = max_marg_err(P @ raiz(Xv), k, v0, [0.0] * (5 * k))
    chk.check(err_E < 1e-6, f"k = {k}: hay un estado en E con las marginales de v(0) = {v0:.10f} (error {err_E:.1e})")

    # Promedios: intercambios u_m <-> w_m y rotación global alrededor de z (en E).
    def swap_full(m):
        perm = list(range(n))
        perm[3 * m + 1], perm[3 * m + 2] = perm[3 * m + 2], perm[3 * m + 1]
        idx = np.arange(2**n).reshape((2,) * n).transpose(perm).reshape(-1)
        return idx
    Sw = []
    for m in range(k):
        idx = swap_full(m)
        Sw.append(P.T @ P[idx, :])
    err_sw = max(np.abs(S @ S.T - np.eye(d)).max() for S in Sw)
    chk.check(err_sw < 1e-10, f"k = {k}: los intercambios u_m <-> w_m dejan E invariante ({err_sw:.1e})")
    Xb = np.zeros_like(Xv)
    for bits in itertools.product((0, 1), repeat=k):
        G = np.eye(d)
        for m, bt in enumerate(bits):
            if bt:
                G = Sw[m] @ G
        Xb += G @ Xv @ G.T
    Xb /= 2**k
    Sz = P.T @ (zs(n).sum(0)[:, None] * P)
    wz, Vz = np.linalg.eigh(Sz)
    blocks = np.round(wz).astype(int)
    Xz = Vz.T @ Xb @ Vz
    Xz[blocks[:, None] != blocks[None, :]] = 0
    Xb = Vz @ Xz @ Vz.T

    Mb = P @ raiz(Xb)
    err0 = max_marg_err(Mb, k, v0, [0.0] * (5 * k))
    chk.check(err0 < 1e-7, f"k = {k}: el promedio sigue siendo factible en psi = 0 (error {err0:.1e})")

    # Sectores: Q_sigma = prod_m (1 -+ SWAP_m)/2 en E; el anillo (todas T) tiene peso nulo.
    Qs = {}
    for sig in itertools.product("ST", repeat=k):
        Q = np.eye(d)
        for m, ch in enumerate(sig):
            Q = Q @ ((np.eye(d) - Sw[m]) / 2 if ch == "S" else (np.eye(d) + Sw[m]) / 2)
        Qs[sig] = Q
    anillo = np.trace(Qs[("T",) * k] @ Xb @ Qs[("T",) * k])
    chk.check(abs(anillo) < 1e-10, f"k = {k}: peso del sector sin singletes {anillo:.1e}")
    suma = sum(Q @ Xb @ Q for Q in Qs.values())
    chk.check(np.abs(suma - Xb).max() < 1e-10, f"k = {k}: el promedio es la suma de sus bloques de sector")

    Z = zs(n)
    for psi in PSIS:
        s = psi / (2 * k)
        phis = [s, s, 0.0, s, s] * k
        cols = []
        for sig, Q in Qs.items():
            if "S" not in sig:
                continue
            Ms = P @ raiz(Q @ Xb @ Q)
            if Ms.shape[1] == 0:
                continue
            g = gauge(k, sig.index("S"), s)
            cols.append(np.exp(0.5j * (g[:, None] * Z).sum(0))[:, None] * Ms)
        Mg = np.hstack(cols)
        tr = np.trace(Mg.conj().T @ Mg).real
        err = max_marg_err(Mg, k, v0, phis)
        neg = max_marg_err(Mb, k, v0, phis)
        chk.check(abs(tr - 1) < 1e-9 and err < 1e-7 and neg > 1e-2,
                  f"k = {k}, psi = {psi / np.pi:.4f} pi: estado girado factible con v(0) "
                  f"(error {err:.1e}, traza {tr:.12f}); sin girar, error {neg:.2e}")
        rows.append([k, f"{psi:.12f}", d, f"{v0:.12f}", f"{v0:.12f}", f"{err:.3e}", f"{neg:.3e}"])


def main():
    chk = Checker("diamond_igualdad")
    a_s, b_s, c = constantes()
    chk.info(f"a* = {a_s:.12f}, b* = {b_s:.12f}, c = {c:.12f}, 3c = {3 * c:.12f}")

    # 1. exceso del anillo en psi = 0
    for k in (2, 3, 4):
        X = 4 * a_s * eps_ring(k) + 2 * k - 4 * a_s * k
        lo = 3 * c if k % 2 else 0.0
        chk.check(X > lo + 0.1, f"k = {k}: exceso del anillo en psi = 0, X = {X:.6f} > {lo:.6f}")

    # 2. k = 3: el anillo girado en [0, pi], con constante de Lipschitz
    nb = enlace_derivado()
    chk.check(abs(nb - np.sqrt(2) / 2) < 1e-12, f"norma del enlace derivado = {nb:.12f} = sqrt(2)/2")
    lip = 4 * a_s * nb
    Xs = [4 * a_s * np.linalg.eigvalsh(ring_twisted(3, p).toarray())[0] + 6 - 12 * a_s for p in MALLA]
    h = MALLA[1] - MALLA[0]
    cota = min(Xs) - lip * h / 2
    chk.check(cota > 3 * c + 0.1,
              f"k = 3: min X(psi) en la malla = {min(Xs):.6f}, menos {lip:.4f}*h/2 = {cota:.6f} > 3c = {3 * c:.6f}")
    chk.info(f"k = 3: X(0) = {Xs[0]:.6f}, X(pi) = {Xs[-1]:.6f}")

    # 3. de punta a punta
    rows = []
    for k in (2, 3, 4):
        de_punta_a_punta(chk, k, a_s, b_s, c, rows)
    write_csv("diamond_igualdad.csv", ["k", "psi", "dim_E", "v_primal", "v0", "err_girado", "err_sin_girar"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

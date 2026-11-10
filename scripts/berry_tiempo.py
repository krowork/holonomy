"""Sin curvatura de Berry: la inversión temporal T = (iY)^{x n} K conmuta con todos los observables del pegado.

Los A_e, B_e, C_e (y las 15 correlaciones de dos sitios del caso no abeliano) son productos de dos Paulis, y T cambia el
signo de cada Pauli: T conmuta con todos ellos y con todo testigo y todo estado de máxima entropía, en cada punto del
espacio de holonomías. T^2 = (-1)^n. Con n par, un estado óptimo puro es T-real y su curvatura de Berry es nula; con n
impar (Kramers) todo autovalor de un testigo es doble. El estado de máxima entropía es único, luego T-invariante, y la
parte imaginaria de tr(rho L_mu L_nu) (SLD) es nula. Solo quedan fases de Berry Z_2 (0 o pi) en lazos, y en el ejemplo
el estado óptimo solo es puro en una ventana alrededor del flujo nulo.

Comprobaciones:
  1. [T, O_k] = 0 en la escalera abierta 2 x 3 (n = 6), el anillo de 5 y la escalera periódica 2 x 4; T^2 = (-1)^n;
     con n = 5, el espectro de un testigo al azar con flujo es doble.
  2. Escalera abierta 2 x 3 (dos plaquetas, flujos f1, f2), malla 16 x 16 del toro: el flujo de Berry por plaqueta de
     los autovectores máximos del testigo del SDP solo toma los valores 0 y +-pi.
  3. Entropía residual (máxima entropía en 1 - v/vmax = 1e-2, 1e-3, 1e-4) en varios flujos: pura cerca del flujo nulo,
     mixta en el resto.
  4. Máxima entropía en v = 0,3 y flujo (pi/2, pi/3): Im tr(rho L_1 L_2) = 0 frente a la parte real.

Escribe data/berry_tiempo{,_entropia,_resumen}.csv.

Uso: (ulimit -v 10000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/berry_tiempo.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.special import logsumexp

from fisher_kubo_mori import MaxEnt, escalera
from holonomy.report import write_csv

ABIERTA = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]  # patas 0-1-2 y 3-4-5, peldaños 0-3, 1-4, 2-5
M = 16
RESUMEN = []


def T_unitario(n):
    """Parte unitaria de T = (iY)^{x n} K."""
    iY = np.array([[0.0, 1.0], [-1.0, 0.0]])
    U = np.array([[1.0]])
    for _ in range(n):
        U = np.kron(U, iY)
    return U


def fases(f1, f2):
    """Flujo f1 en la plaqueta izquierda (en el peldaño 0-3) y f2 en la derecha (en el peldaño 2-5)."""
    ph = np.zeros(len(ABIERTA))
    ph[4], ph[6] = f1, -f2
    return ph


def testigo(me, ph):
    K, N = len(me.O), me.O.shape[1]
    t1 = me.objetivo(1.0, ph)
    y, lam = cp.Variable(K), cp.Variable()
    Wr = sum(y[k] * me.O[k].real for k in range(K))
    Wi = sum(y[k] * me.O[k].imag for k in range(K))
    pr = cp.Problem(cp.Minimize(lam), [lam * np.eye(2 * N) - cp.bmat([[Wr, -Wi], [Wi, Wr]]) >> 0, y @ t1 == 1])
    pr.solve(solver="CLARABEL")
    return float(lam.value), np.einsum("k,kab->ab", y.value, me.O), pr.status


def simetria(chk):
    rng = np.random.default_rng(5)
    for nombre, n, edges in (("escalera abierta 2x3", 6, ABIERTA), ("anillo de 5", 5, [(i, (i + 1) % 5) for i in range(5)]),
                             ("escalera 2x4", 8, escalera(4)[0])):
        me = MaxEnt(n, edges)
        U = T_unitario(n)
        err = max(np.abs(U @ O.conj() - O @ U).max() for O in me.O)
        T2 = (U @ U.conj())[0, 0].real
        chk.check(err < 1e-14 and abs(T2 - (-1)**n) < 1e-14,
                  f"{nombre} (n = {n}): [T, O_k] = {err:.0e} en los {len(me.O)} observables; T^2 = {T2:+.0f} = (-1)^n")
        if n % 2:
            W = np.einsum("k,kab->ab", rng.normal(size=len(me.O)), me.O)
            ev = np.linalg.eigvalsh(W)
            chk.check(np.abs(ev[0::2] - ev[1::2]).max() < 1e-12,
                      f"{nombre}: testigo al azar (fases al azar): los {len(ev)} autovalores van por pares "
                      f"(diferencia máxima {np.abs(ev[0::2] - ev[1::2]).max():.0e}): Kramers")


def plaquetas(chk):
    me = MaxEnt(6, ABIERTA)
    g = 2 * np.pi * np.arange(M) / M
    psi, est = {}, []
    for a, f1 in enumerate(g):
        for b, f2 in enumerate(g):
            _, W, st = testigo(me, fases(f1, f2))
            est.append(st)
            psi[a, b] = np.linalg.eigh(W)[1][:, -1]
    F = np.zeros((M, M))
    for a in range(M):
        for b in range(M):
            c = [(a, b), ((a + 1) % M, b), ((a + 1) % M, (b + 1) % M), (a, (b + 1) % M)]
            F[a, b] = np.angle(np.prod([np.vdot(psi[c[i]], psi[c[(i + 1) % 4]]) for i in range(4)]))
    dist = np.minimum(np.abs(F), np.abs(np.abs(F) - np.pi)).max()
    n_pi = int((np.abs(np.abs(F) - np.pi) < 1e-4).sum())
    est = {e: est.count(e) for e in set(est)}
    chk.check(dist < 1e-4 and set(est) <= {"optimal", "optimal_inaccurate"},
              f"escalera abierta 2x3, malla {M} x {M}: el flujo de Berry por plaqueta solo vale 0 o +-pi (distancia "
              f"máxima {dist:.0e}, del orden de la precisión del SDP; {n_pi} celdas con pi); estados del SDP: {est}")
    write_csv("berry_tiempo.csv", ("f1_indice", "f2_indice", "flujo_berry"),
              [[a, b, F[a, b]] for a in range(M) for b in range(M)])
    RESUMEN.extend([["distancia_0_pi", dist], ["celdas_pi", n_pi]])
    return n_pi, dist


def ventana(chk):
    me = MaxEnt(6, ABIERTA)
    filas = []
    for f1, f2 in ((0, 0), (0.05, 0), (0.2, 0), (0.5, 0), (np.pi / 2, np.pi / 3), (np.pi, 0), (np.pi, np.pi)):
        ph = fases(f1, f2)
        vmax, _, st = testigo(me, ph)
        x = np.zeros(len(me.O))
        S = {}
        for eps in (1e-2, 1e-3, 1e-4):
            x, _, S[eps], res = me.resuelve(me.objetivo(vmax * (1 - eps), ph), x)
        filas.append([f1, f2, vmax, S[1e-2], S[1e-3], S[1e-4], res])
        chk.info(f"escalera abierta 2x3, (f1, f2) = ({f1:.3f}, {f2:.3f}): vmax = {vmax:.8f}, S_G = {S[1e-2]:.4f}, "
                 f"{S[1e-3]:.4f}, {S[1e-4]:.4f} en eps = 1e-2, 1e-3, 1e-4 (residuo {res:.0e})")
    S4 = {(round(r[0], 3), round(r[1], 3)): r[5] for r in filas}
    S3 = {(round(r[0], 3), round(r[1], 3)): r[4] for r in filas}
    puros = [(0, 0), (0.05, 0)]
    mixtos = [(0.5, 0), (round(np.pi / 2, 3), round(np.pi / 3, 3)), (round(np.pi, 3), 0), (round(np.pi, 3), round(np.pi, 3))]
    chk.check(all(S4[p] < 0.02 and S4[p] < S3[p] / 5 for p in puros) and all(S4[p] > 0.3 for p in mixtos),
              "escalera abierta 2x3: estado óptimo puro cerca del flujo nulo (S_G en 1e-4: "
              + ", ".join(f"{S4[p]:.4f}" for p in puros) + ") y mixto en (0,5, 0), (pi/2, pi/3), (pi, 0), (pi, pi) ("
              + ", ".join(f"{S4[p]:.4f}" for p in mixtos) + ")")
    write_csv("berry_tiempo_entropia.csv", ("f1", "f2", "vmax", "S_1e-2", "S_1e-3", "S_1e-4", "residuo"), filas)


def qgt_mixto(chk):
    me = MaxEnt(6, ABIERTA)
    ph0 = fases(np.pi / 2, np.pi / 3)
    v = 0.3
    x, g, S, res = me.resuelve(me.objetivo(v, ph0))
    E, U = eigh(np.einsum("k,kab->ab", x, me.O))
    p = np.exp(E - logsumexp(E))
    dE = E[:, None] - E[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(dE) > 1e-10, (p[:, None] - p[None, :]) / np.where(dE == 0, 1, dE), (p[:, None] + p[None, :]) / 2)
    L = []
    for i in (4, 6):  # derivada respecto de f1 y de f2
        dph = np.zeros(len(ABIERTA))
        dph[i] = 1.0 if i == 4 else -1.0
        dm = v * np.array([[-2 * np.sin(f) * d, 2 * np.cos(f) * d, 0.0] for f, d in zip(ph0, dph)]).ravel()
        dH = U.conj().T @ np.einsum("k,kab->ab", np.linalg.solve(g, dm), me.O) @ U
        drho = w * dH - np.diag(p) * np.trace(np.diag(p) @ dH).real
        L.append(2 * drho / (p[:, None] + p[None, :]))
    q = np.sum(p[:, None] * L[0] * L[1].T)
    RESUMEN.extend([["tr_rho_L1_L2_real", q.real], ["tr_rho_L1_L2_imag", q.imag]])
    chk.check(abs(q.imag) < 1e-12 * abs(q.real) and res < 1e-10,
              f"escalera abierta 2x3, v = 0,3, flujo (pi/2, pi/3): tr(rho L_1 L_2) = {q.real:+.6e} {q.imag:+.1e} i "
              "(SLD): la parte imaginaria es nula")


def main():
    chk = Checker("berry_tiempo")
    simetria(chk)
    plaquetas(chk)
    ventana(chk)
    qgt_mixto(chk)
    write_csv("berry_tiempo_resumen.csv", ("magnitud", "valor"), RESUMEN)
    chk.finish()


if __name__ == "__main__":
    main()

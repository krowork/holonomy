"""Curvatura de la entropía de máxima entropía frente al flujo y métrica de Kubo--Mori (información de Fisher cuántica).

El estado de máxima entropía con marginales de Werner rotadas es rho = exp(sum_k x_k O_k)/Z, una familia exponencial:
m_k = <O_k> = d_k ln Z y g = Hess ln Z es la métrica de Kubo--Mori. S = ln Z - x.m es la transformada de Legendre, de modo
que dS/dm = -x y d2S/dm2 = -g^{-1}. A lo largo de un camino de flujo m(t), con phi_e(t) = t u_e,
    S'' = -m'^T g^{-1} m' - x.m'',
con m'_{B_e} = 2 v u_e y m''_{A_e} = -2 v u_e^2 en t = 0. El primer término (Fisher) es <= 0; el segundo, el de los
multiplicadores, compite con él, y de ahí que el signo de Delta S_hol no esté fijado (sección 7 de esfera).

Implementación densa e independiente de `sphere_entropia.Gibbs`: escalera periódica 2 x L (L = 3, 4), tres operadores
por arista, sin reducción por simetría. Camino: fase +t/2 en cada arista de la pata inferior y -t/2 en la superior
(plaqueta t, patas +-L t/2), que es `sphere_entropia_holonomia.escalera(L, F)` con psi = 0 y t = F/L.

Comprobaciones:
  1. S'' por diferencias finitas de cinco puntos frente a la fórmula, en v = 0,1, 0,2, 0,3 y en una dirección al azar;
     control: en una dirección pura de gauge (u_e = g_j - g_i) las dos dan 0.
  2. 1/2 S'' t^2 frente a Delta S_hol de data/sphere_entropia_holonomia_escaleras.csv con F = pi/8 (difiere en O(t^4)).
  3. Cerca de vmax (dual por SDP), x.O se alinea con el testigo óptimo W* y el término de multiplicadores por unidad
     de ||x.O|| tiende a -v d2N_t(W*)/dt2 / ||W*||. En 2 x 3 (patas triangulares, vmax = 1/3) el testigo es degenerado y
     la alineación es lenta; el límite del término de multiplicadores solo se comprueba en 2 x 4 (el cubo).
No se comprueba la relación con la susceptibilidad de fidelidad del fundamental del testigo.

Escribe data/fisher_kubo_mori.csv y data/fisher_kubo_mori_vmax.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/fisher_kubo_mori.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.special import logsumexp

from holonomy.paulis import pauli_string
from holonomy.report import read_csv, write_csv

VS = (0.1, 0.2, 0.3)
H = 2e-3
EPSS = (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3)


def escalera(L):
    """Aristas orientadas i -> j y fase por unidad de t: patas +-1/2, peldaños 0."""
    edges, u = [], []
    for base, s in ((0, 0.5), (L, -0.5)):
        for i in range(L):
            edges.append((base + i, base + (i + 1) % L))
            u.append(s)
    for i in range(L):
        edges.append((i, L + i))
        u.append(0.0)
    return edges, np.array(u)


class MaxEnt:
    """Estado de máxima entropía denso con operadores A_e, B_e, C_e por arista, en ese orden."""

    def __init__(self, n, edges):
        ops = []
        for i, j in edges:
            ops.append(-(pauli_string(n, ((i, 0), (j, 0))) + pauli_string(n, ((i, 1), (j, 1)))))
            ops.append(pauli_string(n, ((i, 0), (j, 1))) - pauli_string(n, ((i, 1), (j, 0))))
            ops.append(-pauli_string(n, ((i, 2), (j, 2))))
        self.O = np.array([np.asarray(o.todense() if hasattr(o, "todense") else o, dtype=complex) for o in ops])

    @staticmethod
    def objetivo(v, phis):
        return v * np.array([[2 * np.cos(f), 2 * np.sin(f), 1.0] for f in phis]).ravel()

    def gibbs(self, x):
        """Medias, métrica de Kubo--Mori y entropía."""
        E, U = eigh(np.einsum("k,kab->ab", x, self.O))
        p = np.exp(E - logsumexp(E))
        # Con np.einsum en vez de matmul, U^dag O U es O(N^4) por operador y no termina en 2 x 4.
        M = U.conj().T[None] @ self.O @ U[None]
        m = np.einsum("i,kii->k", p, M).real
        dE = E[:, None] - E[None, :]
        dp = p[:, None] - p[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(dE) > 1e-10, dp / np.where(dE == 0, 1, dE), (p[:, None] + p[None, :]) / 2)
        K = len(M)
        g = ((M * w[None]).reshape(K, -1) @ M.transpose(0, 2, 1).reshape(K, -1).T).real - np.outer(m, m)
        S = -(p[p > 0] * np.log(p[p > 0])).sum()
        return m, g, S

    def resuelve(self, t, x0=None, tol=1e-12):
        """Newton sobre las medias, amortiguado por el residuo (una búsqueda de línea sobre ln Z - x.t se atasca en el
        ruido cerca de la tolerancia)."""
        x = np.zeros(len(self.O)) if x0 is None else x0.copy()
        m, g, S = self.gibbs(x)
        for _ in range(60):
            r = m - t
            if np.abs(r).max() < tol:
                break
            paso = np.linalg.solve(g, -r)
            a = 1.0
            while a > 1e-6:
                mn, gn, Sn = self.gibbs(x + a * paso)
                if np.abs(mn - t).max() < np.abs(r).max():
                    break
                a /= 2
            x, m, g, S = x + a * paso, mn, gn, Sn
        return x, g, S, float(np.abs(m - t).max())


def derivadas(v, u):
    dm = v * np.array([[0.0, 2 * ue, 0.0] for ue in u]).ravel()
    d2m = v * np.array([[-2 * ue**2, 0.0, 0.0] for ue in u]).ravel()
    return dm, d2m


def curvatura(me, v, u):
    """S'' en t = 0 por diferencias finitas de cinco puntos y por la fórmula; S0 y residuo."""
    x0, g, S0, res = me.resuelve(me.objetivo(v, 0 * u))
    Ss = {}
    for k in (-2, -1, 1, 2):
        _, _, S, r = me.resuelve(me.objetivo(v, k * H * u), x0)
        res = max(res, r)
        Ss[k] = S
    fd = (-Ss[2] + 16 * Ss[1] - 30 * S0 + 16 * Ss[-1] - Ss[-2]) / (12 * H**2)
    dm, d2m = derivadas(v, u)
    fis, mul = -dm @ np.linalg.solve(g, dm), -x0 @ d2m
    return fd, fis, mul, S0, res


def formula_frente_a_fd(chk):
    filas = []
    rng = np.random.default_rng(1)
    previas = {(int(r["L"]), round(r["v"], 6)): r for r in read_csv("sphere_entropia_holonomia_escaleras.csv")
               if abs(r["F"] - np.pi / 8) < 1e-9}
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        for v in VS:
            fd, fis, mul, S0, res = curvatura(me, v, u)
            bkm = fis + mul
            rel = abs(fd - bkm) / abs(bkm)
            chk.check(res < 1e-10 and rel < 1e-5,
                      f"2x{L}, v = {v}: S'' = {fd:+.10e} (dif. finitas), {bkm:+.10e} (Kubo-Mori), dif. rel. {rel:.1e}; "
                      f"Fisher {fis:+.4e}, multiplicadores {mul:+.4e}, residuo {res:.1e}")
            # Contraste con el Gibbs por órbitas del repo: la misma S sin flujo y Delta S_hol a F = pi/8.
            r = previas[L, v]
            t = np.pi / 8 / L
            pred = bkm * t**2 / 2
            chk.check(abs(S0 - r["S_trivial"]) < 1e-8 and abs(pred - r["dS"]) / abs(r["dS"]) < 1e-2,
                      f"2x{L}, v = {v}: S0 = {S0:.10f} (CSV {r['S_trivial']:.10f}); 1/2 S'' t^2 = {pred:+.5e} frente a "
                      f"Delta S_hol = {r['dS']:+.5e} con F = pi/8, dif. rel. {abs(pred - r['dS']) / abs(r['dS']):.1e}")
            filas.append([L, 2 * L, "plaquetas", v, S0, fd, bkm, fis, mul, pred, r["dS"], res])
        gg = rng.normal(size=2 * L)
        ug = np.array([gg[j] - gg[i] for i, j in edges])
        fd, fis, mul, S0, res = curvatura(me, 0.2, ug)
        chk.check(abs(fd) < 1e-6 and abs(fis + mul) < 1e-10,
                  f"2x{L}, gauge, v = 0,2: S'' = {fd:+.1e} (dif. finitas), {fis + mul:+.1e} (Kubo-Mori)")
        filas.append([L, 2 * L, "gauge", 0.2, S0, fd, fis + mul, fis, mul, np.nan, np.nan, res])
        ur = rng.normal(size=len(edges))
        fd, fis, mul, S0, res = curvatura(me, 0.2, ur)
        rel = abs(fd - fis - mul) / abs(fis + mul)
        chk.check(rel < 1e-5, f"2x{L}, dirección al azar, v = 0,2: S'' = {fd:+.10e}, {fis + mul:+.10e}, dif. rel. {rel:.1e}")
        filas.append([L, 2 * L, "azar", 0.2, S0, fd, fis + mul, fis, mul, np.nan, np.nan, res])
    write_csv("fisher_kubo_mori.csv", ("L", "n", "direccion", "v", "S0", "S2_fd", "S2_kubo_mori", "fisher",
                                       "multiplicadores", "pred_dS_F_pi8", "dS_csv", "residuo"), filas)


def cerca_de_vmax(chk):
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        K, N = len(me.O), me.O.shape[1]
        t1 = me.objetivo(1.0, 0 * u)
        y, lam = cp.Variable(K), cp.Variable()
        Wr = sum(y[k] * me.O[k].real for k in range(K))
        Wi = sum(y[k] * me.O[k].imag for k in range(K))
        prob = cp.Problem(cp.Minimize(lam), [lam * np.eye(2 * N) - cp.bmat([[Wr, -Wi], [Wi, Wr]]) >> 0, y @ t1 == 1])
        prob.solve(solver="CLARABEL")
        vmax, ys = float(lam.value), y.value
        Wst = np.einsum("k,kab->ab", ys, me.O)
        ev = np.linalg.eigvalsh(Wst)
        chk.check(abs(ev[-1] - vmax) < 1e-6,
                  f"2x{L}: vmax = {vmax:.8f} = lambda_max(W*) = {ev[-1]:.8f}, gap del testigo {ev[-1] - ev[-2]:.2e}"
                  + chk.status_note(prob.status))
        if L == 3:
            chk.check(abs(vmax - 1 / 3) < 1e-7, f"2x3: vmax = 1/3 (patas triangulares), {vmax - 1 / 3:+.1e}")
        x = np.zeros(K)
        for eps in EPSS:
            v = vmax * (1 - eps)
            x, g, S, res = me.resuelve(me.objetivo(v, 0 * u), x)
            X = np.einsum("k,kab->ab", x, me.O)
            cos = np.vdot(X, Wst).real / np.linalg.norm(X) / np.linalg.norm(Wst)
            dm, d2m = derivadas(v, u)
            fis, mul = -dm @ np.linalg.solve(g, dm), -x @ d2m
            chk.info(f"2x{L}, 1 - v/vmax = {eps:.0e}: ||x.O|| = {np.linalg.norm(X):7.2f}, cos(x.O, W*) = {cos:.6f}, "
                     f"S = {S:.4f}, Fisher {fis:+.4e}, multiplicadores {mul:+.4e}, residuo {res:.1e}")
            filas.append([L, vmax, eps, v, np.linalg.norm(X), cos, S, fis, mul, fis + mul, res])
        chk.check(res < 1e-10 and cos > 0.998, f"2x{L}: en 1 - v/vmax = 1e-3, x.O se alinea con W*: cos = {cos:.6f}")
        if L == 4:
            d2N = -2 * sum(ys[3 * i] * u[i]**2 for i in range(len(u)))
            pred, med = -vmax * d2N / np.linalg.norm(Wst), mul / np.linalg.norm(X)
            chk.check(abs(med - pred) / abs(pred) < 1e-2,
                      f"2x4: multiplicadores / ||x.O|| = {med:+.5e} frente a -v d2N(W*)/dt2 / ||W*|| = {pred:+.5e}")
    write_csv("fisher_kubo_mori_vmax.csv", ("L", "vmax", "eps", "v", "norma_xO", "cos_W", "S", "fisher",
                                            "multiplicadores", "S2", "residuo"), filas)


def main():
    chk = Checker("fisher_kubo_mori")
    formula_frente_a_fd(chk)
    cerca_de_vmax(chk)
    chk.finish()


if __name__ == "__main__":
    main()

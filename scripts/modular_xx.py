"""Nota «Tiempo modular y holonomía», sección 3.1 (el caso XX, Proposiciones 4 y 5).

Con c_e = 0, K_phi = sum_e a_e (cos phi_e A_e + sin phi_e B_e).

1. Anillo (Jordan-Wigner): C_j(s) = tr(Z_0(s) Z_j)/N = (|U_Theta(s)_{0j}|^2 + |U_{Theta+pi}(s)_{0j}|^2)/2, U = e^{-ihs} con h la
   matriz de una partícula, n = 4..7; las vueltas impares se cancelan: en el anillo de 5 el primer momento con dependencia
   es k = 2n y proporcional a 1 - cos 2 Theta, y en Theta = pi no hay dependencia hasta k = 2n + 4; anillos de 4, 6, 8: 2n.
2. Cota XX: el flujo no entra antes de min(2V*, 2V*_2 - 2), con V*_2 el menor |T| que contiene 0 y j con subgrafo inducido
   conexo, de número ciclomático >= 2 y con un ciclo no trivial. Diez grafos, fases genéricas y en una sola arista,
   todos los j; contraejemplo de «XX => 2V*»: K4 con fase solo en la arista 23 y j = 1 (V* = 4, primer orden 6).
3. Palabras de Pauli en el triángulo: con XX no hay palabras de longitud 2V* - 2 = 4 cuyas aristas contengan el ciclo;
   con XXZ, sí.

Escribe data/modular_xx{,_anillo,_palabras}.csv.

Uso: (ulimit -v 8000000; OMP_NUM_THREADS=8 .venv/bin/python scripts/modular_xx.py)
"""
import _common  # noqa: F401
from _common import Checker

import itertools

import numpy as np
from scipy.linalg import eigh, expm

from holonomy.paulis import pauli_string
from holonomy.report import write_csv
from modular_cota import anillo, dense, no_triviales, vstar, zdiag

GRAFOS = {
    "triángulo": (3, [(0, 1), (1, 2), (2, 0)]),
    "anillo 5": (5, anillo(5)),
    "triángulo con cola": (4, [(1, 2), (2, 3), (3, 1), (0, 1)]),
    "K4": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    "K4 menos arista": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]),
    "casa": (5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]),
    "escalera abierta 2x3": (6, [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]),
    "rueda W4 (centro 4)": (5, [(4, 0), (4, 1), (4, 2), (4, 3), (0, 1), (1, 2), (2, 3), (3, 0)]),
    "dos triángulos por un vértice": (5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]),
    "theta (0 en un camino)": (5, [(1, 0), (0, 2), (1, 3), (3, 2), (1, 4), (4, 2)]),
}


def K_xx(n, edges, phis, amps):
    K = 0
    for (i, j), f, a in zip(edges, phis, amps):
        A = -(pauli_string(n, ((i, 0), (j, 0))) + pauli_string(n, ((i, 1), (j, 1))))
        B = pauli_string(n, ((i, 0), (j, 1))) - pauli_string(n, ((i, 1), (j, 0)))
        K = K + a * (np.cos(f) * A + np.sin(f) * B)
    return dense(K)


def momentos(n, edges, phis, amps, j, kmax):
    """Delta mu_k = mu_k(phi) - mu_k(0) y |mu_k(0)|, k = 0..kmax, mu_k = tr(ad_K^k(Z_0) Z_j)/N."""
    N = 2 ** n
    K1, K0 = K_xx(n, edges, phis, amps), K_xx(n, edges, 0 * phis, amps)
    X1 = X0 = np.diag(zdiag(n, 0)).astype(complex)
    d, m = [], []
    for k in range(kmax + 1):
        d.append(((X1.diagonal() - X0.diagonal()) * zdiag(n, j)).sum().real / N)
        m.append(abs((X0.diagonal() * zdiag(n, j)).sum().real / N))
        X1, X0 = K1 @ X1 - X1 @ K1, K0 @ X0 - X0 @ K0
    return np.array(d), np.array(m)


def primer_orden(n, edges, phis, amps, j, kmax):
    d, m = momentos(n, edges, phis, amps, j, kmax)
    return next((k for k in range(kmax + 1) if abs(d[k]) > 1e-9 * (1 + m[k])), None)


def vstar2(n, edges, phis, j):
    nt = no_triviales(n, edges, phis)
    for size in range(2, n + 1):
        for T in itertools.combinations(range(n), size):
            if 0 not in T or j not in T:
                continue
            sub = [e for e in edges if e[0] in T and e[1] in T]
            ady = {v: set() for v in T}
            for a, b in sub:
                ady[a].add(b)
                ady[b].add(a)
            vis, pila = {T[0]}, [T[0]]
            while pila:
                for w in ady[pila.pop()] - vis:
                    vis.add(w)
                    pila.append(w)
            if len(vis) == len(T) and len(sub) - len(T) + 1 >= 2 and any(V <= set(T) for V in nt):
                return size
    return None


def h_una(n, Theta):
    h = np.zeros((n, n), complex)
    for i in range(n):
        h[i, (i + 1) % n] = -2 * np.exp(1j * Theta / n)
        h[(i + 1) % n, i] = -2 * np.exp(-1j * Theta / n)
    return h


def anillos(chk):
    filas = []
    ss = np.linspace(0, 3, 13)
    for n in (4, 5, 6, 7):
        for Th in (0.7, 2.0):
            E, U = eigh(K_xx(n, anillo(n), np.full(n, Th / n), np.ones(n)))
            Z0 = (U.conj().T * zdiag(n, 0)[None, :]) @ U
            Z2 = (U.conj().T * zdiag(n, 2)[None, :]) @ U
            dE = E[None, :] - E[:, None]
            Cs = np.array([(Z0 * np.exp(1j * s * dE) * Z2.T).sum().real / 2 ** n for s in ss])
            pred = np.array([0.5 * sum(abs(expm(-1j * h_una(n, T) * s)[0, 2]) ** 2 for T in (Th, Th + np.pi)) for s in ss])
            err = np.abs(Cs - pred).max()
            chk.check(err < 1e-12, f"anillo {n}, Theta = {Th}: C_2(s) = (|U_Theta|^2 + |U_Theta+pi|^2)/2, error {err:.1e}")
            filas.append(("formula_JW", n, Th, err))
    ths = np.array([0.5, 1.0, np.pi / 2, 2.0])
    for n in (5, 6):
        d = np.array([momentos(n, anillo(n), np.full(n, t / n), np.ones(n), 0, 2 * n)[0][2 * n] for t in ths])
        k = [primer_orden(n, anillo(n), np.full(n, t / n), np.ones(n), 0, 2 * n) for t in ths]
        r = d / (1 - np.cos(2 * ths))
        chk.check(all(x == 2 * n for x in k) and np.ptp(r) < 1e-8 * abs(r).max(),
                  f"anillo {n}, j = 0: primer orden {k} = 2n; Delta mu_2n / (1 - cos 2 Theta) = {r[0]:.6g} para todo Theta")
        kp = primer_orden(n, anillo(n), np.full(n, np.pi / n), np.ones(n), 0, 2 * n + 4)
        chk.check(kp is None, f"anillo {n}, Theta = pi: sin dependencia hasta k = {2 * n + 4}")
        filas += [("dos_vueltas", n, t, x) for t, x in zip(ths, r)]
    for n in (4, 8):
        k = primer_orden(n, anillo(n), np.full(n, 1.0 / n), np.ones(n), 0, 2 * n + 2)
        chk.check(k == 2 * n, f"anillo {n}, Theta = 1, j = 0: primer orden {k} = 2n")
    write_csv("modular_xx_anillo.csv", ("magnitud", "n", "Theta", "valor"), filas)


def cota(chk):
    rng = np.random.default_rng(11)
    filas, tot, ok, igual = [], 0, 0, 0
    for nombre, (n, edges) in GRAFOS.items():
        m = len(edges)
        for patron in [rng.uniform(0.3, 2.8, m) for _ in range(2)] + [np.eye(m)[i] * 1.3 for i in range(m)]:
            nt = no_triviales(n, edges, patron)
            if not nt:
                continue
            amps = rng.uniform(0.5, 1.5, m)
            for j in range(n):
                vs, v2 = vstar(nt, j), vstar2(n, edges, patron, j)
                c = min(2 * vs, 2 * v2 - 2) if v2 else 2 * vs
                k = primer_orden(n, edges, patron, amps, j, c + 2)
                tot += 1
                ok += k is not None and k >= c
                igual += k == c
                filas.append((nombre, " ".join(f"{x:.3f}" for x in patron), j, vs, v2 if v2 else "", c, k if k else ""))
    chk.check(ok == tot, f"cota XX min(2V*, 2V*_2 - 2): se cumple en {ok}/{tot} casos, igualdad en {igual}")
    n, edges = GRAFOS["K4"]
    phis = np.array([0, 0, 0, 0, 0, 1.3])
    nt = no_triviales(n, edges, phis)
    k = primer_orden(n, edges, phis, np.ones(6), 1, 10)
    chk.check(vstar(nt, 1) == 4 and vstar2(n, edges, phis, 1) == 4 and k == 6,
              f"K4, fase solo en la arista 23, j = 1: V* = 4, V*_2 = 4, primer orden {k} = 2V* - 2 (no 2V*)")
    write_csv("modular_xx.csv", ("grafo", "fases", "j", "Vstar", "Vstar2", "cota", "k_min"), filas)


def palabras(chk):
    """Palabras Q_1..Q_k (Q = sigma_u sigma_w) que llevan Z_0 a Z_j anticonmutando en cada paso, en el triángulo."""
    P = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    tri = [(0, 1), (1, 2), (2, 0)]

    def buscar(j, k, tipos):
        Qs = [((u, w), {u: P[a], w: P[b]}) for (u, w) in tri for a in tipos for b in tipos]
        sols = []

        def rec(s, pal):
            if len(pal) == k:
                if s == {j: P["Z"]}:
                    sols.append(list(pal))
                return
            for e, q in Qs:
                if sum(s.get(v, (0, 0))[0] * q[v][1] + s.get(v, (0, 0))[1] * q[v][0] for v in q) % 2:
                    t = dict(s)
                    for v, (x, z) in q.items():
                        a = t.get(v, (0, 0))
                        t[v] = ((a[0] + x) % 2, (a[1] + z) % 2)
                    pal.append(e)
                    rec({v: f for v, f in t.items() if f != (0, 0)}, pal)
                    pal.pop()
        rec({0: P["Z"]}, [])
        return sols
    filas = []
    for j in (0, 1):
        n_xx = sum(set(w) >= set(tri) for w in buscar(j, 4, "XY"))
        n_xxz = sum(set(w) >= set(tri) for w in buscar(j, 4, "XYZ"))
        chk.check(n_xx == 0 and n_xxz > 0,
                  f"triángulo, j = {j}, k = 4 = 2V* - 2: palabras con el ciclo, XX {n_xx}, XXZ {n_xxz}")
        filas.append((j, 4, n_xx, n_xxz))
    write_csv("modular_xx_palabras.csv", ("j", "k", "con_ciclo_XX", "con_ciclo_XXZ"), filas)


def main():
    chk = Checker("modular_xx")
    anillos(chk)
    palabras(chk)
    cota(chk)
    chk.finish()


if __name__ == "__main__":
    main()

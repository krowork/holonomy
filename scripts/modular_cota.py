"""Nota «Tiempo modular y holonomía», secciones 3, 4 y 6: orden en que la holonomía entra en el flujo modular.

Giro puro: K_phi = sum_e a_e (cos phi_e A_e + sin phi_e B_e) + c_e C_e, las mismas amplitudes y solo las fases. Con
V* = min |V(C) u {0, j}| sobre los ciclos simples con holonomía no trivial:
  1. Teorema 1: mu_k = tr(ad_K^k(Z_0) Z_j)/N no depende de las fases para k < 2V* - 2. Primer k con dependencia, por
     momentos exactos densos, en anillos de 6 y 8 y escaleras periódicas 2 x 4 y 2 x 5 (tabla tab:momentos): igual a
     2V* - 2 con c_e != 0 (también a = 0,7, c = 0,3); con XX puro (c = 0) es 2V*.
  2. Proposición 2: c_{p,k} = tr(K^p ad_K^k(Z_0) Z_j)/N no depende de las fases si k + 2p < 2V* - 2; el mínimo de
     k + 2p con dependencia es 2V* - 2, la parte estática (k = 0) empieza en p >= V*, y Z(phi) en p = l (ciclo no trivial
     más corto).
  3. Proposición 6: en la máxima entropía, x(v, phi) - giro(x(v, 0)) = O(v^{l-1}); pendientes log-log en
     v = 0,02-0,15 (tabla tab:dx). Anillo y escalera se resuelven por invariancia por traslación (3 y 9 multiplicadores).

Escribe data/modular_cota_{momentos,pk,dx}.csv.

Uso: (ulimit -v 8000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/modular_cota.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np

from fisher_kubo_mori import MaxEnt, escalera
from holonomy.paulis import pauli_string
from holonomy.report import write_csv


def dense(o):
    return np.asarray(o.todense() if hasattr(o, "todense") else o, dtype=complex)


def zdiag(n, j):
    """Diagonal de Z_j (qubit 0 = bit más significativo, como en pauli_string)."""
    return 1.0 - 2 * ((np.arange(2 ** n) >> (n - 1 - j)) & 1)


def terminos(n, i, j):
    A = -(pauli_string(n, ((i, 0), (j, 0))) + pauli_string(n, ((i, 1), (j, 1))))
    B = pauli_string(n, ((i, 0), (j, 1))) - pauli_string(n, ((i, 1), (j, 0)))
    C = -pauli_string(n, ((i, 2), (j, 2)))
    return A, B, C


def K_giro(n, edges, phis, a=1.0, c=1.0):
    K = 0
    for (i, j), f in zip(edges, phis):
        A, B, C = terminos(n, i, j)
        K = K + a * (np.cos(f) * A + np.sin(f) * B) + c * C
    return dense(K)


def anillo(n):
    return [(i, (i + 1) % n) for i in range(n)]


def ops_anillo(n):
    """A, B, C sumados sobre las aristas del anillo (multiplicadores iguales por traslación)."""
    tot = [0, 0, 0]
    for i, j in anillo(n):
        tot = [t + o for t, o in zip(tot, terminos(n, i, j))]
    return np.array([dense(o) for o in tot])


def ops_clases(L):
    """Escalera 2 x L: O[clase, operador] sumado en la clase (0 pata inferior, 1 superior, 2 peldaño)."""
    n = 2 * L
    O = np.zeros((3, 3, 2 ** n, 2 ** n), dtype=complex)
    for e, (i, j) in enumerate(escalera(L)[0]):
        cl = 0 if e < L else (1 if e < 2 * L else 2)
        for k, o in enumerate(terminos(n, i, j)):
            O[cl, k] += dense(o)
    return O.reshape(9, 2 ** n, 2 ** n)


def simetrico(O):
    me = MaxEnt.__new__(MaxEnt)
    me.O = O
    return me


def objetivo_escalera(L, v, t):
    return np.concatenate([L * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]) for f in (t / 2, -t / 2, 0.0)])


def girar(y0, t):
    """Multiplicadores de t = 0 de la escalera con la fase de cada clase: (a, 0, c) -> (a cos f, a sin f, c)."""
    y = y0.reshape(3, 3).copy()
    for cl, f in enumerate((t / 2, -t / 2, 0.0)):
        a = y[cl, 0]
        y[cl, 0], y[cl, 1] = a * np.cos(f), a * np.sin(f)
    return y.ravel()


def ciclos(n, edges, phis):
    """Ciclos simples (conjunto de vértices) con su holonomía, por DFS desde el vértice mínimo."""
    ady = {v: [] for v in range(n)}
    for (i, j), f in zip(edges, phis):
        ady[i].append((j, f))
        ady[j].append((i, -f))
    out, vistos = [], set()

    def dfs(s, v, camino, fase):
        for w, f in ady[v]:
            if w == s and len(camino) >= 3:
                clave = (frozenset(camino), min(camino[1], camino[-1]))
                if clave not in vistos:
                    vistos.add(clave)
                    out.append((frozenset(camino), fase + f))
            elif w > s and w not in camino:
                dfs(s, w, camino + [w], fase + f)
    for s in range(n):
        dfs(s, s, [s], 0.0)
    return out


def no_triviales(n, edges, phis):
    return [V for V, h in ciclos(n, edges, phis) if abs(np.angle(np.exp(1j * h))) > 1e-9]


def vstar(nt, j):
    return min(len(V | {0, j}) for V in nt)


def casos_momentos():
    out = []
    for n, Th in ((6, np.pi), (6, 1.0), (8, np.pi)):
        out.append((f"anillo {n}, Theta={Th:.4f}", n, anillo(n), np.full(n, Th / n), list(range(n // 2 + 1)), 1.0, 1.0))
    for L, t in ((4, np.pi / 2), (4, np.pi), (5, 4 * np.pi / 5)):
        edges, u = escalera(L)
        out.append((f"escalera 2x{L}, t={t:.4f}", 2 * L, edges, t * u, [0, 1, 2, L, L + 1, L + 2], 1.0, 1.0))
    edges, u = escalera(4)
    out.append(("escalera 2x4, t=1, a=0,7, c=0,3", 8, edges, u, [0, 1, 2, 4, 5, 6], 0.7, 0.3))
    out.append(("escalera 2x4, t=1, XX (c=0)", 8, edges, u, [0, 1, 2, 4, 5, 6], 1.0, 0.0))
    return out


def momentos(chk):
    filas = []
    for nombre, n, edges, phis, js, a, c in casos_momentos():
        N = 2 ** n
        K1, K0 = K_giro(n, edges, phis, a, c), K_giro(n, edges, 0 * phis, a, c)
        nt = no_triviales(n, edges, phis)
        kmax = 2 * max(vstar(nt, j) for j in js) + 2
        X1 = X0 = np.diag(zdiag(n, 0)).astype(complex)
        D = {j: [] for j in js}
        for k in range(kmax + 1):
            for j in js:
                m1 = (X1.diagonal() * zdiag(n, j)).sum().real / N
                m0 = (X0.diagonal() * zdiag(n, j)).sum().real / N
                D[j].append((abs(m1 - m0), abs(m0)))
            X1, X0 = K1 @ X1 - X1 @ K1, K0 @ X0 - X0 @ K0
        for j in js:
            cota = 2 * vstar(nt, j) - 2
            kmin = next(k for k, (d, m0) in enumerate(D[j]) if d > 1e-10 * (1 + m0))
            esperado = cota + 2 if c == 0 else cota
            chk.check(kmin == esperado,
                      f"{nombre}, j = {j}: primer orden con dependencia k = {kmin}, 2V* - 2 = {cota}"
                      + (" (XX puro: 2V*)" if c == 0 else ""))
            filas.append((nombre, j, cota, kmin))
    write_csv("modular_cota_momentos.csv", ("caso", "j", "2Vstar_menos_2", "k_min"), filas)


def coeficientes(chk):
    filas = []
    casos = [(f"anillo 6, Theta={Th:.4f}", 6, anillo(6), np.full(6, Th / 6), range(4), 8, 12, 1.0, 1.0)
             for Th in (np.pi, 1.0)]
    for L, t in ((4, np.pi / 2), (4, np.pi)):
        edges, u = escalera(L)
        casos.append((f"escalera 2x{L}, t={t:.4f}", 8, edges, t * u, [0, 1, 2, 4, 6], 6, 9, 1.0, 1.0))
    edges, u = escalera(4)
    casos.append(("escalera 2x4, t=1, a=0,7, c=0,3", 8, edges, u, [0, 1, 2, 4, 6], 6, 9, 0.7, 0.3))
    for nombre, n, edges, phis, js, P, KK, a, c in casos:
        N = 2 ** n
        K1, K0 = K_giro(n, edges, phis, a, c), K_giro(n, edges, 0 * phis, a, c)
        nt = no_triviales(n, edges, phis)
        ell = min(len(V) for V in nt)
        A1 = A0 = np.eye(N, dtype=complex)
        pz = None
        for p in range(P + 1):
            if abs(np.trace(A1) - np.trace(A0)) > 1e-9 * (1 + abs(np.trace(A0))):
                pz = p
                break
            A1, A0 = K1 @ A1, K0 @ A0
        chk.check(pz == ell, f"{nombre}: Z(phi) depende desde p = {pz}, l = {ell}")
        Z0 = np.diag(zdiag(n, 0)).astype(complex)
        X1, X0 = [Z0], [Z0]
        for k in range(KK):
            X1.append(K1 @ X1[-1] - X1[-1] @ K1)
            X0.append(K0 @ X0[-1] - X0[-1] @ K0)
        for j in js:
            Vs = vstar(nt, j)
            M1 = M0 = np.diag(zdiag(n, j)).astype(complex)
            dep = []
            for p in range(P + 1):
                for k in range(KK + 1):
                    c1, c0 = np.trace(X1[k] @ M1) / N, np.trace(X0[k] @ M0) / N
                    if abs(c1 - c0) > 1e-9 * (1 + abs(c0) + abs(c1)):
                        dep.append((p, k))
                M1, M0 = M1 @ K1, M0 @ K0
            fr = min(k + 2 * p for p, k in dep)
            pest = min((p for p, k in dep if k == 0), default=None)
            borde = sorted((p, k) for p, k in dep if k + 2 * p == fr)
            chk.check(fr == 2 * Vs - 2 and (pest is None or pest >= Vs),
                      f"{nombre}, j = {j}: min(k + 2p) con dependencia = {fr} = 2V* - 2 = {2 * Vs - 2}, en {borde}; "
                      f"parte estática desde p = {pest} (>= V* = {Vs}; None: más allá de p = {P})")
            filas.append((nombre, j, 2 * Vs - 2, fr, " ".join(f"{p}:{k}" for p, k in borde), pest, ell, pz))
    write_csv("modular_cota_pk.csv", ("caso", "j", "2Vstar_menos_2", "min_k_2p", "borde_p_k", "p_estatico", "l", "p_Z"),
              filas)


def renormalizacion(chk):
    vs = np.array([0.02, 0.03, 0.05, 0.07, 0.1, 0.15])
    filas = []
    for n in (6, 8):
        me = simetrico(ops_anillo(n))
        f = np.pi / n
        d = []
        for v in vs:
            y0 = me.resuelve(n * v * np.array([2.0, 0.0, 1.0]))[0]
            y1 = me.resuelve(n * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]))[0]
            d.append(np.abs(y1 - np.array([y0[0] * np.cos(f), y0[0] * np.sin(f), y0[2]])).max())
        filas += [(f"anillo {n}, Theta=pi", n, v, dv) for v, dv in zip(vs, d)]
    me = simetrico(ops_clases(4))
    for t in (np.pi, np.pi / 2):
        d = []
        for v in vs:
            y0 = me.resuelve(objetivo_escalera(4, v, 0.0))[0]
            y1 = me.resuelve(objetivo_escalera(4, v, t), x0=girar(y0, t))[0]
            d.append(np.abs(y1 - girar(y0, t)).max())
        filas += [(f"escalera 2x4, t={t:.4f}", 4, v, dv) for v, dv in zip(vs, d)]
    for nombre in dict.fromkeys(f[0] for f in filas):
        ell = next(f[1] for f in filas if f[0] == nombre)
        d = np.array([f[3] for f in filas if f[0] == nombre])
        pend = np.diff(np.log(d)) / np.diff(np.log(vs))
        chk.check(np.all(np.abs(pend - (ell - 1)) < 0.3) and abs(pend[0] - (ell - 1)) < 0.12,
                  f"{nombre}: |dx| ~ v^(l-1), l = {ell}: pendientes locales {np.round(pend, 2)}")
    write_csv("modular_cota_dx.csv", ("caso", "l", "v", "dx_max"), filas)


def main():
    chk = Checker("modular_cota")
    momentos(chk)
    coeficientes(chk)
    renormalizacion(chk)
    chk.finish()


if __name__ == "__main__":
    main()

"""Estrellas y árboles sin holonomía (exploración).

Un árbol no tiene ciclos: por el lema de invariantes del texto de escaleras
hay un gauge con T^e = -I en todas las aristas, y v_max no depende de los
datos. En la estrella K_{1,d}, la proposición de la estrella da
v <= (d+2)/(3d). Comprobaciones:
  1. la cota se alcanza: el estado rho = Pi / tr Pi, con Pi el proyector sobre
     las hojas en el subespacio simétrico (S = d/2) acopladas al centro a espín
     total J = (d-1)/2, tiene en cada arista las 15 correlaciones de
     rho_e((d+2)/(3d)) y marginales de un qubit I/2 (d = 1..6, <= 1e-12);
  2. el primal reducido por U(1) (holonomy.sdp.primal_u1_graph) da
     (d+2)/(3d) en la estrella (d = 1..7, <= 1e-6);
  3. en todos los árboles no isomorfos de n = 2..7 vértices, v_max <= (D+2)/(3D),
     con D el grado máximo (la proposición de la estrella). La igualdad no vale
     en general: se dan los árboles en los que v_max < (D+2)/(3D) (el primero, el
     camino de 5 sitios). En n = 8 el primal reducido por U(1) pide ~8 GB.
Se comparan además los caminos con los anillos sin holonomía (dual del anillo):
el camino de 2m-1 y el de 2m sitios con el anillo de 2m.

Escribe data/tree_star.csv (n, aristas, grado máximo, v_max, (D+2)/(3D)).

Uso: (ulimit -v 6000000; python scripts/tree_star.py [nmax])   por defecto nmax = 7, ~3 min
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import sys
from functools import reduce

import numpy as np

from holonomy import dual, xxz
from holonomy.report import write_csv
from holonomy.sdp import primal_u1_graph

TOL = 1e-6
PAULI = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex))


def bound(d):
    return (d + 2) / (3 * d)


def op(n, d):
    return reduce(np.kron, [d.get(i, np.eye(2)) for i in range(n)])


def star_state(d):
    """Pi / tr Pi en el centro 0 y las hojas 1..d."""
    n = d + 1
    S = [sum(op(n, {i: P / 2}) for i in range(1, n)) for P in PAULI]
    J = [S[a] + op(n, {0: PAULI[a] / 2}) for a in range(3)]
    S2 = sum(M @ M for M in S)
    J2 = sum(M @ M for M in J)
    s, j = d / 2, (d - 1) / 2
    w, V = np.linalg.eigh(S2)
    Psym = V[:, np.abs(w - s * (s + 1)) < 1e-9]
    Psym = Psym @ Psym.conj().T
    w, V = np.linalg.eigh(J2)
    PJ = V[:, np.abs(w - j * (j + 1)) < 1e-9]
    PJ = PJ @ PJ.conj().T
    Pi = PJ @ Psym
    return Pi / np.trace(Pi).real


def canonical(adj):
    """Forma canónica de un árbol no enraizado (AHU desde el centro)."""
    n = len(adj)
    deg = [len(a) for a in adj]
    leaves = [i for i in range(n) if deg[i] <= 1]
    left = n
    while left > 2:
        left -= len(leaves)
        new = []
        for u in leaves:
            for w in adj[u]:
                deg[w] -= 1
                if deg[w] == 1:
                    new.append(w)
            deg[u] = 0
        leaves = new

    def enc(u, parent):
        return "(" + "".join(sorted(enc(w, u) for w in adj[u] if w != parent)) + ")"

    return min(enc(c, -1) for c in leaves)


def trees(nmax):
    """Árboles no isomorfos de 2..nmax vértices, como listas de aristas."""
    level = {canonical([[1], [0]]): [(0, 1)]}
    out = {2: list(level.values())}
    for n in range(3, nmax + 1):
        nxt = {}
        for edges in level.values():
            for u in range(n - 1):
                e = edges + [(u, n - 1)]
                adj = [[] for _ in range(n)]
                for a, b in e:
                    adj[a].append(b)
                    adj[b].append(a)
                nxt.setdefault(canonical(adj), e)
        level = nxt
        out[n] = list(level.values())
    return out


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    chk = Checker("tree_star")

    print("== Estado explícito de la estrella")
    for d in range(1, 7):
        n = d + 1
        rho = star_state(d)
        v = bound(d)
        err = max(abs(np.trace(rho @ op(n, {0: PAULI[a], j: PAULI[b]})).real - (-v if a == b else 0.0))
                  for j in range(1, n) for a in range(3) for b in range(3))
        err1 = max(abs(np.trace(rho @ op(n, {i: P})).real) for i in range(n) for P in PAULI)
        psd = np.linalg.eigvalsh(rho).min()
        chk.check(err < 1e-12 and err1 < 1e-12 and psd > -1e-12,
                  f"K_1,{d}: v = {v:.6f}; |correlaciones - Werner| = {err:.1e}, |marginales| = {err1:.1e}, "
                  f"autovalor mínimo {psd:.1e}")

    print("== Primal en estrellas y árboles")
    rows, worst, below, paths = [], 0.0, 0, {}
    for n, ts in trees(nmax).items():
        for edges in ts:
            D = max(sum(1 for e in edges if u in e) for u in range(n))
            v, _, st = primal_u1_graph(n, edges, [0.0] * len(edges), return_blocks=True, **CLARABEL_OPTS)
            star = D == n - 1
            rows.append((n, " ".join(f"{a}-{b}" for a, b in edges), D, v, bound(D)))
            worst = max(worst, v - bound(D))
            if D <= 2 and n > 2:
                paths[n] = v
            if star:
                chk.check(abs(v - bound(D)) < TOL, f"K_1,{D}: primal = {v:.8f}, (d+2)/(3d) = {bound(D):.8f}"
                          f"{chk.status_note(st)}")
            elif abs(v - bound(D)) >= TOL:
                below += 1
                chk.info(f"n = {n}, aristas {edges}: v = {v:.8f}, (D+2)/(3D) = {bound(D):.8f} (D = {D})")
        chk.info(f"n = {n}: {len(ts)} árboles")
    write_csv("tree_star.csv", ["n", "aristas", "grado_max", "v_max", "cota_estrella"], rows)
    chk.check(worst < TOL, f"{len(rows)} árboles (n = 2..{nmax}): v_max <= (D+2)/(3D) en todos "
              f"(máx v - cota = {worst:.1e}); por debajo en {below}")
    for n, v in sorted(paths.items()):
        m = 2 * ((n + 1) // 2)
        vr = dual.v_free(xxz.eps, m, 0.0)[0]
        chk.info(f"camino de {n} sitios: {v:.8f}; anillo de {m} sin holonomía: {vr:.8f}, dif = {v - vr:+.1e}")
    chk.finish()


if __name__ == "__main__":
    main()

"""Borrador «Parches causales»: nervios con un qudit SU(d) por parche, sin holonomía.

Marginal de arista rho(v) = (1 - v) I/d^2 + v P_A/binom(d, 2) (`qudit.werner`), separable si y solo si v <= 1/(d + 1).
En un grafo transitivo en aristas, sin datos (Allerstorfer, ec. 58; Proposición `sinhol` del artículo de rigidez),
    v_max = d/(d + 1) (1/d - E_0(sum_e P_e)/|E|),
con P el intercambio. En la clique K_m, E_0 es la menor suma de contenidos de un diagrama de Young de m cajas con a lo
sumo d filas: v_max(K_m) = 1 si m <= d, y v_max(K_3) = 1/3 con d = 2.

Se calcula E_0 con Lanczos en el sector de colores más equilibrado, que contiene todo el espectro: un peso mu aparece en
la irrep lambda si y solo si mu está dominado por lambda (Kostka > 0), y la partición más equilibrada de n en d partes
está dominada por todas. Si el espacio es pequeño, se recorren todos los sectores como control. Control positivo: con
d = 2 se recuperan 1/3 (octaedro), 0,2750173 (icosaedro) y 7/27 (toro 3 x 3).

Escribe data/parches_qudits.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 python scripts/parches_qudits.py)
"""
import _common  # noqa: F401

import itertools
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from holonomy import sphere
from holonomy.report import write_csv
import parches_nervios as pn


def clique(m):
    return m, tuple(itertools.combinations(range(m), 2))


def contenidos_min(m, d):
    """Menor suma de contenidos (columna - fila) entre los diagramas de m cajas con <= d filas."""
    mejor = None

    def parts(n, maxp, filas):
        if n == 0:
            yield ()
            return
        if filas == 0:
            return
        for p in range(min(n, maxp), 0, -1):
            for resto in parts(n - p, p, filas - 1):
                yield (p,) + resto

    for lam in parts(m, m, d):
        c = sum(j - i for i, fila in enumerate(lam) for j in range(fila))
        mejor = c if mejor is None else min(mejor, c)
    return mejor


def sector(n, d, counts):
    """Estados (tuplas de colores) con los números de color dados, y su índice."""
    estados = sorted(set(itertools.permutations(sum(([a] * c for a, c in enumerate(counts)), []))))
    return estados, {s: k for k, s in enumerate(estados)}


def suma_P(n, d, edges, counts):
    estados, idx = sector(n, d, counts)
    R, C = [], []
    for k, s in enumerate(estados):
        for i, j in edges:
            t = list(s)
            t[i], t[j] = t[j], t[i]
            R.append(idx[tuple(t)])
            C.append(k)
    M = sp.csr_matrix((np.ones(len(R)), (R, C)), shape=(len(estados), len(estados)))
    return M


def e0(n, d, edges, todos):
    if todos:
        sectores = [c for c in itertools.product(range(n + 1), repeat=d) if sum(c) == n and list(c) == sorted(c, reverse=True)]
    else:
        base = [n // d + (1 if a < n % d else 0) for a in range(d)]
        sectores = [tuple(base)]
    mejor = np.inf
    for c in sectores:
        M = suma_P(n, d, edges, c)
        if M.shape[0] <= 3000:
            w = np.linalg.eigvalsh(M.toarray())[0]
        else:
            w = eigsh(M, k=1, which="SA", tol=1e-12, ncv=40)[0][0]
        mejor = min(mejor, w)
    return mejor


def main():
    grafos = [("triángulo K3", *clique(3)), ("tetraedro K4", *clique(4))]
    for nombre in ("octaedro", "icosaedro"):
        p = sphere.polyhedron(nombre)
        grafos.append((nombre, p.n, p.edges))
    g, *_ = pn.toro(3, 3)
    grafos.append(("toro 3x3", g.n, g.edges))
    filas = []
    for d in (2, 3, 4):
        for nombre, n, edges in grafos:
            if d**n > 2e7:
                continue
            t0 = time.time()
            E0 = e0(n, d, edges, todos=d**n <= 5000)
            v = d / (d + 1) * (1 / d - E0 / len(edges))
            extra = ""
            if nombre.startswith(("triángulo", "tetraedro")):
                Ec = contenidos_min(n, d)
                assert abs(Ec - E0) < 1e-9, (nombre, d, Ec, E0)
                extra = f" (contenidos: {Ec})"
            umbral = 1 / (d + 1)
            filas.append([nombre, n, len(edges), d, E0, v, umbral, v > umbral + 1e-12])
            print(f"d = {d}, {nombre:13s}: E0(sum P) = {E0:.8f}{extra}, v_max = {v:.7f}, separable hasta {umbral:.4f}: "
                  f"{'entrelazable' if v > umbral + 1e-12 else 'separable'}  ({time.time() - t0:.1f} s)", flush=True)
    write_csv("parches_qudits.csv", ("grafo", "N", "E", "d", "E0_sumaP", "v_max", "umbral_separable", "entrelazable"), filas)


if __name__ == "__main__":
    main()

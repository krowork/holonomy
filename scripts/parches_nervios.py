"""Borrador «Parches causales»: v_max, entropía de máxima entropía y entrelazamiento entre regiones en nervios.

Nervios de recubrimientos finos (con triángulos, así que v_max <= 1/3):
  * toros triangulares L1 x L2 (vecinos +-(1,0), +-(0,1), +-(1,1)), nervios de recubrimientos de un toro plano, sin datos;
  * tetraedro, octaedro e icosaedro, nervios de casquetes de la esfera, sin datos y con datos de curvatura.

Para cada uno:
  1. v_max por el dual de órbitas (`OrbitDual`), con cotas superior e inferior;
  2. S_G(v), entropía del estado de máxima entropía con las marginales de Werner (`Gibbs` de sphere_entropia), en
     v = r v_max y en v = 0,1 y 0,2, y la densidad s = S_G/N;
  3. con N <= 12, el estado denso rho_G = exp(K)/Z construido aparte con matrices de Pauli: residuo de las marginales,
     información mutua I(A:B) entre dos mitades y log-negatividad E_N = ln ||rho^{T_A}||_1 entre ellas. Las marginales de
     dos qubits son separables (v <= 1/3); E_N > 0 dice que el estado global está entrelazado entre las mitades.

Escribe data/parches_nervios_vmax.csv, data/parches_nervios_entropia.csv y data/parches_nervios_mitades.csv.

Uso: (ulimit -v 16000000; OMP_NUM_THREADS=8 python scripts/parches_nervios.py [--grande])
  --grande añade el toro 4 x 4 (16 qubits), solo al paso 1. Los CSV se reescriben tras cada caso.
"""
import _common  # noqa: F401

import csv
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from holonomy import sphere
from holonomy.report import write_csv
import sphere_casquetes as cas
from sphere_entropia import Gibbs

DATA = Path(__file__).resolve().parent.parent / "data"
LN2 = np.log(2)
RS = (0.25, 0.5, 0.75, 0.9)
VS_ABS = (0.1, 0.2)


# ---------------------------------------------------------------------------
# Grafos
# ---------------------------------------------------------------------------

def toro(L1, L2):
    """Toro triangular. Aristas (i, j) con i < j; orbitas por dirección (sin datos, b = 0 por la inversión)."""
    idx = lambda x, y: (x % L1) * L2 + (y % L2)  # noqa: E731
    dirs = ((1, 0), (0, 1), (1, 1))
    edges, dir_of = [], []
    for x in range(L1):
        for y in range(L2):
            for d, (dx, dy) in enumerate(dirs):
                i, j = idx(x, y), idx(x + dx, y + dy)
                e = (min(i, j), max(i, j))
                if e not in edges:
                    edges.append(e)
                    dir_of.append(d)
    order = sorted(range(len(edges)), key=lambda m: edges[m])
    edges = tuple(edges[m] for m in order)
    dir_of = [dir_of[m] for m in order]
    n = L1 * L2
    deg = np.zeros(n, int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    assert (deg == 6).all() and len(edges) == 3 * n, (L1, L2)
    g = sphere.Polyhedron(f"toro{L1}x{L2}", np.zeros((n, 3)), edges, ())
    orbs = []
    for d in range(3):
        mem = [m for m in range(len(edges)) if dir_of[m] == d]
        orbs.append(sphere.Orbit(mem, 1, True, {m: 0 for m in mem}))
    mitad = [idx(x, y) for x in range(L1) for y in range(L2) if y < L2 // 2] if L2 % 2 == 0 else \
        [idx(x, y) for x in range(L1) for y in range(L2) if x < L1 // 2 + (L1 % 2)]
    return g, np.zeros(len(edges)), orbs, sorted(mitad)


def poliedro(nombre, caso):
    p = sphere.polyhedron(nombre)
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    ph = cas.data_phases(p, caso)
    # mitad: los vértices con z > 0 y, en el plano, x > 0 (o y > 0), para un corte en dos mitades iguales
    V = p.verts
    clave = V[:, 2] + 1e-3 * V[:, 0] + 1e-6 * V[:, 1]
    mitad = sorted(np.argsort(-clave)[: p.n // 2].tolist())
    return p, ph, orbs, mitad


# ---------------------------------------------------------------------------
# Estado denso (independiente del código por sectores)
# ---------------------------------------------------------------------------

I2 = np.eye(2)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1.0 + 0j, -1.0])


def dos(n, i, Pi, j, Pj):
    ops = [sp.identity(2, format="csr")] * n
    ops[i], ops[j] = sp.csr_matrix(Pi), sp.csr_matrix(Pj)
    return reduce(lambda a, b: sp.kron(a, b, format="csr"), ops)


def operador_K(graph, phis, coef):
    """K = sum_e a_e (cos A + sin B) + b_e (-sin A + cos B) + c_e C, con A = -(XX+YY), B = XY-YX, C = -ZZ."""
    a, b, c = coef
    n = graph.n
    K = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for (i, j), f, ae, be, ce in zip(graph.edges, phis, a, b, c, strict=True):
        A = -(dos(n, i, PX, j, PX) + dos(n, i, PY, j, PY))
        B = dos(n, i, PX, j, PY) - dos(n, i, PY, j, PX)
        C = -dos(n, i, PZ, j, PZ)
        K += (ae * np.cos(f) - be * np.sin(f)) * A + (ae * np.sin(f) + be * np.cos(f)) * B + ce * C
    return K.toarray()


def gibbs_denso(K):
    w, U = np.linalg.eigh(K)
    p = np.exp(w - w.max())
    p /= p.sum()
    return (U * p) @ U.conj().T


def residuo_denso(rho, graph, phis, v):
    n, err = graph.n, 0.0
    for (i, j), f in zip(graph.edges, phis):
        A = -(dos(n, i, PX, j, PX) + dos(n, i, PY, j, PY))
        B = dos(n, i, PX, j, PY) - dos(n, i, PY, j, PX)
        C = -dos(n, i, PZ, j, PZ)
        val = np.array([O.multiply(rho.T).sum().real for O in (A, B, C)])
        err = max(err, *np.abs(val - v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0])))
    return err


def reduce_a(rho, n, keep):
    keep = sorted(keep)
    rest = [q for q in range(n) if q not in keep]
    t = rho.reshape([2] * (2 * n))
    perm = keep + rest + [n + q for q in keep] + [n + q for q in rest]
    t = t.transpose(perm).reshape(2**len(keep), 2**len(rest), 2**len(keep), 2**len(rest))
    return np.einsum("ajbj->ab", t)


def entropia(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return float(-(w * np.log(w)).sum())


def log_negatividad(rho, n, A):
    A = sorted(A)
    B = [q for q in range(n) if q not in A]
    t = rho.reshape([2] * (2 * n))
    perm = A + B + [n + q for q in A] + [n + q for q in B]
    t = t.transpose(perm).reshape(2**len(A), 2**len(B), 2**len(A), 2**len(B))
    pt = t.transpose(2, 1, 0, 3).reshape(2**n, 2**n)
    return float(np.log(np.abs(np.linalg.eigvalsh(pt)).sum()))


# ---------------------------------------------------------------------------

def vmax_de(nombre, graph, phis, orbs, serie):
    if nombre in serie:
        return serie[nombre], 0.0, "sphere_series.csv"
    d = sphere.OrbitDual(graph, phis, orbs, dense_max=3000)
    res = d.minimize(target=2e-9)
    return res.v, res.gap, "OrbitDual"


def main():
    grande = "--grande" in sys.argv
    serie = {}
    for r in csv.DictReader(open(DATA / "sphere_series.csv")):
        serie[f"{r['nombre']} trivial"] = float(r["v_trivial"])
        serie[f"{r['nombre']} curvado"] = float(r["v_curvado"])
    casos = [(f"{nom} {caso}", *poliedro(nom, caso)) for nom in ("tetraedro", "octaedro", "icosaedro")
             for caso in ("trivial", "curvado")]
    tamaños = [(3, 3), (3, 4), (3, 5)] + ([(4, 4)] if grande else [])
    casos += [(f"toro {L1}x{L2}", *toro(L1, L2)) for L1, L2 in tamaños]

    filas_v, filas_s, filas_m = [], [], []
    for nombre, g, ph, orbs, mitad in casos:
        t0 = time.time()
        vG, gap, fuente = vmax_de(nombre, g, ph, orbs, serie)
        filas_v.append([nombre, g.n, len(g.edges), vG, gap, fuente])
        print(f"{nombre}: N = {g.n}, v_max = {vG:.7f} (gap {gap:.1e}, {fuente})", flush=True)
        if g.n > 15:
            guarda(filas_v, filas_s, filas_m)
            continue
        G = Gibbs(g, ph, orbs)
        y = None
        vs = [(r, r * vG) for r in RS] + [(None, v) for v in VS_ABS if v < vG]
        for r, v in sorted(vs, key=lambda t: t[1]):
            y, S, res = G.resuelve(v, y0=y)
            s = S / g.n
            filas_s.append([nombre, g.n, vG, "" if r is None else r, v, S, s, s / LN2, res])
            print(f"   v = {v:.4f}{'' if r is None else f' ({r} v_max)'}: S = {S:.6f}, S/N = {s:.6f}, "
                  f"f = {s / LN2:.4f}, residuo {res:.1e}", flush=True)
            if g.n <= 12 and r is not None:
                coef = G.dual.edge_coefficients(y)
                rho = gibbs_denso(operador_K(g, ph, coef))
                rd = residuo_denso(rho, g, ph, v)
                SA = entropia(reduce_a(rho, g.n, mitad))
                SB = entropia(reduce_a(rho, g.n, [q for q in range(g.n) if q not in mitad]))
                Stot = entropia(rho)
                I = SA + SB - Stot
                EN = log_negatividad(rho, g.n, mitad)
                corte = sum(1 for i, j in g.edges if (i in mitad) != (j in mitad))
                filas_m.append([nombre, g.n, len(mitad), corte, r, v, Stot, SA, SB, I, EN, rd])
                print(f"      denso: |S - S_sect| = {abs(Stot - S):.1e}, residuo {rd:.1e}, aristas de corte {corte}, "
                      f"I(A:B) = {I:.5f}, E_N = {EN:.5f}", flush=True)
        print(f"   ({time.time() - t0:.1f} s)", flush=True)
        guarda(filas_v, filas_s, filas_m)


def guarda(filas_v, filas_s, filas_m):
    write_csv("parches_nervios_vmax.csv", ("grafo", "N", "E", "v_max", "gap", "fuente"), filas_v)
    write_csv("parches_nervios_entropia.csv",
              ("grafo", "N", "v_max", "r", "v", "S", "S_por_parche", "f", "residuo"), filas_s)
    write_csv("parches_nervios_mitades.csv",
              ("grafo", "N", "NA", "aristas_corte", "r", "v", "S", "SA", "SB", "I", "E_N", "residuo"), filas_m)


if __name__ == "__main__":
    main()

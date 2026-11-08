"""Entropía máxima de una vista en el rango de conflicto: ¿es compatible con una hoja casi saturada?

Una vista de una región R con marginales de Werner a v tiene S <= S_Gibbs(R, v), la del estado exp(W(y))/Z con esas marginales,
y S_Gibbs baja con v. Así, en el rango v(G) < v <= v(R), toda vista tiene S <= S_Gibbs(R, v(G)). Se mide la fracción
f = S_Gibbs/(|R| ln 2); (H1) y la saturación piden f cercano a 1.

Gibbs: un multiplicador por órbita de aristas, W(y) de `OrbitDual`, y bloques de Sz k <= n//2, con k y n - k del mismo espectro y
los mismos A_e, B_e, C_e, por R_x(pi) y conjugación, que además anulan Bloch. La forma exponencial y las marginales exactas
caracterizan el máximo de entropía, así que el residuo por arista certifica el resultado sin depender de la simetría. Newton con
hessiano de Kubo-Mori truncado a los estados de más peso: el gradiente es exacto y solo se acelera la dirección.

Comprobaciones:
  1. frente a `sphere_cmi.gibbs`, denso y sin simetría, en el cubo y sus casquetes;
  2. orden v^2: 1 - f = 3 |E| v^2 / (2 n ln 2) en v pequeño;
  3. residuo por arista y Bloch nulo por la simetría; S decreciente en v;
  4. cota de subaditividad con un emparejamiento máximo de m aristas: S <= m S_e(v) + (n - 2m) ln 2;
  5. familia de anillos: v(G) del anillo de 4 es 2/3, y el de n -> infinito tiende al de Bethe; con flujo pi, el de 4 es 1/2.

Escribe data/sphere_entropia_casquetes.csv, data/sphere_entropia_poliedros.csv, data/sphere_entropia_cadenas.csv,
data/sphere_entropia_cadenas_flujo.csv; con --dodecaedro añade los casquetes de 15 qubits, y se puede reanudar; con --flujo
rehace solo la familia con flujo pi. Cada modo conserva en el CSV de casquetes las filas de los otros.

Uso: (ulimit -v 25000000; OMP_NUM_THREADS=8 python scripts/sphere_entropia.py [--dodecaedro | --flujo])
"""
import _common  # noqa: F401
from _common import Checker

import csv
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from holonomy import sphere
from holonomy.report import write_csv
import sphere_casquetes as cas
import sphere_cmi as cmi

DATA = Path(__file__).resolve().parent.parent / "data"
LN2 = np.log(2)


def s_arista(v):
    p = np.array([(1 + 3 * v) / 4] + [(1 - v) / 4] * 3)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def emparejamiento_maximo(n, edges):
    """Tamaño del mayor emparejamiento, por máscara de vértices."""
    vecinos = [[] for _ in range(n)]
    for i, j in edges:
        vecinos[i].append(j)
        vecinos[j].append(i)

    @lru_cache(maxsize=None)
    def m(libres):
        if not libres:
            return 0
        i = (libres & -libres).bit_length() - 1
        resto = libres & ~(1 << i)
        mejor = m(resto)
        for j in vecinos[i]:
            if resto >> j & 1:
                mejor = max(mejor, 1 + m(resto & ~(1 << j)))
        return mejor

    return m((1 << n) - 1)


def cota_subaditividad(n, edges, v):
    m = emparejamiento_maximo(n, edges)
    return m * s_arista(v) + (n - 2 * m) * LN2


class Gibbs:
    def __init__(self, graph, phis, orbits, kmax=1200):
        self.dual = sphere.OrbitDual(graph, phis, orbits)
        self.graph, self.phis, self.n = graph, np.asarray(phis, dtype=float), graph.n
        self.kmax = kmax
        self.mult = [1 if 2 * k == self.n else 2 for k in range(self.n // 2 + 1)]

    def _espectro(self, y):
        out = []
        for k in range(self.n // 2 + 1):
            H, d = self.dual._block(k, y)
            W = H.toarray()
            if np.abs(W.imag).max() == 0:
                W = W.real
            W[np.diag_indices_from(W)] += d
            out.append(eigh(W))
        return out

    def _pesos(self, esp):
        lnZ = logsumexp(np.concatenate([E for E, _ in esp] + [E for (E, _), m in zip(esp, self.mult) if m == 2]))
        return lnZ, [np.exp(E - lnZ) for E, _ in esp]

    def medias(self, esp, ps):
        """Medias de los operadores de órbita, sobre los estados de peso apreciable."""
        g = np.zeros(len(self.dual.params))
        for k, ((_, U), p, m) in enumerate(zip(esp, ps, self.mult)):
            c = p > 1e-17
            Uc = U[:, c]
            for q, O in enumerate(self.dual.blocks[k]):
                if isinstance(O, np.ndarray):
                    d = (np.abs(Uc)**2 * O[:, None]).sum(0)
                else:
                    d = np.einsum("ij,ij->j", Uc.conj(), O @ Uc).real
                g[q] += m * (p[c] * d).sum()
        return g

    def hessiano(self, esp, ps, g):
        Q = len(g)
        Hs = np.zeros((Q, Q))
        for k, ((E, U), p, m) in enumerate(zip(esp, ps, self.mult)):
            orden = np.argsort(-p)[:self.kmax]
            c = orden[p[orden] > 1e-14]
            if len(c) == 0:
                continue
            Uc = U[:, c]
            M = []
            for O in self.dual.blocks[k]:
                OU = O[:, None] * Uc if isinstance(O, np.ndarray) else O @ Uc
                M.append(U.conj().T @ OU)
            dE = E[:, None] - E[c][None, :]
            dp = p[:, None] - p[c][None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(np.abs(dE) > 1e-9, dp / np.where(dE == 0, 1, dE), (p[:, None] + p[c][None, :]) / 2)
            # Cada par con un solo estado retenido cuenta en los dos órdenes.
            f = np.full(len(E), 2.0)
            f[c] = 1.0
            w = w * f[:, None]
            for a in range(Q):
                wa = w * M[a].conj()
                for b in range(a, Q):
                    Hs[a, b] += m * (wa * M[b]).sum().real
        Hs = Hs + np.triu(Hs, 1).T
        return Hs - np.outer(g, g)

    def resuelve(self, v, y0=None, tol=1e-9, maxit=120, log=False):
        t = v * self.dual.norm
        y = np.zeros(len(t)) if y0 is None else y0.copy()

        def valor(y):
            esp = self._espectro(y)
            lnZ, ps = self._pesos(esp)
            return lnZ - y @ t, esp, ps

        F, esp, ps = valor(y)
        for it in range(maxit):
            g = self.medias(esp, ps)
            r = g - t
            if np.abs(r).max() < tol:
                break
            Hs = self.hessiano(esp, ps, g)
            paso = np.linalg.lstsq(Hs + 1e-12 * np.trace(Hs) / len(t) * np.eye(len(t)), -r, rcond=1e-12)[0]
            if r @ paso >= 0:
                paso = -r
            paso *= min(1.0, 1.0 / np.abs(paso).max())
            a = 1.0
            while a > 1e-8:
                Fn, espn, psn = valor(y + a * paso)
                if Fn <= F + 1e-4 * a * (r @ paso):
                    break
                a /= 2
            y, F, esp, ps = y + a * paso, Fn, espn, psn
            if log:
                print(f"      it {it}: |r| {np.abs(r).max():.2e}, a {a:.3g}, F {F:.8f}", flush=True)
        S = self.entropia(ps)
        return y, S, self.residuo(esp, ps, v)

    def entropia(self, ps):
        return float(sum(-m * (p[p > 0] * np.log(p[p > 0])).sum() for p, m in zip(ps, self.mult)))

    def residuo(self, esp, ps, v):
        """Mayor error de <A_e>, <B_e>, <C_e> por arista, y de la traza."""
        err = abs(sum(m * p.sum() for p, m in zip(ps, self.mult)) - 1)
        for e, f in zip(self.graph.edges, self.phis, strict=True):
            val = np.zeros(3)
            for k, ((_, U), p, m) in enumerate(zip(esp, ps, self.mult)):
                c = p > 1e-17
                Uc = U[:, c]
                A = sphere.hopping_block(self.n, k, [e], [1.0], [0.0])
                B = sphere.hopping_block(self.n, k, [e], [0.0], [1.0])
                C = sphere.zz_diagonal(self.n, k, [e])
                val[0] += m * (p[c] * np.einsum("ij,ij->j", Uc.conj(), A @ Uc).real).sum()
                val[1] += m * (p[c] * np.einsum("ij,ij->j", Uc.conj(), B @ Uc).real).sum()
                val[2] += m * (p[c] * (np.abs(Uc)**2 * C[:, None]).sum(0)).sum()
            err = max(err, *np.abs(val - v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0])))
        return float(err)


def casquete(pol, g, caso):
    p = sphere.polyhedron(pol)
    vs = cas.subsets(pol, p)[g]
    sub, ph, _ = sphere.induced_subgraph(p, cas.data_phases(p, caso), vs)
    grupo = sphere.subset_group(p, sphere.symmetry_group(p), vs)
    return sub, ph, sphere.edge_orbits(sub, grupo)


def controles(chk):
    for caso in ("trivial", "curvado"):
        p, phis, dat = cmi.cubo(caso)
        orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
        G = Gibbs(p, phis, orbs)
        y = None
        for v in (0.15, 0.3, 0.4):
            y, S, res = G.resuelve(v, y)
            rho, _ = cmi.gibbs(dat, v)
            S0 = cmi.entropia(rho)
            chk.check(abs(S - S0) < 1e-7 and res < 1e-8 and dat.residuo(rho, v) < 1e-8,
                      f"cubo {caso}, v = {v}: S por órbitas {S:.9f}, denso sin simetría {S0:.9f}, residuo {res:.1e}")
    for pol, corte, g, *_ in cas.CORTES[:2]:
        for caso in ("trivial", "curvado"):
            sub, ph, orbs = casquete(pol, g, caso)
            dat = cmi.Datos(sub.n, sub.edges, ph)
            G = Gibbs(sub, ph, orbs)
            for v in (0.2, 0.4):
                _, S, res = G.resuelve(v)
                rho, _ = cmi.gibbs(dat, v)
                S0 = cmi.entropia(rho)
                chk.check(abs(S - S0) < 1e-7 and res < 1e-8 and dat.residuo(rho, v) < 1e-8,
                          f"{pol} {corte} {caso}, {sub.n} qubits, v = {v}: S {S:.9f}, denso {S0:.9f}, residuo {res:.1e}")
    p = sphere.polyhedron("cubo")
    G = Gibbs(p, np.zeros(p.E), sphere.edge_orbits(p, sphere.symmetry_group(p)))
    v = 0.01
    _, S, _ = G.resuelve(v)
    ref = 3 * p.E * v * v / 2
    chk.check(abs((p.n * LN2 - S) / ref - 1) < 1e-2, f"orden v^2 en el cubo: n ln 2 - S = {p.n * LN2 - S:.4e}, 3 |E| v^2/2 = {ref:.4e}")


def fila(chk, nombre, G, n, edges, v, y, datos, previa=None):
    t0 = time.time()
    y, S, res = G.resuelve(v, y)
    cota = cota_subaditividad(n, edges, v)
    baja = previa is None or S <= previa + 1e-9
    chk.check(res < 1e-8 and S <= cota + 1e-9 and baja,
              f"{nombre}: v = {v:.6f}, S = {S:.6f}, f = {S / (n * LN2):.4f}, cota de subaditividad {cota / (n * LN2):.4f}, residuo {res:.1e}, {time.time() - t0:.0f} s")
    return y, S, [*datos, n, v, S, S / (n * LN2), cota / (n * LN2), res]


def corre_casquetes(chk, polis):
    defecto = {(r["poliedro"], r["corte"], r["caso"]): r for r in csv.DictReader(open(DATA / "sphere_casquetes_defecto.csv"))}
    ruta = DATA / "sphere_entropia_casquetes.csv"
    cab = ("poliedro", "corte", "caso", "v_G", "v_R", "delta", "punto", "n", "v", "S", "f", "cota", "residuo")
    # Conserva las filas de los demás poliedros; las de --dodecaedro, además, permiten reanudar.
    filas = [list(r.values()) for r in csv.DictReader(open(ruta))] if ruta.exists() else []
    filas = [r for r in filas if r[0] not in polis or polis == ("dodecaedro",)]
    hechos = {tuple(r[:3]) + (r[6],) for r in filas}
    for pol, corte, g, _ in cas.CORTES:
        if pol not in polis:
            continue
        for caso in ("trivial", "curvado"):
            d = defecto[pol, corte, caso]
            vG, vR, delta = float(d["v_union"]), float(d["v_G1"]), float(d["delta"])
            sub, ph, orbs = casquete(pol, g, caso)
            G = Gibbs(sub, ph, orbs)
            pts = [(f"{fr} v_R", fr * vR) for fr in ((0.1, 0.25, 0.5, 0.7, 0.85) if sub.n < 12 else (0.85,))]
            if delta > 1e-6:
                pts += [("v_G", vG), ("medio", (vG + vR) / 2)]
            pts += [("0.99 v_R", 0.99 * vR)]
            y, prev = None, None
            for nombre, v in sorted(pts, key=lambda x: x[1]):
                if (pol, corte, caso, nombre) in hechos:
                    continue
                y, prev, row = fila(chk, f"{pol} {corte} {caso} [{nombre}]", G, sub.n, sub.edges, v, y,
                                    [pol, corte, caso, vG, vR, delta, nombre], prev)
                filas.append(row)
                write_csv(ruta.name, cab, filas)


def corre_poliedros(chk):
    serie = {r["nombre"]: r for r in csv.DictReader(open(DATA / "sphere_series.csv"))}
    filas = []
    for pol in ("tetraedro", "octaedro", "cubo", "cuboctaedro", "icosaedro", "rombododecaedro"):
        p = sphere.polyhedron(pol)
        orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
        for caso in ("trivial", "curvado"):
            vG = float(serie[pol][f"v_{caso}"])
            G = Gibbs(p, cas.data_phases(p, caso), orbs)
            y, prev = None, None
            for fr in (0.25, 0.5, 0.75, 0.9):
                y, prev, row = fila(chk, f"{pol} {caso} [{fr} v(G)]", G, p.n, p.edges, fr * vG, y, [pol, caso, vG, fr], prev)
                filas.append(row)
    write_csv("sphere_entropia_poliedros.csv", ("poliedro", "caso", "v_G", "v_rel", "n", "v", "S", "f", "cota", "residuo"), filas)


def grafo_camino(m):
    edges = tuple((i, i + 1) for i in range(m - 1))
    return sphere.Polyhedron(f"camino{m}", np.zeros((m, 3)), edges, ())


def orbitas_camino(E):
    """Por la reflexión, b = 0: sin datos, el óptimo y el Gibbs son invariantes bajo SU(2)."""
    out = []
    for e in range((E + 1) // 2):
        mem = sorted({e, E - 1 - e})
        out.append(sphere.Orbit(mem, 1, True, {m: 0 for m in mem}))
    return out


def v_anillo(n, th=0.0):
    """Sin flujo, el testigo de Heisenberg es óptimo; con flujo th (fase th/n por arista) b no se anula y se minimiza."""
    edges = tuple((i, (i + 1) % n) for i in range(n))
    g = sphere.Polyhedron(f"anillo{n}", np.zeros((n, 3)), edges, ())
    if th == 0:
        orb = [sphere.Orbit(list(range(n)), 1, True, {m: 0 for m in range(n)})]
        d = sphere.OrbitDual(g, np.zeros(n), orb, dense_max=3000)
        return d.value(d.heisenberg(), all_blocks=True, add_cuts=False)[0], 0.0
    orb = [sphere.Orbit(list(range(n)), 1, False, {m: 1 for m in range(n)})]
    res = sphere.OrbitDual(g, np.full(n, th / n), orb).minimize(target=2e-8)
    return res.v, res.gap


def corre_cadenas(chk):
    """Familia que crece sin mezclar clases: anillo par de n sitios y caminos de n - 1, sin datos."""
    chk.check(abs(v_anillo(4)[0] - 2 / 3) < 1e-10, f"anillo de 4: v(G) = {v_anillo(4)[0]:.10f}")
    filas = []
    for n in (4, 6, 8, 10, 12, 14, 16):
        m = n - 1
        vG = v_anillo(n)[0]
        g = grafo_camino(m)
        orb = orbitas_camino(m - 1)
        res = sphere.OrbitDual(g, np.zeros(m - 1), orb).minimize(target=2e-8)
        vR = res.v
        chk.check(res.gap < 1e-7 and vG <= vR + 1e-7, f"anillo {n}: v(G) = {vG:.8f}, camino de {m}: {vR:.8f}, delta = {vR - vG:.6f}, dif de cotas {res.gap:.1e}")
        G = Gibbs(g, np.zeros(m - 1), orb)
        y, prev = None, None
        pts = [fr * vR for fr in (0.25, 0.5, 0.75, 0.9)]
        vs = sorted(p for p in pts if p < vG - 1e-6)
        if vR - vG > 1e-6:
            vs += [vG, (vG + vR) / 2]
        for v in vs:
            punto = "v_G" if v == vG else ("medio" if v > vG else f"{v / vR:.2f} v_R")
            y, prev, row = fila(chk, f"camino de {m} [{punto}]", G, m, g.edges, v, y, [n, vG, vR, vR - vG, punto], prev)
            filas.append(row)
        write_csv("sphere_entropia_cadenas.csv", ("n_anillo", "v_G", "v_R", "delta", "punto", "n", "v", "S", "f", "cota", "residuo"), filas)
    v_bethe = (4 * LN2 - 1) / 3
    chk.info(f"Bethe: v(infinito) = {v_bethe:.6f}")


def corre_cadenas_flujo(chk):
    """Anillo par de n sitios con flujo pi y caminos de n - 1. En un camino la fase se quita con rotaciones locales en z,
    que conservan la entropía: el Gibbs del camino es el de sin datos, y solo v(G) cambia. Aquí delta > 0 para todo n."""
    v4, _ = v_anillo(4, np.pi)
    chk.check(abs(v4 - 0.5) < 1e-7, f"anillo de 4 con flujo pi: v(G) = {v4:.8f}, 1/2 en la sec. 6")
    filas = []
    for n in (4, 6, 8, 10, 12, 14, 16):
        m = n - 1
        vG, gap = v_anillo(n, np.pi)
        g = grafo_camino(m)
        orb = orbitas_camino(m - 1)
        res = sphere.OrbitDual(g, np.zeros(m - 1), orb).minimize(target=2e-8)
        vR = res.v
        chk.check(max(gap, res.gap) < 1e-7 and vG < vR, f"anillo {n} con flujo pi: v(G) = {vG:.8f}, camino de {m}: {vR:.8f}, delta = {vR - vG:.6f}, n^2 delta = {n * n * (vR - vG):.4f}")
        G = Gibbs(g, np.zeros(m - 1), orb)
        y, prev = None, None
        for punto, v in (("0.75 v_R", 0.75 * vR), ("0.90 v_R", 0.9 * vR), ("v_G", vG), ("medio", (vG + vR) / 2)):
            if v > vG + 1e-9 and punto != "medio":
                continue
            y, prev, row = fila(chk, f"camino de {m}, flujo pi [{punto}]", G, m, g.edges, v, y, [n, vG, vR, vR - vG, punto], prev)
            filas.append(row)
        write_csv("sphere_entropia_cadenas_flujo.csv", ("n_anillo", "v_G", "v_R", "delta", "punto", "n", "v", "S", "f", "cota", "residuo"), filas)


def main():
    chk = Checker("sphere_entropia")
    if "--dodecaedro" in sys.argv:
        corre_casquetes(chk, ("dodecaedro",))
        chk.finish()
    if "--flujo" in sys.argv:
        corre_cadenas_flujo(chk)
        chk.finish()
    controles(chk)
    corre_casquetes(chk, ("cubo",))
    corre_poliedros(chk)
    corre_cadenas(chk)
    corre_cadenas_flujo(chk)
    chk.finish()


if __name__ == "__main__":
    main()

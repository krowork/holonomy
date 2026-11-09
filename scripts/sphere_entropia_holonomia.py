"""Parte de holonomía de la entropía a v fijo: ¿término topológico (tipo Jacobson--Myers) o corrección que se apaga?

Con N y v fijos, Delta S_hol(v) = S_Gibbs(v; datos) - S_Gibbs(v; triviales) es la parte de la entropía de máxima entropía que
depende solo de la holonomía, no del número de celdas. En gravedad, un término lineal en la curvatura intrínseca de la hoja
integraría a 2 pi chi (Gauss--Bonnet) y daría una constante topológica; uno cuadrático, sum_p f_p^2 ~ F^2/L, se apagaría.

Paridad: el gauge global R_x(pi) en todos los sitios lleva R_z(phi_e) a R_z(-phi_e), así que S es par en el flujo y no puede
tener término lineal. Se comprueba aquí numéricamente.

Familias:
  1. poliedros con datos de curvatura (flujo de cara = ángulo sólido, de orden uno), a v fijo;
  2. escalera periódica 2 x L con flujo total F fijo, f = F/L por plaqueta, en el ecuador (psi = 0): patas con holonomía
     +-F/2. Es la familia de N creciente sin mezclar clases.

Comprobaciones: residuo por arista < 1e-8 (certifica el máximo de entropía sin depender de la simetría) y S(-phi) = S(phi).
El signo de Delta S no se comprueba: con flujos de orden uno la curvatura sube S en el tetraedro, el octaedro, el
cuboctaedro, el icosaedro (v <= 0,2) y la escalera 2 x 3.

Con F = 4 pi las patas tienen holonomía trivial y toda la curvatura está en las plaquetas.

Escribe data/sphere_entropia_holonomia_poliedros.csv y data/sphere_entropia_holonomia_escaleras.csv; con --plaquetas,
solo la familia F = 2 pi, psi = pi (patas triviales), en data/sphere_entropia_holonomia_plaquetas.csv.

Uso: (ulimit -v 16000000; OMP_NUM_THREADS=8 python scripts/sphere_entropia_holonomia.py [--plaquetas])
"""
import _common  # noqa: F401
from _common import Checker

import csv
import sys
import time
from pathlib import Path

import numpy as np

from holonomy import sphere
from holonomy.report import write_csv
import sphere_casquetes as cas
from sphere_entropia import Gibbs

DATA = Path(__file__).resolve().parent.parent / "data"

POLIS = ("tetraedro", "octaedro", "cubo", "cuboctaedro", "icosaedro", "rombododecaedro")
V_POLI = (0.1, 0.2, 0.25)
V_ESC = (0.1, 0.2, 0.3)
LS = (3, 4, 5, 6, 7)
FS = (np.pi / 8, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi)


def envuelve(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def escalera(L, F, psi=0.0):
    """Escalera periódica 2 x L con fase (psi + F/2)/L en la pata inferior y (psi - F/2)/L en la superior, peldaños sin
    fase: plaquetas F/L, patas psi +- F/2. Las aristas se guardan con i < j: la de cierre de cada pata va al revés, con la
    fase y el signo de b cambiados."""
    edges, phis = [], []
    patas = []
    for base, ph in ((0, (psi + F / 2) / L), (L, (psi - F / 2) / L)):
        miembros, signos = [], {}
        for i in range(L):
            a, b = base + i, base + (i + 1) % L
            s = 1 if a < b else -1
            signos[len(edges)] = s
            miembros.append(len(edges))
            edges.append((min(a, b), max(a, b)))
            phis.append(s * ph)
        patas.append(sphere.Orbit(miembros, 1, False, signos))
    peld = list(range(len(edges), len(edges) + L))
    for i in range(L):
        edges.append((i, L + i))
        phis.append(0.0)
    orbs = patas + [sphere.Orbit(peld, 1, False, {m: 1 for m in peld})]
    g = sphere.Polyhedron(f"escalera{L}", np.zeros((2 * L, 3)), tuple(edges), ())
    return g, np.array(phis), orbs


def holonomias_escalera(L, phis):
    """Plaqueta p -> p+1 -> L+p+1 -> L+p -> p y las dos patas, desde las fases guardadas."""
    pb = phis[:L] * np.array([1 if i < L - 1 else -1 for i in range(L)])
    pt = phis[L:2 * L] * np.array([1 if i < L - 1 else -1 for i in range(L)])
    return pb - pt, pb.sum(), pt.sum()


def resuelve(chk, nombre, G, vs):
    out, y = [], None
    for v in vs:
        t0 = time.time()
        try:
            y, S, res = G.resuelve(v, y)
        except Exception as exc:  # noqa: BLE001
            chk.info(f"{nombre}, v = {v}: sin solución ({exc})")
            out.append((v, np.nan, np.inf))
            y = None
            continue
        out.append((v, S, res))
        chk.info(f"    {nombre}, v = {v}: S = {S:.10f}, residuo {res:.1e}, {time.time() - t0:.0f} s")
    return out


def paridad(chk):
    for pol in ("cubo", "octaedro"):
        p = sphere.polyhedron(pol)
        orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
        phis = cas.data_phases(p, "curvado")
        (_, S1, r1), = resuelve(chk, f"{pol} +phi", Gibbs(p, phis, orbs), (0.2,))
        (_, S2, r2), = resuelve(chk, f"{pol} -phi", Gibbs(p, -phis, orbs), (0.2,))
        chk.check(max(r1, r2) < 1e-8 and abs(S1 - S2) < 1e-9, f"paridad en el {pol}: S(phi) - S(-phi) = {S1 - S2:.1e}")
    g, phis, orbs = escalera(5, np.pi)
    (_, S1, r1), = resuelve(chk, "escalera 5, F = pi", Gibbs(g, phis, orbs), (0.2,))
    (_, S2, r2), = resuelve(chk, "escalera 5, F = -pi", Gibbs(g, -phis, orbs), (0.2,))
    chk.check(max(r1, r2) < 1e-8 and abs(S1 - S2) < 1e-9, f"paridad en la escalera 2x5: S(F) - S(-F) = {S1 - S2:.1e}")
    for L in (4, 5):
        plaq, hb, ht = holonomias_escalera(L, escalera(L, 2.0)[1])
        chk.check(np.allclose(plaq, 2.0 / L) and abs(hb - 1.0) < 1e-12 and abs(ht + 1.0) < 1e-12,
                  f"escalera 2x{L}: plaquetas {plaq[0]:.4f} = F/L, patas {hb:.4f}, {ht:.4f} = +-F/2")


def poliedros(chk):
    filas = []
    for pol in POLIS:
        p = sphere.polyhedron(pol)
        orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
        flujos = envuelve(sphere.face_fluxes(p))
        suma2 = float((flujos**2).sum())
        r0 = resuelve(chk, f"{pol} trivial", Gibbs(p, np.zeros(p.E), orbs), V_POLI)
        r1 = resuelve(chk, f"{pol} curvado", Gibbs(p, cas.data_phases(p, "curvado"), orbs), V_POLI)
        for (v, S0, e0), (_, S1, e1) in zip(r0, r1, strict=True):
            ok = max(e0, e1) < 1e-8
            dS = S1 - S0
            chk.check(ok, f"{pol} (N = {p.n}), v = {v}: Delta S_hol = {dS:+.6e}, por qubit {dS / p.n:+.3e}, "
                                         f"sum f_p^2 = {suma2:.3f}, Delta S / sum f^2 = {dS / suma2:+.3e}")
            filas.append([pol, p.n, p.E, p.F, suma2, v, S0, S1, dS, dS / p.n, max(e0, e1)])
    write_csv("sphere_entropia_holonomia_poliedros.csv",
              ("poliedro", "n", "E", "F", "suma_f2", "v", "S_trivial", "S_curvado", "dS", "dS_por_qubit", "residuo"), filas)


def escaleras(chk):
    filas = []
    for L in LS:
        g, _, orbs = escalera(L, 0.0)
        base = {v: (S, e) for v, S, e in resuelve(chk, f"2x{L} F = 0", Gibbs(g, np.zeros(len(g.edges)), orbs), V_ESC)}
        for F in FS:
            g, phis, orbs = escalera(L, F)
            for v, S, e in resuelve(chk, f"2x{L} F = {F / np.pi:.3f} pi", Gibbs(g, phis, orbs), V_ESC):
                S0, e0 = base[v]
                ok = max(e, e0) < 1e-8
                dS = S - S0 if ok else np.nan
                f = F / L
                chk.check(ok,
                          f"2x{L}, F = {F / np.pi:.3f} pi, v = {v}: Delta S_hol = {dS:+.6e}, L Delta S / F^2 = {L * dS / F**2:+.4e}, "
                          f"Delta S / F^2 = {dS / F**2:+.4e}, residuo {max(e, e0):.1e}")
                filas.append([L, 2 * L, F, f, v, S0, S, dS, max(e, e0)])
        write_csv("sphere_entropia_holonomia_escaleras.csv",
                  ("L", "n", "F", "f_plaqueta", "v", "S_trivial", "S", "dS", "residuo"), filas)


def plaquetas(chk):
    """Solo curvatura de plaquetas: F = 2 pi, psi = pi, patas con holonomía 2 pi y 0, triviales. Con F = 4 pi y psi = 0
    (en `escaleras`) también lo son. Da g(f) = Delta S / L en f = 2 pi / L."""
    # Las referencias sin flujo son las de `escaleras` (mismo grafo y mismas órbitas): la de 2 x 7 en v = 0,1 tarda 3 h.
    previas = {}
    for r in csv.DictReader(open(DATA / "sphere_entropia_holonomia_escaleras.csv")):
        # El residuo guardado es el mayor de la fila (con y sin flujo): es una cota del de la referencia.
        k = int(r["L"]), float(r["v"])
        previas[k] = (float(r["S_trivial"]), max(float(r["residuo"]), previas.get(k, (0, 0.0))[1]))
    filas = []
    F, psi = 2 * np.pi, np.pi
    for L in LS:
        g, phis, orbs = escalera(L, F, psi)
        plaq, hb, ht = holonomias_escalera(L, phis)
        chk.check(np.allclose(plaq, F / L) and abs(hb - 2 * np.pi) < 1e-12 and abs(ht) < 1e-12,
                  f"2x{L}: plaquetas {plaq[0]:.4f} = 2 pi/L, patas {hb:.4f} y {ht:.4f}")
        base = [(v, *previas[L, v]) for v in V_ESC]
        for (v, S0, e0), (_, S, e) in zip(base, resuelve(chk, f"2x{L} plaquetas", Gibbs(g, phis, orbs), V_ESC), strict=True):
            dS, f = S - S0, F / L
            chk.check(max(e, e0) < 1e-8, f"2x{L}, f = {f:.4f}, v = {v}: Delta S_hol = {dS:+.6e}, g(f) = {dS / L:+.4e}, "
                                         f"g/f^2 = {dS / L / f**2:+.4e}, residuo {max(e, e0):.1e}")
            filas.append([L, 2 * L, F, psi, f, v, S0, S, dS, dS / L, max(e, e0)])
        write_csv("sphere_entropia_holonomia_plaquetas.csv",
                  ("L", "n", "F", "psi", "f_plaqueta", "v", "S_trivial", "S", "dS", "g", "residuo"), filas)


def main():
    chk = Checker("sphere_entropia_holonomia")
    if "--plaquetas" in sys.argv:
        plaquetas(chk)
        chk.finish()
    paridad(chk)
    poliedros(chk)
    escaleras(chk)
    chk.finish()


if __name__ == "__main__":
    main()

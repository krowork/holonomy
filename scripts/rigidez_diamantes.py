"""Rigidez nula del punto A de la cadena cerrada de diamantes, por fragmentación.

Artículo (~/fisica/holonomia_diamantes.tex): §8 (Proposición del punto A,
«prop:puntoA»); cota tipo Kohn de holonomy/rigidez.py (tareas 1 y 3 del
trabajo de rigidez), aplicada al testigo SU(2) de la Proposición.

Enunciado. En la cadena cerrada de k diamantes con f = 0 (fA = fB = 0), los
datos del gauge simétrico llevan la fase s = psi/(2k) en las cuatro aristas
exteriores de cada celda, (j,u), (j,w), (u,j'), (w,j'), y 0 en la central
(u,w); la holonomía de cada vuelta, por los u y por los w, es psi. Entonces,
para todo psi y todo k impar,
    v(psi) <= 1/3 - c/k = v(0),
con c = (8 + 2e)/(3(2 - e)) y e = eps_2 = -3,3815016433 (el valor de la
Proposición). Para k par, v(psi) <= 1/3 = v(0) es el techo de la Proposición
del techo, que no depende de psi.

Demostración (cadena de implicaciones; lo que comprueba este guion).
 (1) Cota de Kohn. Sea W el testigo óptimo de psi = 0, el de la Proposición,
     W = -H(a*, b*) con a* = 1/(2 - e), b* = 1 - 4a*, normalizado a N = 3k.
     Es invariante bajo la rotación global alrededor de z (de hecho bajo todo
     SU(2)), así que por la Proposición de holonomy/rigidez.py el testigo con
     sus fases giradas lo mismo que los datos, W[delta], conserva la
     normalización y cumple
         v(psi) <= lambda_max(W[delta]) = -lambda_min(H_psi(a*, b*)) / (3k),
     donde H_psi es H con el término de cada arista exterior (i,j) girado,
         J [ S^z_i S^z_j + (1/2)(e^{i s} S^+_i S^-_j + e^{-i s} S^-_i S^+_j) ],
     s = psi/(2k), sin tocar el ZZ ni la arista central (diamond_H_twisted).
 (2) El giro conserva cada T_m^2 = (S_u + S_w)^2. Las aristas (j,u) y (j,w)
     llevan la misma fase, de modo que su suma es
         J [ S^z_j T^z_m + (1/2)(e^{is} S^+_j T^-_m + h.c.) ],
     que solo contiene los T_m^a, y lo mismo vale para (u,j') y (w,j'); T_m^2
     conmuta con T_m^a y con los operadores de las demás celdas. Con fases
     distintas en (j,u) y (j,w) (un flujo fA != 0) no es cierto.
 (3) Fragmentación bajo el giro. Como en la Proposición, cada sector de los
     T_m^2 (celdas singlete 'S' y triplete 'T') se parte:
       - con alguna celda singlete, el anillo está cortado y los fragmentos son
         cadenas abiertas 1/2-1-...-1/2. En una cadena abierta un giro es un
         gauge: exp(i sum_i g_i S^z_i) con g_{i+1} - g_i = fase de la arista
         lo elimina, así que la energía eps_m no cambia;
       - el sector sin singletes es el anillo de k espines 1/2 (j_m) y k espines
         1 (T_m) con una fase s en cada uno de los 2k enlaces, de fase total
         psi, y energía 4a eps(psi) + k b. Para k >= 5, la cuarta parte de la
         suma de sus k cadenas abiertas de cuatro espines 1 (que no ven el
         giro) da eps(psi) >= (k/4) eps_4 para todo psi, de modo que el
         argumento de la Proposición (exceso X > 3c) sigue valiendo; con k = 3
         se comprueba directamente en una malla de 7 puntos de psi en [0, pi]
         (numérico en la malla, no una demostración para todo psi).
     Por tanto lambda_min(H_psi(a*, b*)) = -k + 3c = lambda_min(H_0(a*, b*))
     (en la normalización de diamond_H, dividido por k).
 (4) De (1) y (3): v(psi) <= -lambda_min(H_0(a*))/(3k) = v(0) = 1/3 - c/k.

Comprueba (k = 3 y 5, psi en una malla de 7 puntos de [0, pi]):
  1. las fases de diamond_H_twisted son las de los datos: fA = fB = 0 y
     holonomía psi por cada vuelta, frente a las columnas fA, fB, vuelta_u,
     vuelta_w de data/diamond_ring_L3_igual.csv;
  2. [H_psi, T_m^2] = 0 para cada celda m;
  3. lambda_min(H_psi(a*)) = lambda_min(H_0(a*)) = (-k + 3c)/k, con eigsh de
     arranque aleatorio (los sectores no se fijan por un arranque en
     caliente); la cota v(psi) <= -lambda_min/3 = 1/3 - c/k es esa igualdad
     dividida por -3 y no se comprueba aparte (columna v_cota del CSV);
  4. fragmentación con giro: lambda_min del hamiltoniano completo = mínimo
     sobre sectores con eps_m de la cadena abierta y el fundamental del anillo
     girado; el anillo girado cumple el exceso X > 3c, y para k = 5
     eps(psi) >= (k/4) eps_4; en k = 3 también para a != a* (la fragmentación
     vale para todo a);
  5. contraste con la planitud de data/diamond_ring_L3_igual.csv en f = 0:
     v(psi) <= v(0) + tolerancia de las cotas de la generación de columnas;
  6. comprobación negativa (k = 3): con fases distintas en (j,u) y (j,w) el
     giro no conserva T_m^2 y lambda_min sí cambia; la invariancia no es trivial;
  7. k = 4 (informativo): lambda_min(H_psi(a*)) en psi = 0, pi/2, pi.

Escribe data/rigidez_diamantes.csv.

Uso: python scripts/rigidez_diamantes.py   (tiempo y memoria: ver README; medidos
con /usr/bin/time -f "%M KB %e s")
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import sympy

from diamond_punto_a import (eps_open, heisenberg, sector_min, spin_ops,
                             diamond_H)  # noqa: F401  (diamond_H: referencia de psi = 0, la usan las pruebas)
from holonomy.diamond import abelian_phases, chain_edges, triangle_fluxes
from holonomy.report import DATA, write_csv

TOL = 1e-10
DATA_TOL = 1e-8
GRID = np.linspace(0.0, np.pi, 7)


def twisted_heisenberg(spins, bonds, J, theta):
    """sum_b J_b [S^z_i S^z_j + (1/2)(e^{i theta_b} S^+_i S^-_j + h.c.)] sobre
    los enlaces orientados (i, j); espines de cualquier tamaño. Disperso,
    complejo hermítico. Con theta = 0 es heisenberg(spins, bonds, J)."""
    ops = [spin_ops(s) for s in spins]
    dims = [int(round(2 * s + 1)) for s in spins]
    cache = {}

    def emb(i, name):
        if (i, name) not in cache:
            sx, sy, sz = ops[i]
            o = {"+": sx + 1j * sy, "-": sx - 1j * sy, "z": sz}[name]
            left = sp.identity(int(np.prod(dims[:i])), format="csr")
            right = sp.identity(int(np.prod(dims[i + 1:])), format="csr")
            cache[(i, name)] = sp.kron(sp.kron(left, o), right, format="csr")
        return cache[(i, name)]

    H = sp.csr_matrix((int(np.prod(dims)),) * 2, dtype=complex)
    for (i, j), w, th in zip(bonds, J, theta):
        zz = emb(i, "z") @ emb(j, "z")
        xy = np.exp(1j * th) * (emb(i, "+") @ emb(j, "-"))
        H = H + w * (zz + 0.5 * (xy + xy.getH()))
    return H.tocsr()


def diamond_phases(k, psi):
    """Fases de las 5k aristas orientadas de chain_edges(k, periodic=True) de los
    datos con f = 0 en el gauge simétrico: psi/(2k) en las exteriores, 0 en la central."""
    return abelian_phases(k, 0, 0, periodic=True, psi=psi)


def diamond_H_twisted(k, a, b, psi, a2=None, phis=None):
    """Testigo de la Proposición 10 (diamond_H: a en ju/jw, a2 = a en uj/wj, b en uw,
    s.s = 4 S.S) con el giro de los datos: el término de cada arista (i,j) es
    J [S^z S^z + (1/2)(e^{i th} S^+_i S^-_j + h.c.)], th = fase de la arista en el
    gauge simétrico con f = 0 y holonomía psi (diamond_phases). psi = 0 es diamond_H.
    `phis` permite fases arbitrarias por arista (comprobación negativa)."""
    a2 = a if a2 is None else a2
    J = [4 * x for x in (a, a, b, a2, a2)] * k
    phis = diamond_phases(k, psi) if phis is None else phis
    return twisted_heisenberg([0.5] * (3 * k), chain_edges(k, periodic=True), J, phis)


def ring_twisted(k, psi):
    """Anillo alternado 1/2 (j_m), 1 (T_m) con acoplamiento 1 y fase psi/(2k) en
    cada uno de sus 2k enlaces (el sector sin singletes); fundamental eps(psi)."""
    spins = [0.5, 1] * k
    bonds = [(i, (i + 1) % (2 * k)) for i in range(2 * k)]
    return twisted_heisenberg(spins, bonds, [1.0] * (2 * k), [psi / (2 * k)] * (2 * k))


def lam_min_aleatorio(H, seed=0):
    """Autovalor mínimo: denso si cabe; si no, Lanczos con arranque aleatorio
    y más vectores que el mínimo (sin arranque en caliente)."""
    n = H.shape[0]
    if n <= 4096:
        return np.linalg.eigvalsh(H.toarray())[0]
    v0 = np.random.default_rng(seed).standard_normal(n)
    return sla.eigsh(H, k=2, which="SA", tol=1e-13, v0=v0, ncv=min(60, n - 1))[0][0]


def comm_norm(A, B):
    return abs(A @ B - B @ A).max()


def t2_cells(k):
    """(S_u + S_w)^2 = 3/2 + 2 S_u.S_w de cada celda: basta S_u.S_w, que da el mismo conmutador."""
    return [heisenberg([0.5] * (3 * k), [(3 * m + 1, 3 * m + 2)]) for m in range(k)]


def main():
    chk = Checker("rigidez_diamantes")
    x = sympy.symbols("x")
    e_exacto = float(min(sympy.Poly(2 * x**3 + 8 * x**2 + 3 * x - 4, x).nroots(n=30),
                         key=lambda z: sympy.re(z)))
    eps = {m: eps_open(m) for m in range(5)}
    e = eps[2]
    chk.check(abs(e - e_exacto) < TOL, f"e = eps_2 = {e:.12f} (raíz de 2x^3+8x^2+3x-4)")
    a_s = 1 / (2 - e)
    b_s = 1 - 4 * a_s
    c = 4 * a_s - 2 / 3
    chk.info(f"a* = {a_s:.12f}, b* = {b_s:.12f}, c = {c:.12f}")

    # 1. fases frente a los datos
    with open(DATA / "diamond_ring_L3_igual.csv") as fh:
        todas = list(csv.DictReader(fh))
    data = [r for r in todas if float(r["f"]) == 0.0]
    # Una fila de f = 0 por cada psi de la malla del mapa (la de todas las f).
    psis_mapa = sorted({float(r["psi"]) for r in todas})
    n_psi = len(psis_mapa)
    err = 0.0
    for r in data:
        psi = float(r["psi"])
        fA, fB, hol = triangle_fluxes(3, diamond_phases(3, psi), periodic=True)
        err = max(err, max(abs(np.array(fA) - float(r["fA"]))), max(abs(np.array(fB) - float(r["fB"]))),
                  abs(hol[0] - float(r["vuelta_u"])), abs(hol[1] - float(r["vuelta_w"])))
    chk.check(sorted(float(r["psi"]) for r in data) == psis_mapa and err < 1e-12,
              f"fases del testigo = fases de los datos (f = 0): fA = fB = 0, holonomía psi por u y por w "
              f"({len(data)} filas, desvío {err:.1e})")

    # 2-4. k = 3 y 5 en la malla de psi
    rows = []
    ring_ref = {}
    for k in (3, 5):
        lam_ref = (-k + 3 * c) / k
        T2 = t2_cells(k)
        e4 = eps[4]
        for psi in GRID:
            H = diamond_H_twisted(k, a_s / k, b_s / k, psi)
            herm = abs(H - H.getH()).max()
            comm = max(comm_norm(H, T) for T in T2)
            lam = lam_min_aleatorio(H, seed=1)
            lam_alt = lam_min_aleatorio(H, seed=2) if H.shape[0] > 4096 else lam
            er = lam_min_aleatorio(ring_twisted(k, psi), seed=3)
            ring_ref.setdefault(k, []).append(er)
            sec = sector_min(k, a_s / k, b_s / k, eps, er)
            X_ring = 4 * a_s * er + 2 * k - 4 * a_s * k
            v_cota = -lam / 3  # paso (4); v_cota = 1/3 - c/k se sigue del check de lam
            chk.check(herm < 1e-12 and comm < 1e-10,
                      f"k = {k}, psi = {psi:.4f}: H hermítico ({herm:.0e}) y [H, T_m^2] = 0 ({comm:.1e})")
            chk.check(abs(lam - lam_ref) < 1e-9 and abs(lam - lam_alt) < 1e-9,
                      f"k = {k}, psi = {psi:.4f}: lambda_min = {lam:.12f} = (-k+3c)/k = {lam_ref:.12f}")
            chk.check(abs(lam - sec) < 1e-9,
                      f"k = {k}, psi = {psi:.4f}: = mínimo por sectores con el anillo girado "
                      f"(eps(psi) = {er:.10f}, X = {X_ring:.6f} vs 3c = {3 * c:.6f})")
            ok_ring = X_ring > 3 * c + 0.1 and (k == 3 or er >= k / 4 * e4 - TOL)
            chk.check(ok_ring, f"k = {k}, psi = {psi:.4f}: anillo girado con exceso X > 3c"
                               + (f" y eps(psi) >= (k/4) eps_4 = {k / 4 * e4:.6f}" if k == 5 else ""))
            rows.append([k, f"{psi:.15f}", f"{lam:.15f}", f"{v_cota:.15f}", f"{1 / 3 - c / k:.15f}",
                         f"{er:.15f}", f"{X_ring:.15f}", f"{comm:.2e}"])
        er = np.array(ring_ref[k])
        chk.info(f"k = {k}: eps(psi) del anillo en [{er.min():.10f}, {er.max():.10f}] "
                 f"(el anillo sí depende de psi; la cota no)")

    # la fragmentación con giro vale para todo a (k = 3, psi = pi)
    k, psi = 3, np.pi
    er = lam_min_aleatorio(ring_twisted(k, psi), seed=4)
    for a in (0.05, 0.12, a_s, 0.22, 0.24):
        full = lam_min_aleatorio(diamond_H_twisted(k, a / k, (1 - 4 * a) / k, psi))
        sec = sector_min(k, a / k, (1 - 4 * a) / k, eps, er)
        chk.check(abs(full - sec) < 1e-9, f"k = 3, psi = pi, a = {a:.4f}/k: lambda_min = {full:.12f} = mínimo por sectores")

    # 5. contraste con los datos del mapa (f = 0, k = 3)
    v0 = float(next(r["v_max"] for r in data if float(r["psi"]) == 0.0))
    vs = np.array([float(r["v_max"]) for r in data])
    chk.check(abs(v0 - (1 / 3 - c / 3)) < DATA_TOL,
              f"datos k = 3, f = 0, psi = 0: v = {v0:.12f} = 1/3 - c/3 = {1 / 3 - c / 3:.12f}")
    chk.check(np.all(vs <= v0 + DATA_TOL) and np.all(vs <= 1 / 3 - c / 3 + DATA_TOL),
              f"datos: v(psi) <= v(0) + {DATA_TOL:.0e} en las {len(vs)} filas (una por psi del mapa, "
              f"{n_psi}); máx v(psi) - v(0) = {np.max(vs - v0):.1e}")
    chk.info(f"datos: planitud, max |v(psi) - v(0)| = {np.max(abs(vs - v0)):.1e} (cotas del dual por "
             f"generación de columnas, de tolerancia {DATA_TOL:.0e})")

    # 6. comprobación negativa: fases distintas en (j,u) y (j,w)
    k, psi, d = 3, np.pi / 3, 0.3
    phis = list(diamond_phases(k, psi))
    for m in range(k):
        phis[5 * m] += d
        phis[5 * m + 1] -= d
    Hn = diamond_H_twisted(k, a_s / k, b_s / k, psi, phis=phis)
    commn = max(comm_norm(Hn, T) for T in t2_cells(k))
    lam_n = lam_min_aleatorio(Hn)
    lam_0 = (-k + 3 * c) / k
    chk.check(commn > 1e-3 and abs(lam_n - lam_0) > 1e-3,
              f"negativa k = 3: con fases s + {d} en (j,u) y s - {d} en (j,w) (flujo fA = {2 * d:.1f}) "
              f"[H, T^2] = {commn:.2e} y lambda_min = {lam_n:.9f} != {lam_0:.9f} (dif {lam_n - lam_0:.2e})")

    # 7. k = 4 (informativo)
    k = 4
    for psi in (0.0, np.pi / 2, np.pi):
        lam = lam_min_aleatorio(diamond_H_twisted(k, a_s / k, b_s / k, psi), seed=5)
        chk.info(f"k = 4, psi = {psi:.4f}: lambda_min(H_psi(a*)) = {lam:.12f} (el techo da v <= 1/3; datos planos a 1e-8)")

    write_csv("rigidez_diamantes.csv",
              ["k", "psi", "lambda_min", "v_cota", "v_cota_psi0", "eps_anillo", "X_anillo", "conmutador"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

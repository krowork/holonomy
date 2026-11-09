"""Punto A de la cadena cerrada de k diamantes: forma cerrada de v_max.

Artículo (~/fisica/holonomia_diamantes.tex): §8, El límite k -> infinito
(Proposición del punto A y su demostración).

En el punto A (f = 0, psi = 0) todos los datos son R_e = I. El promedio sobre
SU(2) global, el intercambio u_m <-> w_m de cada celda, la inversión de la
cadena y la traslación da un testigo óptimo W = -H/(3k(4a+b)) con
    H(a, b) = a sum_{aristas exteriores} s.s + b sum_m s_{u_m}.s_{w_m}
(s.s = producto escalar de matrices de Pauli), de modo que
    v_max = min_{4a+b=1} -lambda_min(H)/(3k).
Con T_m = (s_u + s_w)/2, el espín de cada par central se conserva:
    a (s_j + s_j').(s_u + s_w) = 4a (S_j + S_j').T_m,   s_u.s_w = 2 T_m^2 - 3.
Una celda singlete (T = 0) vale -3b y corta la cadena; una triplete (T = 1)
vale +b y es un espín 1. Los fragmentos son cadenas abiertas alternadas
1/2-1-...-1/2 con m espines 1 y energía 4a eps_m (eps_m = fundamental de
sum S.T con acoplamiento 1), y el sector sin singletes es el anillo de k
espines 1/2 y k espines 1.

Resultado: con e = eps_2, la menor raíz de 2x^3 + 8x^2 + 3x - 4,
    a* = 1/(2 - e),  c = 4a* - 2/3 = (8 + 2e)/(3(2 - e)) = 0,0766202939...,
    v_max = 1/3 - c/k (k impar),  v_max = 1/3 (k par),
y c es raíz de 675c^3 - 1836c^2 + 972c - 64.

Comprueba:
  1. e por diagonalización del pentámero y como raíz de la cúbica; c raíz de
     su polinomio;
  2. eps_0 = 0, eps_1 = -2, eps_2 = e; las cotas por trímeros y pentámeros de
     eps_m (m <= 4); el coeficiente por celda de la cota, 2a*e - 4a* + 2 = 0;
  3. los excesos X(m) en a*: X(0) = X(2) = 3c, X(1) = 0, X(m) >= su cota;
  4. el sector sin singletes: k = 3 por diagonalización; k >= 5 por el
     recubrimiento con cadenas de 4 espines 1, k (4a*(eps_4/4 - 1) + 2) > 3c;
  5. fragmentación: lambda_min del hamiltoniano completo = mínimo sobre
     sectores, en k = 3 para varios a, y en a* para k = 3 y 5; en a* los dos
     defectos están degenerados;
  6. testigo general en k = 3 (a en ju/jw, a' en uj/wj, b en uw): el mínimo
     es 1/3 - c/3, en a = a' = a*/3;
  7. k par (2, 4): el sector (TS)^(k/2) da lambda_min <= -k para todo a;
  8. 1/3 - c/k frente a data/diamond_limit_A.csv (k = 3, 5, 7): dentro de la
     diferencia de cotas de la generación de columnas (1e-8).

Escribe data/diamond_punto_a.csv.

Uso: python scripts/diamond_punto_a.py   (23 s y 0,43 GB de pico, medidos con
/usr/bin/time -f "%M KB %e s")
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import sympy
from scipy.optimize import minimize, minimize_scalar

from holonomy.diamond import chain_edges
from holonomy.report import DATA, write_csv

TOL = 1e-10
DATA_TOL = 1e-8


def spin_ops(s):
    """(S_x, S_y, S_z) de espín s, dispersas."""
    d = int(round(2 * s + 1))
    m = s - np.arange(d)
    sp_ = np.diag(np.sqrt(s * (s + 1) - m[1:] * (m[1:] + 1)), 1)
    return [sp.csr_matrix(x) for x in ((sp_ + sp_.T) / 2, (sp_ - sp_.T) / 2j, np.diag(m))]


def heisenberg(spins, bonds, J=None):
    """sum_b J_b S_i.S_j sobre los enlaces dados (disperso, real)."""
    ops = [spin_ops(s) for s in spins]
    dims = [int(round(2 * s + 1)) for s in spins]

    def emb(o, i):
        left = sp.identity(int(np.prod(dims[:i])), format="csr")
        right = sp.identity(int(np.prod(dims[i + 1:])), format="csr")
        return sp.kron(sp.kron(left, o), right, format="csr")

    H = sp.csr_matrix((int(np.prod(dims)),) * 2)
    for t, (i, j) in enumerate(bonds):
        w = 1.0 if J is None else J[t]
        H = H + w * sum(emb(ops[i][x], i) @ emb(ops[j][x], j) for x in range(3))
    return H.real.tocsr()


def lam_min(H):
    if H.shape[0] <= 4096:
        return np.linalg.eigvalsh(H.toarray())[0]
    return sla.eigsh(H, k=1, which="SA", tol=1e-13)[0][0]


def eps_open(m):
    """Fundamental de la cadena abierta 1/2-1-...-1/2 con m espines 1."""
    if m == 0:
        return 0.0
    spins = [0.5, 1] * m + [0.5]
    return lam_min(heisenberg(spins, [(i, i + 1) for i in range(2 * m)]))


def eps_ring(k):
    spins = [0.5, 1] * k
    return lam_min(heisenberg(spins, [(i, (i + 1) % (2 * k)) for i in range(2 * k)]))


def diamond_H(k, a, b, a2=None):
    """H en qubits de la cadena cerrada: a en ju/jw, a2 (= a) en uj/wj, b en uw;
    s.s = 4 S.S."""
    a2 = a if a2 is None else a2
    J = [4 * x for x in (a, a, b, a2, a2)] * k
    return heisenberg([0.5] * (3 * k), chain_edges(k, periodic=True), J)


def sector_energy(pattern, a, b, eps, ering):
    """Energía del sector con celdas 'S' (singlete) y 'T' (triplete)."""
    if "S" not in pattern:
        return 4 * a * ering + len(pattern) * b
    r = pattern.index("S")
    p = pattern[r:] + pattern[:r]
    runs = [len(x) for x in "".join(p).split("S")[1:]]
    return -3 * b * p.count("S") + sum(4 * a * eps[m] + m * b for m in runs)


def sector_min(k, a, b, eps, ering):
    return min(sector_energy(p, a, b, eps, ering) for p in itertools.product("ST", repeat=k))


def main():
    chk = Checker("diamond_punto_a")
    x = sympy.symbols("x")
    cubic = 2 * x**3 + 8 * x**2 + 3 * x - 4
    e = float(min(sympy.Poly(cubic, x).nroots(n=30), key=lambda z: sympy.re(z)))
    a_s = 1 / (2 - e)
    c = 4 * a_s - 2 / 3
    b_s = 1 - 4 * a_s
    chk.info(f"e = {e:.15f}, a* = {a_s:.15f}, b* = {b_s:.15f}, c = {c:.15f}")

    # 1. e y c
    eps = {m: eps_open(m) for m in range(5)}
    chk.check(abs(eps[2] - e) < TOL, f"pentámero: eps_2 = {eps[2]:.15f} = raíz de 2x^3+8x^2+3x-4")
    chk.check(abs(c - (8 + 2 * e) / (3 * (2 - e))) < 1e-14, "c = (8+2e)/(3(2-e))")
    res = 675 * c**3 - 1836 * c**2 + 972 * c - 64
    chk.check(abs(res) < 1e-11, f"c es raíz de 675c^3-1836c^2+972c-64 (residuo {res:.1e})")
    chk.check(abs(b_s - (1 / 3 - c)) < 1e-14, f"b* = 1/3 - c = {b_s:.10f}")

    # 2. fragmentos y sus cotas
    chk.check(abs(eps[0]) < TOL and abs(eps[1] + 2) < TOL, "eps_0 = 0, eps_1 = -2")
    for m in range(1, 5):
        bound = (m // 2) * e - (2 if m % 2 else 0)
        chk.check(eps[m] >= bound - TOL, f"eps_{m} = {eps[m]:.10f} >= cota por pentámeros y trímeros {bound:.10f}")
    coef = 2 * a_s * e - 4 * a_s + 2
    chk.check(abs(coef) < 1e-14, f"coeficiente por celda de la cota en a*: 2a*e - 4a* + 2 = {coef:.1e}")

    # 3. excesos X(m) = 4a eps_m - 4a(m-3) + 2m - 2 en a*
    def X(m):
        return 4 * a_s * eps[m] - 4 * a_s * (m - 3) + 2 * m - 2
    chk.check(abs(X(0) - 3 * c) < TOL and abs(X(2) - 3 * c) < TOL and abs(X(1)) < TOL,
              f"X(0) = X(2) = 3c = {3 * c:.10f}, X(1) = 0")
    for m in (3, 4):
        lo = 3 * c if m % 2 == 0 else 0.0
        chk.check(X(m) >= lo - TOL, f"X({m}) = {X(m):.6f} >= {lo:.6f}")

    # 4. sector sin singletes
    er3 = eps_ring(3)
    xr3 = 4 * a_s * er3 + 6 - 12 * a_s
    chk.check(xr3 > 3 * c + 0.1, f"anillo k = 3: eps = {er3:.10f}, X = {xr3:.6f} > 3c = {3 * c:.6f}")
    per_cell = 4 * a_s * (eps[4] / 4 - 1) + 2
    chk.check(5 * per_cell > 3 * c + 0.1,
              f"anillo k >= 5 (cadenas de 4 espines 1): X >= k * {per_cell:.6f} >= {5 * per_cell:.6f} > 3c")

    # 5. fragmentación frente al hamiltoniano completo
    ering = {3: er3, 5: eps_ring(5)}
    eps_k = eps  # con un singlete, el tramo más largo de k = 5 tiene 4 tripletes
    rows = []
    for k, alist in ((3, (0.05, 0.12, a_s, 0.22, 0.24)), (5, (a_s,))):
        for a in alist:
            b = 1 - 4 * a
            full = lam_min(diamond_H(k, a / k, b / k))
            sec = sector_min(k, a / k, b / k, eps_k, ering[k])
            chk.check(abs(full - sec) < 1e-9, f"k = {k}, a = {a:.4f}/k: lambda_min = {full:.12f} = mínimo por sectores")
            if a == a_s:
                full_s = full
        E1 = (-k - 2 + 12 * a_s) / k
        E2 = (-k + 2 + 4 * a_s * (e + 1)) / k
        chk.check(abs(full_s - E1) < 1e-9 and abs(E1 - E2) < 1e-12,
                  f"k = {k}, a*: espín libre y pentámero degenerados, lambda_min = {full_s:.12f} = {E1:.12f}")

    # 6. testigo general en k = 3
    k = 3

    def F(y):
        a, a2 = y
        b = 1 / k - 2 * a - 2 * a2
        return -lam_min(diamond_H(k, a, b, a2)) / 3
    r = minimize(F, [0.06, 0.06], method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-14, maxiter=4000))
    chk.check(abs(r.fun - (1 / 3 - c / k)) < 1e-9 and np.allclose(r.x, a_s / k, atol=1e-5),
              f"k = 3, testigo de tres clases: min = {r.fun:.12f} en a = {r.x[0] * k:.6f}/k, a' = {r.x[1] * k:.6f}/k")
    r1 = minimize_scalar(lambda a: -sector_min(5, a / 5, (1 - 4 * a) / 5, eps_k, ering[5]) / 3,
                         bounds=(0.05, 0.249), method="bounded", options=dict(xatol=1e-12))
    chk.check(abs(r1.fun - (1 / 3 - c / 5)) < 1e-10, f"k = 5, mínimo en a por sectores = {r1.fun:.12f}")

    # 7. k par
    for k in (2, 4):
        worst = max(lam_min(diamond_H(k, a / k, (1 - 4 * a) / k)) for a in (0.0, 0.1, a_s, 0.3, 0.5))
        chk.check(worst <= -1 + 1e-10, f"k = {k}: lambda_min <= -1 (v >= 1/3) en cinco valores de a (máx {worst:.12f})")

    # 8. contra los datos del dual por generación de columnas
    with open(DATA / "diamond_limit_A.csv") as fh:
        data = list(csv.DictReader(fh))
    for row in data:
        k, vd = int(row["k"]), float(row["v_max"])
        vf = 1 / 3 - c / k
        d = vf - vd
        chk.check(abs(d) < DATA_TOL, f"k = {k}: 1/3 - c/k = {vf:.12f}, dual = {vd:.12f}, dif {d:.1e}")
        rows.append([k, f"{vf:.15f}", row["v_max"], f"{d:.3e}"])
    write_csv("diamond_punto_a.csv", ["k", "v_formula", "v_dual_cg", "dif"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

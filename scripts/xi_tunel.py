"""Problema abierto 3 de robustez_mps.tex: la longitud del desdoblamiento túnel.

Para n par y Delta = cosh(gamma) > 1, R_n = E0(Delta, pi) - E0(Delta, 0) es el
desdoblamiento del doblete de Néel. Se compara la tasa de su caída exponencial,
R_n ~ A n^{-beta} e^{-n/xi}, con dos candidatos:
    xi_as   = (1/4) e^{pi^2/(2 gamma)}               (forma asintótica cerca de Delta = 1),
    2 xi_JKM, con 1/xi_JKM = gamma + 2 sum_m (-1)^m tanh(m gamma)/m   (Johnson-Krinsky-McCoy).
Difieren en un O(e^{-pi^2/gamma}) relativo: 0,3 % en Delta = 2,66, 2,5 % en Delta = 6 y
5 % en Delta = 10, así que se usan Delta grandes. R_n se calcula con Bethe (control:
diagonalización exacta en n <= 14) hasta que llega a R_FLOOR, cerca del redondeo.

Ajuste: log R_n = a - beta log n - n/xi + c/n por mínimos cuadrados en una ventana
de n grandes, con beta libre y con beta = 1/2.
Salida: data/xi_tunel.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from mpmath import mp, acosh, exp, log, nsum, pi, tanh, inf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from holonomy import bethe, xxz  # noqa: E402

R_FLOOR = 3e-11
OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def candidates(D):
    mp.dps = 30
    g = acosh(D)
    xa = exp(pi ** 2 / (2 * g)) / 4
    inv = g - 2 * log(2) + 2 * nsum(lambda m: (-1) ** m / m * (tanh(m * g) - 1), [1, inf])
    return float(xa), float(2 / inv)


def splitting(n, D):
    S = bethe.solver(n)
    return S.energy(D, np.pi, 0) - S.energy(D, 0.0, 0)


def fit(ns, R, beta=None):
    ns = np.asarray(ns, float)
    y = np.log(R)
    cols = [np.ones_like(ns), -ns, 1 / ns]
    if beta is None:
        cols.insert(1, -np.log(ns))
    else:
        y = y + beta * np.log(ns)
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    k = 2 if beta is None else 1
    return 1 / coef[k], (coef[1] if beta is None else beta), np.max(np.abs(X @ coef - y))


if __name__ == "__main__":
    rows = []
    for D in (2.0, 2.66, 4.0, 6.0, 10.0):
        err = max(abs(splitting(n, D) - (xxz.ground_energy(n, D, np.pi) - xxz.ground_energy(n, D, 0.0)))
                  for n in (8, 10, 12, 14))
        check(err < 1e-11, f"Delta = {D}: R_n de Bethe = diagonalización exacta, n = 8..14 ({err:.1e})")
        xa, xj = candidates(D)
        ns, R = [], []
        n = 8
        while True:
            r = splitting(n, D)
            if r < R_FLOOR:
                break
            ns.append(n)
            R.append(r)
            n += 2 if n < 60 else 4 if n < 200 else 10
        m = len(ns)
        w = slice(m // 3, m)
        xf, bf, ef = fit(ns[w], R[w])
        xh, _, eh = fit(ns[w], R[w], beta=0.5)
        print(f"Delta = {D}: n = {ns[w.start]}..{ns[-1]} ({m - w.start} puntos), R_min = {R[-1]:.1e}")
        print(f"   xi_as = {xa:.5f}   2 xi_JKM = {xj:.5f}   (diferencia {100 * (xa / xj - 1):.2f} %)")
        print(f"   ajuste beta libre: xi = {xf:.5f}, beta = {bf:.3f}, residuo {ef:.1e}")
        print(f"   ajuste beta = 1/2: xi = {xh:.5f}, residuo {eh:.1e}")
        print(f"   xi_ajuste / 2 xi_JKM - 1 = {100 * (xh / xj - 1):+.3f} %,  xi_ajuste / xi_as - 1 = "
              f"{100 * (xh / xa - 1):+.3f} %", flush=True)
        rows.append(dict(Delta=D, n_min=ns[w.start], n_max=ns[-1], xi_as=xa, xi_2jkm=xj, xi_libre=xf,
                         beta_libre=bf, res_libre=ef, xi_beta05=xh, res_beta05=eh))
    out = ROOT / "data" / "xi_tunel.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"escrito {out}")
    sys.exit(0 if OK else 1)

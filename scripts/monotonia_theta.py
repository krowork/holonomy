"""Problema abierto 7 de robustez_mps.tex: monotonía de v(n, .) en [Theta_0, Theta_0 + pi].

El Corolario cor:paridad da el máximo de v(n, Theta) en Theta_0 (0 con n par, pi
con n impar), pero no que el mínimo esté en Theta_0 + pi. Aquí se calcula
v(n, Theta) en una malla de Theta en [0, pi] con el dual completo de la Sección
sec:neel,
    v(n, Theta) = min{ inf_{Delta, Phi} 4 eps_n(Delta, Phi) / (Delta + 2 kappa cos((Phi - Theta)/n)),  v_I },
con Phi en [Theta - pi, Theta + pi] (el representante del flujo más cercano a la
holonomía; los demás tienen el mismo eps_n y menor denominador). Es el valor
exacto del SDP: el testigo óptimo es invariante por la U(1) y la traslación, y el
campo uniforme no mejora la cota para ninguna Theta (comprobación (c) abajo; la
demostración está en el docstring de campo_inutil).

v(n, .) es par y de periodo 2 pi, así que [0, pi] basta. Se mide
    viol = max_k  sgn * (v(Theta_{k+1}) - v(Theta_k)),   sgn = +1 (n par), -1 (n impar),
que debe ser <= 0 si v es monótona, relativo a la variación total |v(0) - v(pi)|.

eps_n con Bethe (Delta > 0), contrastado con la diagonalización exacta.
Controles:
  (a) eps_n de Bethe = diagonalización exacta, n = 8..11, Phi en [0, 2 pi);
  (b) v(n, 0) y v(n, pi) reproducen data/neel_u1.csv (n = 8..18);
  (c) el campo no mejora el dual, n = 8..11, malla de Theta (diagonalización exacta);
  (d) fuerza bruta en una malla fina (Delta, Phi), n = 10: no hay mínimos perdidos.
Salida: data/monotonia_theta.csv (una fila por (kappa, n, Theta)).
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from holonomy import bethe, xxz  # noqa: E402

D_LO = 0.05
GRID_D = np.concatenate([np.linspace(D_LO, 6, 24), [9, 14, 25, 45, 80, 150, 300, 700, 2000]])
OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def fold(Phi):
    """Representante de Phi en [0, pi]: eps_n es par y de periodo 2 pi en Phi."""
    P = np.mod(Phi, 2 * np.pi)
    return 2 * np.pi - P if P > np.pi else P


def eps_bethe(n, D, Phi):
    return bethe.eps(n, D, fold(Phi))


def eps_ed(n, D, Phi):
    return xxz.eps(n, D, Phi)


def v_at_flux(n, Theta, kappa, Phi, eps):
    c = np.cos((Phi - Theta) / n)
    f = lambda D: 4 * eps(n, D, Phi) / (D + 2 * kappa * c) if D + 2 * kappa * c > 0 else np.inf
    vals = [f(D) for D in GRID_D]
    k = int(np.argmin(vals))
    a, b = GRID_D[max(k - 1, 0)], GRID_D[min(k + 1, len(GRID_D) - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-9})
    v, D = (r.fun, r.x) if r.fun < vals[k] else (vals[k], GRID_D[k])
    vI = (n - 2 * (n % 2)) / n
    return (v, D) if v <= vI else (vI, np.inf)


def v_free(n, Theta, kappa, eps, nphi=25):
    Phis = Theta + np.linspace(-np.pi, np.pi, nphi)
    res = [(v_at_flux(n, Theta, kappa, P, eps), P) for P in Phis]
    (v0, D0), P0 = min(res, key=lambda t: t[0][0])
    h = 2 * np.pi / (nphi - 1)
    r = minimize_scalar(lambda P: v_at_flux(n, Theta, kappa, P, eps)[0],
                        bounds=(P0 - h, P0 + h), method="bounded", options={"xatol": 1e-8})
    best = (v0, D0, P0)
    if r.fun < v0:
        v1, D1 = v_at_flux(n, Theta, kappa, r.x, eps)
        best = (v1, D1, r.x)
    # Las cúspides de eps_n (cruce de niveles: Phi = pi mod 2 pi con n par, 0 mod 2 pi con
    # n impar) son mínimos en V que el refinamiento solo alcanza a xatol: se prueban exactas.
    for P in np.pi * np.arange(-2, 3):
        if Theta - np.pi - 1e-12 <= P <= Theta + np.pi + 1e-12:
            vc, Dc = v_at_flux(n, Theta, kappa, P, eps)
            if vc < best[0]:
                best = (vc, Dc, P)
    return best


def campo_inutil(n, D, Phi):
    """eps_n con el multiplicador de <Z_i> = 0 (campo uniforme mu sum Z_i).

    El campo conmuta con H, así que lambda_max(-4H + mu sum Z) = max_M [lambda_M + mu (2M - n)],
    una función convexa de mu. La inversión X^{(x)n} lleva el sector M al n - M y
    H(Phi) a H(-Phi), con el mismo espectro por conjugación compleja: lambda_M = lambda_{n-M}.
    La función es par y convexa, y su mínimo está en mu = 0. Se comprueba aquí.
    """
    E = np.array([xxz.sector_spectrum(n, M, D, Phi)[0] for M in range(n + 1)])
    m = 2 * np.arange(n + 1) - n
    mus = [0.0] + [(E[i] - E[j]) / (m[j] - m[i]) for i in range(n + 1) for j in range(i + 1, n + 1)]
    return -max(np.min(E + mu * m) for mu in mus) / n


def barrido(args):
    kappa, n, thetas = args
    eps = eps_bethe
    out = []
    for T in thetas:
        v, D, P = v_free(n, T, kappa, eps)
        out.append(dict(kappa=kappa, n=n, Theta=T, v=v, Delta=D, Phi=P,
                        frontera=int(np.isfinite(D) and D <= D_LO + 1e-6)))
    return out


def controles(kappas):
    # (a) eps_n: lo único que el dual toma de Bethe. Se compara punto a punto, con Phi
    # en [0, 2 pi) (incluido el plegado) y Delta en el rango de los óptimos.
    err = max(abs(eps_ed(n, D, P) - eps_bethe(n, D, P))
              for n in (8, 9, 10, 11) for D in (0.3, 0.6, 0.9, 1.0, 1.3, 1.9, 2.7, 4.0, 9.0)
              for P in np.linspace(0, 2 * np.pi, 13))
    check(err < 1e-12, f"(a) eps_n de Bethe = diagonalización exacta, n = 8..11, Phi en [0, 2 pi) (error {err:.1e})")
    err = 0.0
    with open(ROOT / "data" / "neel_u1.csv") as fh:
        for r in csv.DictReader(fh):
            k, n = float(r["kappa"]), int(r["n"])
            if n < 8 or k not in kappas:
                continue
            eps = eps_bethe
            err = max(err, abs(v_free(n, 0.0, k, eps)[0] - float(r["v0"])),
                      abs(v_free(n, np.pi, k, eps)[0] - float(r["vpi"])))
    check(err < 1e-8, f"(b) v(n, 0) y v(n, pi) = data/neel_u1.csv, n = 8..18 (error {err:.1e})")
    err = 0.0
    for n in (8, 9, 10, 11):
        for D in (0.5, 1.0, 1.9, 2.7):
            for P in np.linspace(0, np.pi, 7):
                err = max(err, abs(campo_inutil(n, D, P) - xxz.eps(n, D, P)))
    check(err < 1e-12, f"(c) el campo uniforme no cambia eps_n, n = 8..11, todo Phi (error {err:.1e})")
    n, worst = 10, -np.inf
    for k in kappas:
        for T in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            v = v_free(n, T, k, eps_bethe)[0]
            brute = min(4 * eps_bethe(n, D, P) / (D + 2 * k * np.cos((P - T) / n))
                        for D in np.linspace(0.1, 5, 50) for P in T + np.linspace(-np.pi, np.pi, 49))
            worst = max(worst, v - brute)
    check(worst < 1e-12, f"(d) ninguna malla (Delta, Phi) baja del óptimo, n = 10 (máximo v - malla {worst:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.25])
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 24, 25, 32, 33, 48, 49, 64, 65,
                             96, 97, 128, 129])
    ap.add_argument("--ntheta", type=int, default=33)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--sin-controles", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "monotonia_theta.csv")
    a = ap.parse_args()
    if not a.sin_controles:
        controles(a.kappas)
    thetas = np.linspace(0, np.pi, a.ntheta)
    tareas = [(k, n, thetas) for k in a.kappas for n in sorted(a.ns, reverse=True)]
    with Pool(a.procs) as pool:
        res = pool.map(barrido, tareas, chunksize=1)
    rows = [r for blk in res for r in blk]
    print("\n kappa    n   v(0)-v(pi)    viol/|var|     argmin_T   Phi*-Theta (max)   frontera")
    for blk in sorted(res, key=lambda b: (b[0]["kappa"], b[0]["n"])):
        k, n = blk[0]["kappa"], blk[0]["n"]
        v = np.array([r["v"] for r in blk])
        sgn = 1 if n % 2 == 0 else -1
        var = abs(v[0] - v[-1])
        viol = np.max(sgn * np.diff(v))
        amin = thetas[np.argmin(v)]
        dphi = max(abs(r["Phi"] - r["Theta"]) for r in blk)
        front = sum(r["frontera"] for r in blk)
        print(f" {k:5.2f} {n:4d}  {v[0] - v[-1]:+.3e}   {viol / var if var > 0 else np.nan:+.2e}   "
              f"{amin / np.pi:5.3f} pi   {dphi:.3e}         {front}")
        if viol > 1e-12 + 1e-6 * var:
            check(False, f"kappa={k}, n={n}: v no es monótona (viol = {viol:.2e}, variación {var:.2e})")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"escrito {a.out}")
    sys.exit(0 if OK else 1)

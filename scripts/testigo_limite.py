"""Problema abierto 2 de robustez_mps.tex: la vía del testigo límite.

Datos U(1) del anillo de qubits con kappa < 1 y n par. El testigo límite es el
XXZ con Delta_inf(kappa) (Proposición prop:limite). Se comprueban numéricamente
las dos hipótesis de la vía:

  (i)  la energía por enlace del anillo retorcido, eps_n(Delta_inf, Theta),
       converge a eps_inf(Delta_inf) exponencialmente, uniformemente en Theta;
  (ii) la marginal de arista del fundamental del anillo retorcido, en el marco
       girado de la arista, converge a los datos con v_inf.

Con ellas, para todo Theta,
    L_n = v_inf / (1 + 19 d^2 delta_n)  <=  v(n, Theta)  <=  U_n = 4 eps_n / (Delta_inf + 2 kappa),
la superior por el dual con el XXZ alineado (Phi = Theta) y la inferior por el
Lema lem:dilucionti aplicado al fundamental retorcido, con
delta_n = || rho_GS - rho_datos(v_inf) ||_1.

Marginal del fundamental. Es invariante por la U(1), con <Z_i> = 0 (sector
M = n/2 e invariancia por traslación), así que la fijan
    zz = <Sz_i Sz_{i+1}> = (1/n) dE0/dDelta      (Hellmann-Feynman, Richardson),
    g  = e^{i Phi/n} <S+_i S-_{i+1}>,  Re g = E0/n - Delta zz,  Im g = -dE0/dPhi,
y en el marco girado rho = (I + 4 zz ZZ)/4 + conj(g) S+S- + g S-S+. Los datos con
v son zz = -v/4 y g = -kappa v / 2. Con dzz y dg las diferencias,
    delta = 2 |dzz| + 2 max(|dzz|, |dg|).
La estacionariedad del mínimo que define Delta_inf hace que la cadena infinita
reproduzca exactamente los datos: zz_inf = -eps_inf', g_inf = -2 kappa eps_inf'.

Control: con diagonalización exacta (n <= 16) se calcula la marginal del
autovector y se compara con la reconstruida desde Bethe.

Salida: data/testigo_limite.csv y una tabla por pantalla.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import bethe, xxz  # noqa: E402
from neel_bethe import eps_inf, limit  # noqa: E402
from mpmath import mp, mpf  # noqa: E402

D2 = 4          # d^2 para qubits
R_DIL = 1 / (19 * D2)
OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def limit_mp(kappa):
    """Delta_inf, v_inf, eps_inf y eps_inf' con 30 cifras (serie exacta, Delta > 1).

    eps_inf = sinh(g) [1/2 + 2 sum_m q^{2m}/(1+q^{2m})] - Delta/4, q = e^{-g}, Delta = cosh g;
    Delta_inf resuelve eps' (Delta + 2 kappa) = eps (estacionariedad del cociente).
    """
    mp.dps = 30
    e = lambda D: mp.sinh(mp.acosh(D)) * (mpf(1) / 2 + 2 * mp.nsum(
        lambda m: mp.exp(-2 * m * mp.acosh(D)) / (1 + mp.exp(-2 * m * mp.acosh(D))), [1, mp.inf])) - D / 4
    D0, _ = limit(kappa)
    Dinf = mp.findroot(lambda D: mp.diff(e, D) * (D + 2 * kappa) - e(D), mpf(D0))
    ei, de = e(Dinf), mp.diff(e, Dinf)
    return float(Dinf), float(4 * ei / (Dinf + 2 * kappa)), float(ei), float(de)


def bethe_marginal(n, D, Phi, h=2e-3):
    """(E0/n, zz, g) del fundamental del anillo, desde Bethe."""
    S = bethe.solver(n)
    b = S.ground_branch()
    E, dEdPhi = S.energy_and_flux_derivative(D, Phi, b)
    E2 = lambda x: S.energy(x, Phi, b)
    d1 = (E2(D + h) - E2(D - h)) / (2 * h)
    d2 = (E2(D + h / 2) - E2(D - h / 2)) / h
    zz = (4 * d2 - d1) / 3 / n
    S.solve(D, Phi, b)  # devuelve el arranque en caliente al punto central
    g = complex(E / n - D * zz, -dEdPhi)
    return E / n, zz, g


def ed_marginal(n, D, Phi):
    """(E0/n, zz, g) desde el autovector de diagonalización exacta (sector M = n/2)."""
    H = xxz.hamiltonian(n, n // 2, D, Phi).toarray()
    w, V = np.linalg.eigh(H)
    psi = V[:, 0]
    diag, K = xxz.sector_operators(n, n // 2)
    zz = float(np.real(np.vdot(psi, diag * psi))) / n
    g = np.exp(1j * Phi / n) * np.vdot(psi, K @ psi) / n
    return w[0] / n, zz, complex(g)


def delta(zz, g, zz0, g0):
    dzz, dg = abs(zz - zz0), abs(g - g0)
    return 2 * dzz + 2 * max(dzz, dg)


def sizes(nmax):
    ns = list(range(8, 41, 2)) + list(range(44, 101, 4)) + list(range(110, 301, 10)) \
        + list(range(320, 1001, 40))
    return [n for n in ns if n <= nmax]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--nmax", type=int, default=1000)
    ap.add_argument("--floor", type=float, default=1e-10)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "testigo_limite.csv")
    args = ap.parse_args()
    thetas = np.linspace(0, np.pi, 9)
    rows = []
    for kappa in args.kappas:
        Dinf, vinf, e_inf, de = limit_mp(kappa)
        check(abs(e_inf - eps_inf(Dinf)) < 1e-12, "eps_inf: serie con mpmath = guion de Bethe")
        zz0, g0 = -vinf / 4, -kappa * vinf / 2
        print(f"\nkappa = {kappa}: Delta_inf = {Dinf:.10f}, v_inf = {vinf:.12f}")
        # La cadena infinita reproduce los datos (estacionariedad)
        check(abs(-de - zz0) < 1e-13 and abs((-e_inf + Dinf * de) - g0) < 1e-13,
              f"marginal de la cadena infinita = datos con v_inf "
              f"(|dzz| = {abs(-de - zz0):.1e}, |dg| = {abs(-e_inf + Dinf * de - g0):.1e})")
        # Control contra diagonalización exacta
        err = 0.0
        # En Theta = pi los dos niveles del doblete se cruzan y el autovector de la
        # diagonalización es una combinación arbitraria; se compara en [0, pi).
        for n in (8, 10, 12, 14):
            for T in thetas[:-1]:
                eb, zb, gb = bethe_marginal(n, Dinf, T)
                ee, ze, ge = ed_marginal(n, Dinf, T)
                err = max(err, abs(eb - ee), abs(zb - ze), abs(gb - ge))
        check(err < 1e-8, f"marginal desde Bethe = marginal del autovector, n = 8..14, Theta < pi (error {err:.1e})")
        print("    n   max_T|eps_n-eps_inf|  max_T delta_n   max_T U_n - v_inf   v_inf - min_T L_n   "
              "max U - min L    xi_e     xi_delta")
        prev = None
        for n in sizes(args.nmax):
            de_n, dl_n, U, L = [], [], [], []
            for T in thetas:
                e, zz, g = bethe_marginal(n, Dinf, T)
                en = -e
                dl = delta(zz, g, zz0, g0)
                de_n.append(abs(en - e_inf))
                dl_n.append(dl)
                U.append(4 * en / (Dinf + 2 * kappa))
                L.append(vinf / (1 + dl / R_DIL))
            r = dict(kappa=kappa, n=n, Dinf=Dinf, vinf=vinf, de=max(de_n), delta=max(dl_n),
                     Umax=max(U) - vinf, Lmin=vinf - min(L), sens=max(U) - min(L),
                     delta_T0=dl_n[0], delta_Tpi=dl_n[-1], de_T0=de_n[0], de_Tpi=de_n[-1])
            xe = xd = np.nan
            if prev is not None:
                xe = -(n - prev["n"]) / np.log(r["de"] / prev["de"])
                xd = -(n - prev["n"]) / np.log(r["delta"] / prev["delta"])
            print(f" {n:4d}   {r['de']:.3e}           {r['delta']:.3e}      {r['Umax']:+.3e}         "
                  f"{r['Lmin']:.3e}           {r['sens']:.3e}    {xe:7.2f}  {xd:7.2f}", flush=True)
            rows.append(r)
            prev = r
            if r["delta"] < args.floor and r["de"] < args.floor:
                break
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nescrito {args.out}")
    sys.exit(0 if OK else 1)

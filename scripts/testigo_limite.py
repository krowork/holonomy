"""Problema abierto 2 de robustez_mps.tex: la vía del testigo límite.

Datos U(1) del anillo de qubits con kappa < 1 y n par. El testigo límite es el
XXZ con Delta_inf(kappa) (Proposición prop:limite). Se comprueban numéricamente
las dos hipótesis de la vía:

  (E)  la energía por enlace del anillo retorcido, eps_n(Delta_inf, Theta),
       converge a eps_inf(Delta_inf) exponencialmente, uniformemente en Theta;
  (M)  la marginal de arista del fundamental del anillo retorcido, en el marco
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

Reducción (M) <= (E) (problema 2): con eta_n la corrección de tamaño finito
|E0/n + e_inf| del sector S^z = 0, uniforme en Theta y en Delta_inf y Delta_inf +- s,
    delta_n <= 2 eta_n + (2 sqrt2 + 4 (1 + Delta_inf) sqrt K) sqrt(eta_n),
con K = max |e_inf''| en [Delta_inf - R, Delta_inf + R] y s = 2 sqrt(eta_n / K) <= R.
Se elige eta_n por punto fijo: se parte de eta en Delta_inf, se evalúa en
Delta_inf +- s y, si allí |f| supera eta, se toma eta = 1,01 |f| y se repite.
Los n en que s > R (los pequeños) se omiten (eta = cota_red = nan en el CSV).

Control: con diagonalización exacta (n = 8..14, Theta < pi) se calcula la
marginal del autovector y se compara con la reconstruida desde Bethe.

Resultados con las opciones por defecto (kappa = 0,4, 0,6, 0,8; nueve Theta en
[0, pi]; suelo 3e-11, que corta en n = 120, 320 y 1000):

  kappa  Delta_inf     v_inf      K       c      error Bethe-ED  sens. (n final)
  0,4    2,6600374602  0,8736054  0,02660  5,216  1,7e-12         3,4e-10 (120)
  0,6    1,8501723949  0,7709179  0,05582  5,522  2,4e-11         1,4e-9  (320)
  0,8    1,4001744953  0,6750233  0,06093  5,198  2,0e-11         3,8e-7  (1000)

  Reducción (M) <= (E): se cumple en todos los n con s <= R, desde n = 22, 32 y
  38; el cociente delta_n / (2 eta_n + c sqrt(eta_n)) es como mucho 0,039, 0,074
  y 0,21 (en ese primer n) y baja a 4,8e-6, 2,6e-5 y 5,2e-4 en el n final: la
  cota cae como e^{-n/(2 xi)} y delta_n como e^{-n/xi}.

Salida: data/testigo_limite.csv (columnas de la vía más eta, s_red y cota_red)
y una tabla por pantalla. Tarda decenas de minutos, casi todo en kappa = 0,8.
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


R_RED = 0.2     # semianchura del entorno de Delta_inf en la reducción (M) <= (E)


def e_mp(D):
    """eps_inf(Delta) con mpmath, Delta > 1."""
    g = mp.acosh(D)
    return mp.sinh(g) * (mpf(1) / 2 + 2 * mp.nsum(
        lambda m: mp.exp(-2 * m * g) / (1 + mp.exp(-2 * m * g)), [1, mp.inf])) - D / 4


def curvatura(Dinf, R=R_RED, k=41):
    """K = max |eps_inf''| en [Dinf - R, Dinf + R] (malla de k puntos, con margen del 5 %)."""
    mp.dps = 30
    return 1.05 * max(abs(float(mp.diff(e_mp, mpf(D), 2))) for D in np.linspace(Dinf - R, Dinf + R, k))


def reduccion(n, Dinf, K, thetas, eta0, R=R_RED, it=6):
    """Punto fijo de eta_n para la reducción (M) <= (E). Devuelve (eta, s) o (nan, nan) si s > R."""
    S = bethe.solver(n)
    b = S.ground_branch()
    eta = eta0
    for _ in range(it):
        s = 2 * np.sqrt(eta / K)
        if s > R:
            return np.nan, np.nan
        es = max(abs(S.energy(Dinf + sg * s, T, b) / n + eps_inf(Dinf + sg * s))
                 for T in thetas for sg in (1, -1))
        for T in thetas:  # arranque en caliente de vuelta al punto central
            S.solve(Dinf, T, b)
        if es <= eta:  # consistente: eta acota |f| en Delta_inf y en Delta_inf +- s(eta)
            return eta, s
        eta = 1.01 * es  # holgura del 1 %: la iteración converge en pocas vueltas
    return np.nan, np.nan


def limit_mp(kappa):
    """Delta_inf, v_inf, eps_inf y eps_inf' con 30 cifras (serie exacta, Delta > 1).

    eps_inf = sinh(g) [1/2 + 2 sum_m q^{2m}/(1+q^{2m})] - Delta/4, q = e^{-g}, Delta = cosh g;
    Delta_inf resuelve eps' (Delta + 2 kappa) = eps (estacionariedad del cociente).
    """
    mp.dps = 30
    e = e_mp
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
    ap.add_argument("--floor", type=float, default=3e-11)
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
        K = curvatura(Dinf)
        c_red = 2 * np.sqrt(2) + 4 * (1 + Dinf) * np.sqrt(K)
        print(f"    reducción (M) <= (E): K = {K:.6f}, c = {c_red:.6f}")
        ok_red, n_red = True, 0
        print("    n   max_T|eps_n-eps_inf|  max_T delta_n   max_T U_n - v_inf   v_inf - min_T L_n   "
              "max U - min L    xi_e     xi_delta   eta_n      cota_red   delta/cota")
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
            eta, s_red = reduccion(n, Dinf, K, thetas, r["de"])
            cota = 2 * eta + c_red * np.sqrt(eta)
            r.update(eta=eta, s_red=s_red, cota_red=cota)
            if np.isfinite(cota):
                n_red += 1
                ok_red &= r["delta"] <= cota
            xe = xd = np.nan
            if prev is not None:
                xe = -(n - prev["n"]) / np.log(r["de"] / prev["de"])
                xd = -(n - prev["n"]) / np.log(r["delta"] / prev["delta"])
            print(f" {n:4d}   {r['de']:.3e}           {r['delta']:.3e}      {r['Umax']:+.3e}         "
                  f"{r['Lmin']:.3e}           {r['sens']:.3e}    {xe:7.2f}  {xd:7.2f}   {eta:.3e}  {cota:.3e}  "
                  f"{r['delta'] / cota:.3e}", flush=True)
            rows.append(r)
            prev = r
            if r["delta"] < args.floor and r["de"] < args.floor:
                break
        check(ok_red and n_red > 0, f"reducción (M) <= (E): delta_n <= 2 eta_n + c sqrt(eta_n) "
                                    f"en los {n_red} tamaños con s <= {R_RED}")
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nescrito {args.out}")
    sys.exit(0 if OK else 1)

"""Correcciones de tamaño finito del anillo XXZ con Delta grande (docs/xxz_anillo_retorcido.tex).

Con n par y Delta = 1/epsilon grande, el Teorema de la energía del artículo predice
    |E0(n, 0) + n e_inf(Delta)| <= C (c/Delta)^{n/2},
y el desdoblamiento R_n = E0(n, pi) - E0(n, 0) es del mismo orden. Comprobaciones, con
diagonalización exacta en los sectores de magnetización mínima (n = 4..18 par) y la e_inf exacta
de Yang-Yang:
  1. E0(n, 0) + n e_inf < 0 (el anillo sin flujo queda por debajo de la cadena infinita);
  2. el cociente cada dos sitios de |E0(n, 0) + n e_inf| es < 2/Delta y crece con n (la tasa
     exacta, e^{-1/xi} con xi = 2 xi_JKM, es algo menor que 2/Delta);
  3. con flujo pi la corrección es de orden superior: |E0(n, pi) + n e_inf| <= 0,05 R_n desde n = 10;
  4. la tasa exacta: el cociente del último par de n (16 -> 18) frente a e^{-2/xi} sqrt(16/18), con
     xi = 2 xi_JKM y el prefactor n^{-1/2} de R_n, está a menos de un 2 % de 1.
  5. el orden principal en epsilon = 1/Delta (Observación del Teorema de la energía): en las unidades
     de H~ = (4/Delta) H, [E0(n,0) + n e_inf] 4 epsilon / epsilon^{n/2} -> -n C_{n/2-1} 2^{2-n/2},
     con C_k los números de Catalan, a menos de un 0,1 % con epsilon = 1/160 (n <= 10) y 1/80 (n = 12).
  6. la cota del Teorema de la energía con sus constantes, |E0(n,0) + n e_inf| <= 1,14 n (45/Delta)^{n/2-1},
     con Delta = 50, 100 y 200 y n = 4..14 par, la constante de contracción K(45) <= 1/2 del Lema de
     contracción y K'(54) <= 1/2 del de contracción con flujo.
  7. marginales retorcidas (Teorema del estado de prueba): la distancia en norma traza
     delta_n(Theta) = 2|dzz| + 2 max(|dzz|, |dg|) entre la marginal del fundamental del sector S^z = 0
     de H_n(Delta, Theta) (gauge uniforme) y la girada de la cadena infinita (zz = -e_inf',
     g = -e_inf + Delta e_inf') cumple delta_n <= 2 (2/Delta)^{n/2} desde n = 10, con Theta = 0, pi/2
     y pi, y delta_n(pi/2)/delta_n(0) difiere de 1 en menos de un 3 % (uniformidad en Theta).
Escribe data/xxz_ising_tamfinito.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import xxz  # noqa: E402
from neel_bethe import eps_inf  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def xi_jkm(D):
    g = np.arccosh(D)
    m = np.arange(1, 400)
    return 1 / (g + 2 * np.sum((-1.0) ** m / m * np.tanh(m * g)))


rows = []
NS = list(range(4, 19, 2))
for D in [3.0, 5.0, 8.0, 12.0]:
    einf = eps_inf(D)
    d0, dpi, R = [], [], []
    for n in NS:
        e0 = -n * xxz.eps(n, D, 0.0)
        epi = -n * xxz.eps(n, D, np.pi)
        d0.append(e0 + n * einf)
        dpi.append(epi + n * einf)
        R.append(epi - e0)
        rows.append(dict(Delta=D, n=n, E0_0_mas_n_einf=d0[-1], E0_pi_mas_n_einf=dpi[-1], R_n=R[-1]))
    d0, dpi, R = map(np.array, (d0, dpi, R))
    rat = np.abs(d0[1:] / d0[:-1])
    print(f"\nDelta = {D}: |E0(n,0)+n e_inf| = " + ", ".join(f"{x:.2e}" for x in np.abs(d0)))
    print("  cociente cada dos sitios / (2/Delta): " + ", ".join(f"{x:.4f}" for x in rat / (2 / D)))
    check(np.all(d0 < 0), f"Delta = {D}: E0(n,0) + n e_inf < 0 en n = 4..18")
    check(np.all(rat < 2 / D) and np.all(np.diff(rat) > 0),
          f"Delta = {D}: cociente cada dos sitios < 2/Delta y creciente")
    sel = np.array(NS) >= 10
    check(np.all(np.abs(dpi[sel]) <= 0.05 * R[sel]),
          f"Delta = {D}: |E0(n,pi) + n e_inf| <= 0,05 R_n desde n = 10 "
          f"(máximo {np.max(np.abs(dpi[sel]) / R[sel]):.1e})")
    tasa = np.exp(-2 / (2 * xi_jkm(D)))
    q = rat[-1] / (tasa * np.sqrt(NS[-2] / NS[-1]))
    check(abs(q - 1) < 0.02, f"Delta = {D}: cociente(n=16->18) / [e^(-2/xi) sqrt(16/18)] = {q:.4f} "
          f"(e^(-2/xi) = {tasa:.5f}, 2/Delta = {2 / D:.5f})")

# --- 5: orden principal (números de Catalan) ---------------------------------------
from math import comb  # noqa: E402

print()
for n, D in [(4, 160.0), (6, 160.0), (8, 160.0), (10, 160.0), (12, 80.0)]:
    k = n // 2
    eps = 1 / D
    d = (-n * xxz.eps(n, D, 0.0) + n * eps_inf(D)) * 4 / D
    pred = -n * (comb(2 * k - 2, k - 1) // k) / 2 ** (k - 2)
    check(abs(d / eps ** k / pred - 1) < 1e-3,
          f"n = {n}: coeficiente de epsilon^{k} = {d / eps ** k:.4f} (Catalan: {pred})")

# --- 6: cota del teorema con constantes explícitas ---------------------------------
def K(M):
    d = 8 / M
    return np.exp(d) - 1 + d * np.exp(d) + (3 + d) * np.exp(d) / M


def B(M):
    s4 = 4 / M
    return 2 * ((np.exp(s4) - 1 - s4) + (1 + np.exp(s4)) / M)


print()
check(K(45) <= 0.5 < K(44), f"K(45) = {K(45):.4f} <= 1/2 < K(44) = {K(44):.4f}")
check(B(45) * 45 / 4 <= 1.14, f"B(45) * 45/4 = {B(45) * 45 / 4:.4f} <= 1,14")


def Kflujo(M):
    d = 8 / M
    return (np.exp(d) - 1 + d * np.exp(d) + (3 + d) * np.exp(d) / M
            + np.pi * (1.5 + d / 2) * np.exp(d / 2) / M)


check(Kflujo(54) <= 0.5 < Kflujo(53),
      f"contracción con flujo: K'(54) = {Kflujo(54):.4f} <= 1/2 < K'(53) = {Kflujo(53):.4f}")
peor = 0.0
for D in [50.0, 100.0, 200.0]:
    for n in range(4, 15, 2):
        d = abs(-n * xxz.eps(n, D, 0.0) + n * eps_inf(D))
        peor = max(peor, d / (1.14 * n * (45 / D) ** (n / 2 - 1)))
check(peor <= 1, f"|E0 + n e_inf| / [1,14 n (45/Delta)^(n/2-1)] <= {peor:.2e} (Delta = 50..200, n = 4..14)")

# --- 7: marginales retorcidas ------------------------------------------------------
def de_inf(D, h=1e-6):
    return (eps_inf(D + h) - eps_inf(D - h)) / (2 * h)


print()
for D in [5.0, 8.0, 12.0]:
    zzi, gi = -de_inf(D), -eps_inf(D) + D * de_inf(D)
    for n in range(10, 19, 2):
        diag, Kop = xxz.sector_operators(n, n // 2)
        dl = {}
        for Th in [0.0, np.pi / 2, np.pi]:
            _, v = xxz.ground_state(n, D, Th)
            zz = np.real(np.vdot(v, diag * v)) / n
            g = np.exp(1j * Th / n) * np.vdot(v, Kop @ v) / n
            dzz, dg = abs(zz - zzi), abs(g - gi)
            dl[Th] = 2 * dzz + 2 * max(dzz, dg)
        cota = 2 * (2 / D) ** (n / 2)
        check(max(dl.values()) <= cota and abs(dl[np.pi / 2] / dl[0.0] - 1) < 0.03,
              f"Delta = {D}, n = {n}: delta_n = {dl[0.0]:.2e}, {dl[np.pi / 2]:.2e}, {dl[np.pi]:.2e} "
              f"<= 2 (2/Delta)^(n/2) = {cota:.2e}")

out = ROOT / "data" / "xxz_ising_tamfinito.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nescrito {out.relative_to(ROOT)}")
sys.exit(0 if OK else 1)

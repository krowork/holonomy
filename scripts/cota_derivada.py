"""Cotas rigurosas de tamaño finito para e_n y su derivada en Delta (robustez_mps.tex, Lemas
lem:tamfinito y lem:convexa, Corolario cor:velocidad).

  Lema lem:tamfinito. Para todo n >= 3, Delta real y flujo Phi,
      |e_n(Delta, Phi) - e_inf(Delta)| <= 2 ||h|| / n = (1 + |Delta|/2) / n,
  con h = 1/2 (S+S- + h.c.) + Delta SzSz (||h|| = 1/2 + |Delta|/4). La cadena abierta de N sitios
  cumple |E0 + N e_inf| <= ||h|| por subaditividad (lema de Fekete en los dos sentidos), y el anillo
  retorcido es la cadena abierta más un enlace (el flujo se concentra en él con un gauge).

  Lema lem:convexa. e_n es convexa en Delta. Si |e_n - e_inf| <= eta en [Delta - r, Delta + r] y
  |e_inf''| <= M allí, todo subgradiente g de e_n en Delta cumple |g - e_inf'(Delta)| <= M t/2 + 2 eta/t
  para todo t en (0, r]; con t = min(2 sqrt(eta/M), r), es 2 sqrt(M eta) si 2 sqrt(eta/M) <= r.

  Corolario cor:velocidad. Con F = 4 e/(Delta + 2 kappa), eps'_n <= 4 cota_e'/(Delta + 2 kappa)
  + 4 eta/(Delta + 2 kappa)^2 = O(n^{-1/2}), y si a = F_inf''(Delta_inf) > 0,
  |Delta*_i - Delta_inf| <= 2 eps'_n / a.

Comprobaciones:
  1. la cota del Lema: diagonalización exacta en todos los sectores (n = 3..8, Delta en [-0,95; 3],
     Phi en [0, 2 pi]) y Bethe (n hasta 201, Delta de -0,8 a 6);
  2. la cota de la derivada frente a |e_n' - e_inf'| (diferencias finitas de Bethe) en Delta_inf;
  3. la de la velocidad frente a |Delta* - Delta_inf| de data/neel_bethe.csv;
  4. a = F_inf''(Delta_inf) > 0 en los kappa de la tabla (con la e_inf exacta);
  5. el sándwich de la Conjetura conj:sandwich, e_n(Delta, pi) <= e_inf(Delta) <= e_n(Delta, 0) con n
     par: diagonalización exacta en todos los sectores (n = 4..12) y Bethe (n hasta 200), Delta de
     -0,9 a 10 (con Delta <= 0, holonomy.bethe_p). Se descartan las diferencias por debajo de 1e-13
     (redondeo).
"""
import csv
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import bethe, bethe_p, xxz  # noqa: E402
from neel_bethe import eps_inf  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def d1(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def d2(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def eps_n(n, D, P):
    return bethe.eps(n, D, P) if D > 0.05 else bethe_p.eps(n, D, P)


if __name__ == "__main__":
    print("1. n |e_n - e_inf| / (1 + |Delta|/2) <= 1")
    w_ed = max(abs(xxz.eps_full(n, D, P) - eps_inf(D)) * n / (1 + abs(D) / 2)
               for n in range(3, 9) for D in (-0.95, -0.5, 0.0, 0.7, 1.0, 3.0) for P in np.linspace(0, 2 * np.pi, 7))
    w_b = max(abs(eps_n(n, D, P) - eps_inf(D)) * n / (1 + abs(D) / 2)
              for D in (-0.8, -0.4, 0.3, 0.5, 1.0, 1.4, 2.66, 6.0) for n in (10, 11, 40, 41, 200, 201)
              for P in np.linspace(0, np.pi, 5))
    print(f"   máximo con diagonalización exacta (n = 3..8): {w_ed:.3f}; con Bethe (n = 10..201): {w_b:.3f}")
    check(w_ed <= 1 and w_b <= 1, "   cota del Lema lem:tamfinito")

    print("2-4. derivada y velocidad, en Delta_inf (cota rigurosa frente a valor real)")
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "neel_bethe.csv")) if r["frontera"] == "0"]
    ok2 = ok3 = ok4 = True
    for par, kappa in (("par", 0.4), ("impar", 0.4), ("impar", 0.8), ("par", 1.25), ("impar", 1.25), ("par", 1.5)):
        g = [r for r in rows if r["paridad"] == par and float(r["kappa"]) == kappa]
        Dinf = float(g[0]["Dinf"])
        a = d2(lambda x: 4 * eps_inf(x) / (x + 2 * kappa), Dinf)
        ok4 &= a > 0
        r_int = min(0.5, 0.98 * (Dinf - 1)) if Dinf > 1 else min(0.5, 0.98 * (1 - Dinf), 0.98 * (Dinf + 1))
        M = max(abs(d2(eps_inf, x)) for x in np.linspace(Dinf - r_int, Dinf + r_int, 41))
        for r in (g[len(g) // 3], g[-1]):
            n = int(r["n"])
            T0 = 0.0 if par == "par" else np.pi
            eta = (1 + (abs(Dinf) + r_int) / 2) / n
            t = min(2 * np.sqrt(eta / M), r_int)
            cota_d = M * t / 2 + 2 * eta / t
            real_d = max(abs(d1(lambda x: bethe.eps(n, x, T), Dinf) - d1(eps_inf, Dinf)) for T in (T0, np.pi - T0))
            epsp = 4 * cota_d / (Dinf + 2 * kappa) + 4 * eta / (Dinf + 2 * kappa) ** 2
            real_v = max(abs(float(r["D0"]) - Dinf), abs(float(r["D1"]) - Dinf))
            ok2 &= real_d <= cota_d
            ok3 &= real_v <= 2 * epsp / a
            print(f"   {par:5s} kappa = {kappa:4.2f} n = {n:5d}: |e_n' - e_inf'| = {real_d:.1e} <= {cota_d:.1e};  "
                  f"|Delta* - Delta_inf| = {real_v:.1e} <= {2 * epsp / a:.1e}  (a = {a:.4f})")
    check(ok2, "   cota de la derivada (Lema lem:convexa)")
    check(ok3, "   cota de la velocidad (Corolario cor:velocidad)")
    check(ok4, "   a = F_inf''(Delta_inf) > 0")

    print("5. sándwich e_n(pi) <= e_inf <= e_n(0), n par")
    viol, npts = [], 0
    for D in (-0.9, -0.6, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.66, 4.0, 6.0, 10.0):
        for n in (4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 100, 200):
            if n <= 12:
                e0, epi = xxz.eps_full(n, D, 0.0), xxz.eps_full(n, D, np.pi)
            else:
                e0, epi = eps_n(n, D, 0.0), eps_n(n, D, np.pi)
            ei = eps_inf(D)
            npts += 1
            if e0 - ei < -1e-13 or ei - epi < -1e-13:
                viol.append((D, n, e0 - ei, ei - epi))
    print(f"   {npts} puntos; violaciones: {viol}")
    check(not viol, "   e_n(pi) <= e_inf <= e_n(0) en todos los puntos")
    sys.exit(0 if OK else 1)

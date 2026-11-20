"""Problema abierto 3 de robustez_mps.tex: el paso de R_n a la caída s (Proposición prop:paso).

Con K(Delta) = 4 [E0(Delta, Theta_1) - E0(Delta, Theta_0)] / (n (Delta + 2 kappa)) y Delta*_0, Delta*_1
los óptimos del Lema de reducción en Theta_0 y Theta_1:
  (i)   K(Delta*_0) <= s <= K(Delta*_1)           (cota de Kohn por los dos lados, rigurosa);
  (ii)  |Delta*_i - Delta_inf| <= 2 eps'_n / F_inf''(Delta_inf) (1 + o(1)) si |F'_{n,i} - F'_inf| <= eps'_n,
        y |s/K(Delta_inf) - 1| <= (2 eps'_n / F_inf'') sup |d log K / d Delta| (1 + o(1));
  (iii) a segundo orden, s = K(Delta_medio), Delta_medio = (Delta*_0 + Delta*_1)/2, y
        s/K(Delta_inf) - 1 = (d log K/d Delta)(Delta_medio - Delta_inf), con
        Delta_medio - Delta_inf = -(d/d Delta)[(delta_0 + delta_1)/2] / F_inf'', delta_i = F_{n,i} - F_inf.
Con las correcciones de tamaño finito de cada régimen, el error relativo es exponencial con n par y gap,
c_1/n con n impar y gap, y c_2/n^2 en el caso crítico, con

    c_1 = (1/F_inf'') d/dDelta[4 E_gap/(Delta + 2 kappa)] d/dDelta log[c/(Delta + 2 kappa)]   (n impar, gap),
    c_2 = -(1/F_inf'') d/dDelta[4 (pi v/6 - pi^2 D/4)/(Delta + 2 kappa)] d/dDelta log[D/(Delta + 2 kappa)]
                                                                                    (crítico, n par),

E_gap = E_sp(0) y c = (pi^2/8) E_sp''(0) del espinón (Observación obs:espinon), v = pi sin(gamma)/(2 gamma)
la velocidad y D la rigidez de Shastry-Sutherland.

Comprobaciones con data/neel_bethe.csv (y holonomy.bethe para K):
  1. s = K(Delta_medio) en el mayor n de cada grupo, a 1e-6 (frente a ~4e-3 con K(Delta_inf), n impar);
  2. n par con gap (kappa = 0,4, 0,6): s/K(Delta_inf) - 1 cae como n^{1/2} e^{-n/xi};
  3. n impar con gap (kappa = 0,4, 0,6): n (s/K(Delta_inf) - 1) -> c_1, a 0,5 %;
  4. crítico, n par (kappa = 1,25, 1,5): n^2 (s/K(Delta_inf) - 1) -> c_2, a 0,5 %.
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
from holonomy import bethe  # noqa: E402
from mpmath import mp, mpf  # noqa: E402
from neel_bethe import eps_inf  # noqa: E402
from xi_tunel import candidates  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def F2(D, kappa, h=1e-4):
    F = lambda x: 4 * eps_inf(x) / (x + 2 * kappa)  # noqa: E731
    return (F(D + h) - 2 * F(D) + F(D - h)) / h ** 2


def espinon(D):
    """(E_gap, c) del XXZ con Delta = cosh(gamma) > 1."""
    g = mp.acosh(mpf(D))
    q = mp.exp(-g)
    k = (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q)) ** 2
    kp = mp.sqrt(1 - k ** 2)
    A = mp.sinh(g) / mp.pi * mp.ellipk(k ** 2)
    return A * kp, mp.pi / 8 * mp.sinh(g) * mp.ellipk(k ** 2) * k ** 2 / kp


def K(n, D, kappa, par):
    T0 = 0.0 if par == "par" else np.pi
    return 4 * (bethe.eps(n, D, T0) - bethe.eps(n, D, np.pi - T0)) / (D + 2 * kappa)


def cociente(r, par, kappa, D):
    return float(r["s"]) / K(int(r["n"]), D, kappa, par) - 1


if __name__ == "__main__":
    mp.dps = 30
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "neel_bethe.csv")) if r["frontera"] == "0"]
    grupo = lambda par, kappa: [r for r in rows if r["paridad"] == par and float(r["kappa"]) == kappa]  # noqa: E731

    print("1. s = K(Delta_medio) en el mayor n")
    ok1 = True
    for par, kappa in (("impar", 0.4), ("impar", 0.6), ("par", 1.25), ("par", 1.5)):
        r = grupo(par, kappa)[-1]
        n, Dinf = int(r["n"]), float(r["Dinf"])
        Dm = (float(r["D0"]) + float(r["D1"])) / 2
        a, b = cociente(r, par, kappa, Dinf), cociente(r, par, kappa, Dm)
        ok1 &= abs(b) < 1e-6
        print(f"   {par} kappa = {kappa}, n = {n}: s/K(Delta_inf) - 1 = {a:+.2e}, s/K(Delta_medio) - 1 = {b:+.2e}")
    check(ok1, "   s = K(Delta_medio) a 1e-6")

    print("2. n par con gap: s/K(Delta_inf) - 1")
    ok2 = True
    for kappa in (0.4, 0.6):
        g = [r for r in grupo("par", kappa) if float(r["s"]) > 1e-9]
        Dinf = float(g[0]["Dinf"])
        sel = [g[i] for i in np.unique(np.linspace(len(g) // 3, len(g) - 1, 4).astype(int))]
        q = [abs(cociente(r, "par", kappa, Dinf)) for r in sel]
        ns = [int(r["n"]) for r in sel]
        xi = candidates(Dinf)[1]
        pend = np.log(q[-2] / q[-1]) / (ns[-1] - ns[-2])
        pred = 1 / xi - 1 / (ns[-1] + ns[-2])          # n^{1/2} e^{-n/xi}: 1/xi - 1/(2 n medio)
        ok2 &= q[-1] < 1e-4 and abs(pend / pred - 1) < 0.05
        print(f"   kappa = {kappa}: " + "  ".join(f"n = {n}: {x:.1e}" for n, x in zip(ns, q))
              + f";  pendiente {pend:.4f}, predicha {pred:.4f}")
    check(ok2, "   cae como n^{1/2} e^{-n/xi}, xi = 2 xi_JKM (pendiente a 5 %)")

    print("3. n impar con gap: n (s/K(Delta_inf) - 1) -> c_1")
    ok3 = True
    for kappa in (0.4, 0.6):
        g = grupo("impar", kappa)
        Dinf = float(g[0]["Dinf"])
        c1 = float(mp.diff(lambda x: 4 * espinon(x)[0] / (x + 2 * kappa), Dinf)
                   * mp.diff(lambda x: mp.log(espinon(x)[1] / (x + 2 * kappa)), Dinf)) / F2(Dinf, kappa)
        pts = [(int(r["n"]), int(r["n"]) * cociente(r, "impar", kappa, Dinf)) for r in g[-5:]]
        x = np.array([1 / p[0] for p in pts])
        lim = np.polyfit(x, [p[1] for p in pts], 1)[1]
        ok3 &= abs(lim / c1 - 1) < 5e-3
        print(f"   kappa = {kappa}: n (s/K - 1) = {pts[-1][1]:.4f} (n = {pts[-1][0]}), extrapolado {lim:.4f}, "
              f"c_1 = {c1:.4f}")
    check(ok3, "   coincide con c_1 a 0,5 %")

    print("4. crítico, n par: n^2 (s/K(Delta_inf) - 1) -> c_2")
    ok4 = True
    for kappa in (1.25, 1.5):
        g = grupo("par", kappa)
        Dinf = float(g[0]["Dinf"])
        v = lambda x: mp.pi * mp.sin(mp.acos(x)) / (2 * mp.acos(x))  # noqa: E731
        Ds = lambda x: mp.pi * mp.sin(mp.acos(x)) / (4 * mp.acos(x) * (mp.pi - mp.acos(x)))  # noqa: E731
        dbar = lambda x: 4 * (mp.pi * v(x) / 6 - Ds(x) * mp.pi ** 2 / 4) / (x + 2 * kappa)  # noqa: E731
        c2 = float(-mp.diff(dbar, Dinf) / F2(Dinf, kappa) * mp.diff(lambda x: mp.log(Ds(x) / (x + 2 * kappa)), Dinf))
        r = g[-1]
        n = int(r["n"])
        val = n * n * cociente(r, "par", kappa, Dinf)
        ok4 &= abs(val / c2 - 1) < 5e-3
        print(f"   kappa = {kappa}: n^2 (s/K - 1) = {val:.4f} (n = {n}), c_2 = {c2:.4f}")
    check(ok4, "   coincide con c_2 a 0,5 %")
    sys.exit(0 if OK else 1)

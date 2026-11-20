"""Problema abierto 3 de robustez_mps.tex, n par y kappa < 1: la ley del doblete de Néel.

Derivación (Observación obs:doblete, sin rigor). Con Delta = cosh(gamma) > 1 y n par hay dos vacíos
de Néel; los conecta un solo espinón virtual que da la vuelta al anillo (la pared de dominio
intercambia las subredes). Su amplitud es la de una partícula del canal espejo (E -> i p~,
theta -> i E~): con E_sp(theta) = A sqrt(1 - k^2 cos^2 theta), A = sinh(gamma) K(k)/pi, la energía
espejo es E~(p~) = arccosh(sqrt(1 + p~^2/A^2)/k), mínima en E~_0 = arccosh(1/k) = 1/(2 xi_JKM), y
el punto de silla da

    I_n = int dp~/(2 pi) e^{-n E~(p~)} ~ A sqrt(k'/(2 pi n)) e^{-n/(2 xi_JKM)}.

Hay dos especies (S^z = +-1/2), con fases e^{+-i Phi/2}: el elemento entre los dos vacíos es
t(Phi) = 2 I_n cos(Phi/2), los niveles son Ebar -+ |t(Phi)|, y

    E0(Phi) - E0(0) = R_n (1 - cos(Phi/2))  (Phi en [0, pi]),
    R_n = E0(pi) - E0(0) = 2 I_n = (sinh(gamma) K(k)/pi) sqrt(2 k'/(pi n)) e^{-n/(2 xi_JKM)} (1 + O(1/n)).

La normalización int dp~/(2 pi) por especie es la del término F de Lüscher; aquí se toma como dato
y la comprueba la constante C = 2 del punto 2.

Comprobaciones (holonomy.bethe):
  1. arccosh(1/k) = 1/(2 xi_JKM), con k de nomo e^{-gamma} y xi_JKM de xi_tunel.candidates;
  2. C = lim R_n sqrt(n) e^{n E~_0} / (A sqrt(k'/(2 pi))) = 2, extrapolando en 1/n (grado 2) con
     1e-9 < R_n < 1e-3, Delta = 1,5, 2, 2,66, 4, 6 y 10;
  3. forma en Phi: [E0(Phi) - E0(0)]/R_n = 1 - cos(Phi/2) con desviación que cae con n (Delta = 2,66);
  4. la sensibilidad de data/neel_bethe.csv (n par, kappa = 0,4, 0,6, 0,8) frente a
     4 R_n / (n (Delta_inf + 2 kappa)) con la R_n de arriba: el cociente es 1 + c/n.
Salida: data/doblete_par.csv y una tabla por pantalla.
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
from mpmath import mp  # noqa: E402
from xi_tunel import candidates  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def constantes(D):
    """(k, k', A, E~_0) del XXZ con Delta = cosh(gamma) > 1."""
    mp.dps = 30
    g = mp.acosh(D)
    q = mp.exp(-g)
    k = (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q)) ** 2
    return float(k), float(mp.sqrt(1 - k ** 2)), float(mp.sinh(g) / mp.pi * mp.ellipk(k ** 2)), float(mp.acosh(1 / k))


def R_pred(n, D):
    k, kp, A, E0 = constantes(D)
    return 2 * A * np.sqrt(kp / (2 * np.pi * n)) * np.exp(-n * E0)


def R_bethe(n, D):
    s = bethe.solver(int(n))
    return s.energy(D, np.pi, 0) - s.energy(D, 0.0, 0)


if __name__ == "__main__":
    rows = []
    err = max(abs(1 / constantes(D)[3] / candidates(D)[1] - 1) for D in (1.2, 1.5, 2.0, 2.66, 4.0, 6.0, 10.0))
    check(err < 1e-12, f"1. arccosh(1/k) = 1/(2 xi_JKM) en Delta de 1,2 a 10 (dif. rel. {err:.1e})")

    print("2. C = lim R_n sqrt(n) e^{n E~_0} / (A sqrt(k'/(2 pi)))")
    Cs = []
    for D in (1.5, 2.0, 2.66, 4.0, 6.0, 10.0):
        k, kp, A, E0 = constantes(D)
        xi = 1 / E0
        ns = np.unique(np.round(np.linspace(max(10, 3 * xi), 20 * xi, 24) / 2).astype(int) * 2)
        pts = []
        for n in ns:
            R = R_bethe(n, D)
            if 1e-9 < R < 1e-3:
                pts.append((int(n), R * np.sqrt(n) * np.exp(n * E0) / (A * np.sqrt(kp / (2 * np.pi))), R))
        n_ = np.array([p[0] for p in pts], float)
        c_ = np.array([p[1] for p in pts])
        X = np.column_stack([n_ ** -i for i in range(3)])
        c, *_ = np.linalg.lstsq(X, c_, rcond=None)
        Cs.append(c[0])
        print(f"   Delta = {D:5.2f}, 2 xi_JKM = {xi:7.3f}, n = {int(n_[0])}..{int(n_[-1])}: C = {c[0]:.5f}, "
              f"c1 = {c[1]:.2f}")
        for n, cc, R in pts:
            rows.append(dict(tipo="R_n", Delta=D, kappa=np.nan, n=n, valor=R, cociente=cc / 2))
    check(max(abs(C - 2) for C in Cs) < 1e-2, f"   C = 2 en los seis Delta (máx. |C - 2| = {max(abs(C - 2) for C in Cs):.1e})")

    print("3. forma en Phi (Delta = 2,66)")
    devs = []
    for n in (30, 40, 50, 60):
        s = bethe.solver(n)
        E0, R = s.energy(2.66, 0.0, 0), R_bethe(n, 2.66)
        dev = max(abs((s.energy(2.66, P, 0) - E0) / R - (1 - np.cos(P / 2))) for P in np.linspace(0, np.pi, 9))
        devs.append(dev)
        print(f"   n = {n}: max_Phi |[E0(Phi) - E0(0)]/R_n - (1 - cos(Phi/2))| = {dev:.2e}")
    check(devs[-1] < 1e-5 and all(b < a for a, b in zip(devs, devs[1:])),
          "   la forma es 1 - cos(Phi/2), con desviación decreciente")

    print("4. sensibilidad de neel_bethe.csv frente a 4 R_n / (n (Delta_inf + 2 kappa))")
    datos = [r for r in csv.DictReader(open(ROOT / "data" / "neel_bethe.csv"))
             if r["paridad"] == "par" and r["frontera"] == "0"]
    ok4 = True
    for kappa in (0.4, 0.6, 0.8):
        g = [r for r in datos if float(r["kappa"]) == kappa and float(r["s"]) > 1e-10]
        D = float(g[0]["Dinf"])
        pts = [(int(r["n"]), float(r["s"]) / (4 * R_pred(int(r["n"]), D) / (int(r["n"]) * (D + 2 * kappa))))
               for r in g[-8:]]
        n_ = np.array([p[0] for p in pts], float)
        q_ = np.array([p[1] for p in pts])
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(n_), 1 / n_]), q_, rcond=None)
        ok4 &= abs(c[0] - 1) < 5e-3
        print(f"   kappa = {kappa}: n = {int(n_[0])}..{int(n_[-1])}, cociente {q_[0]:.4f} .. {q_[-1]:.4f}, "
              f"ajuste a + b/n: a = {c[0]:.5f}, b = {c[1]:.2f}")
        for n, qq in pts:
            rows.append(dict(tipo="s", Delta=D, kappa=kappa, n=n, valor=np.nan, cociente=qq))
    check(ok4, "   el cociente tiende a 1 (|a - 1| < 5e-3)")

    with open(ROOT / "data" / "doblete_par.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nescrito {ROOT / 'data' / 'doblete_par.csv'}")
    sys.exit(0 if OK else 1)

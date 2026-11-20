"""Problema abierto 3 de robustez_mps.tex: el caso crítico con kappa > pi/2 (Delta_inf < 0).

neel_bethe.py usa holonomy.bethe, que solo llega a Delta > 0: con flujo pi una rapidez escapa a
infinito en Delta = 0. Aquí el mismo dual reducido (Lema lem:reduccion),

    v(n, Theta) = min{ inf_Delta 4 eps_n(Delta, Theta) / (Delta + 2 kappa),  v_I },

se evalúa con holonomy.bethe_p (ecuaciones de Bethe en variables de momento, -1 < Delta < 1), y
se contrastan las dos constantes del caso crítico (Delta_inf = cos gamma, D la rigidez de
Shastry-Sutherland, D = pi sin(gamma) / (4 gamma (pi - gamma))):

    n R_n -> pi^2 D / 2,   R_n = E0(Delta_inf, Theta_1) - E0(Delta_inf, Theta_0),
    n^2 s -> C = 2 pi^2 D / (Delta_inf + 2 kappa),   s = v(n, Theta_0) - v(n, Theta_1),

con Theta_0 = 0 (n par) o pi (n impar) y Theta_1 = Theta_0 + pi. Extrapolación en 1/n con
y = y_inf + a/n + b/n^2, como en neel_bethe_ajuste.py.

Alcance. Con Delta < 0 el estado retorcido forma, por debajo de Delta_c(Phi), un par de raíces
complejas conjugadas: las dos raíces extremas chocan en p = pi - gamma (lambda = infinito), con
pi - gamma_c = Phi/4 si n es par (Delta_c = -1/sqrt 2 con Phi = pi) y (pi - Phi)/5 si es impar,
sin dependencia de n. Más abajo se forman más pares, en pi - gamma = Phi/(4k) (n par) o
(pi - Phi)/(4k + 1) (impar), k = 2, 3, ...; bethe_p los sigue con momentos complejos. Con
kappa = 5, 6 y 8 (Delta_inf = -0,933, -0,956, -0,977) el fundamental con n par y Phi = pi tiene
2, 2 y 3 pares. La búsqueda en Delta no baja de D_MIN.

Controles:
  (a) bethe_p = holonomy.bethe con Delta > 0 (n = 20, 21, 100, 101; siete Phi en [0, pi]);
  (b) bethe_p = diagonalización exacta con todos los sectores de S^z, Delta en [-0,97; -0,05]
      (sin pares y con uno, dos o tres), n = 8..12, Phi en {0, pi/2, pi};
  (c) kappa = 1,5 (Delta_inf = 0,0945 > 0): el dual con bethe_p reproduce data/neel_bethe.csv;
  (d) en cada punto, el signo del Corolario cor:paridad (s > 0) y la cota de Kohn por los dos lados.
Salida: data/neel_bethe_negativo.csv y una tabla por pantalla.
"""
import csv
import os
import sys
from pathlib import Path

# Matrices pequeñas: con muchos hilos de BLAS (o con otro proceso compitiendo) cada Newton se
# vuelve cientos de veces más lento.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import bethe, bethe_p, xxz  # noqa: E402
from neel_bethe import limit, stiffness, v_ising  # noqa: E402

OK = True
D_MIN = -0.985
KAPPAS = (1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
SIZES = {"par": [100, 150, 200, 300, 400, 600, 800, 1000],
         "impar": [101, 151, 201, 301, 401, 601, 801, 1001]}


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def v_dual(n, Theta, kappa, guess, h=0.15):
    """(v, Delta*) del dual reducido; Brent acotado alrededor de guess, ampliando si toca el borde."""
    f = lambda D: 4 * bethe_p.eps(n, D, Theta) / (D + 2 * kappa)  # noqa: E731
    while True:
        a, b = max(D_MIN, guess - h), min(0.95, guess + h)
        r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
        borde_a = r.x - a < 1e-3 * h and a > D_MIN
        borde_b = b - r.x < 1e-3 * h and b < 0.95
        if not (borde_a or borde_b):
            break
        guess, h = r.x, 2 * h
    if r.x - D_MIN < 1e-6:
        raise RuntimeError(f"Delta* en D_MIN: n={n} kappa={kappa}")
    vI = v_ising(n)
    return (r.fun, r.x) if r.fun <= vI else (vI, np.inf)


def punto(n, kappa, Dinf, prev):
    T0 = 0.0 if n % 2 == 0 else np.pi
    T1 = np.pi - T0
    v0, D0 = v_dual(n, T0, kappa, prev[0] if prev else Dinf)
    v1, D1 = v_dual(n, T1, kappa, prev[1] if prev else Dinf)
    kohn = lambda D: 4 * (bethe_p.eps(n, D, T0) - bethe_p.eps(n, D, T1)) / (D + 2 * kappa)  # noqa: E731
    nR = n * n * (bethe_p.eps(n, Dinf, T0) - bethe_p.eps(n, Dinf, T1))  # n [E0(T1) - E0(T0)]
    return dict(n=n, v0=v0, v1=v1, s=v0 - v1, D0=D0, D1=D1, kohn_lo=kohn(D0), kohn_hi=kohn(D1), nR=nR)


def extrapola(ns, ys, k=5):
    x = 1 / np.asarray(ns[-k:], float)
    X = np.column_stack([np.ones_like(x), x, x * x])
    c, *_ = np.linalg.lstsq(X, np.asarray(ys[-k:]), rcond=None)
    return c[0]


def controles():
    th = np.linspace(0, np.pi, 7)
    err = max(abs(bethe.eps(n, D, T) - bethe_p.eps(n, D, T))
              for n in (20, 21, 100, 101) for D in (0.2, 0.5, 0.9) for T in th)
    check(err < 1e-12, f"(a) bethe_p = holonomy.bethe con Delta > 0 (error {err:.1e})")
    err = 0.0
    for n in range(8, 13):
        for D in (-0.05, -0.2, -0.4, -0.55, -0.7, -0.75, -0.85, -0.9, -0.94, -0.97):
            for T in (0.0, np.pi / 2, np.pi):
                E = min(np.linalg.eigvalsh(xxz.hamiltonian(n, M, D, T).toarray())[0] for M in range(n + 1))
                err = max(err, abs(-E / n - bethe_p.eps(n, D, T)))
    check(err < 1e-11, f"(b) bethe_p = diagonalización exacta (todos los sectores), Delta en [-0,97; -0,05], "
                       f"n = 8..12 (error {err:.1e})")
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "neel_bethe.csv"))
            if float(r["kappa"]) == 1.5 and int(r["n"]) in (100, 101, 300, 301)]
    err = 0.0
    for r in rows:
        n = int(r["n"])
        T0 = 0.0 if n % 2 == 0 else np.pi
        v0, _ = v_dual(n, T0, 1.5, float(r["D0"]), h=0.05)
        v1, _ = v_dual(n, np.pi - T0, 1.5, float(r["D1"]), h=0.05)
        err = max(err, abs(v0 - float(r["v0"])), abs(v1 - float(r["v1"])))
    check(len(rows) >= 2 and err < 1e-10,
          f"(c) kappa = 1,5: el dual con bethe_p reproduce neel_bethe.csv en {len(rows)} puntos (error {err:.1e})")


if __name__ == "__main__":
    controles()
    out = []
    for kappa in KAPPAS:
        Dinf, vinf = limit(kappa)
        D = stiffness(Dinf)
        C, cR = 2 * np.pi ** 2 * D / (Dinf + 2 * kappa), np.pi ** 2 * D / 2
        print(f"\nkappa = {kappa}: Delta_inf = {Dinf:.6f}, v_inf = {vinf:.6f}, D = {D:.6f}, "
              f"C = {C:.6f}, pi^2 D / 2 = {cR:.6f}")
        for par in ("par", "impar"):
            prev, pts = None, []
            print(f"  {par}:      n     v(T0)        s          n^2 s     n R_n     Delta*_0  Delta*_1")
            for n in SIZES[par]:
                r = punto(n, kappa, Dinf, prev)
                prev = (r["D0"], r["D1"])
                r.update(paridad=par, kappa=kappa, Dinf=Dinf, C=C, cR=cR)
                pts.append(r)
                out.append(r)
                print(f"           {n:5d}  {r['v0']:.8f}  {r['s']:.4e}  {n * n * r['s']:.6f}  {r['nR']:.6f}  "
                      f"{r['D0']:.5f}  {r['D1']:.5f}", flush=True)
            ns = [r["n"] for r in pts]
            okk = all(r["kohn_lo"] - 1e-12 <= r["s"] <= r["kohn_hi"] + 1e-12 and r["s"] > 0 for r in pts)
            check(okk, f"  (d) {par}: s > 0 y cota de Kohn por los dos lados en los {len(pts)} tamaños")
            lim_s = extrapola(ns, [n * n * r["s"] for n, r in zip(ns, pts)])
            lim_R = extrapola(ns, [r["nR"] for r in pts])
            check(abs(lim_s / C - 1) < 1e-5, f"  {par}: n^2 s -> {lim_s:.5f}, C = 2 pi^2 D/(Delta_inf + 2 kappa) = "
                                             f"{C:.5f} (dif. rel. {lim_s / C - 1:+.1e})")
            check(abs(lim_R / cR - 1) < 1e-5, f"  {par}: n R_n -> {lim_R:.5f}, pi^2 D / 2 = {cR:.5f} "
                                              f"(dif. rel. {lim_R / cR - 1:+.1e})")
    keys = ["paridad", "kappa", "n", "v0", "v1", "s", "D0", "D1", "kohn_lo", "kohn_hi", "nR", "Dinf", "C", "cR"]
    with open(ROOT / "data" / "neel_bethe_negativo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\nescrito {ROOT / 'data' / 'neel_bethe_negativo.csv'}")
    sys.exit(0 if OK else 1)

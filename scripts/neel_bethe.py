"""Sensibilidad a la holonomía del anillo de qubits con datos U(1), a tamaños grandes,
por Bethe ansatz: Néel débil (n par, kappa < 1), n impar y caso crítico (kappa >= 1).

Por el Lema de reducción (robustez_mps.tex, Lema lem:reduccion), en Theta en
{0, pi} el dual es exacto con un solo parámetro:

    v(n, Theta) = min{ inf_Delta 4 eps_n(Delta, Theta) / (Delta + 2 kappa),  v_I },

con eps_n = -E0/n del XXZ con flujo Phi = Theta y v_I = 1 (n par) o (n - 2)/n
(n impar, techo de Ising frustrado). El cociente es cuasiconvexo en Delta (eps_n
es convexa, el denominador afín). eps_n se calcula con holonomy.bethe; la
diagonalización exacta (holonomy.xxz) sirve de control.

Por el Corolario cor:paridad, el máximo está en Theta_0 = 0 (n par) o pi (n
impar); la sensibilidad es s = v(n, Theta_0) - v(n, Theta_1) >= 0, con
Theta_1 = Theta_0 + pi.

Controles:
  * Bethe = diagonalización exacta (n = 7..16, Delta en [0,3; 10], Phi en {0, pi/2, pi});
  * el barrido reproduce data/neel_u1.csv (n = 6..18, ínfimo libre en (Delta, Phi));
  * cota de Kohn por los dos lados (Lema lem:giro en Theta_0 y en Theta_1):
        4 [eps(D0, T0) - eps(D0, T1)] / (D0 + 2 kappa)  <=  s
                                  <=  4 [eps(D1, T0) - eps(D1, T1)] / (D1 + 2 kappa);
  * Delta*(n) -> Delta_inf, el mínimo del mismo cociente para la energía exacta de
    la cadena infinita (Yang-Yang con Delta >= 1, integral de Hulthén-Orbach-Walker
    con |Delta| < 1).

Predicción crítica (kappa >= 1, Delta_inf = cos gamma en [0, 1]): con la rigidez de
Shastry-Sutherland D = pi sin(gamma) / (4 gamma (pi - gamma)) y E0(Phi) - E0(0) =
D Phi^2 / (2 n) en |Phi| <= pi,  n^2 s -> 2 pi^2 D / (Delta_inf + 2 kappa). Con
kappa = 1, Delta_inf = 1 y el límite es pi^2/6 (con correcciones logarítmicas).

Con kappa >~ 1,55, Delta_inf < 0, fuera del alcance de holonomy.bethe: esos kappa se
omiten.

Salida: data/neel_bethe.csv (columnas paridad, kappa, n, ...) y una tabla por
pantalla. El barrido de cada (kappa, paridad) se detiene cuando s < S_FLOOR o en nmax.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holonomy import bethe, xxz  # noqa: E402

OK = True
S_FLOOR = 1e-12
D_LO = 0.05
GRID = np.concatenate([np.linspace(D_LO, 6.0, 30), [9, 14, 25, 45, 80, 150, 300]])


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


# --- cadena infinita ---------------------------------------------------------
def eps_inf(D):
    """-E0/N del XXZ infinito, Delta > -1."""
    if abs(D - 1) < 1e-12:
        return np.log(2) - 0.25
    if D > 1:
        g = np.arccosh(D)
        m = np.arange(1, int(40 / g) + 2)
        return np.sinh(g) * (0.5 + 2 * np.sum(np.exp(-2 * m * g) / (1 + np.exp(-2 * m * g)))) - D / 4
    g = np.arccos(D)
    a = np.pi - g

    def f(x):  # sinh(a x) / (2 sinh(pi x) cosh(g x)), sin desbordamiento
        if x < 1e-8:
            return a / (2 * np.pi)
        return np.exp(-2 * g * x) * (-np.expm1(-2 * a * x)) / ((-np.expm1(-2 * np.pi * x))
                                                              * (1 + np.exp(-2 * g * x)))
    I = 2 * quad(f, 0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    return np.sin(g) * I - np.cos(g) / 4


def limit(kappa):
    r = minimize_scalar(lambda D: 4 * eps_inf(D) / (D + 2 * kappa), bounds=(-0.99, 20.0),
                        method="bounded", options={"xatol": 1e-11})
    return r.x, r.fun


def stiffness(D):
    """Rigidez de espín de Shastry-Sutherland, |Delta| <= 1."""
    g = np.arccos(min(D, 1.0))
    return 0.25 if g < 1e-9 else np.pi * np.sin(g) / (4 * g * (np.pi - g))


def xi_asint(D):
    """Longitud de correlación de Néel, forma asintótica cerca de Delta = 1."""
    return 0.25 * np.exp(np.pi ** 2 / (2 * np.arccosh(D))) if D > 1 else np.inf


# --- dual reducido -----------------------------------------------------------
def v_ising(n):
    return (n - 2 * (n % 2)) / n


def v_reducido(n, Theta, kappa, eps=bethe.eps, guess=None, half=0.08):
    f = lambda D: 4 * eps(n, D, Theta) / (D + 2 * kappa)
    vI = v_ising(n)
    if guess is not None and np.isfinite(guess):
        # Arranque desde el Delta* del tamaño anterior; si el mínimo queda en el
        # borde de la ventana, se repite con la malla completa.
        h = half * max(1.0, guess / 3)
        a, b = max(D_LO, guess - h), guess + h
        r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
        if min(r.x - a, b - r.x) > 1e-4 * h and r.fun <= vI:
            return r.fun, r.x
    vals = [f(D) for D in GRID]
    k = int(np.argmin(vals))
    a, b = GRID[max(k - 1, 0)], GRID[min(k + 1, len(GRID) - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    v, D = (r.fun, r.x) if r.fun < vals[k] else (vals[k], GRID[k])
    return (v, D) if v <= vI else (vI, np.inf)


def punto(n, kappa, prev=None):
    T0 = 0.0 if n % 2 == 0 else np.pi
    T1 = np.pi - T0
    g0, g1 = (prev["D0"], prev["D1"]) if prev is not None else (None, None)
    v0, D0 = v_reducido(n, T0, kappa, guess=g0)
    v1, D1 = v_reducido(n, T1, kappa, guess=g1)

    def kohn(D):  # respuesta al giro del XXZ alineado; el de Ising no responde
        if not np.isfinite(D):
            return 0.0
        return 4 * (bethe.eps(n, D, T0) - bethe.eps(n, D, T1)) / (D + 2 * kappa)
    # Si un Delta* toca D_LO el óptimo verdadero puede estar más abajo (fuera del
    # alcance de Bethe): el valor es solo una cota y el punto no entra en los ajustes.
    frontera = int(min(D0, D1) <= D_LO + 1e-6)
    return dict(n=n, v0=v0, v1=v1, s=v0 - v1, D0=D0, D1=D1, kohn_lo=kohn(D0), kohn_hi=kohn(D1),
                frontera=frontera)


def tamanos(nmax, paridad):
    ns = list(range(6, 41, 2)) + list(range(44, 101, 4)) + list(range(110, 301, 10)) \
        + list(range(320, 1001, 20)) + list(range(1100, 3001, 100))
    if paridad == 1:
        ns = [n + 1 for n in ns]
    return [n for n in ns if n <= nmax]


def controles():
    err = max(abs(xxz.eps(n, D, P) - bethe.eps(n, D, P))
              for n in (7, 8, 9, 10, 11, 12, 13, 14, 16) for D in (0.3, 0.7, 1.05, 1.4, 1.85, 2.7, 5.0, 10.0)
              for P in (0.0, np.pi / 2, np.pi))
    check(err < 1e-12, f"Bethe = diagonalización exacta, n = 7..16, Delta en [0,3; 10] (error {err:.1e})")
    ref = Path(__file__).resolve().parents[1] / "data" / "neel_u1.csv"
    if not ref.exists():
        return
    err = 0.0
    with open(ref) as fh:
        for r in csv.DictReader(fh):
            k, n = float(r["kappa"]), int(r["n"])
            if n < 6:
                continue
            err = max(err, abs(v_reducido(n, 0.0, k)[0] - float(r["v0"])),
                      abs(v_reducido(n, np.pi, k)[0] - float(r["vpi"])))
    check(err < 1e-8, f"dual reducido (Bethe) = ínfimo libre de neel_u1.csv, n = 6..18, "
          f"kappa = 0,4..1 (error {err:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.4, 0.6, 0.8, 0.9, 1.0, 1.25, 1.5])
    ap.add_argument("--paridad", choices=["par", "impar", "ambas"], default="ambas")
    ap.add_argument("--nmax", type=int, default=3001)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "neel_bethe.csv")
    ap.add_argument("--sin-controles", action="store_true")
    args = ap.parse_args()
    if not args.sin_controles:
        controles()

    paridades = {"par": [0], "impar": [1], "ambas": [0, 1]}[args.paridad]
    rows = []
    for par in paridades:
        for kappa in args.kappas:
            Dinf, vinf = limit(kappa)
            if Dinf <= D_LO:
                print(f"\nkappa = {kappa}: Delta_inf = {Dinf:.4f} fuera del alcance de Bethe; se omite")
                continue
            if kappa == 1.0:
                Dinf = 1.0  # exacto por SU(2); el minimizador da 1 + O(xatol)
            pred = 2 * np.pi ** 2 * stiffness(Dinf) / (Dinf + 2 * kappa) if Dinf <= 1 + 1e-6 else np.nan
            print(f"\n{'n impar' if par else 'n par'}, kappa = {kappa}: Delta_inf = {Dinf:.6f}, "
                  f"v_inf = {vinf:.8f}, xi_asint = {xi_asint(Dinf):.3g}, "
                  f"n^2 s crítico -> {pred:.4f}")
            print("    n   v(T0)          s = vT0 - vT1   n^2 s      n^3 s      D0        D1       "
                  "Kohn lo/s  Kohn hi/s  xi_loc   alpha_loc")
            prev, first = None, None
            for n in tamanos(args.nmax, par):
                t = time.time()
                r = punto(n, kappa, prev)
                r.update(paridad="impar" if par else "par", kappa=kappa, Dinf=Dinf, vinf=vinf, pred_crit=pred)
                rows.append(r)
                first = first or r
                s = r["s"]
                xi = al = np.nan
                if prev is not None and s > 0 and prev["s"] > 0:
                    q = np.log(s / prev["s"])
                    xi = -(n - prev["n"]) / q
                    al = -q / np.log(n / prev["n"])
                lo = r["kohn_lo"] / s if s > 0 else np.nan
                hi = r["kohn_hi"] / s if s > 0 else np.nan
                print(f" {n:4d}  {r['v0']:.10f}  {s:.4e}   {n * n * s:8.4f}  {n ** 3 * s:9.3f}  "
                      f"{r['D0']:8.5f}  {r['D1']:8.5f}   {lo:.4f}     {hi:.4f}    "
                      f"{xi:7.2f}  {al:7.2f}   ({time.time() - t:.0f} s)"
                      + ("  [Delta* en la frontera]" if r["frontera"] else ""), flush=True)
                if r["frontera"]:
                    prev = r
                    continue
                if not (r["kohn_lo"] <= s * (1 + 1e-6) + 1e-14 and s <= r["kohn_hi"] * (1 + 1e-6) + 1e-14):
                    check(False, f"kappa={kappa}, n={n}: cota de Kohn por los dos lados")
                if s < -1e-13:
                    check(False, f"kappa={kappa}, n={n}: s >= 0 (Corolario de paridad)")
                prev = r
                if 0 <= s < S_FLOOR:
                    break
            last = rows[-1]
            finitos = [x for x in rows if x["kappa"] == kappa and x["paridad"] == last["paridad"]
                       and np.isfinite(x["D0"])]
            if finitos and np.isfinite(last["D0"]):
                # Con kappa = 1 y n par, Delta* = 1 exactamente para todo n (SU(2)).
                check(abs(last["D0"] - Dinf) < max(abs(finitos[0]["D0"] - Dinf), 1e-6),
                      f"{last['paridad']}, kappa={kappa}: Delta*(n, Theta_0) se acerca a Delta_inf "
                      f"({last['D0']:.5f} en n={last['n']} frente a {Dinf:.5f})")

    keys = ["paridad", "kappa", "n", "v0", "v1", "s", "D0", "D1", "kohn_lo", "kohn_hi",
            "frontera", "Dinf", "vinf", "pred_crit"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    print(f"\nescrito {args.out}")
    sys.exit(0 if OK else 1)

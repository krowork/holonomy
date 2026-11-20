"""Problema abierto 3 de robustez_mps.tex, n impar: la energía de giro del espinón.

Derivación (Observación obs:espinon). Con n impar y Delta = cosh(gamma) > 1, el
fundamental tiene un hueco en el mar de Bethe (un espinón de S^z = 1/2). Como las
vacantes llenan un periodo entero de la rapidez, sin puntos de Fermi, las sumas sobre el
mar son reglas del trapecio y la energía es la del mar más la del hueco salvo términos
exponenciales; el flujo solo cuantiza el momento del hueco, que lo ve con carga 1/2, y el
mínimo en Phi = pi (Marshall) fija el desfase. Resultado:

    E0(n, Phi) = -n e_inf(Delta) + E_sp(dist(Phi - pi, 2 pi Z) / (2n)) + O(n^-1/2 e^{-n/xi}),

con E_sp(theta) = sinh(gamma)/pi K(k) sqrt(1 - k^2 cos^2 theta), k de nomo e^{-gamma}, y
la longitud xi = 2 xi_JKM del desdoblamiento túnel con n par. En particular
    R_n = E0(Delta, 0) - E0(Delta, pi) = E_sp(pi/(2n)) - E_sp(0) = c/n^2 + O(n^-4),
    c = (pi^2/8) E_sp''(0) = (pi/8) sinh(gamma) K(k) k^2/k'          (eq:espinon).

Comprobaciones, en Delta_inf(kappa), kappa = 0,4, 0,6, 0,8 (n impar, Bethe, nueve Phi en [0, pi]):
  1. forma cerrada de c = (pi^2/8) E_sp''(0) por derivación numérica;
  2. el residuo max_Phi |E0 + n e_inf - E_sp(...)| baja del 10 % del valor de E_sp(0) a
     menos de 1e-11 (kappa = 0,4 y 0,6) o 1e-7 (kappa = 0,8): la fórmula es exacta salvo
     términos exponenciales, para todo Phi y no solo en la parábola;
  3. ajuste log(sqrt(n) r_n) = a - n/xi en los n grandes: xi = 2 xi_JKM al 1 %;
  4. n^2 R_n de Bethe = el de E_sp dentro de 2 n^2 r_n, y su distancia O(n^-2) a la parábola c;
  5. informativo: kappa = 0,9, donde 2 xi_JKM es comparable a n y el residuo aún no es
     pequeño en n <= 3001.
Salida: data/espinon_giro.csv y una tabla por pantalla.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import bethe  # noqa: E402
from mpmath import mp  # noqa: E402
from neel_bethe import eps_inf, limit  # noqa: E402
from xi_tunel import candidates  # noqa: E402

OK = True
FLOOR = 1e-12


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def espinon(D):
    """(E_sp, c, k) del XXZ con Delta = cosh(gamma) > 1, con mpmath."""
    mp.dps = 30
    g = mp.acosh(D)
    q = mp.exp(-g)
    k = (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q)) ** 2
    A = mp.sinh(g) / mp.pi * mp.ellipk(k ** 2)
    Esp = lambda th: A * mp.sqrt(1 - k ** 2 * mp.cos(th) ** 2)  # noqa: E731
    c = mp.pi / 8 * mp.sinh(g) * mp.ellipk(k ** 2) * k ** 2 / mp.sqrt(1 - k ** 2)
    return Esp, c, k


def theta_permitido(n, Phi):
    """Momento del hueco más próximo al mínimo de la banda: dist(Phi - pi, 2 pi Z) / (2n)."""
    d = abs((Phi - np.pi + np.pi) % (2 * np.pi) - np.pi)
    return d / (2 * n)


def residuo(n, D, Esp, phis):
    S = bethe.solver(n)
    b = S.ground_branch()
    return max(abs(S.energy(D, P, b) + n * eps_inf(D) - float(Esp(theta_permitido(n, P)))) for P in phis)


def n2R(n, D):
    S = bethe.solver(n)
    b = S.ground_branch()
    return n * n * (S.energy(D, 0.0, b) - S.energy(D, np.pi, b))


if __name__ == "__main__":
    phis = np.linspace(0, np.pi, 9)
    rows = []
    casos = {0.4: (list(range(11, 202, 10)), 1e-11),
             0.6: (list(range(11, 402, 10)), 1e-11),
             0.8: (list(range(21, 1002, 20)), 1e-7)}
    for kappa, (ns, tol) in casos.items():
        D, _ = limit(kappa)
        Esp, c, k = espinon(D)
        _, xj2 = candidates(D)
        c_num = mp.pi ** 2 / 8 * mp.diff(Esp, 0, 2)
        print(f"\nkappa = {kappa}: Delta_inf = {D:.10f}, k = {float(k):.8f}, E_sp(0) = {float(Esp(0)):.10f}, "
              f"c = {float(c):.6f}, 2 xi_JKM = {xj2:.4f}")
        check(abs(c_num / c - 1) < 1e-12, f"c = (pi^2/8) E_sp''(0) en forma cerrada (dif. rel. {float(abs(c_num / c - 1)):.1e})")
        print("     n    max_Phi residuo    n^2 R_n (Bethe)   n^2 R_n (E_sp)   c")
        res = []
        for n in ns:
            r = residuo(n, D, Esp, phis)
            nb = n2R(n, D)
            nf = n * n * float(Esp(np.pi / (2 * n)) - Esp(0))
            rows.append(dict(kappa=kappa, n=n, Dinf=D, residuo=r, n2R_bethe=nb, n2R_formula=nf, c=float(c),
                             xi_JKM2=xj2))
            res.append(r)
            print(f"  {n:5d}   {r:.3e}          {nb:12.6f}     {nf:12.6f}    {float(c):.6f}", flush=True)
            if r < FLOOR:
                break
        n_ok = ns[:len(res)]
        check(res[-1] < tol and res[-1] < 1e-3 * res[0],
              f"residuo de {res[0]:.1e} (n = {n_ok[0]}) a {res[-1]:.1e} (n = {n_ok[-1]}) < {tol:.0e}")
        sel = [(n, r) for n, r in zip(n_ok, res) if r > 30 * FLOOR][-6:]
        x = np.array([n for n, _ in sel], float)
        y = np.log([np.sqrt(n) * r for n, r in sel])
        xi = -1 / np.polyfit(x, y, 1)[0]
        check(abs(xi / xj2 - 1) < 0.01, f"ajuste log(sqrt(n) r_n) = a - n/xi en n = {int(x[0])}..{int(x[-1])}: "
                                        f"xi = {xi:.4f}, 2 xi_JKM = {xj2:.4f} ({100 * (xi / xj2 - 1):+.2f} %)")
        last = rows[-1]
        # R_n = diferencia de dos energías, cada una con error <= residuo: |dif| <= 2 n^2 r_n.
        check(abs(last["n2R_bethe"] - last["n2R_formula"]) <= 2 * last["n"] ** 2 * last["residuo"] + 1e-9,
              f"n^2 R_n = {last['n2R_bethe']:.6f} (Bethe) = {last['n2R_formula']:.6f} (E_sp) en n = {last['n']}; "
              f"frente a la parábola c = {float(c):.6f}, dif. rel. {abs(last['n2R_formula'] / float(c) - 1):.1e} (O(n^-2))")

    # kappa = 0,9: 2 xi_JKM grande; solo informativo.
    D, _ = limit(0.9)
    Esp, c, _ = espinon(D)
    _, xj2 = candidates(D)
    print(f"\nkappa = 0.9 (informativo): Delta_inf = {D:.6f}, 2 xi_JKM = {xj2:.1f}, c = {float(c):.2f}")
    for n in (301, 1001, 3001):
        nb = n2R(n, D)
        nf = n * n * float(Esp(np.pi / (2 * n)) - Esp(0))
        rows.append(dict(kappa=0.9, n=n, Dinf=D, residuo=np.nan, n2R_bethe=nb, n2R_formula=nf, c=float(c),
                         xi_JKM2=xj2))
        print(f"  n = {n}: n^2 R_n = {nb:.2f} (Bethe), {nf:.2f} (E_sp), n / (2 xi_JKM) = {n / xj2:.2f}")

    out = ROOT / "data" / "espinon_giro.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nescrito {out}")
    sys.exit(0 if OK else 1)

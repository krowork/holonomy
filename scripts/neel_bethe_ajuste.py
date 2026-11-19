"""Leyes de la sensibilidad (lee data/neel_bethe.csv, escrito por neel_bethe.py).

Para cada (paridad, kappa), con Theta_0 el máximo (0 si n par, pi si impar):
  * R_n = E0(Delta_inf, Theta_1) - E0(Delta_inf, Theta_0), la respuesta al giro del
    testigo límite, por Bethe; s debe tender a 4 R_n / (n (Delta_inf + 2 kappa)),
    la cota de Kohn evaluada en ese testigo;
  * n par, kappa < 1 (Néel, gap): ajuste s = A n^-beta e^(-n/xi), comparado con
    xi_asint(Delta_inf) = (1/4) e^(pi^2 / (2 gamma));
  * n impar, kappa < 1 (un solo espinón masivo): n^2 R_n -> c y n^3 s -> 4 c / (Delta_inf + 2 kappa);
  * kappa >= 1 (crítico): n R_n -> pi^2 D / 2 y n^2 s -> 2 pi^2 D / (Delta_inf + 2 kappa),
    con D la rigidez de Shastry-Sutherland; extrapolación lineal en 1/n (en 1/ln n
    con kappa = 1, por el operador marginal).
Se excluyen los puntos con Delta* en la frontera de la búsqueda y los de s < 1e-11.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holonomy import bethe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ajuste_exp(ns, ys):
    X = np.column_stack([np.ones_like(ns), -np.log(ns), -ns])
    c, *_ = np.linalg.lstsq(X, np.log(ys), rcond=None)
    return c[1], 1 / c[2], np.max(np.abs(X @ c - np.log(ys)))


def ajuste_pot(ns, ys):
    X = np.column_stack([np.ones_like(ns), -np.log(ns)])
    c, *_ = np.linalg.lstsq(X, np.log(ys), rcond=None)
    return c[1], np.max(np.abs(X @ c - np.log(ys)))


def extrapola(xs, ys):
    """y = y_inf + a x + b x^2 en los puntos dados; devuelve y_inf."""
    X = np.column_stack([np.ones_like(xs), xs, xs ** 2])
    c, *_ = np.linalg.lstsq(X, ys, rcond=None)
    return c[0]


if __name__ == "__main__":
    rows = list(csv.DictReader(open(ROOT / "data" / "neel_bethe.csv")))
    out = []
    grupos = sorted({(r["paridad"], float(r["kappa"])) for r in rows})
    for par, kappa in grupos:
        R = [r for r in rows if r["paridad"] == par and float(r["kappa"]) == kappa
             and r["frontera"] == "0" and float(r["s"]) > 1e-11]
        ns = np.array([int(r["n"]) for r in R], float)
        s = np.array([float(r["s"]) for r in R])
        Dinf = float(R[0]["Dinf"])
        T0 = 0.0 if par == "par" else np.pi
        T1 = np.pi - T0
        sel = ns >= max(20, ns.max() / 3)
        nt = ns[sel][:: max(1, sel.sum() // 12)]
        Rn = np.array([int(n) * (bethe.eps(int(n), Dinf, T0) - bethe.eps(int(n), Dinf, T1)) for n in nt])
        pred = 4 * Rn / (nt * (Dinf + 2 * kappa))
        sn = np.interp(nt, ns, s)
        fila = dict(paridad=par, kappa=kappa, Dinf=Dinf, n_min=int(ns[sel].min()), n_max=int(ns.max()),
                    s_sobre_kohn_inf=sn[-1] / pred[-1])
        print(f"\n{par}, kappa = {kappa}  (Delta_inf = {Dinf:.5f}, n = {int(ns[sel].min())}..{int(ns.max())})")
        print(f"  s / [4 R_n / (n (Delta_inf + 2 kappa))] = {sn[0] / pred[0]:.5f} (n={int(nt[0])}) -> "
              f"{sn[-1] / pred[-1]:.5f} (n={int(nt[-1])})")
        alpha, erra = ajuste_pot(ns[sel], s[sel])
        fila.update(alpha_pot=alpha, res_pot=erra)
        if kappa < 1 and par == "par":
            beta, xi, err = ajuste_exp(ns[sel], s[sel])
            xa = 0.25 * np.exp(np.pi ** 2 / (2 * np.arccosh(Dinf)))
            print(f"  exponencial: xi = {xi:.3f} (xi_asint = {xa:.3f}), beta = {beta:.3f}, residuo {err:.1e};"
                  f" potencia pura: alpha = {alpha:.2f}, residuo {erra:.1e}")
            fila.update(xi=xi, xi_asint=xa, beta=beta, res_exp=err)
        elif kappa < 1:
            c = nt ** 2 * Rn
            lim_c = extrapola(1 / nt, c)
            lim_s = 4 * lim_c / (Dinf + 2 * kappa)
            n3s = ns[sel] ** 3 * s[sel]
            lim_n3s = extrapola(1 / ns[sel], n3s)
            print(f"  potencia: alpha = {alpha:.3f} (residuo {erra:.1e}); n^3 s = {n3s[0]:.3f} -> "
                  f"{n3s[-1]:.3f}, extrapolado {lim_n3s:.3f}")
            print(f"  n^2 R_n = {c[0]:.4f} -> {c[-1]:.4f}, extrapolado {lim_c:.4f}; "
                  f"4 c / (Delta_inf + 2 kappa) = {lim_s:.3f}")
            fila.update(n3s_inf=lim_n3s, c_espinon=lim_c, n3s_pred=lim_s)
        else:
            D = float(R[0]["pred_crit"]) * (Dinf + 2 * kappa) / (2 * np.pi ** 2)
            nR = nt * Rn
            n2s = ns[sel] ** 2 * s[sel]
            x = 1 / np.log(ns[sel]) if kappa == 1 else 1 / ns[sel]
            xr = 1 / np.log(nt) if kappa == 1 else 1 / nt
            lim = extrapola(x, n2s)
            print(f"  potencia: alpha = {alpha:.3f} (residuo {erra:.1e})")
            print(f"  n R_n = {nR[0]:.5f} -> {nR[-1]:.5f}, extrapolado {extrapola(xr, nR):.5f}; "
                  f"pi^2 D / 2 = {np.pi ** 2 * D / 2:.5f}")
            print(f"  n^2 s = {n2s[0]:.4f} -> {n2s[-1]:.4f}, extrapolado {lim:.4f}; "
                  f"predicción 2 pi^2 D / (Delta_inf + 2 kappa) = {float(R[0]['pred_crit']):.4f}")
            fila.update(n2s_inf=lim, n2s_pred=float(R[0]["pred_crit"]), D_SS=D)
        out.append(fila)
    keys = sorted({k for f in out for k in f}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
    with open(ROOT / "data" / "neel_bethe_ajuste.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"\nescrito {ROOT / 'data' / 'neel_bethe_ajuste.csv'}")

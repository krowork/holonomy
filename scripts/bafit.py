"""Ajustes de escalado con correcciones logarítmicas (secciones 6.2 y 6.3).

Usa data/bethe_large.csv (scripts/baedge2.py). Ajusta, con ambas paridades a la
vez (asíntota común, correcciones propias de cada paridad):
  * la rigidez finita D_n (exacto: 1/4),
  * el coeficiente |n^2 [v(n,0) - v(n,pi)]| (predicción: pi^2/6),
y, por separado en cada paridad, la anchura psi_n/pi (predicción:
1 - 3/(8 (ln 2 - 1/4))). Se recorren varios modelos de corrección y varios
n mínimos: en el punto isótropo la dispersión entre modelos es mayor que el
error estadístico de cada ajuste, y es la que debe citarse.

Para D_n se hace además, por paridad, el ajuste de forma RG
  D_n = D_inf [1 + a / (ln n + c + b ln ln n)],
con la amplitud libre y con la amplitud universal a = 1/2 fijada (operador
marginal del punto isótropo; Laflorencie, Capponi y Sørensen, EPJB 24, 77).
Fijar la amplitud reduce la dispersión entre modelos.
"""
import numpy as np
from scipy.optimize import curve_fit

from holonomy import asymptotics
from holonomy.report import read_csv

N_MINS = (64, 128, 256)
LOG_ORDERS = {"1/ln n": 1, "1/ln n + 1/ln^2 n": 2, "1/ln n + 1/ln^2 n + 1/ln^3 n": 3}


def joint_log_fit(n, y, parity, order):
    """y = y_inf + sum_k a_{p,k} / ln^k n, con y_inf común. Devuelve (y_inf, error)."""
    L = np.log(n)
    cols = [np.ones_like(L)]
    for p in (0, 1):
        m = (parity == p).astype(float)
        cols += [m / L ** k for k in range(1, order + 1)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma2 = float(np.sum((X @ beta - y) ** 2) / (len(y) - X.shape[1]))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta[0], np.sqrt(cov[0, 0])


def joint_shifted_log_fit(n, y, parity, y0):
    """y = y_inf + a_p / (ln n + c_p)."""
    def f(X, yinf, a0, c0, a1, c1):
        nn, pp = X
        return yinf + np.where(pp == 0, a0 / (np.log(nn) + c0), a1 / (np.log(nn) + c1))
    popt, pcov = curve_fit(f, (n, parity), y, p0=(y0, 0.1, 0.0, 0.1, 0.0), maxfev=50000)
    return popt[0], np.sqrt(pcov[0, 0])


def free_asymptote(n, y, model):
    models = {
        "1/ln n": (lambda n, c, a: c + a / np.log(n), (0.15, -0.5)),
        "1/ln n + 1/ln^2 n": (lambda n, c, a, b: c + a / np.log(n) + b / np.log(n) ** 2, (0.15, -0.5, 0.0)),
        "n^-a": (lambda n, c, a, b: c + b * n ** (-a), (0.15, 0.3, -0.3)),
    }
    f, p0 = models[model]
    popt, pcov = curve_fit(f, n, y, p0=p0, maxfev=20000)
    return popt[0], np.sqrt(pcov[0, 0])


def rg_fit(n, D, amplitude=None):
    """D_inf [1 + a/(ln n + c + b ln ln n)]; con amplitude fija, a = amplitude.
    Devuelve (D_inf, error, a)."""
    L, LL = np.log(n), np.log(np.log(n))
    if amplitude is None:
        f = lambda n, D_inf, a, c: D_inf * (1 + a / (L + c))
        p0 = (0.25, 0.5, 1.0)
    else:
        f = lambda n, D_inf, c, b: D_inf * (1 + amplitude / (L + c + b * LL))
        p0 = (0.25, 2.0, 0.0)
    popt, pcov = curve_fit(f, n, D, p0=p0, maxfev=50000)
    a = popt[1] if amplitude is None else amplitude
    return popt[0], np.sqrt(pcov[0, 0]), a


def rg_scan():
    print("\nD_inf con la forma RG D_inf [1 + a/(ln n + c [+ b ln ln n])], por paridad")
    rows = read_csv("bethe_large.csv")
    fixed = []
    for n_min in N_MINS:
        for p, name in ((0, "par"), (1, "impar")):
            sel = [r for r in rows if r["n"] >= n_min and int(r["n"]) % 2 == p]
            n = np.array([r["n"] for r in sel], dtype=float)
            D = np.array([r["D_n"] for r in sel])
            v, e, a = rg_fit(n, D)
            vf, ef, _ = rg_fit(n, D, amplitude=0.5)
            fixed.append(vf)
            print(f"  n >= {n_min:<4} {name:5s}  a libre: D_inf = {v:.4f}±{e:.4f} (a = {a:.3f})"
                  f"   a = 1/2: D_inf = {vf:.4f}±{ef:.4f}")
    print(f"  intervalo con a = 1/2: [{min(fixed):.4f}, {max(fixed):.4f}]")


def scan(label, key, exact, transform=lambda v: v):
    print(f"\n{label} (exacto {exact:.4f}); valor ± error estadístico")
    values = []
    for n_min in N_MINS:
        rows = [r for r in read_csv("bethe_large.csv") if r["n"] >= n_min]
        n = np.array([r["n"] for r in rows])
        par = (n % 2).astype(int)
        y = transform(np.array([r[key] for r in rows]))
        cells = []
        for name, order in LOG_ORDERS.items():
            v, e = joint_log_fit(n, y, par, order)
            values.append(v)
            cells.append(f"{v:.4f}±{e:.4f}")
        v, e = joint_shifted_log_fit(n, y, par, exact)
        values.append(v)
        cells.append(f"{v:.4f}±{e:.4f}")
        print(f"  n >= {n_min:<4} " + "  ".join(cells))
    print(f"  modelos: {', '.join(LOG_ORDERS)}, 1/(ln n + c)")
    print(f"  intervalo entre modelos y rangos: [{min(values):.4f}, {max(values):.4f}]")


def main():
    rows = read_csv("bethe_large.csv")
    n = np.array([r["n"] for r in rows])
    print(f"{len(rows)} valores de n entre {int(n.min())} y {int(n.max())}")

    scan("D_inf", "D_n", 0.25)
    rg_scan()
    scan("|coef|_inf", "coef", np.pi ** 2 / 6, transform=np.abs)
    for r in rows:
        if r["n"] in (1537, 2048):
            print(f"  n = {int(r['n'])}: |coef| por encima de pi^2/6 en un "
                  f"{100 * (abs(r['coef']) / (np.pi ** 2 / 6) - 1):.1f} %")

    print(f"\npsi_inf/pi exacto (D = 1/4, eps = ln 2 - 1/4) = {asymptotics.psi_infinity_over_pi():.4f}")
    sel = [r for r in rows if r["n"] >= 64]
    n = np.array([r["n"] for r in sel])
    psi = np.array([r["psi_over_pi"] for r in sel])
    parity = (n % 2).astype(int)
    for model in ("1/ln n", "1/ln n + 1/ln^2 n", "n^-a"):
        for p, name in ((0, "par"), (1, "impar")):
            m = parity == p
            c, dc = free_asymptote(n[m], psi[m], model)
            print(f"  psi_inf/pi, {name:5s}, modelo {model:18s} = {c:.3f} ± {dc:.3f}")


if __name__ == "__main__":
    main()

"""Gap con simetría rota en qubits: datos U(1) anisótropos y testigo de Néel.

Datos en cada arista del anillo (gauge uniforme, phi = Theta/n):
    <A_e> = 2 v_perp cos phi, <B_e> = 2 v_perp sin phi, <C_e> = v_z,
con A = -(XX+YY), B = XY-YX, C = -ZZ, sobre el rayo v_perp = kappa v,
v_z = v. kappa = 1 son los datos de Werner girados.

Dual (el testigo cos psi A + sin psi B + Delta C es -4 H_XXZ con flujo n psi):
    v_max(n, Theta; kappa) = inf_{Delta, Phi}
        4 eps_n(Delta, Phi) / (Delta + 2 kappa cos((Phi - Theta)/n)).
Con kappa < 1 los datos favorecen ZZ y el testigo óptimo puede caer en
Delta > 1 (Néel, gap con simetría rota). Pregunta: ¿la sensibilidad
v(n, 0) - v(n, pi) cae exponencialmente (Néel) o como 1/n^2 (crítico)?

Salida: data/neel_u1.csv y una tabla por pantalla.

Notas numéricas. (1) Para n impar y kappa pequeño el ínfimo está en Delta -> inf
(techo de Ising frustrado (n-2)/n, independiente de Theta); v_at_flux lo incluye.
(2) Hasta noviembre de 2026 el CSV tenía v(pi) sobrestimado (hasta ~3e-4) en
n >= 14: el arranque en caliente de Lanczos en holonomy.xxz reutilizaba un
autovector de momento definido y, tras el cruce de niveles en Phi = pi, devolvía
el fundamental del sector de momento equivocado. Con la componente aleatoria
del arranque (xxz._warm_start) el CSV se ha regenerado con nphi = 33. La cota
v(n, pi) <= 4 eps_n(1, pi) / 3 (giro del testigo de Heisenberg, kappa = 1, n par)
sirve de control: ver la comprobación al final del barrido.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holonomy import xxz  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def ratio(n, Theta, kappa, Delta, Phi, eps=xxz.eps):
    den = Delta + 2 * kappa * np.cos((Phi - Theta) / n)
    if den <= 0:
        return np.inf
    return 4 * eps(n, Delta, Phi) / den


def v_at_flux(n, Theta, kappa, Phi, eps=xxz.eps):
    """Mínimo en Delta (el cociente es cuasiconvexo en Delta)."""
    f = lambda D: ratio(n, Theta, kappa, D, Phi, eps)
    lo = -2 * kappa * np.cos((Phi - Theta) / n) + 1e-6
    best = None
    # acotación gruesa y refinamiento
    grid = np.concatenate([np.linspace(max(lo, -0.9), 6, 24),
                           [9, 14, 25, 45, 80, 150, 300, 700, 2000]])
    vals = [f(D) for D in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-9})
    best = (r.fun, r.x) if r.fun < vals[k] else (vals[k], grid[k])
    # límite de Ising (Delta -> inf): 4 eps / Delta -> (n - 2 (n mod 2)) / n,
    # el techo clásico del antiferro de Ising (frustrado si n es impar).
    v_ising = (n - 2 * (n % 2)) / n
    return best if best[0] <= v_ising else (v_ising, np.inf)


def v_free(n, Theta, kappa, eps=xxz.eps, nphi=33):
    """Mínimo sobre (Delta, Phi), con Phi en [Theta - pi, Theta + pi]: es el
    representante del flujo más cercano a la holonomía. Malla y refinamiento."""
    Phis = Theta + np.linspace(-np.pi, np.pi, nphi)
    res = [(v_at_flux(n, Theta, kappa, P, eps), P) for P in Phis]
    (v0, D0), P0 = min(res, key=lambda t: t[0][0])
    h = 2 * np.pi / (nphi - 1)
    r = minimize_scalar(lambda P: v_at_flux(n, Theta, kappa, P, eps)[0],
                        bounds=(P0 - h, P0 + h), method="bounded",
                        options={"xatol": 1e-7})
    if r.fun < v0:
        v1, D1 = v_at_flux(n, Theta, kappa, r.x, eps)
        return v1, D1, r.x
    return v0, D0, P0


def gap_over_E(n, Delta, Phi):
    """Gap del testigo (energía fundamental del XXZ) en todos los sectores,
    por enlace: (E1 - E0) con E1 el primer excitado global."""
    if not np.isfinite(Delta):  # óptimo en el límite de Ising: sin gap finito que medir
        return np.nan, np.nan
    levels = []
    for M in range(n + 1):
        k = 2 if xxz.sector_operators(n, M)[0].shape[0] > 1 else 1
        levels += list(xxz.sector_spectrum(n, M, Delta, Phi, k=k))
    levels = np.sort(levels)
    return levels[1] - levels[0], levels[2] - levels[0] if len(levels) > 2 else np.nan


def eps_field(n, Delta, Phi):
    """eps_n con el multiplicador de <Z_i> = 0: campo uniforme h sum Z_i en el
    testigo. -E0/n con E0 = max_h min_M [E_M(Delta, Phi) + h (2M - n)]; el
    máximo de una envolvente inferior de rectas está en un cruce o en h = 0."""
    E = np.array([xxz.sector_spectrum(n, M, Delta, Phi)[0] for M in range(n + 1)])
    m = 2 * np.arange(n + 1) - n
    hs = [0.0] + [(E[i] - E[j]) / (m[j] - m[i]) for i in range(n + 1) for j in range(i + 1, n + 1)]
    return -max(np.min(E + h * m) for h in hs) / n


def primal(n, Theta, kappa):
    import cvxpy as cp
    from holonomy.sdp import u1_sector_operators, ring_edges
    edges = ring_edges(n)
    v = cp.Variable()
    lin, cons, tr = {}, [], 0
    for k in range(n + 1):
        basis, ops = u1_sector_operators(n, edges, k)
        rb = cp.Variable((len(basis), len(basis)), hermitian=True)
        cons.append(rb >> 0)
        tr = tr + cp.real(cp.trace(rb))
        for key, Mo in ops.items():
            lin[key] = lin.get(key, 0) + cp.real(cp.trace(Mo.toarray() @ rb))
    cons.append(tr == 1)
    phi = Theta / n
    for e in range(n):
        cons += [lin[("A", e)] == 2 * kappa * v * np.cos(phi),
                 lin[("B", e)] == 2 * kappa * v * np.sin(phi),
                 lin[("C", e)] == v]
    cons += [lin[("Z", i)] == 0 for i in range(n)]
    cp.Problem(cp.Maximize(v), cons).solve(solver="CLARABEL")
    return float(v.value)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "neel_u1.csv")
    ap.add_argument("--no-primal", action="store_true", help="omite la validación primal")
    args = ap.parse_args()
    # --- validación contra el primal U(1) en n pequeño --------------------
    try:
        if args.no_primal:
            raise ImportError
        err = 0.0
        for n in (4, 5, 6):
            for kappa in (1.0, 0.6, 0.3):
                for Theta in (0.0, np.pi / 2, np.pi):
                    vp = primal(n, Theta, kappa)
                    vd = v_free(n, Theta, kappa, eps=xxz.eps_full, nphi=17)[0]
                    err = max(err, abs(vp - vd))
        check(err < 1e-6, f"dual (Delta, Phi) = primal U(1), n = 4..6, kappa = 1, 0.6, 0.3 "
              f"(error {err:.1e})")
    except ImportError:
        print("(omito la validación primal)")

    # --- barrido ----------------------------------------------------------
    kappas = args.kappas
    ns = list(range(4, 19))
    rows = []
    print("\n kappa  n   v(0)        v(pi)       v0-vpi       Delta*(0) Delta*(pi)  Phi*(0)/pi Phi*(pi)/pi  gap1(0)  gap2(0)",
          flush=True)
    for kappa in kappas:
        for n in ns:
            t = time.time()
            v0, D0, P0 = v_free(n, 0.0, kappa)
            vp, Dp, Pp = v_free(n, np.pi, kappa)
            g1, g2 = gap_over_E(n, D0, P0)
            rows.append(dict(kappa=kappa, n=n, v0=v0, vpi=vp, D0=D0, Dpi=Dp,
                             P0=P0, Ppi=Pp, gap1=g1, gap2=g2))
            print(f" {kappa:4.2f} {n:3d}  {v0:.8f}  {vp:.8f}  {v0 - vp:+.3e}   {D0:7.4f}  {Dp:7.4f}"
                  f"   {P0 / np.pi:+7.4f}   {Pp / np.pi:+7.4f}   {g1:.2e} {g2:.2e}  ({time.time() - t:.0f} s)",
                  flush=True)
    # --- controles: signo (Marshall) y cota de Kohn con kappa = 1 ----------
    for r in rows:
        n = r["n"]
        if n % 2 == 0:
            check(r["v0"] >= r["vpi"] - 1e-9, f"kappa={r['kappa']}, n={n}: v(0) >= v(pi) (Marshall)")
        if r["kappa"] == 1.0 and n % 2 == 0:
            cota = 4 * xxz.eps(n, 1.0, np.pi) / 3
            check(r["vpi"] <= cota + 1e-9, f"n={n}: v(pi) <= 4 eps_n(1, pi)/3 (giro del testigo)")
    out = args.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nescrito {out}")
    sys.exit(0 if OK else 1)

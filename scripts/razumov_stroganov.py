"""El punto kappa = 5/4 con n impar: v(n, pi) frente a v_inf = 1/2 (robustez_mps.tex,
Proposición prop:rs).

Con n impar y Phi = pi, el giro de Marshall lleva H_XXZ(1/2, pi) al hamiltoniano de
Razumov-Stroganov periódico. Dos resultados exactos de ese punto:
  * E0 = -3n/8 (Fridkin-Stroganov-Zagier), es decir, eps_n(1/2, pi) = 3/8;
  * <S^z_i S^z_{i+1}> = -1/8 + 3/(8 n^2) (Stroganov), es decir,
    d eps_n / d Delta (1/2, pi) = 1/8 - 3/(8 n^2).
Con el Lema de reducción y la convexidad de eps_n dan
    1/2 - 3/(2 n^2) <= v(n, pi) < 1/2     (n >= 5 impar).
Numéricamente, 1/2 - v(n, pi) = 3 / (32 eps_inf''(1/2) n^4) (1 + o(1)), la
predicción del desarrollo de segundo orden de log F en Delta = 1/2.

Controles: los dos resultados exactos con diagonalización exacta (n = 5..13) y con
Bethe (hasta n = 801); el mínimo en Delta con las dos energías en n <= 11.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from holonomy import bethe, xxz  # noqa: E402
import neel_bethe as nb  # noqa: E402

OK = True
KAPPA = 1.25


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def v_pi(n, eps, lo=0.3, hi=3.0):
    r = minimize_scalar(lambda D: 4 * eps(n, D, np.pi) / (D + 2 * KAPPA), bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-12})
    return min(r.fun, (n - 2) / n), r.x


if __name__ == "__main__":
    # resultados exactos del punto de Razumov-Stroganov
    for name, eps, ns in (("diag. exacta", xxz.eps, range(5, 14, 2)),
                          ("Bethe", bethe.eps, (5, 11, 21, 51, 101, 201, 401, 801))):
        e_err = d_err = 0.0
        for n in ns:
            h = 1e-5
            e_err = max(e_err, abs(eps(n, 0.5, np.pi) - 3 / 8))
            de = (eps(n, 0.5 + h, np.pi) - eps(n, 0.5 - h, np.pi)) / (2 * h)
            d_err = max(d_err, abs(de - (1 / 8 - 3 / (8 * n * n))))
        check(e_err < 1e-13 and d_err < 1e-9,
              f"{name}: eps_n(1/2, pi) = 3/8 (err {e_err:.1e}) y eps_n' = 1/8 - 3/(8n^2) "
              f"(err {d_err:.1e}, diferencias finitas)")

    # el mínimo en Delta: diagonalización exacta frente a Bethe
    err = max(abs(v_pi(n, xxz.eps)[0] - v_pi(n, bethe.eps)[0]) for n in (5, 7, 9, 11))
    check(err < 1e-10, f"v(n, pi) con diag. exacta = Bethe, n = 5..11 (err {err:.1e})")

    h = 1e-4
    epp = (nb.eps_inf(0.5 + h) - 2 * nb.eps_inf(0.5) + nb.eps_inf(0.5 - h)) / h ** 2
    pred = 3 / (32 * epp)
    print(f"\n      eps_inf''(1/2) = {epp:.6f};  3/(32 eps'') = {pred:.5f};  3/(8 eps'') = {3 / (8 * epp):.4f}")
    print("\n     n   Delta*        v(n, pi)          n^4 (1/2 - v)   n^2 (Delta* - 1/2)")
    cotas = True
    for n in (5, 7, 9, 11, 13, 21, 51, 101, 201, 401, 801):
        lo, hi = (0.3, 3.0) if n < 21 else (0.45, 0.6)
        v, D = v_pi(n, bethe.eps, lo, hi)
        cotas &= 0.5 - 3 / (2 * n * n) <= v < 0.5
        print(f"  {n:4d}   {D:.6f}   {v:.15f}   {n ** 4 * (0.5 - v):.5f}         {n * n * (D - 0.5):.4f}")
    check(cotas, "1/2 - 3/(2 n^2) <= v(n, pi) < 1/2 en todos los n")
    sys.exit(0 if OK else 1)

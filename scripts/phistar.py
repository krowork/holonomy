"""Flujo óptimo libre Phi*(Theta) y barridos completos en el flujo (sección 5.2).

1. Barridos completos: para n = 4..13 y holonomías fuera de la ventana, el
   objetivo minimizado en Delta, como función del flujo en [0, pi], tiene su
   mínimo en la cúspide (diagonalización exacta).
2. Phi*(Theta) para n impar = 7, 9, 11, 13 cerca de Theta = pi (figura 1), con
   las ecuaciones de Bethe y comprobación puntual por diagonalización exacta:
   fuera de la ventana Phi* = 0 (cúspide); dentro se desplaza de forma continua
   hasta el punto liso, al que llega en Theta = pi. Escribe data/phistar.csv.
"""
import numpy as np

from holonomy import bethe, dual, xxz
from holonomy.report import Checker, write_csv

SWEEP_FLUXES = np.linspace(0.0, np.pi, 33)


def sweeps(ck):
    for n in range(4, 14):
        psi, _ = dual.window_edge(bethe.eps, n, slope=bethe.cusp_slope)
        smooth = dual.smooth_flux(n)
        # holonomías fuera de la ventana, de la cúspide al borde
        dists = [d for d in (np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.25 * np.pi, psi + 0.02 * np.pi) if d > psi]
        worst = np.inf
        for d in dists:
            Theta = abs(smooth - d)
            v_c = dual.v_cusp(xxz.eps, n, Theta)[0]
            others = [dual.v_at_flux(xxz.eps, n, Theta, P)[0] for P in SWEEP_FLUXES
                      if abs(P - dual.cusp_flux(n)) > 1e-12]
            worst = min(worst, min(others) - v_c)
        ck.check(worst > 0, f"n={n}: fuera de la ventana el mínimo en el flujo está en la cúspide "
                            f"({len(dists)} holonomías, margen mínimo {worst:.1e})")


def phistar_curves(ck):
    rows = []
    for n in (7, 9, 11, 13):
        psi, _ = dual.window_edge(bethe.eps, n, slope=bethe.cusp_slope)
        thetas = np.linspace(0.95, 1.0, 101) * np.pi
        prev = None
        jumps = 0.0
        for T in thetas:
            v, P, D = dual.v_free(bethe.eps, n, T)
            rows.append([n, T / np.pi, P / np.pi, v, D])
            if prev is not None:
                jumps = max(jumps, abs(P - prev))
            prev = P
        inside = [r for r in rows if r[0] == n and r[1] > 1 - psi / np.pi + 1e-3]
        outside = [r for r in rows if r[0] == n and r[1] < 1 - psi / np.pi - 1e-3]
        ck.check(all(r[2] < 1e-6 for r in outside), f"n={n}: Phi* = 0 fuera de la ventana (Theta < {1 - psi / np.pi:.4f} pi)")
        ck.check(all(r[2] > 1e-6 for r in inside), f"n={n}: Phi* > 0 dentro de la ventana")
        v_end = rows[-1][3]
        v_smooth = dual.v_smooth(bethe.eps, n, np.pi)[0]
        ck.check(abs(rows[-1][2] - 1) < 1e-4 and abs(v_end - v_smooth) < 1e-12,
                 f"n={n}: en Theta = pi el óptimo está en el punto liso (Phi*/pi = {rows[-1][2]:.6f}, "
                 f"mínimo plano; v - v_liso = {v_end - v_smooth:.0e})")
        ck.check(jumps < 0.1 * np.pi, f"n={n}: Phi* continuo (salto máximo entre puntos {jumps / np.pi:.3f} pi)")
        # comprobación puntual por diagonalización exacta
        for T in (thetas[40], thetas[90]):
            v_ed = dual.v_free(xxz.eps, n, T)[0]
            v_ba = dual.v_free(bethe.eps, n, T)[0]
            ck.close(v_ed, v_ba, 1e-10, f"n={n}, Theta = {T / np.pi:.4f} pi: v libre, diagonalización vs Bethe")
    write_csv("phistar.csv", ["n", "theta_over_pi", "phistar_over_pi", "v", "delta"], rows)


def main():
    ck = Checker()
    sweeps(ck)
    phistar_curves(ck)
    ck.finish()


if __name__ == "__main__":
    main()

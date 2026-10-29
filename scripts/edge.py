"""Bordes de la ventana de desanclaje por el criterio de la Proposición 5.

Para n = 4..20 por diagonalización exacta (sector de magnetización mínima,
Lanczos con arranque en caliente para n grande) y, por separado, con las
ecuaciones de Bethe; ambos métodos deben coincidir. Comprueba la tabla 1 del
artículo y el carácter marginal de n = 4 y n = 5. Escribe data/edges_ed.csv
(con el coeficiente de 1/n^2 y la rigidez finita, para la figura 2).
"""
import numpy as np

from holonomy import asymptotics, bethe, dual, xxz
from holonomy.report import Checker, write_csv

# Tabla 1 publicada: n -> (psi/pi, Delta* en el borde)
PUBLISHED = {
    6: (0.018587, 0.820), 7: (0.017261, 1.354), 8: (0.032868, 0.878),
    9: (0.026632, 1.171), 10: (0.043237, 0.911), 11: (0.033113, 1.102),
    12: (0.050947, 0.932), 13: (0.038055, 1.068), 14: (0.056863, 0.946),
    15: (0.042025, 1.049), 16: (0.061534, 0.957), 17: (0.045324, 1.037),
    18: (0.065312, 0.964), 19: (0.048131, 1.029), 20: (0.068431, 0.970),
}


def marginal_cases(ck):
    """n = 4 (Theta = 0) y n = 5 (Theta = pi): g_n = 0 en el punto liso."""
    golden = (1 + np.sqrt(5)) / 2
    for n, D_expected, label in ((4, 1 / np.sqrt(2), "1/sqrt 2"), (5, 3 * golden / 2, "3 phi/2")):
        Theta = dual.smooth_flux(n)
        g, Ds = dual.edge_function(bethe.eps, n, Theta, slope=bethe.cusp_slope)
        ck.check(abs(g) < 1e-8, f"n={n}: g_n = 0 en el punto liso con pendiente exacta (g = {g:.1e}, "
                                f"limitado por la precisión de Delta*)")
        ck.close(Ds, D_expected, 1e-6, f"n={n}: Delta* en el borde = {label}")
        # Con diferencias laterales de segundo orden, el residuo cae como h^2.
        Pc = dual.cusp_flux(n)
        sgn = -1.0 if n % 2 == 0 else 1.0
        e0 = xxz.eps(n, Ds, Pc)
        res = []
        hs = (1e-2, 3e-3, 1e-3)
        for h in hs:
            e1 = xxz.eps(n, Ds, Pc + sgn * h)
            e2 = xxz.eps(n, Ds, Pc + 2 * sgn * h)
            s = (-3 * e0 + 4 * e1 - e2) / (2 * h)
            d = abs(Pc - Theta) / n
            res.append(abs(s * (Ds + 2 * np.cos(d)) - (2 / n) * e0 * np.sin(d)))
        order = np.polyfit(np.log(hs), np.log(res), 1)[0]
        ck.check(abs(order - 2) < 0.2,  # el h más pequeño toca el suelo de precisión de Delta*
                 f"n={n}: residuo con diferencias laterales de 2.º orden {', '.join(f'{r:.1e}' for r in res)} "
                 f"para h = 1e-2, 3e-3, 1e-3 (orden {order:.2f})")
        v_flat = [dual.v_at_flux(xxz.eps, n, Theta, P)[0] for P in np.linspace(0, np.pi, 9)]
        ck.check(np.ptp(v_flat) < 1e-9, f"n={n}: objetivo dual constante en el flujo ({v_flat[0]:.10f}, variación {np.ptp(v_flat):.1e})")


def main():
    ck = Checker()
    marginal_cases(ck)
    rows = []
    for n in range(4, 21):
        r_ed = asymptotics.row(xxz.eps, n)
        rows.append([r_ed[k] for k in asymptotics.FIELDS])
        line = (f"n={n:>2}: psi/pi = {r_ed['psi_over_pi']:.6f}, Delta* = {r_ed['delta_edge']:.4f}, "
                f"coef = {r_ed['coef']:+.4f}, D_n = {r_ed['D_n']:.4f}")
        if n >= 6:  # en n = 4, Theta = pi, el óptimo está en Delta = 0, fuera del Bethe
            r_ba = asymptotics.row(bethe.eps, n, slope=bethe.cusp_slope)
            # psi a ~5e-8: lo limita la tolerancia relativa de Brent en Delta* (~1.5e-8),
            # que pasa a psi a través de g_n. coef y D_n llevan un factor
            # n^2 y la diagonalización usa diferencias finitas para s_n.
            d_psi = abs(r_ed["psi_over_pi"] - r_ba["psi_over_pi"])
            d_coef = abs(r_ed["coef"] - r_ba["coef"])
            d_D = abs(r_ed["D_n"] - r_ba["D_n"])
            ck.check(d_psi < 1e-7 and d_coef < 1e-5 and d_D < 1e-6,
                     f"{line}; Bethe difiere en psi/pi {d_psi:.0e}, coef {d_coef:.0e}, D_n {d_D:.0e}")
        else:
            print(f"      {line}")
        if n in PUBLISHED:
            psi_p, D_p = PUBLISHED[n]
            ok = round(r_ed["psi_over_pi"], 6) == psi_p and round(r_ed["delta_edge"], 3) == D_p
            ck.check(ok, f"n={n}: coincide con la tabla 1 ({psi_p}, {D_p})")
    write_csv("edges_ed.csv", asymptotics.FIELDS, rows)
    ck.finish()


if __name__ == "__main__":
    main()

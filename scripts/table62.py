"""Tabla de valores v_max(n, Theta) por diagonalización exacta (tabla 2).

Fuera de la ventana de desanclaje se usa la fórmula de un parámetro (flujo en la
cúspide); en los extremos Theta = 0 (n par) y Theta = pi (n impar), dentro de la
ventana, el óptimo está en el punto liso. Ambos son valores duales admisibles,
así que v_max es el menor de los dos. Para n <= 10 se compara además con el
mínimo libre sobre (Delta, Phi).
"""
import numpy as np

from holonomy import dual, xxz
from holonomy.report import Checker, write_csv

THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi)
LABELS = ("0", "pi/4", "pi/2", "3pi/4", "pi")

# Valores publicados (4 decimales) para comprobar la tabla del artículo.
PUBLISHED = {
    3: (0.3333, 0.3333, 0.3333, 0.3333, 0.3333),
    4: (0.6667, 0.5912, 0.5395, 0.5097, 0.5000),
    5: (0.4981, 0.5022, 0.5143, 0.5335, 0.5578),
    6: (0.6228, 0.5964, 0.5770, 0.5655, 0.5617),
    7: (0.5438, 0.5461, 0.5530, 0.5643, 0.5794),
    8: (0.6085, 0.5950, 0.5847, 0.5785, 0.5765),
    9: (0.5626, 0.5640, 0.5683, 0.5754, 0.5848),
    10: (0.6021, 0.5938, 0.5874, 0.5836, 0.5823),
    11: (0.5720, 0.5730, 0.5759, 0.5808, 0.5871),
    12: (0.5986, 0.5931, 0.5887, 0.5861, 0.5852),
    13: (0.5774, 0.5781, 0.5802, 0.5837, 0.5882),
    14: (0.5965, 0.5926, 0.5894, 0.5875, 0.5868),
    15: (0.5808, 0.5813, 0.5829, 0.5856, 0.5889),
    16: (0.5952, 0.5922, 0.5898, 0.5883, 0.5878),
}


def vmax(n, Theta):
    return min(dual.v_cusp(xxz.eps, n, Theta)[0], dual.v_smooth(xxz.eps, n, Theta)[0])


def main():
    ck = Checker()
    rows = []
    print(f"{'n':>3} " + " ".join(f"{l:>8}" for l in LABELS) + "   v(0)-v(pi)")
    for n in range(3, 17):
        vals = [vmax(n, T) for T in THETAS]
        rows.append([n, *vals])
        print(f"{n:>3} " + " ".join(f"{v:8.4f}" for v in vals) + f"   {vals[0] - vals[-1]:+.4f}", flush=True)
        dev = max(abs(round(v, 4) - p) for v, p in zip(vals, PUBLISHED[n]))
        ck.check(dev < 1e-12, f"n={n}: coincide con la tabla publicada a 4 decimales")
        if n <= 10:
            worst = max(abs(dual.v_free(xxz.eps, n, T)[0] - v) for T, v in zip(THETAS, vals))
            ck.check(worst < 1e-9, f"n={n}: mínimo libre sobre (Delta, Phi) = tabla (error máx {worst:.1e})")
    v5 = vmax(5, np.pi)
    ck.close(v5, (5 + 4 * np.sqrt(5)) / 25, 1e-12, "v_max(5, pi) = (5 + 4 sqrt 5)/25")
    _, D_pi = dual.v_at_flux(xxz.eps, 5, np.pi, np.pi)
    _, D_0 = dual.v_at_flux(xxz.eps, 5, np.pi, 0.0)
    ck.close(D_pi, 3.0, 1e-6, "n=5, Theta=pi: minimizador Delta = 3 en la rama de flujo pi")
    ck.close(D_0, 3 * (1 + np.sqrt(5)) / 4, 1e-6, "n=5, Theta=pi: minimizador Delta = 3 phi/2 en la rama de flujo 0")
    write_csv("table62.csv", ["n", *[f"v_{l}" for l in LABELS]], rows)
    ck.finish()


if __name__ == "__main__":
    main()

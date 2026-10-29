"""Rango del óptimo primal (sección sec:optimalidad, Conjetura conj:rango).

Resuelve holonomy.sdp.primal_u1 (bloques de número de giros fijo) y calcula el
rango numérico de la rho diagonal por bloques (holonomy.sdp.numerical_rank
sobre la matriz ensamblada). Valores del artículo:
  n = 5: rango 4 en Theta = pi/4, pi/2; rango 2 en Theta = pi (umbral marginal);
  n = 6: rango 2 fuera de la ventana (pi/4, pi/2, pi; pesos 1/2, 1/2 en pi),
         rango 1 dentro (Theta/pi = 0.005, 0.01);
  n = 7: rango 4 fuera de la ventana (Theta/pi = 0, 0.5, 0.9, 0.98), con dos
         autovalores que tienden a 0 cerca del borde (0.024 en 0.9pi,
         0.0008 en 0.98pi); rango 2 dentro (0.985, 0.99, 1).
El punto interior devuelve un elemento del interior relativo de la cara
óptima, así que el rango acota el de la cara pero no prueba la unicidad. Se
imprimen los autovalores para que el lector juzgue el corte.

Uso: python scripts/rank.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np
from scipy.linalg import block_diag

from holonomy.sdp import numerical_rank, primal_u1

RANK_TOL = 1e-5

# (n, Theta/pi, rango esperado, autovalores de referencia del artículo o None)
CASES = [
    (5, 0.25, 4, None), (5, 0.5, 4, None), (5, 1.0, 2, None),
    (6, 0.25, 2, None), (6, 0.5, 2, None), (6, 1.0, 2, (0.5, 0.5)),
    (6, 0.005, 1, None), (6, 0.01, 1, None),
    (7, 0.0, 4, None), (7, 0.5, 4, None), (7, 0.9, 4, None), (7, 0.98, 4, None),
    (7, 0.985, 2, None), (7, 0.99, 2, None), (7, 1.0, 2, None),
]
# Autovalor pequeño citado en el artículo (los dos que tienden a 0 en n = 7).
SMALL_EIG = {0.9: 0.024, 0.98: 0.0008}


def main():
    chk = Checker("rank")
    for n, t, expected, ref in CASES:
        v, blocks, status = primal_u1(n, t * np.pi, return_blocks=True, **CLARABEL_OPTS)
        rho = block_diag(*blocks)
        rank, w = numerical_rank(rho, tol=RANK_TOL)
        shown = " ".join(f"{x:.3e}" for x in w[:max(rank + 2, 6)])
        sectors = [k for k, b in enumerate(blocks)
                   if np.max(np.linalg.eigvalsh((b + b.conj().T) / 2), initial=0) > RANK_TOL]
        chk.check(rank == expected, f"n={n} Theta/pi={t:<5}: v={v:.10f} rango {rank} (esperado {expected}); "
                  f"sectores k={sectors}{chk.status_note(status)}")
        chk.info(f"autovalores: {shown}")
        if ref is not None:
            err = np.max(np.abs(w[:len(ref)] - np.array(ref)))
            chk.check(err < 1e-5, f"n={n} Theta/pi={t}: pesos {w[0]:.8f} {w[1]:.8f} vs "
                      f"{ref} err={err:.1e}")
        if n == 7 and t in SMALL_EIG:
            # Para n impar los autovalores van por pares (S^z = +-1/2): los dos
            # pequeños son w[2] y w[3]. Se compara con 2 cifras significativas.
            rel = abs(w[2] + w[3] - 2 * SMALL_EIG[t]) / (2 * SMALL_EIG[t])
            chk.check(rel < 0.1, f"n=7 Theta/pi={t}: autovalores pequeños {w[2]:.4g}, {w[3]:.4g} "
                      f"vs artículo {SMALL_EIG[t]} (dif. rel. {rel:.1e})")
    chk.finish()


if __name__ == "__main__":
    main()

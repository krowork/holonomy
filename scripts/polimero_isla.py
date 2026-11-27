"""Estructura de polímero del vacío exacto (celda_k_notas.md, Sec. 7.13, paso 6).

Si una isla (I, c) del vacío de la fase 0 es un polímero con el vestido de su propia fase dentro,
  phi_gamma = a_gamma chi_{c 1_I} exp((G_c - G_0)|_I + ...),
entonces, al orden más bajo de Mayer, g_0(c 1_I + nu) / g_0(c 1_I) = (g_c - g_0)(nu) para nu en el interior de I.
Se compara en el vacío exacto (Kirkwood-Thomas completo con Anderson) para los nu de dos sitios en los enlaces
más interiores de una isla de longitud L.
Uso: python scripts/polimero_isla.py modelo n eps L
"""
import sys

import numpy as np

from base_fases import vacio
from dual_celda import anillo
from kt_zk import K, marco, termino


def main():
    modelo, n, eps, L = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    hcl, v, q = termino(modelo)
    dig, am, dm = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    H = anillo(n, K, q, hcl + eps * v, 0.0)
    pw = K ** np.arange(n)
    gs = {}
    for c in range(K):
        chi = np.exp(2j * np.pi / K * c * dig.sum(1))
        gs[c] = vacio(lambda x, chi=chi: chi.conj() * am(H @ dm(chi * x)), n, np.where(borde > 0, borde, 0))[0]
    idx = lambda m: int((np.asarray(m) % K * pw).sum())
    for c in range(1, K):
        isla = np.zeros(n, int)
        isla[1:1 + L] = c
        g_isla = gs[0][idx(isla)]
        print(f"{modelo} n={n} eps={eps} isla c={c}, L={L}: g_0(isla) = {abs(g_isla):.3e}")
        filas = []
        centro = 1 + L // 2
        for j in (centro - 1, centro):
            for a in range(1, K):
                for b in range(1, K):
                    nu = np.zeros(n, int)
                    nu[j], nu[j + 1] = a, b
                    izq = gs[0][idx(isla + nu)] / g_isla
                    der = gs[c][idx(nu)] - gs[0][idx(nu)]
                    filas.append((j, a, b, izq, der))
        err = max(abs(i - d) for *_, i, d in filas)
        esc = max(abs(d) for *_, i, d in filas)
        for j, a, b, i, d in filas:
            print(f"   nu en ({j},{j + 1}) = ({a},{b}): cociente {i.real:+.4f}{i.imag:+.4f}i   g_c - g_0 = "
                  f"{d.real:+.4f}{d.imag:+.4f}i")
        print(f"   máx |diferencia| = {err:.2e} frente a una escala {esc:.2e}", flush=True)


if __name__ == "__main__":
    main()

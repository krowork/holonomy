"""Radio espectral de la linealización de Jacobi de Kirkwood-Thomas en el punto fijo (celda_k_notas.md, Sec. 7.13).

J v = d/dt [g* + t v - R^(g* + t v)/|d mu|] en t = 0 (mu no constante), por diferencias finitas complejas;
el radio espectral se estima con iteración de potencias (y Arnoldi corto). Si rho(J) >= 1, ninguna norma hace
contractiva la iteración de Jacobi, aunque el punto fijo exista (lo encuentra Anderson, vacio_islas.py).
Uso: python scripts/radio_jacobi.py modelo n eps [eps ...]
"""
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from base_fases import vacio
from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def main():
    modelo, n = sys.argv[1], int(sys.argv[2])
    hcl, v, q = termino(modelo)
    dig, am, dm = marco(n)
    N = K**n
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    libre = borde > 0
    for eps in [float(x) for x in sys.argv[3:]]:
        H = anillo(n, K, q, hcl + eps * v, 0.0)
        Hop = lambda x: am(H @ dm(x))
        g, _ = vacio(Hop, n, np.where(libre, borde, 0))

        def paso(g):
            G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * N
            Rh = chi_hat(Hop(np.exp(G)) / np.exp(G), n)
            return np.where(libre, Rh / np.where(libre, borde, 1), 0)

        p0 = paso(g)
        h = 1e-7

        def Jfull(x):
            x = np.where(libre, x, 0)  # sin los caracteres constantes, que la iteración no mueve
            s = max(np.abs(x).max(), 1e-300)
            return np.where(libre, x - (paso(g + h * x / s) - p0) / h * s, 0)  # J = I - d(R/|d|)

        op = LinearOperator((N, N), matvec=Jfull, dtype=complex)
        w = eigs(op, k=3, which="LM", tol=1e-6, maxiter=400, return_eigenvectors=False)
        print(f"{modelo} n={n} eps={eps}: radio espectral de Jacobi = {np.abs(w).max():.3f}  "
              f"(autovalores de mayor módulo: {', '.join(f'{z:.3f}' for z in w)})", flush=True)


if __name__ == "__main__":
    main()

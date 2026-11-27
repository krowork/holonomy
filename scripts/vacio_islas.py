"""Vacío completo de Kirkwood-Thomas del modelo de carga entera y sus coeficientes de isla (celda_k_notas.md,
Sec. 7.13, hipótesis (V)).

Resuelve R^(mu) = 0 (mu no constante) con la iteración de Jacobi amortiguada y acelerada por Anderson (memoria
MEM), que converge donde Jacobi puro diverge. Da:
  - si converge, iteraciones, E/n y el residuo final;
  - para cada longitud L de tramo, I_L = max |g(mu)| sobre los mu cuyo mayor tramo constante no nulo mide L
    (las "islas" de longitud L con su decoración), y la razón I_{L+1}/I_L;
  - la norma de Z1 sin pesos, sup_x sum_{mu ∋ x} |g(mu)| |d mu|.
Uso: python scripts/vacio_islas.py modelo n eps   (MEM, BETA en el entorno)
"""
import os
import sys

import numpy as np

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino
from vestido_fases import tramo_max

MEM = int(os.environ.get("MEM", 8))
BETA = float(os.environ.get("BETA", 0.5))


def main():
    modelo, n, eps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    hcl, v, q = termino(modelo)
    H = anillo(n, K, q, hcl + eps * v, 0.0)
    dig, am, dm = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    libre = borde > 0

    def F(g):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(am(H @ dm(np.exp(G))) / np.exp(G), n)
        return np.where(libre, -Rh / np.where(libre, borde, 1), 0), Rh

    g = np.zeros(K**n, complex)
    X, Fs = [], []
    for it in range(3000):
        f, Rh = F(g)  # f = paso de Jacobi
        err = np.abs(f).max()
        if not np.isfinite(err) or err > 1e6:
            print(f"{modelo} n={n} eps={eps}: diverge en la iteración {it}")
            return
        if err < 1e-11:
            break
        X.append(g.copy())
        Fs.append(f.copy())
        X, Fs = X[-MEM:], Fs[-MEM:]
        if len(Fs) > 1:  # Anderson: combinación de los últimos pasos que minimiza el residuo
            dF = np.array([Fs[i + 1] - Fs[i] for i in range(len(Fs) - 1)]).T
            dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
            gam = np.linalg.lstsq(dF, f, rcond=None)[0]
            g = g + BETA * f - (dX + BETA * dF) @ gam
        else:
            g = g + BETA * f
    print(f"{modelo} n={n} eps={eps}: converge en {it} it., E/n = {Rh[0].real / n:.10f}, paso final {err:.1e}")
    tm = tramo_max(dig)
    I = [np.abs(g[(tm == L) & libre]).max() if ((tm == L) & libre).any() else 0 for L in range(1, n)]
    print("   I_L (máx |g| con tramo mayor L): " + "  ".join(f"{L + 1}:{x:.1e}" for L, x in enumerate(I)))
    print("   razones I_{L+1}/I_L: " + "  ".join(f"{I[i + 1] / I[i]:.3f}" for i in range(len(I) - 1) if I[i] > 0))
    nz = dig != 0
    nrm = max((np.abs(g) * borde)[nz[:, x]].sum() for x in range(n))
    print(f"   sup_x sum_(mu ∋ x) |g| |d mu| = {nrm:.4f}", flush=True)


if __name__ == "__main__":
    main()

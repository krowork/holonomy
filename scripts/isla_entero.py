"""Obstrucción del modelo de carga entera: columnas de islas del generador vestido (celda_k_notas.md, Sec. 7.13).

Con el punto fijo de Kirkwood-Thomas G del marco de reloj (kt_zk.py) se calcula, para el carácter de una isla
mu = c en un intervalo de longitud L (resto 0), la columna de H~ = e^{-G} H' e^{G} en la base de caracteres:
  - diag = H~_{mu mu} - E, que vale |d mu| + O(eps) = 2 + O(eps);
  - radio = sum_{mu' != mu} |H~_{mu' mu}|, el radio del disco de Gershgorin.
Con perturbación sin simetría de reloj (S+S-), el radio crece como eps * sum_j K_c(j) ~ (8/3)|eps| L (Lema Z8);
con el intercambio SU(3) no crece con L.
Uso: python scripts/isla_entero.py modelo n eps
"""
import sys

import numpy as np

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def main():
    modelo, n, eps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    hcl, v, q = termino(modelo)
    H = anillo(n, K, q, hcl + eps * v, 0.0)
    dig, am, dm = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    const = borde == 0
    g = np.zeros(K**n, complex)
    for it in range(1000):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(am(H @ dm(np.exp(G))) / np.exp(G), n)
        paso = np.where(const, 0, Rh / np.where(const, 1, borde))
        g -= paso
        if not np.isfinite(paso).all() or np.abs(paso).max() > 1e3:
            sys.exit(f"KT diverge (it {it})")
        if np.abs(paso).max() < 1e-12:
            break
    E = Rh[0]
    eG, emG = np.exp(G), np.exp(-G)
    pw = K ** np.arange(n)
    print(f"{modelo} n={n} eps={eps}: KT en {it} iteraciones, E/n = {E.real / n:.8f}")
    for c in (1, 2):
        fila = []
        for L in range(1, n):
            mu_dig = np.zeros(n, int)
            mu_dig[1:1 + L] = c
            mu = int((mu_dig * pw).sum())
            chi = np.exp(2j * np.pi / K * (dig @ mu_dig))
            col = chi_hat(emG * am(H @ dm(eG * chi)), n)
            radio = np.abs(col).sum() - abs(col[mu])
            fila.append(f"L={L}: {(col[mu] - E).real:.3f}/{radio:.3f}")
        print(f"  isla c={c} (diag/radio): " + "  ".join(fila), flush=True)


if __name__ == "__main__":
    main()

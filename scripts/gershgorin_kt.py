"""Gershgorin para el generador vestido H~ = e^{-G} H' e^{G} en la base de caracteres (celda_k_notas.md, Sec. 7.11).

Con el punto fijo de Kirkwood-Thomas G del intercambio SU(3) (kt_zk.py), H~ chi_mu = chi_mu * [E + sum_c r_c chi_c +
F_mu + |d mu|], con F_mu local en el borde de mu. Para cada columna mu se calcula
  eta_mu = (|M_mumu - E - |d mu|| + sum_{mu' != mu} |M_mu'mu|) / |d mu|      (mu no constante),
y la cota de Gershgorin del hueco, min_mu (Re M_mumu - E - sum_{mu' != mu} |M_mu'mu|), frente al hueco exacto.
Uso: python scripts/gershgorin_kt.py theta_sobre_pi n eps [eps ...]   (MUESTRA = número de columnas, 0 = todas)
"""
import os
import sys

import numpy as np

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def punto_fijo(n, eps, theta, maxit=500, tol=1e-13):
    hcl, v, q = termino("llenado")
    H = anillo(n, K, q, hcl + eps * v, theta)
    dig, am, dm = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    const = borde == 0
    g = np.zeros(K**n, complex)
    for _ in range(maxit):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(am(H @ dm(np.exp(G))) / np.exp(G), n)
        paso = np.where(const, 0, Rh / np.where(const, 1, borde))
        g -= paso
        if np.abs(paso).max() < tol:
            break
    G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
    return H, dig, am, dm, G, borde, Rh[0]


def main():
    th, n = float(sys.argv[1]) * np.pi, int(sys.argv[2])
    muestra = int(os.environ.get("MUESTRA", 0))
    for eps in [float(x) for x in sys.argv[3:]]:
        H, dig, am, dm, G, borde, E = punto_fijo(n, eps, th)
        eG, emG = np.exp(G), np.exp(-G)
        N = K**n
        cols = np.arange(N) if muestra == 0 or muestra >= N else np.random.default_rng(0).choice(N, muestra, False)
        eta, cota = 0.0, np.inf
        for mu in cols:
            chi = np.exp(2j * np.pi / K * (dig @ dig[mu]))  # chi_mu(s) = w^{mu.s}; dig[mu] son las componentes de mu
            col = chi_hat(emG * am(H @ dm(eG * chi)), n)
            diag = col[mu]
            off = np.abs(col).sum() - abs(diag)
            if borde[mu] > 0:
                eta = max(eta, (abs(diag - E - borde[mu]) + off) / borde[mu])
                cota = min(cota, (diag - E).real - off)
        ev = np.linalg.eigvalsh(H.toarray()) if N <= 20000 else None
        hueco = "" if ev is None else f"  hueco exacto (4º - 1º) = {ev[3] - ev[0]:.4f}"
        print(f"n={n} eps={eps:+.3f} Theta/pi={th / np.pi}: eta = {eta:.4f}  cota de Gershgorin del hueco = {cota:.4f}"
              f"  (columnas: {len(cols)}){hueco}", flush=True)


if __name__ == "__main__":
    main()

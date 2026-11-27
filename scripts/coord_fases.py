"""Coordenadas de chi_gamma Phi_beta en la base por fases (celda_k_notas.md, Sec. 7.13, Lema Z11).

Para pares (beta, gamma) con Y = {x : beta_x != gamma_x}, se calcula c = B^{-1}(chi_gamma Phi_beta) en la base
b_mu = chi_mu Phi_mu de base_fases.py, y se da, por |Y|, el máximo de ||c - e_gamma||_1 y la tasa efectiva
sigma_ef = max log(1 + ||c - e_gamma||_1)/|Y| (el lema da ||c - e_gamma||_1 <= e^{delta |Y|} - 1).
Uso: python scripts/coord_fases.py modelo n eps [eps ...]
"""
import sys

import numpy as np

from base_fases import dominios, vacio
from dual_celda import anillo
from kt_zk import K, marco, termino


def main():
    modelo, n = sys.argv[1], int(sys.argv[2])
    hcl, v, q = termino(modelo)
    dig, am, dm = marco(n)
    N = K**n
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    sop = dig != 0
    Gf = lambda g: np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
    rng = np.random.default_rng(1)
    for eps in [float(x) for x in sys.argv[3:]]:
        H = anillo(n, K, q, hcl + eps * v, 0.0)
        gs = {}
        for c in range(K):
            chi = np.exp(2j * np.pi / K * c * dig.sum(1))
            gs[c] = vacio(lambda x, chi=chi: chi.conj() * am(H @ dm(chi * x)), n, np.where(borde > 0, borde, 0))[0]
        Phi = np.empty((N, N), complex)
        for mu in range(N):
            m = dig[mu]
            lab = dominios(m)
            g = gs[0].copy()
            for d in np.unique(lab):
                en_d = lab == d
                dentro = ~(sop & ~en_d[None, :]).any(1) & sop.any(1)
                g = np.where(dentro, gs[int(m[en_d][0])], g)
            Phi[:, mu] = np.exp(Gf(g))
        chis = np.exp(2j * np.pi / K * (dig @ dig.T))  # chis[:, mu] = chi_mu
        B = chis * Phi
        Binv = np.linalg.inv(B)
        res = {}
        for _ in range(4000):
            beta, gamma = rng.integers(N, size=2)
            Y = int((dig[beta] != dig[gamma]).sum())
            c = Binv @ (chis[:, gamma] * Phi[:, beta])
            c[gamma] -= 1
            x = np.abs(c).sum()
            r = res.setdefault(Y, [0.0, 0.0])
            r[0] = max(r[0], x)
            r[1] = max(r[1], np.log1p(x) / max(Y, 1))
        print(f"{modelo} n={n} eps={eps}: ||c - e_gamma||_1 máx (sigma_ef) por |Y|: " +
              "  ".join(f"{Y}: {res[Y][0]:.3f} ({res[Y][1]:.3f})" for Y in sorted(res)), flush=True)


if __name__ == "__main__":
    main()

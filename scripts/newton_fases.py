"""Kirkwood-Thomas precondicionado en la base por fases (celda_k_notas.md, Sec. 7.13, paso 6′).

En el punto fijo, la linealización de R^ es J = H~ - E (H~ = e^{-G} H' e^{G}). En la base por fases B
(b_mu = chi_mu e^{G_mu}, base_fases.py), M = B^{-1} H' B. El quasi-Newton G <- G - B D^{-1} B^{-1} R^(G), con D la
diagonal de M - E en los caracteres no constantes, tiene como linealización I - D^{-1}(M - E). Se compara su radio
espectral con el de Jacobi (radio_jacobi.py).
  - n <= 6: M completa (doble precisión), D exacta, autovalores densos.
  - n = 9: LU de B en precisión simple; D tomada de la columna local de Z10, (e^{-G_mu} H' e^{G_mu} chi_mu)^(mu);
    radio espectral por Arnoldi.
Uso: python scripts/newton_fases.py modelo n eps [eps ...]
"""
import sys

import numpy as np
import scipy.linalg as sl
from scipy.sparse.linalg import LinearOperator, eigs

from base_fases import dominios, vacio
from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def main():
    modelo, n = sys.argv[1], int(sys.argv[2])
    hcl, v, q = termino(modelo)
    dig, am, dm = marco(n)
    N = K**n
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    sop = dig != 0
    nc = borde > 0
    Gf = lambda g: np.fft.ifftn(g.reshape((K,) * n)).ravel() * N
    dt = np.complex128 if N <= 3000 else np.complex64
    for eps in [float(x) for x in sys.argv[3:]]:
        H = anillo(n, K, q, hcl + eps * v, 0.0)
        gs, E = {}, None
        for c in range(K):
            chi = np.exp(2j * np.pi / K * c * dig.sum(1))
            g, Ec = vacio(lambda x, chi=chi: chi.conj() * am(H @ dm(chi * x)), n, np.where(nc, borde, 0))
            gs[c] = g
            E = Ec if E is None else E
        B = np.empty((N, N), dt)
        Dloc = np.empty(N, complex)
        for mu in range(N):
            m = dig[mu]
            lab = dominios(m)
            g = gs[0].copy()
            for d in np.unique(lab):
                en_d = lab == d
                dentro = ~(sop & ~en_d[None, :]).any(1) & sop.any(1)
                g = np.where(dentro, gs[int(m[en_d][0])], g)
            G = Gf(g)
            chi = np.exp(2j * np.pi / K * (dig @ m))
            b = chi * np.exp(G)
            B[:, mu] = b
            if N > 3000:
                Dloc[mu] = chi_hat(np.exp(-G) * am(H @ dm(b)), n)[mu] - E
        idx = np.nonzero(nc)[0]
        if N <= 3000:
            Mm = sl.solve(B, np.column_stack([am(H @ dm(B[:, j])) for j in range(N)]))
            A = Mm[np.ix_(idx, idx)] - E * np.eye(idx.size)
            D = np.diag(A).copy()
            T = np.eye(idx.size) - A / D[:, None]
            rho = np.abs(np.linalg.eigvals(T)).max()
            print(f"{modelo} n={n} eps={eps}: radio espectral del quasi-Newton por fases = {rho:.3f}", flush=True)
        else:
            Bc = B.copy()
            lu = sl.lu_factor(B, overwrite_a=True, check_finite=False)
            del B
            D = Dloc[idx]

            def T(x):
                y = np.zeros(N, complex)
                y[idx] = x
                z = sl.lu_solve(lu, am(H @ dm(Bc @ y.astype(dt))).astype(dt), check_finite=False)
                return x - (z[idx] - E * x) / D

            w = eigs(LinearOperator((idx.size, idx.size), matvec=T, dtype=complex), k=4, which="LM", tol=1e-4,
                     maxiter=300, return_eigenvectors=False)
            print(f"{modelo} n={n} eps={eps}: radio espectral del quasi-Newton por fases = {np.abs(w).max():.3f} "
                  f"(D de la columna local)", flush=True)


if __name__ == "__main__":
    main()

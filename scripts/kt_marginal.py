"""Marginal media del cuasi-modo Z_k frente a la de un anillo mayor sin flujo, girada (celda_k_notas.md, Sec. 7.8).

Para el intercambio SU(3) (modelo 'llenado' de kt_zk.py): calcula la marginal de arista, promediada sobre las
traslaciones, del cuasi-modo de Kirkwood-Thomas con n sitios y flujo Theta, y la compara con
(1 x V_phi) rho_ref (1 x V_phi)^dag, donde rho_ref es la del cuasi-modo sin flujo de NREF sitios.
Uso: python scripts/kt_marginal.py theta_sobre_pi eps n [n ...]   (NREF en el entorno, por defecto 12)
"""
import os
import sys

import numpy as np

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def cuasimodo(n, eps, theta, maxit=400, tol=1e-13):
    hcl, v, q = termino("llenado")
    H = anillo(n, K, q, hcl + eps * v, theta)
    dig, a_marco, desde_marco = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    const = borde == 0
    g = np.zeros(K**n, complex)
    for _ in range(maxit):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        psi_m = np.exp(G)
        Rh = chi_hat(a_marco(H @ desde_marco(psi_m)) / psi_m, n)
        paso = np.where(const, 0, Rh / np.where(const, 1, borde))
        g -= paso
        if np.abs(paso).max() < tol:
            break
    psi = desde_marco(np.exp(np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n))
    return psi / np.linalg.norm(psi), q


def marginal_media(psi, n):
    T = psi.reshape((K,) * n)  # eje 0 = sitio n-1, eje n-1 = sitio 0
    rhos = []
    for j in range(n):
        a, b = n - 1 - j, n - 1 - ((j + 1) % n)  # ejes de los sitios j y j+1
        resto = [x for x in range(n) if x not in (a, b)]
        A = np.moveaxis(T, (a, b), (0, 1)).reshape(K * K, -1)
        rhos.append(A @ A.conj().T)  # índice K*s_j + s_{j+1}
    return sum(rhos) / n


def main():
    th, eps = float(sys.argv[1]) * np.pi, float(sys.argv[2])
    nref = int(os.environ.get("NREF", 12))
    psi, q = cuasimodo(nref, eps, 0.0)
    rho_ref = marginal_media(psi, nref)
    for n in [int(x) for x in sys.argv[3:]]:
        psi, q = cuasimodo(n, eps, th)
        rho = marginal_media(psi, n)
        V = np.diag(np.exp(-1j * th / n * np.array(q)))
        U = np.kron(np.eye(K), V)
        dif = rho - U @ rho_ref @ U.conj().T
        print(f"Theta/pi={th / np.pi} eps={eps} n={n}: ||rho_n(Theta) - girada(rho_{nref}(0))||_1 = "
              f"{np.abs(np.linalg.eigvalsh((dif + dif.conj().T) / 2)).sum():.3e}",
              flush=True)


if __name__ == "__main__":
    main()

"""Constantes del lema de contracción de Kirkwood-Thomas en el marco de reloj Z_k (celda_k_notas.md, Sec. 7.7).

En el marco (Fourier en cada sitio y gauge prod_j w^{-j s_j}), un operador de dos sitios en el enlace (j, j+1) se
escribe O' = sum_{delta, nu} o(delta, nu) chi_nu tau_delta, con (tau_delta psi)(s) = psi(s + delta) y chi_nu(s) = w^{nu.s}.
  - a = max_j sum |v^(delta, nu)|  (parte con simetría de reloj; se comprueba que solo hay delta_1 + delta_2 = 0);
  - b = sup_{0 < |phi| <= pi/4} (pi/|phi|) sum |w^(delta, nu)|, con w = v^{phi} - v el giro: cota de n sum |w^|
    para |phi| <= pi/n;
  - M_*: el menor M con K(M) <= 1/2, con delta = (4a + b)/M y
    K = 2(e^{2d} - 1 + 2d e^{2d}) + (2(a+b)/M)(1+2d) e^{2d} + ((4a + b)/M) e^{2d}.
Uso: python scripts/kt_constantes.py modelo
"""
import sys

import numpy as np

from kt_zk import K, W, termino


def coeficientes(O, j):
    F = np.array([[W ** (a * b) for b in range(K)] for a in range(K)]) / np.sqrt(K)
    F2 = np.kron(F, F)
    s = np.array([(x // K, x % K) for x in range(K * K)])  # índice 3a + b: a sitio j, b sitio j+1
    D2 = np.diag(W ** (-(j * s[:, 0] + (j + 1) * s[:, 1]) % K))
    Op = D2 @ F2 @ O @ F2.conj().T @ D2.conj().T
    c = {}
    for d0 in range(K):
        for d1 in range(K):
            tau = np.zeros((K * K, K * K))
            for x in range(K * K):
                a, b = divmod(x, K)
                tau[x, K * ((a + d0) % K) + (b + d1) % K] = 1  # (tau psi)(s) = psi(s + delta)
            for n0 in range(K):
                for n1 in range(K):
                    chi = np.diag(W ** ((n0 * s[:, 0] + n1 * s[:, 1]) % K))
                    val = np.trace((chi @ tau).conj().T @ Op) / K**2
                    if abs(val) > 1e-12:
                        c[(d0, d1, n0, n1)] = val
    rec = sum(val * np.diag(W ** ((k[2] * s[:, 0] + k[3] * s[:, 1]) % K)) @ tau_de(k[0], k[1]) for k, val in c.items())
    assert np.allclose(rec, Op)
    return c


def tau_de(d0, d1):
    tau = np.zeros((K * K, K * K))
    for x in range(K * K):
        a, b = divmod(x, K)
        tau[x, K * ((a + d0) % K) + (b + d1) % K] = 1
    return tau


def giro(v, q, phi):
    V = np.diag(np.exp(-1j * phi * np.array(q)))
    U = np.kron(np.eye(K), V)
    return U @ v @ U.conj().T


def main():
    modelo = sys.argv[1]
    _, v, q = termino(modelo)
    a, mezcla = 0.0, 0.0
    for j in range(K):
        c = coeficientes(v.astype(complex), j)
        a = max(a, sum(abs(x) for x in c.values()))
        mezcla = max(mezcla, sum(abs(x) for k, x in c.items() if (k[0] + k[1]) % K != 0))
    b = 0.0
    for phi in np.linspace(1e-3, np.pi / 4, 200):
        for j in range(K):
            c = coeficientes(giro(v, q, phi) - v, j)
            b = max(b, np.pi / phi * sum(abs(x) for x in c.values()))
    print(f"{modelo}: ||v|| = {np.linalg.norm(v, 2):.3f}, a = {a:.4f}, parte que mezcla sectores Z_k = {mezcla:.4f}, "
          f"b = {b:.4f}")

    def Kc(M):
        d = (4 * a + b) / M
        return (2 * (np.exp(2 * d) - 1 + 2 * d * np.exp(2 * d)) + 2 * (a + b) / M * (1 + 2 * d) * np.exp(2 * d)
                + (4 * a + b) / M * np.exp(2 * d))
    M = 1.0
    while Kc(M) > 0.5:
        M *= 1.001
    print(f"M_* = {M:.1f}  (radio garantizado |eps| < {1 / M:.2e}; delta = {(4 * a + b) / M:.4f})")


if __name__ == "__main__":
    main()

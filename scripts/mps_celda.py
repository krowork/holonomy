"""Teorema MPS con celda de k sitios: comprobación numérica con k = 2.

MPS de periodo k: tensores A_i^s (i = 1..k) de D_i x D_{i+1}, con D_{k+1} = D_1,
y simetría U(1) enlace a enlace,

    u^(i)_t A_i^s u^(i+1)_t^dagger = e^{-i t q_s} A_i^s,   u^(i)_t = diag(e^{-i t c^(i)}),

con c^(i) las cargas virtuales del enlace i. Para que A_i^s[a, b] != 0 hace falta
c^(i)_a - c^(i+1)_b = q_s. Datos: sigma = (1/k) sum_i sigma_i, con sigma_i la
marginal (límite termodinámico) de la arista de tipo i.

Estado retorcido en el anillo de n = m k sitios: B_i^s = A_i^s w_{i+1}, con
w_i = u^(i)_phi y phi = Theta / n. Afirmación: la marginal de la arista de tipo i es
(1 ⊗ V) sigma_i (1 ⊗ V)^dagger + O(lambda^(m-1)), uniformemente en Theta, con
V = e^{-i phi Q} y lambda el segundo autovalor (relativo) del canal de la celda.

Ejemplos:
  (a) espín 1/2, cargas virtuales {-1/2, 1/2} y {-1, 0, 1} (D = 2, 3);
  (b) espín 3/2, cargas virtuales {-1, 0, 1} y {-1/2, 1/2} (D = 3, 2): sigma
      tiene rango < d^2 = 16, así que además p <= 1.
En ambos la carga por sitio es semientera: k = 1 es imposible (LSM) y k = 2 no.
"""
import sys
from itertools import product

import numpy as np

OK = True
rng = np.random.default_rng(20261117)


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def random_u1_cell(q, charges):
    """Tensores aleatorios A[i][s] (D_i x D_{i+1}) con la regla de carga."""
    k = len(charges)
    A = []
    for i in range(k):
        ci, cj = charges[i], charges[(i + 1) % k]
        T = np.zeros((len(q), len(ci), len(cj)), dtype=complex)
        for s, a, b in product(range(len(q)), range(len(ci)), range(len(cj))):
            if np.isclose(ci[a] - cj[b], q[s]):
                T[s, a, b] = rng.normal() + 1j * rng.normal()
        A.append(T)
    return A


def u(c, t):
    return np.diag(np.exp(-1j * t * np.asarray(c)))


def cell_product(A, i, ss):
    """A_i^{s_0} A_{i+1}^{s_1} ... (índices de tipo cíclicos)."""
    k = len(A)
    M = A[i % k][ss[0]]
    for l, s in enumerate(ss[1:], 1):
        M = M @ A[(i + l) % k][s]
    return M


def channel(A, i):
    """Matriz de transferencia de doble capa de la celda que empieza en el tipo i."""
    k, d = len(A), A[0].shape[0]
    T = None
    for l in range(k):
        X = A[(i + l) % k]
        Tl = sum(np.kron(X[s], X[s].conj()) for s in range(d))
        T = Tl if T is None else T @ Tl
    return T


def twisted(A, charges, phi):
    k = len(A)
    return [np.array([A[i][s] @ u(charges[(i + 1) % k], phi) for s in range(A[i].shape[0])])
            for i in range(k)]


def edge_marginal_ring(B, i, m):
    """Marginal de la arista (i, i+1) del anillo de m celdas, por transferencia."""
    k, d = len(B), B[0].shape[0]
    rest = None
    for l in range(2, k):
        X = B[(i + l) % k]
        Tl = sum(np.kron(X[s], X[s].conj()) for s in range(d))
        rest = Tl if rest is None else rest @ Tl
    T = channel(B, i)
    Tm1 = np.linalg.matrix_power(T, m - 1)
    norm = np.trace(Tm1 @ T)
    rho = np.zeros((d * d, d * d), dtype=complex)
    for a, b, c, e in product(range(d), repeat=4):
        X = np.kron(B[i % k][a] @ B[(i + 1) % k][b], (B[i % k][c] @ B[(i + 1) % k][e]).conj())
        if rest is not None:
            X = X @ rest
        rho[a * d + b, c * d + e] = np.trace(X @ Tm1)
    return rho / norm


def edge_marginal_thermo(A, i):
    """sigma_i: límite termodinámico (vectores propios dominantes del canal)."""
    T = channel(A, i)
    w, R = np.linalg.eig(T)
    j = np.argmax(np.abs(w))
    wl, L = np.linalg.eig(T.T)
    jl = np.argmax(np.abs(wl))
    R, L, mu = R[:, j], L[:, jl], w[j]
    k, d = len(A), A[0].shape[0]
    rest = None
    for l in range(2, k):
        X = A[(i + l) % k]
        Tl = sum(np.kron(X[s], X[s].conj()) for s in range(d))
        rest = Tl if rest is None else rest @ Tl
    rho = np.zeros((d * d, d * d), dtype=complex)
    for a, b, c, e in product(range(d), repeat=4):
        X = np.kron(A[i % k][a] @ A[(i + 1) % k][b], (A[i % k][c] @ A[(i + 1) % k][e]).conj())
        if rest is not None:
            X = X @ rest
        rho[a * d + b, c * d + e] = L @ X @ R
    return rho / (L @ T @ R), w


def brute_marginals(B, n):
    """Todas las marginales de arista del anillo por contracción completa."""
    k, d = len(B), B[0].shape[0]
    psi = np.zeros(d ** n, dtype=complex)
    for idx, ss in enumerate(product(range(d), repeat=n)):
        psi[idx] = np.trace(cell_product(B, 0, ss))
    psi /= np.linalg.norm(psi)
    P = psi.reshape([d] * n)
    out = []
    for j in range(n):
        Q = np.moveaxis(P, [j, (j + 1) % n], [0, 1]).reshape(d * d, -1)
        out.append(Q @ Q.conj().T)
    return out


def trnorm(X):
    return np.abs(np.linalg.eigvalsh((X + X.conj().T) / 2)).sum()


def run(name, q, charges, n_brute, ms):
    print(f"\n=== {name} ===")
    k, d = len(charges), len(q)
    A = random_u1_cell(q, charges)
    Q = np.diag(q)
    t = 0.613
    check(all(np.allclose(u(charges[i], t) @ A[i][s] @ u(charges[(i + 1) % k], t).conj().T,
                          np.exp(-1j * t * q[s]) * A[i][s]) for i in range(k) for s in range(d)),
          "simetría U(1) enlace a enlace")
    # inyectividad de la celda: los productos de L celdas generan M_{D_1}
    D1 = len(charges[0])
    for L in (1, 2, 3):
        prods = [cell_product(A, 0, ss).ravel() for ss in product(range(d), repeat=k * L)]
        r = np.linalg.matrix_rank(np.array(prods), tol=1e-9)
        if r == D1 * D1:
            break
    check(r == D1 * D1, f"celda inyectiva (los productos de {L} celdas generan M_{D1})")
    # sigma_i y espectro del canal de la celda
    sig, lam = [], None
    for i in range(k):
        s_i, w = edge_marginal_thermo(A, i)
        sig.append(s_i)
        wa = np.sort(np.abs(w))[::-1]
        lam_i = wa[1] / wa[0]
        lam = lam_i if lam is None else lam
        check(np.isclose(lam_i, lam), f"canal de la celda tipo {i}: mismo lambda = {lam_i:.4f}")
    sigma = sum(sig) / k
    check(np.isclose(np.trace(sigma).real, 1) and np.allclose(sigma, sigma.conj().T),
          "sigma = media de las sigma_i es un estado")
    S4 = sigma.reshape(d, d, d, d)
    check(np.allclose(np.einsum("abac->bc", S4), np.einsum("abcb->ac", S4))
          and np.allclose(np.einsum("abac->bc", S4), np.diag(np.diag(np.einsum("abac->bc", S4)))),
          "tr_1 sigma = tr_2 sigma, diagonal en la base de carga (reducciones compatibles)")
    rk = np.linalg.matrix_rank(sigma, tol=1e-10)
    print(f"      rango de sigma = {rk} de {d * d}" + ("  (luego p <= 1)" if rk < d * d else ""))
    # fuerza bruta frente a transferencia, con giro
    for n in n_brute:
        m = n // k
        err = 0
        for Theta in (0.0, 1.1, np.pi):
            B = twisted(A, charges, Theta / n)
            br = brute_marginals(B, n)
            for j in range(n):
                tm = edge_marginal_ring(B, j % k, m)
                err = max(err, np.abs(br[j] - tm).max())
        check(err < 1e-12, f"n = {n}: marginales por fuerza bruta = transferencia ({err:.1e})")
    # eps_n frente a lambda^(m-1), uniforme en Theta
    print("      m    n   max_Theta eps_n   eps_n/lambda^(m-1)   eps media")
    ratios = []
    for m in ms:
        n = k * m
        e_max, e_mix = 0, 0
        for Theta in np.linspace(0, np.pi, 9):
            phi = Theta / n
            V = np.kron(np.eye(d), np.diag(np.exp(-1j * phi * np.asarray(q))))
            B = twisted(A, charges, phi)
            rho = [edge_marginal_ring(B, i, m) for i in range(k)]
            e_max = max(e_max, *(trnorm(rho[i] - V @ sig[i] @ V.conj().T) for i in range(k)))
            e_mix = max(e_mix, trnorm(sum(rho) / k - V @ sigma @ V.conj().T))
        if e_max > 1e-12:  # por debajo manda el redondeo
            ratios.append(e_max / lam ** (m - 1))
        print(f"   {m:4d} {n:4d}   {e_max:.3e}        {e_max / lam ** (m - 1):10.4f}          {e_mix:.3e}")
    check(max(ratios) < 2 * min(ratios),
          f"eps_n / lambda^(m-1) en [{min(ratios):.2f}, {max(ratios):.2f}] (eps_n > 1e-12), uniforme en Theta")
    return lam


if __name__ == "__main__":
    run("espín 1/2, k = 2", np.array([0.5, -0.5]),
        [np.array([-0.5, 0.5]), np.array([-1.0, 0.0, 1.0])], n_brute=(4, 6, 8, 10),
        ms=list(range(3, 31, 3)))
    run("espín 3/2, k = 2", np.array([1.5, 0.5, -0.5, -1.5]),
        [np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5])], n_brute=(4, 6),
        ms=list(range(3, 31, 3)))
    sys.exit(0 if OK else 1)

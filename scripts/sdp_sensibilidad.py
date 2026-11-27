"""p(n, Theta) por SDP para los datos de estado fundamental del modelo S+S- (celda_k_notas.md, 9.16).

p(n, Theta) = max p tal que existe un estado del anillo de n sitios con marginal de arista
I/d^2 + p (1 (x) V) tau (1 (x) V)^dag en todas las aristas, V = e^{-i Theta Q/n}, tau = sigma - I/d^2.
Datos sigma: marginal de dos sitios del fundamental de H(0) en el anillo de NREF sitios (3 | NREF), que es
invariante por U(1) y por traslación. Basta un estado invariante por la U(1) global (los datos lo son): omega se
parte en bloques de carga total.
Uso: python scripts/sdp_sensibilidad.py n eps [eps ...]   (NREF, NTH, SOLVER en el entorno)
"""
import os
import sys

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from celda_k_flujo import build, sector

NREF = int(os.environ.get("NREF", 12))
NTH = int(os.environ.get("NTH", 4))
SOLVER = os.environ.get("SOLVER", "CLARABEL")
Qloc = np.diag([-1.0, 0.0, 1.0])


def datos(eps, n=NREF):
    """sigma = marginal (s_0, s_1) del fundamental de H(0) en el anillo de n sitios."""
    N, diag, r, c = build(n, 0)
    A = sp.coo_matrix((2 * eps * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    H = A + A.T + sp.diags(diag)
    w, v = eigsh(H, k=1, which="SA", tol=1e-13)
    codes, _ = sector(n, 0)
    psi = np.zeros(3**n)
    psi[codes] = v[:, 0]
    X = psi.reshape(3 ** (n - 2), 3, 3)  # [resto, s1, s0]
    rho = np.einsum("rba,rdc->abcd", X, X.conj())  # rho[(s0,s1),(s0',s1')]
    return rho.reshape(9, 9)


def montar(n):
    """Bloques (carga Q, momento k) de un estado invariante por U(1) y por traslación.
    Para cada bloque: B (base de momento, filas) y las matrices C_m = B O_m B^dag de los elementos
    O_m = |a b><a' b'| (en la arista (0, 1), tensor identidad), con tr(omega O_m) = marginal[(a',b'),(a,b)]."""
    N = 3**n
    codes = np.arange(N)
    dig = np.array([(codes // 3**i) % 3 for i in range(n)])
    carga = dig.sum(0) - n
    pw = 3 ** np.arange(n)

    def T(x):  # traslación s_i -> s_{i-1}
        d = (x // pw) % 3
        return int(np.dot(np.roll(d, 1), pw))

    pares = [(a, b, a2, b2) for a in range(3) for b in range(3) for a2 in range(3) for b2 in range(3)
             if a + b == a2 + b2]
    # O_m como matrices dispersas N x N
    Os = []
    for (a, b, a2, b2) in pares:
        xs = np.nonzero((dig[0] == a2) & (dig[1] == b2))[0]  # |a b><a2 b2| actúa sobre x con (a2,b2)
        ys = xs + (a - a2) * pw[0] + (b - b2) * pw[1]
        Os.append(sp.csr_matrix((np.ones(xs.size), (ys, xs)), shape=(N, N)))
    vistos = set()
    bloques = []
    for Q in np.unique(carga):
        reps = []
        for x in np.nonzero(carga == Q)[0]:
            if x in vistos:
                continue
            orb = [int(x)]
            y = T(int(x))
            while y != orb[0]:
                orb.append(y)
                y = T(y)
            vistos.update(orb)
            reps.append(orb)
        for kk in range(n):
            k = 2 * np.pi * kk / n
            filas = []
            for orb in reps:
                L = len(orb)
                if (kk * L) % n != 0:  # el momento debe ser compatible con el periodo de la órbita
                    continue
                v = np.zeros(N, dtype=complex)
                for m in range(n):
                    v[orb[m % L]] += np.exp(-1j * k * m)
                v /= np.linalg.norm(v)
                filas.append(v)
            if not filas:
                continue
            B = sp.csr_matrix(np.array(filas))
            Cs = [(B @ O @ B.conj().T).toarray() for O in Os]
            bloques.append((int(Q), kk, B.shape[0], Cs))
    return bloques, pares


def _independientes(bloques, pares):
    """Partes (Re, Im) de los elementos de la marginal que son linealmente independientes como funciones de W.
    Con W invariante por traslación hay relaciones lineales (p. ej. tr_1 rho = tr_2 rho) que dejan el sistema
    de igualdades con rango deficiente, y el solver interior falla."""
    filas = []
    for m in range(len(pares)):
        for parte in (0, 1):
            fila = []
            for (_, _, d, Cs) in bloques:
                C = Cs[m].T  # tr(W C_m) = sum_ij W_ij C_m[j, i]
                # W = X + iY, X simétrica, Y antisimétrica: coeficientes de X e Y
                cx, cy = (C.real, -C.imag) if parte == 0 else (C.imag, C.real)
                fila.append(np.concatenate([cx.ravel(), cy.ravel()]))
            filas.append(np.concatenate(fila))
    G = np.array(filas)
    from scipy.linalg import qr
    _, R, piv = qr(G.T, pivoting=True, mode="economic")
    rango = int(np.sum(np.abs(np.diag(R)) > 1e-9 * abs(R[0, 0])))
    return sorted((int(k) // 2, int(k) % 2) for k in piv[:rango])


def p_sdp(n, sigma, theta, est):
    bloques, pares = est[0], est[1]
    if len(est) == 2:
        est.append(_independientes(bloques, pares))
    sel = est[2]
    V = np.diag(np.exp(-1j * theta * np.diag(Qloc) / n))
    U = np.kron(np.eye(3), V)  # 1 (x) V sobre (s0, s1): índice s0*3 + s1
    tau = sigma - np.eye(9) / 9
    tauj = U @ tau @ U.conj().T
    p = cp.Variable()
    W = [cp.Variable((d, d), hermitian=True) for (_, _, d, _) in bloques]
    cons = [w >> 0 for w in W]
    for m, parte in sel:
        a, b, a2, b2 = pares[m]
        r_, c_ = a2 * 3 + b2, a * 3 + b  # tr(omega |ab><a2b2|) = rho[(a2,b2),(a,b)]
        val = sum(cp.trace(w @ blk[3][m]) for w, blk in zip(W, bloques))
        obj = (1 / 9 if r_ == c_ else 0.0) + p * tauj[r_, c_]
        cons.append(cp.real(val) == cp.real(obj) if parte == 0 else cp.imag(val) == cp.imag(obj))
    prob = cp.Problem(cp.Maximize(p), cons)
    opts = {}
    if SOLVER == "CLARABEL" and os.environ.get("TOL"):
        t = float(os.environ["TOL"])
        opts = dict(tol_gap_abs=t, tol_gap_rel=t, tol_feas=t, tol_ktratio=t, max_iter=2000)
    prob.solve(solver=SOLVER, **opts)
    return p.value, prob.status


if __name__ == "__main__":
    n = int(sys.argv[1])
    est = list(montar(n))
    for eps in [float(x) for x in sys.argv[2:]]:
        sigma = datos(eps)
        ths = np.linspace(0, np.pi, NTH)
        ps = []
        for th in ths:
            pv, st = p_sdp(n, sigma, th, est)
            ps.append(pv)
            print(f"  n={n} eps={eps} Theta={th:.4f}: p = {pv:.10f} ({st})", flush=True)
        ps = np.array(ps)
        print(f"n={n} eps={eps}: rango de p en Theta = {ps.max() - ps.min():.3e}", flush=True)

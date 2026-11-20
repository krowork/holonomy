"""Qubits con celda de 3 sitios y núcleo: el ejemplo mínimo del teorema con celda.

Con qubits, un MPS U(1) de celda doble cuya sigma tenga núcleo es el Néel clásico
(D = 1) y nada más. El ejemplo cuántico mínimo tiene k = 3 y llenado f = 1/3: cada
celda tiene exactamente un espín arriba, en superposición sobre sus tres sitios, y
nunca hay dos arriba contiguos (en la frontera de celda está prohibido el salto
«arriba en el sitio 3, arriba en el sitio 1 de la siguiente»).

Enlaces (D = 2 en los tres):
  enlace 1 (antes del sitio 1): {anterior abajo, anterior arriba}
  enlace 2 (entre 1 y 2):       {ya colocado (en 1), sin colocar}
  enlace 3 (entre 2 y 3):       {ya colocado, sin colocar}
Cargas con Q' = Q - mu, mu = -1/6 (magnetización por sitio): q' = (2/3, -1/3).

Comprobaciones: las de scripts/mps_celda.py (simetría, inyectividad, fuerza bruta,
eps_n = O(lambda^(m-1)) uniforme en Theta), el peso nulo en |up up>, y p(n, Theta)
exacto por SDP (invariante por traslación y U(1): sdp_p_ti; sdp_p es la
versión completa, que coincide hasta n = 8) para n = 4..9 y Theta = 0, pi/2, pi,
con la cota del teorema
p >= 1 - 6 d^2 n eps_n en n = 3m.
"""
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mps_celda as mc  # noqa: E402

UP, DN = 0, 1
Q_PRIME = np.array([2 / 3, -1 / 3])          # q - mu, mu = -1/6
CHARGES = [np.array([0.0, 0.0]),             # enlace 1: {ant. abajo, ant. arriba}
           np.array([-2 / 3, 1 / 3]),        # enlace 2: {colocado, sin colocar}
           np.array([-1 / 3, 2 / 3])]        # enlace 3: {colocado, sin colocar}


def tensors(alpha=1.0, beta=1.0, gamma=1.0):
    A1 = np.zeros((2, 2, 2), dtype=complex)
    A1[UP, 0, 0] = alpha       # arriba en el sitio 1 (solo si el anterior está abajo)
    A1[DN, 0, 1] = 1
    A1[DN, 1, 1] = 1
    A2 = np.zeros((2, 2, 2), dtype=complex)
    A2[DN, 0, 0] = 1           # ya colocado: abajo
    A2[UP, 1, 0] = beta        # arriba en el sitio 2
    A2[DN, 1, 1] = 1
    A3 = np.zeros((2, 2, 2), dtype=complex)
    A3[DN, 0, 0] = 1           # ya colocado: abajo, el siguiente ve «anterior abajo»
    A3[UP, 1, 1] = gamma       # arriba en el sitio 3, el siguiente ve «anterior arriba»
    return [A1, A2, A3]


def sdp_p(n, Theta, sigma):
    """max p con marginales de arista I/4 + p (1⊗V)(sigma - I/4)(1⊗V)^†.

    Los datos son invariantes por la U(1) de Z, así que basta un estado
    diagonal por bloques de magnetización (promediando sobre el grupo)."""
    import cvxpy as cp
    q = np.array([0.5, -0.5])
    V = np.kron(np.eye(2), np.diag(np.exp(-1j * Theta / n * q)))
    tau = V @ (sigma - np.eye(4) / 4) @ V.conj().T
    p = cp.Variable()
    blocks, idx = {}, {}
    for M in range(n + 1):
        states = [s for s in range(2 ** n) if bin(s).count("1") == M]
        idx[M] = {s: i for i, s in enumerate(states)}
        blocks[M] = cp.Variable((len(states), len(states)), hermitian=True)
    cons = [blocks[M] >> 0 for M in blocks]
    cons.append(sum(cp.real(cp.trace(blocks[M])) for M in blocks) == 1)
    bit = lambda s, j: (s >> (n - 1 - j)) & 1  # sitio 0 = bit más significativo; 1 = abajo
    # pares (ab, cd) con la misma carga en la arista
    pairs = [(x, y) for x in range(4) for y in range(4)
             if bin(x).count("1") == bin(y).count("1")]
    for j in range(n):
        jn = (j + 1) % n
        for x, y in pairs:
            expr = 0
            for M in blocks:
                rows, cols = [], []
                for s, i in idx[M].items():
                    if (bit(s, j) << 1 | bit(s, jn)) != x:
                        continue
                    t = s
                    for site, b in ((j, (y >> 1) & 1), (jn, y & 1)):
                        mask = 1 << (n - 1 - site)
                        t = (t | mask) if b else (t & ~mask)
                    if t in idx[M]:
                        rows.append(i)
                        cols.append(idx[M][t])
                if rows:
                    S = np.zeros((len(idx[M]), len(idx[M])))
                    S[rows, cols] = 1
                    expr = expr + cp.sum(cp.multiply(S, blocks[M]))
            cons += [cp.real(expr) == np.eye(4)[x, y] / 4 + p * tau[x, y].real,
                     cp.imag(expr) == p * tau[x, y].imag]
    prob = cp.Problem(cp.Maximize(p), cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
    return float(p.value), prob.status


def sdp_p_ti(n, Theta, sigma):
    """Como sdp_p, pero con un estado invariante por traslación y por la U(1).

    Los datos (gauge uniforme) son iguales en todas las aristas y T lleva la
    arista j a la j+1, así que promediar sobre traslaciones conserva la
    factibilidad. El estado es diagonal por bloques (M, momento) y basta
    imponer la arista (0, 1)."""
    import cvxpy as cp
    q = np.array([0.5, -0.5])
    V = np.kron(np.eye(2), np.diag(np.exp(-1j * Theta / n * q)))
    tau = V @ (sigma - np.eye(4) / 4) @ V.conj().T
    dim = 2 ** n
    shift = lambda s: ((s >> 1) | ((s & 1) << (n - 1)))  # sitio j -> j+1 (sitio 0 = bit alto)
    # operadores de arista |x><y| en los sitios 0, 1 (x, y en {0..3}; bit 1 = abajo)
    def edge_op(x, y):
        rows, cols = [], []
        for s in range(dim):
            if (s >> (n - 2)) == y:
                rows.append((s & ((1 << (n - 2)) - 1)) | (x << (n - 2)))
                cols.append(s)
        E = np.zeros((dim, dim))
        E[rows, cols] = 1
        return E
    pairs = [(x, y) for x in range(4) for y in range(4) if bin(x).count("1") == bin(y).count("1")]
    ops = {xy: edge_op(*xy) for xy in pairs}
    p = cp.Variable()
    tr, lin, cons = 0, {xy: 0 for xy in pairs}, []
    # bases de momento por sector de magnetización
    for M in range(n + 1):
        reps, done = [], set()
        for s in range(dim):
            if bin(s).count("1") != M or s in done:
                continue
            orb, t = [], s
            while t not in orb:
                orb.append(t)
                t = shift(t)
            done.update(orb)
            reps.append(s)
        for kk in range(n):
            vecs = []
            for r in reps:
                v = np.zeros(dim, dtype=complex)
                t = r
                for j in range(n):
                    v[t] += np.exp(2j * np.pi * kk * j / n)
                    t = shift(t)
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    vecs.append(v / nv)
            if not vecs:
                continue
            B = np.array(vecs).T
            R = cp.Variable((B.shape[1], B.shape[1]), hermitian=True)
            cons.append(R >> 0)
            tr = tr + cp.real(cp.trace(R))
            for xy in pairs:
                # <x|rho_01|y> = tr(rho |y><x|)
                O = B.conj().T @ ops[(xy[1], xy[0])] @ B
                lin[xy] = lin[xy] + cp.trace(O @ R)
    cons.append(tr == 1)
    for (x, y) in pairs:
        cons += [cp.real(lin[(x, y)]) == np.eye(4)[x, y] / 4 + p * tau[x, y].real,
                 cp.imag(lin[(x, y)]) == p * tau[x, y].imag]
    prob = cp.Problem(cp.Maximize(p), cons)
    try:
        prob.solve(solver="CLARABEL")
        return float(p.value), prob.status
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-10, max_iters=500000)
        return float(p.value), "SCS " + prob.status


if __name__ == "__main__":
    A = tensors()
    k, d = 3, 2
    # base de espín: s = 0 arriba (q' = 2/3), s = 1 abajo (q' = -1/3)
    lam = mc.run("qubits, k = 3, f = 1/3", Q_PRIME, CHARGES, n_brute=(6, 9, 12),
                 ms=list(range(2, 31, 2)), A=A)
    sig = [mc.edge_marginal_thermo(A, i)[0] for i in range(k)]
    sigma = sum(sig) / k
    mc.check(abs(sigma[0, 0]) < 1e-14, f"peso nulo en |up up> (sigma[0,0] = {abs(sigma[0, 0]):.1e})")
    print("      espectro de sigma:", np.round(np.linalg.eigvalsh(sigma), 6))
    print("      <Z> por sitio:", np.round(np.real(np.trace(np.kron(np.diag([1, -1]), np.eye(2)) @ sigma)), 6))

    # Proposición celda3n: eps_n = (1 + sqrt13)/sqrt5 * x_m, x_m = lam^m/(1 + lam^m), sin
    # dependencia en Theta (m >= 2); la marginal media tiene la diagonal de los datos y su
    # coherencia es la de los datos por 1 + golden * x_m.
    golden = (1 + 5 ** 0.5) / 2
    lam3 = golden ** -4
    err_eps = err_diag = err_coh = 0.0
    for m in range(2, 16):
        x = lam3 ** m / (1 + lam3 ** m)
        for Theta in np.linspace(0, 2 * np.pi, 13):
            phi = Theta / (k * m)
            V = np.kron(np.eye(2), np.diag(np.exp(-1j * phi * Q_PRIME)))
            B = mc.twisted(A, CHARGES, phi)
            rho = [mc.edge_marginal_ring(B, i, m) for i in range(k)]
            eps = max(mc.trnorm(rho[i] - V @ sig[i] @ V.conj().T) for i in range(k))
            err_eps = max(err_eps, abs(eps - (1 + 13 ** 0.5) / 5 ** 0.5 * x))
            avg, tgt = sum(rho) / k, V @ sigma @ V.conj().T
            err_diag = max(err_diag, np.abs(np.diag(avg - tgt)).max())
            err_coh = max(err_coh, abs(avg[1, 2] - (1 + golden * x) * tgt[1, 2]))
    mc.check(max(err_eps, err_diag, err_coh) < 1e-14,
             f"eps_n exacto, m = 2..15, 13 Theta (err {err_eps:.1e}); media: diagonal exacta "
             f"(err {err_diag:.1e}) y coherencia (1 + phi x_m) por la de los datos (err {err_coh:.1e})")

    if "--sin-sdp" not in sys.argv:
        print("\n   n   Theta/pi   p(n, Theta) SDP   estado          cota del teorema")
        for n in range(4, 10):
            for Theta in (0.0, np.pi / 2, np.pi):
                t = time.time()
                pv, st = sdp_p_ti(n, Theta, sigma)
                bound = ""
                if n % k == 0:
                    phi = Theta / n
                    V = np.kron(np.eye(2), np.diag(np.exp(-1j * phi * Q_PRIME)))
                    B = mc.twisted(A, CHARGES, phi)
                    eps = max(mc.trnorm(mc.edge_marginal_ring(B, i, n // k) - V @ sig[i] @ V.conj().T)
                              for i in range(k))
                    bound = f"{1 - 6 * d * d * n * eps:+.3f}"
                print(f"  {n:2d}   {Theta / np.pi:5.2f}     {pv:.6f}        {st:15s} {bound}   ({time.time() - t:.0f} s)",
                      flush=True)
    sys.exit(0 if mc.OK else 1)

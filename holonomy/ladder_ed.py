"""Fundamental de la escalera XXZ periódica 2 x L con giro en las patas, por
bloques de magnetización de cada pata (Lanczos sin matriz, sector S^z = 0).

Artículo (~/fisica/holonomia.tex): sección «Rigidez del testigo» de las
escaleras, testigo fijo a L grande (guion scripts/escalera_testigo_fijo.py).

Hamiltoniano (espines 1/2, S = sigma/2):
    H = sum_patas [ (1/2)(e^{i chi_e} S^+_i S^-_j + h.c.) + Delta S^z_i S^z_j ]
      + x sum_peldaños [ (1/2)(S^+_i S^-_j + h.c.) + Delta_r S^z_i S^z_j ],
con el mismo giro Psi en las dos patas (sum de chi_e por pata = Psi). Un gauge
que rota el sitio i de las dos patas el mismo ángulo no toca los peldaños y
concentra el giro en las aristas de cierre L-1 -> 0 y 2L-1 -> L; con Psi = 0 o
pi el hamiltoniano es real (signo -1 en el término XY de esas dos aristas).

Relación con el testigo (sección Marco): una arista con (alpha, beta) =
r (cos chi, sin chi) y c aporta -4 r [(1/2)(e^{i chi} S^+S^- + h.c.) + (c/r)
S^zS^z] al testigo, así que el testigo de la familia es W = -4 H (patas con
r = 1, peldaños con alpha_r = x) y, con los datos de f = 0 y holonomía psi = Psi
en las patas (fase psi/L por arista, igual a la del testigo),
    N(W) = L [ 2 (2 + Delta) + 2 x + c_r ],   c_r = x Delta_r,
    v(L, 0, psi) <= lambda_max(W)/N(W) = -4 E_0 / N(W)        (cota).
Con Delta = Delta_r = 1, N = 3 L (2 + x).

Base. Un estado de S^z = 0 es (b, t), con b los L bits de la pata inferior
(bit i = 1: espín arriba en el sitio i) y t los de la superior; si b tiene p
unos, t tiene L - p. El vector es la lista de matrices X_p de forma
(C(L, p), C(L, L - p)), y H actúa como
    X_p -> H_b(p) X_p + X_p H_t(L-p)^T + (rung ZZ) * X_p
           + (x/2) sum_i [ R_i(p-1) X_{p-1} R_i(L-p)^T + R_i(p)^T X_{p+1} R_i(L-p-1) ],
con H_b, H_t las patas (anillos XXZ de L sitios con el giro), R_i(q) la
subida de S^+_i de q a q+1 unos en una pata, y el ZZ de los peldaños una
matriz densa precalculada. Memoria: un vector de C(2L, L) reales (L = 14:
40 M, 0,32 GB).
"""
from math import comb

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh


def _states(L, p):
    """Enteros de L bits con p unos, en orden creciente."""
    if p == 0:
        return np.zeros(1, dtype=np.int64)
    if p == L:
        return np.array([(1 << L) - 1], dtype=np.int64)
    a = _states(L - 1, p)
    b = _states(L - 1, p - 1) | (1 << (L - 1))
    return np.concatenate([a, b])


class TwistedLadder:
    """Escalera XXZ 2 x L periódica con giro Psi en {0, pi} en las dos patas."""

    def __init__(self, L, x=1.0, Delta=1.0, Delta_r=1.0, Psi=0.0):
        if not (np.isclose(Psi, 0.0) or np.isclose(Psi, np.pi)):
            raise ValueError("solo Psi = 0 o pi (hamiltoniano real)")
        self.L, self.Psi = L, Psi
        self.set_params(x, Delta, Delta_r)
        sgn = -1.0 if np.isclose(Psi, np.pi) else 1.0
        self.st = [_states(L, p) for p in range(L + 1)]
        idx = np.empty(1 << L, dtype=np.int64)
        for s in self.st:
            idx[s] = np.arange(s.size)
        bits = [np.array([(s >> i) & 1 for i in range(L)]) for s in self.st]  # (L, C)
        z = [2.0 * b - 1.0 for b in bits]

        # Patas: anillo de L sitios con el giro en la arista L-1 -> 0; la parte XY
        # (dispersa) y la ZZ (diagonal) por separado, con acoplamiento 1.
        self.leg_xy, self.leg_zz = [], []
        for p, s in enumerate(self.st):
            n = s.size
            diag = np.zeros(n)
            rows, cols, vals = [], [], []
            for i in range(L):
                j = (i + 1) % L
                diag += 0.25 * z[p][i] * z[p][j]
                sel = np.nonzero(bits[p][i] != bits[p][j])[0]
                rows.append(idx[s[sel] ^ ((1 << i) | (1 << j))])
                cols.append(sel)
                vals.append(np.full(sel.size, 0.5 * (sgn if j == 0 else 1.0)))
            self.leg_xy.append(sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                             shape=(n, n)))
            self.leg_zz.append(diag)

        # Subidas S^+_i: R[i][q] de q a q+1 unos, forma (C(L,q+1), C(L,q)).
        self.R = [[None] * L for _ in range(L)]
        for i in range(L):
            for q in range(L):
                s = self.st[q]
                sel = np.nonzero(bits[q][i] == 0)[0]
                tgt = idx[s[sel] | (1 << i)]
                self.R[i][q] = sp.csr_matrix((np.ones(sel.size), (tgt, sel)),
                                             shape=(self.st[q + 1].size, s.size))
        # ZZ de los peldaños: sum_i z_i^b z_i^t / 4, denso por bloque.
        self.rung_zz = [0.25 * (z[p].T @ z[L - p]) for p in range(L + 1)]
        self.shapes = [(self.st[p].size, self.st[L - p].size) for p in range(L + 1)]
        self.offsets = np.cumsum([0] + [a * b for a, b in self.shapes])
        self.dim = int(self.offsets[-1])
        assert self.dim == comb(2 * L, L)

    def set_params(self, x, Delta, Delta_r):
        self.set_coefs(x, Delta, x * Delta_r)

    def set_coefs(self, x, Delta, c_r):
        """Peldaños con XY x y ZZ c_r (c_r = x Delta_r): la familia es lineal en (Delta, x, c_r)."""
        self.x, self.Delta, self.c_r = float(x), float(Delta), float(c_r)

    def blocks(self, v):
        return [v[self.offsets[p]:self.offsets[p + 1]].reshape(self.shapes[p]) for p in range(self.L + 1)]

    def apply(self, v, c_lxy, c_lzz, c_rxy, c_rzz):
        """(c_lxy H_patas^XY + c_lzz H_patas^ZZ + c_rxy H_peldaños^XY + c_rzz H_peldaños^ZZ) v."""
        L = self.L
        X = self.blocks(np.asarray(v, dtype=float).ravel())
        out = np.empty(self.dim)
        Y = self.blocks(out)
        for p in range(L + 1):
            q = L - p
            Y[p][...] = (c_lxy * (self.leg_xy[p] @ X[p] + (self.leg_xy[q] @ X[p].T).T)
                         + (c_lzz * (self.leg_zz[p][:, None] + self.leg_zz[q][None, :])
                            + c_rzz * self.rung_zz[p]) * X[p])
        if c_rxy:
            for p in range(L):
                # S^+_b S^-_t: de (p, L-p) a (p+1, L-p-1); y su conjugado.
                acc_up = np.zeros(self.shapes[p + 1])
                acc_dn = np.zeros(self.shapes[p])
                for i in range(L):
                    Rb, Rt = self.R[i][p], self.R[i][L - p - 1]  # Rt: de L-p-1 a L-p unos
                    acc_up += (Rt.T @ (Rb @ X[p]).T).T
                    acc_dn += (Rt @ (Rb.T @ X[p + 1]).T).T
                Y[p + 1] += 0.5 * c_rxy * acc_up
                Y[p] += 0.5 * c_rxy * acc_dn
        return out

    def matvec(self, v):
        return self.apply(v, 1.0, self.Delta, self.x, self.c_r)

    def parts(self, u):
        """<u|.|u> de las cuatro partes (XY y ZZ de patas y de peldaños), con u normalizado:
        E = parts . (1, Delta, x, c_r) (Hellmann-Feynman)."""
        return np.array([u @ self.apply(u, *e) for e in np.eye(4)])

    def ground(self, seed=0, tol=1e-12, ncv=None, v0=None):
        """(E_0, vector, norma del residuo) con Lanczos; arranque aleatorio por defecto.
        ncv = 20, salvo con más de 10^7 estados (L = 14), donde 8: ARPACK guarda dos matrices
        de dim x ncv y con 20 no caben (12 GB); en L = 12, ncv = 8 tarda un 15 % más."""
        if ncv is None:
            ncv = 20 if self.dim < 10**7 else 8
        op = LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)
        if v0 is None:
            v0 = np.random.default_rng(seed).standard_normal(self.dim)
        w, V = eigsh(op, k=1, which="SA", tol=tol, ncv=min(ncv, self.dim - 1), v0=v0)
        u = V[:, 0] / np.linalg.norm(V[:, 0])
        res = float(np.linalg.norm(self.matvec(u) - w[0] * u))
        return float(w[0]), u, res

    def norm(self):
        """N(W) del testigo W = -4 H con los datos de f = 0 y holonomía Psi."""
        return self.L * (2 * (2 + self.Delta) + 2 * self.x + self.c_r)


def dense_hamiltonian(L, x=1.0, Delta=1.0, Delta_r=1.0, Psi=0.0):
    """El mismo H en la base completa de 2^(2L) estados (sitios 0..L-1 la pata
    inferior, L..2L-1 la superior; giro uniforme Psi/L en cada arista de pata,
    complejo): contraste de TwistedLadder para L pequeño."""
    n = 2 * L
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.diag([0.5, -0.5])

    def op(o, i):
        return np.kron(np.kron(np.eye(2**i), o), np.eye(2**(n - i - 1)))
    S = [[op(o, i) for o in (sx, sy, sz)] for i in range(n)]
    H = np.zeros((2**n, 2**n), dtype=complex)
    chi = Psi / L
    for leg in (0, L):
        for i in range(L):
            a, b = leg + i, leg + (i + 1) % L
            sp_a = S[a][0] + 1j * S[a][1]
            sm_b = S[b][0] - 1j * S[b][1]
            t = np.exp(1j * chi) * sp_a @ sm_b
            H += 0.5 * (t + t.conj().T) + Delta * S[a][2] @ S[b][2]
    for i in range(L):
        a, b = i, L + i
        H += x * (S[a][0] @ S[b][0] + S[a][1] @ S[b][1] + Delta_r * S[a][2] @ S[b][2])
    return H

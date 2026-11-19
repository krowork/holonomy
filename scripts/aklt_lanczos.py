"""Problema abierto 4 de robustez_mps.tex: p(n, Theta) de AKLT más allá de n = 8.

Mismo dual que scripts/aklt_sdp_ti.py, pero con Lanczos en lugar de diagonalizar
bloques densos. Con h hermítico de dos sitios, que conserva la carga, de traza
nula y tr(h tau_Theta) = 1,

    p(n, Theta) <= f(h) = lambda_max(H_h) / n,      H_h = sum_j h_{j, j+1},

con igualdad en el ínfimo. Si el autovector superior psi es único, f es
diferenciable y (Hellmann-Feynman) su gradiente en la dirección G es tr(rho_01 G),
con rho_01 la marginal de arista de psi (todas iguales: psi tiene momento definido).
El óptimo es el punto donde rho_01 = I/9 + p tau: psi es entonces un estado primal
y las dos cotas se tocan. Se minimiza con BFGS y se pule con Newton. En Theta = 0 se arranca del testigo de techo
h_c = I - (9/5) (1 ⊗ V) P_2 (1 ⊗ V)^†, para el que f = 1 y psi es el AKLT retorcido; después se avanza en Theta en 4
pasos, arrancando cada uno del óptimo anterior girado (Lema lem:giro).

Cotas. p_sup = f(h) en el h final (rigurosa salvo redondeo). p_inf: la marginal de
psi es I/9 + p' tau + Delta, con p' la proyección sobre tau; el Lema lem:dilucionti
(r' = 1/(19 d^2) = 1/171) da p_inf = p' / (1 + 171 ||Delta||_1).

Sectores. H_h conserva S^z total. Se trabaja en M = 0 y se comprueba que los
sectores M = 1 y 2 quedan por debajo (por la inversión de espín, M y -M tienen el
mismo espectro). En M = 0 se piden dos autovalores: el gap mide la unicidad.

Controles: n = 6, 7, 8 frente a la tabla de docs/robustez_mps.tex (SDP exacto).
Uso: python scripts/aklt_lanczos.py [n ...]. Salida: data/aklt_lanczos.csv.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aklt_sdp_ti import TAU, M_OF, charge_pairs, herm_basis  # noqa: E402

D = 3
R_UNIF = 1 / (19 * D * D)
PAIRS = charge_pairs()                     # (x, y) con la misma carga de dos sitios
REF = {(6, 0): 0.9997978, (6, 4): 0.9996082, (6, 8): 0.9993839,
       (7, 0): 0.9999421, (7, 4): 0.9999605, (7, 8): 0.9999770,
       (8, 0): 0.99999658, (8, 4): 0.99999657, (8, 8): 0.99999624}
OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def sector_ops(n, M):
    """S_a = sum_j |..y_a..><..x_a..| en los sitios (j, j+1), en el sector M.

    <psi|S_a|psi>/n = rho_01[x_a, y_a] si psi es invariante por traslación, y
    H_h = sum_a h[y_a, x_a] S_a."""
    digits = np.array(np.unravel_index(np.arange(D ** n), [D] * n)).T.astype(np.int8)
    m = M_OF[digits].sum(axis=1)
    sel = np.where(m == M)[0]
    dg = digits[sel]
    dim = len(sel)
    pw = D ** np.arange(n - 1, -1, -1)
    ops = []
    for x, y in PAIRS:
        x0, x1 = divmod(x, D)
        y0, y1 = divmod(y, D)
        rows, cols = [], []
        for j in range(n):
            k = (j + 1) % n
            hit = np.where((dg[:, j] == x0) & (dg[:, k] == x1))[0]
            tgt = sel[hit] + (y0 - x0) * pw[j] + (y1 - x1) * pw[k]
            rows.append(np.searchsorted(sel, tgt))
            cols.append(hit)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        ops.append(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim)))
    return ops


class AKLTDual:
    def __init__(self, n, Theta, ops):
        self.n, self.ops = n, ops
        V = np.kron(np.eye(D), np.diag(np.exp(-1j * Theta / n * M_OF)))
        self.V = V
        self.tau = V @ TAU @ V.conj().T
        self.G = herm_basis(self.tau)
        self.h0 = self.tau / np.trace(self.tau @ self.tau).real
        self.v0 = None
        self.cache = {}

    def h(self, c):
        return self.h0 + sum(ci * g for ci, g in zip(c, self.G))

    def H(self, c, ops=None):
        h = self.h(c)
        ops = self.ops[0] if ops is None else ops
        return sum(h[y, x] * S for (x, y), S in zip(PAIRS, ops))

    def top(self, c, k=1, ops=None):
        H = self.H(c, ops)
        v0 = self.v0 if (ops is None and self.v0 is not None) else None
        w, U = spla.eigsh(H, k=max(k, 2), which="LA", tol=0, v0=v0, ncv=40)
        o = np.argsort(w)[::-1]
        return w[o], U[:, o]

    def marginal(self, psi):
        R = np.zeros((9, 9), complex)
        for (x, y), S in zip(PAIRS, self.ops[0]):
            R[x, y] = np.vdot(psi, S @ psi) / self.n
        return R

    def fg(self, c):
        key = tuple(np.round(c, 15))
        if key in self.cache:
            return self.cache[key]
        w, U = self.top(c)
        psi = U[:, 0]
        self.v0 = psi
        R = self.marginal(psi)
        g = np.array([np.trace(R @ G).real for G in self.G])
        out = (w[0] / self.n, g)
        self.cache = {key: out}
        return out

    def ceiling(self):
        """Coeficientes del testigo de techo h_c = I - (9/5) (1⊗V) P2 (1⊗V)^†."""
        Sz = np.diag([1.0, 0.0, -1.0])
        Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        Sx, Sy = (Sp + Sp.T) / 2, (Sp - Sp.T) / (2j)
        SS = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
        P2 = np.eye(9) / 3 + SS / 2 + SS @ SS / 6
        hc = np.eye(9) - 9 / 5 * self.V @ P2 @ self.V.conj().T
        return np.array([np.trace(hc @ G).real for G in self.G])

    def A(self, G):
        """Operador sum_j G_{j, j+1} en el sector M = 0."""
        return sum(G[y, x] * S for (x, y), S in zip(PAIRS, self.ops[0]))

    def hessian(self, c):
        """Hessiano exacto de f = lambda_max / n por respuesta lineal:
        d2f/dc_k dc_l = (2/n) Re <psi| A_k (E - H)^{-1}_perp A_l |psi>,
        con (E - H + |psi><psi|) x = P_perp A_l psi resuelto por gradiente conjugado."""
        H = self.H(c)
        w, U = self.top(c)
        E, psi = w[0], U[:, 0]
        self.v0 = psi
        B = spla.LinearOperator(H.shape, dtype=complex,
                                matvec=lambda x: E * x - H @ x + psi * np.vdot(psi, x))
        Apsi = []
        for G in self.G:
            y = self.A(G) @ psi
            Apsi.append(y - psi * np.vdot(psi, y))
        X = []
        for y in Apsi:
            x, info = spla.cg(B, y, rtol=1e-15, atol=0, maxiter=20000)
            X.append(x)
        K = len(self.G)
        Hs = np.empty((K, K))
        for k in range(K):
            for l in range(k, K):
                Hs[k, l] = Hs[l, k] = 2 * np.vdot(Apsi[k], X[l]).real / self.n
        R = self.marginal(psi)
        g = np.array([np.trace(R @ G).real for G in self.G])
        return E / self.n, g, Hs

    def newton(self, c, iters=20):
        """Newton con hessiano exacto, en el subespacio de curvatura no nula (las
        direcciones a⊗1 - 1⊗a no cambian H). Paso amortiguado sobre f (convexa)."""
        for _ in range(iters):
            f, g, Hs = self.hessian(c)
            if np.abs(g).max() < 1e-15:
                break
            lam, Q = np.linalg.eigh(Hs)
            keep = lam > 1e-13 * lam.max()
            dc = -Q[:, keep] @ ((Q[:, keep].T @ g) / lam[keep])
            t = 1.0
            while t > 1e-6:
                f2 = self.fg(c + t * dc)[0]
                if f2 <= f + 1e-16:
                    break
                t /= 2
            c = c + t * dc
            if t < 1e-6:
                break
        self.last_hess = lam
        return c

    def rotated(self, h_prev, Theta_prev, Theta):
        """Coeficientes del testigo anterior girado, (1⊗W) h (1⊗W)^†, W = e^{-i (Theta - Theta_prev) S^z / n}:
        conserva tr h = 0 y tr(h tau) = 1 (Lema lem:giro), y es un buen arranque."""
        W = np.kron(np.eye(D), np.diag(np.exp(-1j * (Theta - Theta_prev) / self.n * M_OF)))
        hr = W @ h_prev @ W.conj().T
        return np.array([np.trace(hr @ G).real for G in self.G])

    def solve(self, c0):
        res = minimize(lambda c: self.fg(c), c0, jac=True, method="BFGS",
                       options=dict(gtol=1e-7, maxiter=200))
        c = self.newton(res.x)
        w, U = self.top(c, k=2)
        psi = U[:, 0]
        R = self.marginal(psi)
        I9 = np.eye(9)
        pp = np.trace((R - I9 / 9) @ self.tau).real / np.trace(self.tau @ self.tau).real
        resid = np.abs(np.linalg.eigvalsh(R - I9 / 9 - pp * self.tau)).sum()
        p_inf = pp / (1 + resid / R_UNIF)
        return dict(c=c, p_sup=w[0] / self.n, p_inf=p_inf, resid=resid,
                    gap=(w[0] - w[1]) / self.n, iters=res.nit)


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [6, 7, 8, 9, 10, 11, 12, 13]
    thetas = tuple(np.linspace(0, np.pi, 9))
    # Continuación en Theta: cada punto arranca del testigo óptimo anterior girado.
    path = thetas
    rows = []
    print("   n  Theta/pi   p_sup                 p_sup - p_inf   gap(M=0)   top M=1,2 - top M=0   iter   s")
    for n in ns:
        t0 = time.time()
        ops = [sector_ops(n, 0), sector_ops(n, 1), sector_ops(n, 2)]
        print(f"      n = {n}: dim(M=0) = {ops[0][0].shape[0]} ({time.time() - t0:.0f} s)", flush=True)
        h_prev, T_prev = None, None
        for T in path:
            t = time.time()
            d = AKLTDual(n, T, ops)
            c0 = d.ceiling() if h_prev is None else d.rotated(h_prev, T_prev, T)
            r = d.solve(c0)
            h_prev, T_prev = d.h(r["c"]), T
            i = int(np.argmin(np.abs(np.array(thetas) - T)))
            if abs(thetas[i] - T) > 1e-12:
                continue
            # sectores M = 1, 2 en el h óptimo
            wM = [d.top(r["c"], k=1, ops=ops[k])[0][0] / n for k in (1, 2)]
            sep = max(wM) - r["p_sup"]
            print(f"  {n:2d}   {T / np.pi:4.2f}   {r['p_sup']:.15f}   {r['p_sup'] - r['p_inf']:.1e}       "
                  f"{r['gap']:.3e}   {sep:+.3e}           {r['iters']:4d}  {time.time() - t:.0f}", flush=True)
            check(sep < 0, f"n={n}, Theta={T / np.pi:.2f} pi: el máximo está en M = 0")
            check(r["p_sup"] - r["p_inf"] < 1e-10, f"n={n}, Theta={T / np.pi:.2f} pi: cotas a 1e-10 "
                  f"({r['p_sup'] - r['p_inf']:.1e})")
            if (n, i) in REF:
                tol = 5e-8 if n < 8 else 5e-9
                check(abs(r["p_sup"] - REF[(n, i)]) < tol,
                      f"n={n}, Theta={T / np.pi:.2f} pi: coincide con la tabla ({REF[(n, i)]})")
            rows.append(dict(n=n, Theta=T, p_sup=r["p_sup"], p_inf=r["p_inf"], resid=r["resid"],
                             gap=r["gap"], sep_M=sep, iters=r["iters"]))
        p = [x["p_sup"] for x in rows if x["n"] == n]
        print(f"      n = {n}: max - min = {max(p) - min(p):.3e}, p(0) - p(pi) = {p[0] - p[-1]:+.3e}", flush=True)
    out = ROOT / "data" / "aklt_lanczos.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"escrito {out}")
    sys.exit(0 if OK else 1)

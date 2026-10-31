"""Escaleras de dos patas: geometría, holonomías abelianas y dual.

Sitios: pata inferior 0..L-1 y pata superior L..2L-1; el peldaño i une i con
L+i. Aristas orientadas: patas i -> i+1 (y L+i -> L+i+1), peldaños i -> L+i;
con `periodic`, también L-1 -> 0 y 2L-1 -> L.

Datos abelianos: T^e = -R_z(phi_e) en cada arista. La holonomía de la plaqueta
p (recorrida p -> p+1 -> L+p+1 -> L+p -> p) es R_z del flujo
f_p = phi(p->p+1) + phi(p+1->L+p+1) - phi(L+p->L+p+1) - phi(p->L+p).
Gauge simétrico (de Landau): peldaños con fase 0, pata inferior con f_p/2 y
pata superior con -f_p/2. En la escalera periódica hay un ciclo independiente
más, la pata: `psi` se suma (repartido, psi/L por arista) a las dos patas, de
modo que la holonomía de la pata inferior es sum f_p/2 + psi y la de la
superior -sum f_p/2 + psi.

Dual (escalera periódica). Se minimiza lambda_max(W)/N sobre testigos
W = sum_e (alpha_e A_e + beta_e B_e + c_e C_e), con N = sum_e (2 alpha_e cos phi_e
+ 2 beta_e sin phi_e + c_e) > 0. Promediar un testigo óptimo sobre las
simetrías de los datos da otro testigo óptimo: U(1) alrededor de z (solo
quedan A, B, C y Z), la traslación a lo largo de las patas (coeficientes
iguales en todas las aristas de una misma pata y en todos los peldaños, si los
flujos son uniformes) y la reflexión izquierda-derecha compuesta con R_x(pi),
que anula los términos Z y la componente B de los peldaños. Quedan ocho
coeficientes: (alpha, beta, c) en cada pata y (alpha, c) en los peldaños. El
cociente es cuasiconvexo (lambda_max convexa, N lineal). Tanto el dual como
el primal se calculan por bloques de número de giros k y momento q.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

from .sdp import u1_sector_operators


def ladder_edges(L, periodic=False):
    """Aristas en el orden: pata inferior, pata superior, peldaños."""
    m = L if periodic else L - 1
    bottom = [(i, (i + 1) % L) for i in range(m)]
    top = [(L + i, L + (i + 1) % L) for i in range(m)]
    rungs = [(i, L + i) for i in range(L)]
    return bottom + top + rungs


def edge_kinds(L, periodic=False):
    m = L if periodic else L - 1
    return ["b"] * m + ["t"] * m + ["r"] * L


def abelian_phases(L, fluxes, periodic=False, psi=0.0):
    """Fases phi_e en el gauge simétrico, con flujo fluxes[p] en la plaqueta p
    (L-1 plaquetas si es abierta, L si es periódica) y, si es periódica, psi."""
    m = L if periodic else L - 1
    fluxes = np.broadcast_to(np.asarray(fluxes, dtype=float), (m,))
    shift = psi / L if periodic else 0.0
    return ([f / 2 + shift for f in fluxes] + [-f / 2 + shift for f in fluxes]
            + [0.0] * L)


def plaquette_fluxes(L, phis, periodic=False):
    """Flujos de plaqueta y, si es periódica, holonomías de las dos patas."""
    m = L if periodic else L - 1
    b, t, r = phis[:m], phis[m:2 * m], phis[2 * m:]
    f = [b[p] + r[(p + 1) % L] - t[p] - r[p] for p in range(m)]
    if periodic:
        return f, (sum(b), sum(t))
    return f, None


def _translate(s, L):
    """Estado de la base trasladado un sitio a lo largo de las patas (qubit 0
    el más significativo, como en holonomy.sdp)."""
    n = 2 * L
    t = 0
    for i in range(n):
        if (s >> (n - 1 - i)) & 1:
            j = (i + 1) % L + (L if i >= L else 0)
            t |= 1 << (n - 1 - j)
    return t


def momentum_isometries(L, basis):
    """Isometrías dispersas V_q (q = 0..L-1) del sector `basis` sobre los
    estados de momento 2 pi q / L: |s, q> = sum_m e^{-2 pi i q m / L} T^m |s>
    normalizado, uno por órbita de traslaciones compatible con q."""
    idx = {s: m for m, s in enumerate(basis)}
    seen, orbits = set(), []
    for s in basis:
        if s in seen:
            continue
        orb = [s]
        t = _translate(s, L)
        while t != s:
            orb.append(t)
            t = _translate(t, L)
        seen.update(orb)
        orbits.append(orb)
    out = []
    for q in range(L):
        rows, cols, vals = [], [], []
        col = 0
        for orb in orbits:
            ell = len(orb)
            if (q * ell) % L:
                continue
            for m, t in enumerate(orb):
                rows.append(idx[t])
                cols.append(col)
                vals.append(np.exp(-2j * np.pi * q * m / L) / np.sqrt(ell))
            col += 1
        if col:
            out.append((q, sp.csr_matrix((vals, (rows, cols)), shape=(len(basis), col))))
    return out


def symmetric_blocks(L, dense_max=None):
    """Bloques (k giros, momento q) de la escalera periódica 2xL, con los
    operadores sumados por tipo de arista: ("A"|"B"|"C", "b"|"t"|"r") y
    ("Z", "b"|"t") (Z total de cada pata), como matrices densas o, en los
    bloques de dimensión mayor que dense_max, dispersas (CSR)."""
    n = 2 * L
    edges = ladder_edges(L, periodic=True)
    kinds = edge_kinds(L, periodic=True)
    out = []
    for k in range(n + 1):
        basis, ops = u1_sector_operators(n, edges, k)
        summed = {}
        for kind in "btr":
            es = [e for e, kk in enumerate(kinds) if kk == kind]
            for name in "ABC":
                summed[(name, kind)] = sum(ops[(name, e)] for e in es)
        summed[("Z", "b")] = sum(ops[("Z", i)] for i in range(L))
        summed[("Z", "t")] = sum(ops[("Z", i)] for i in range(L, n))
        for q, V in momentum_isometries(L, basis):
            Vh = V.conj().T
            dense = dense_max is None or V.shape[1] <= dense_max
            out.append((k, q, {key: (Vh @ (M @ V)).toarray() if dense else (Vh @ (M @ V)).tocsr()
                               for key, M in summed.items()}))
    return out


def primal_symmetric(L, phis, solver="CLARABEL", return_blocks=False, **kw):
    """Primal de la escalera periódica con flujo uniforme, reducido por U(1) y
    por traslación (rho = suma directa de bloques (k, q)). Las restricciones de
    las aristas de cada tipo se suman: son equivalentes porque los datos son
    invariantes bajo traslación y el estado se puede promediar."""
    import cvxpy as cp

    kinds = edge_kinds(L, periodic=True)
    phase = {kd: phis[kinds.index(kd)] for kd in "btr"}
    v = cp.Variable()
    cons, lin, tr, Xs = [], {}, 0, []
    for k, q, red in symmetric_blocks(L):
        d = red[("C", "b")].shape[0]
        X = cp.Variable((d, d), hermitian=True)
        Xs.append(((k, q), X))
        cons.append(X >> 0)
        tr = tr + cp.real(cp.trace(X))
        for key, M in red.items():
            lin[key] = lin.get(key, 0) + cp.real(cp.sum(cp.multiply(M.T, X)))
    cons.append(tr == 1)
    for kd in "btr":
        cons += [lin[("A", kd)] == 2 * L * v * np.cos(phase[kd]),
                 lin[("B", kd)] == 2 * L * v * np.sin(phase[kd]),
                 lin[("C", kd)] == L * v]
    cons += [lin[("Z", "b")] == 0, lin[("Z", "t")] == 0]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_blocks:
        return float(v.value), prob.status, [(lab, np.asarray(X.value)) for lab, X in Xs]
    return float(v.value), prob.status


class LadderDual:
    """lambda_max del testigo de la escalera periódica, por bloques (k, q).

    Con dense_max, los bloques de dimensión mayor se guardan dispersos y sus
    autovalores superiores se calculan por Lanczos (eigsh): es lo que permite
    L >= 8 (los bloques densos no caben en memoria). Solo lam_max, top_blocks,
    block_gaps y v_dual_cg admiten bloques dispersos."""

    def __init__(self, L, dense_max=None):
        self.L = L
        self.n = 2 * L
        self.edges = ladder_edges(L, periodic=True)
        self.kinds = edge_kinds(L, periodic=True)
        sb = symmetric_blocks(L, dense_max)
        self.labels = [(k, q) for k, q, _ in sb]
        self.blocks = [red for _, _, red in sb]

    def witness_terms(self, y):
        """y = (alpha_b, beta_b, c_b, alpha_t, beta_t, c_t, alpha_r, c_r)."""
        ab, bb, cb, at, bt, ct, ar, cr = y
        return {("A", "b"): ab, ("B", "b"): bb, ("C", "b"): cb,
                ("A", "t"): at, ("B", "t"): bt, ("C", "t"): ct,
                ("A", "r"): ar, ("B", "r"): 0.0, ("C", "r"): cr}

    @staticmethod
    def top_eig(W, r):
        """Los r autovalores mayores de W (ascendentes) y sus autovectores:
        eigh si W es densa o pequeña, Lanczos si es dispersa."""
        d = W.shape[0]
        r = min(r, d)
        if not sp.issparse(W) or d <= max(4 * r, 64):
            w, V = np.linalg.eigh(W.toarray() if sp.issparse(W) else W)
            return w[-r:], V[:, -r:]
        w, V = eigsh(W, k=r, which="LA")
        o = np.argsort(w)
        return w[o], V[:, o]

    def lam_max(self, y):
        coef = self.witness_terms(y)
        best = -np.inf
        for red in self.blocks:
            W = sum(c * red[key] for key, c in coef.items())
            best = max(best, self.top_eig(W, 1)[0][-1])
        return float(best)

    KEYS = (("A", "b"), ("B", "b"), ("C", "b"), ("A", "t"), ("B", "t"), ("C", "t"),
            ("A", "r"), ("C", "r"))

    def lam_max_grad(self, y):
        """lambda_max y su gradiente en y (Hellmann-Feynman con el autovector del
        bloque que da el máximo; es un subgradiente si hay degeneración)."""
        best, vec, blk = -np.inf, None, None
        for red in self.blocks:
            W = sum(c * red[key] for key, c in zip(self.KEYS, y))
            d = W.shape[0]
            w, V = eigh(W, subset_by_index=[d - 1, d - 1])
            if w[0] > best:
                best, vec, blk = w[0], V[:, 0], red
        grad = np.array([np.real(vec.conj() @ blk[key] @ vec) for key in self.KEYS])
        return float(best), grad

    def norm_grad(self, phis):
        """N es lineal en y: devuelve el vector g con N(y) = g . y."""
        g = np.zeros(8)
        for e, kind in enumerate(self.kinds):
            ca, sb = 2 * np.cos(phis[e]), 2 * np.sin(phis[e])
            if kind == "b":
                g[0] += ca; g[1] += sb; g[2] += 1
            elif kind == "t":
                g[3] += ca; g[4] += sb; g[5] += 1
            else:
                g[6] += ca; g[7] += 1
        return g

    def v_dual_sdp(self, phis, solver="CLARABEL", **kw):
        """Dual reducido resuelto como SDP: min t con t I - W_blk(y) >= 0 en
        cada bloque (k, q) y N(y) = 1. Es exacto aunque lambda_max sea
        degenerado en el óptimo. Devuelve (v, y, estado)."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        y = cp.Variable(8)
        t = cp.Variable()
        cons = [g @ y == 1]
        for red in self.blocks:
            d = red[("C", "b")].shape[0]
            W = sum(y[m] * red[key] for m, key in enumerate(self.KEYS))
            cons.append(t * np.eye(d) - W >> 0)
        prob = cp.Problem(cp.Minimize(t), cons)
        prob.solve(solver=solver, **kw)
        return float(t.value), np.asarray(y.value), prob.status

    def v_dual_cg(self, phis, y0=None, tol=1e-8, r0=6, add=4, box=50.0, maxiter=200,
                  rmax=40, solver="CLARABEL", **kw):
        """El SDP de v_dual_sdp por generación de columnas, para bloques en los
        que el SDP completo no cabe en memoria (L >= 6: Clarabel forma bloques
        densos de tamaño (n(n+1)/2)^2 por cono).

        Cada bloque se comprime a un subespacio U: min t con
        t I - U^H W_blk(y) U >= 0, N(y) = 1 y |y_m| <= box. Comprimir no sube
        lambda_max, así que t es una cota inferior de v_max (con la caja, si
        el testigo óptimo está dentro de ella); lambda_max(W(y)) en el y
        obtenido es una cota superior. Se añaden a cada U los `add`
        autovectores de mayor autovalor por encima de t y se repite hasta que
        las cotas difieran menos de tol. Arranca en y0 (o en Heisenberg con
        la fase de los datos) con los r0 autovectores superiores de cada
        bloque. Si un subespacio supera rmax columnas, se poda a los rmax/2
        vectores de Ritz de mayor autovalor de U^H W(y) U (la memoria de
        Clarabel crece como la cuarta potencia del rango); la cota inferior
        sigue siendo válida con cualquier subespacio.

        Devuelve (v, y, estado, diferencia de cotas): v es la cota superior;
        estado 'optimal' si la diferencia es < tol y el testigo no toca la
        caja."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        if y0 is None:
            pb, pt = phis[0], phis[self.L]
            y0 = np.array([np.cos(pb), np.sin(pb), 1.0, np.cos(pt), np.sin(pt), 1.0, 1.0, 1.0])
        y0 = np.asarray(y0, float) / (g @ y0)

        def blocks_at(y):
            return [sum(c * red[key] for key, c in zip(self.KEYS, y)) for red in self.blocks]

        U = [self.top_eig(W, r0)[1] for W in blocks_at(y0)]
        best = (np.inf, y0)
        lower = -np.inf
        for _ in range(maxiter):
            y = cp.Variable(8)
            t = cp.Variable()
            cons = [g @ y == 1, cp.abs(y) <= box]
            for red, Ub in zip(self.blocks, U):
                Wc = sum(y[m] * (Ub.conj().T @ (red[key] @ Ub)) for m, key in enumerate(self.KEYS))
                cons.append(t * np.eye(Ub.shape[1]) - Wc >> 0)
            prob = cp.Problem(cp.Minimize(t), cons)
            prob.solve(solver=solver, **kw)
            if y.value is None:
                return best[0], best[1], prob.status, best[0] - lower
            lower, yv = float(t.value), np.asarray(y.value)
            Ws = blocks_at(yv)
            eig = [self.top_eig(W, add + 1) for W in Ws]
            upper = max(w[-1] for w, _ in eig)
            if upper < best[0]:
                best = (upper, yv)
            if best[0] - lower < tol:
                break
            for b, (w, V) in enumerate(eig):
                new = V[:, w > lower + tol / 2][:, -add:]
                if new.shape[1]:
                    if U[b].shape[1] + new.shape[1] > rmax:
                        _, Z = np.linalg.eigh(U[b].conj().T @ (Ws[b] @ U[b]))
                        U[b] = U[b] @ Z[:, -(rmax // 2):]
                    Q, s, _ = np.linalg.svd(np.hstack([U[b], new]), full_matrices=False)
                    U[b] = Q[:, s > 1e-10 * s[0]]
        self.last_ranks = [Ub.shape[1] for Ub in U]
        gap = best[0] - lower
        on_box = np.max(np.abs(best[1])) > 0.99 * box
        st = "optimal" if gap < tol and not on_box else ("en_caja" if on_box else "no_converge")
        return best[0], best[1], st, gap

    def witness_spread(self, phis, v_opt, eps, solver="CLARABEL", **kw):
        """Rango de cada coeficiente del testigo dentro del conjunto casi óptimo
        {y : N(y) = 1, lambda_max(W(y)) <= v_opt + eps}: mínimo y máximo de
        cada y_m (16 SDP). Si el testigo óptimo es único, el rango tiende a 0
        con eps; si hay una cara de óptimos, queda finito."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        y = cp.Variable(8)
        cons = [g @ y == 1]
        for red in self.blocks:
            d = red[("C", "b")].shape[0]
            W = sum(y[m] * red[key] for m, key in enumerate(self.KEYS))
            cons.append((v_opt + eps) * np.eye(d) - W >> 0)
        out = []
        for m in range(8):
            lo_hi = []
            for sense in (cp.Minimize, cp.Maximize):
                prob = cp.Problem(sense(y[m]), cons)
                try:
                    prob.solve(solver=solver, **kw)
                    lo_hi.append(float(y.value[m]) if y.value is not None else np.nan)
                except cp.error.SolverError:
                    lo_hi.append(np.nan)
            out.append(tuple(lo_hi))
        return out

    def v_dual_grad(self, phis, y0):
        """min lambda_max(W)/N con BFGS (gradiente exacto) desde y0 y pulido
        final con Nelder-Mead. Devuelve (v, y) con N(y) = 1."""
        g = self.norm_grad(phis)

        def fun(z):
            N = g @ z
            if N <= 1e-12:
                return 1e3, -g * 1e3
            lam, dl = self.lam_max_grad(z)
            return lam / N, (dl * N - lam * g) / N ** 2

        y0 = np.asarray(y0, float) / (g @ y0)
        res = minimize(fun, y0, jac=True, method="BFGS", options=dict(gtol=1e-11, maxiter=2000))
        x = res.x / (g @ res.x)
        res2 = minimize(lambda z: fun(z)[0], x, method="Nelder-Mead",
                        options=dict(xatol=1e-11, fatol=1e-14, maxiter=4000, adaptive=True))
        if res2.fun < res.fun:
            x, f = res2.x, res2.fun
        else:
            f = res.fun
        return float(f), x / (g @ x)

    def top_blocks(self, y, tol=1e-6):
        """Bloques (k, q) cuyo autovalor máximo está a menos de tol de
        lambda_max(W(y)), con sus valores: muestra la degeneración del óptimo."""
        tops = []
        for lab, red in zip(self.labels, self.blocks):
            W = sum(c * red[key] for key, c in zip(self.KEYS, y))
            tops.append((lab, self.top_eig(W, 1)[0][-1]))
        lam = max(w for _, w in tops)
        return [(lab, w) for lab, w in tops if w > lam - tol]

    def block_gaps(self, y, labels):
        """lambda_max(W(y)) menos el autovalor máximo de cada bloque de labels."""
        tops = {}
        for lab, red in zip(self.labels, self.blocks):
            W = sum(c * red[key] for key, c in zip(self.KEYS, y))
            tops[lab] = self.top_eig(W, 1)[0][-1]
        lam = max(tops.values())
        return [float(lam - tops[lab]) for lab in labels]

    def norm(self, y, phis):
        coef = self.witness_terms(y)
        N = 0.0
        for e, kind in enumerate(self.kinds):
            N += (2 * coef[("A", kind)] * np.cos(phis[e]) + 2 * coef[("B", kind)] * np.sin(phis[e])
                  + coef[("C", kind)])
        return N

    def ratio(self, y, phis):
        N = self.norm(y, phis)
        if N <= 1e-12:
            return np.inf
        return self.lam_max(y) / N

    def v_dual(self, phis, y0=None, restarts=3, seed=0):
        """min lambda_max(W)/N por Nelder-Mead con reinicios.

        Devuelve (v, y). Es una cota superior de v_max para cualquier y; la
        igualdad se contrasta con el primal (scripts/ladder.py).
        """
        rng = np.random.default_rng(seed)
        # Arranque: Heisenberg en todas las aristas, con la fase de los datos.
        if y0 is None:
            pb, pt = phis[0], phis[self.L]
            y0 = np.array([np.cos(pb), np.sin(pb), 1.0, np.cos(pt), np.sin(pt), 1.0, 1.0, 1.0])
        best = (self.ratio(y0, phis), np.asarray(y0, float))
        starts = [best[1]] + [best[1] + 0.3 * rng.standard_normal(8) for _ in range(restarts)]
        for x in starts:
            for _ in range(4):  # Nelder-Mead reiniciado desde su propio óptimo
                res = minimize(lambda z: self.ratio(z, phis), x, method="Nelder-Mead",
                               options=dict(xatol=1e-10, fatol=1e-13, maxiter=20000, maxfev=20000,
                                            adaptive=True))
                x = res.x / max(abs(self.norm(res.x, phis)), 1e-12)
                if res.fun < best[0] - 1e-14:
                    best = (res.fun, x)
                else:
                    break
        return best


def witness_fluxes(y, L):
    """Flujo de plaqueta del testigo, chi_b - chi_t (los peldaños, sin
    componente B, no aportan), y la holonomía de la pata inferior, L chi_b.
    La fase de una arista es la de alpha + i beta."""
    chi_b = np.arctan2(y[1], y[0])
    chi_t = np.arctan2(y[4], y[3])
    return float(np.angle(np.exp(1j * (chi_b - chi_t)))), float(np.angle(np.exp(1j * L * chi_b)))

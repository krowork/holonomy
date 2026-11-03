"""Dual por bloques de un grafo periódico con datos abelianos (U(1) y traslación).

Grafo de n sitios con aristas orientadas `edges`, cada una de un tipo
(`kinds`), y una traslación dada por la permutación de sitios `perm` (sitio
i -> perm[i]) de orden `n_cells` que lleva cada arista a otra del mismo tipo.
Si los datos son T^e = -R_z(phi_e) con la fase constante en cada tipo (salvo
gauge), promediar un testigo óptimo sobre U(1) alrededor de z y sobre la
traslación da otro testigo óptimo: W = sum_kind (alpha A + beta B + c C)
sumados sobre las aristas de cada tipo, más términos Z sumados sobre grupos de
sitios invariantes. Las claves que se conservan (KEYS) las fija cada
subclase: otras simetrías (reflexiones) pueden anular algunas.

A_e = X_i X_j + Y_i Y_j, B_e = X_i Y_j - Y_i X_j y C_e = Z_i Z_j, como en
holonomy.sdp.u1_sector_operators; la normalización es
N = sum_e (2 alpha_e cos phi_e + 2 beta_e sin phi_e + c_e). Se minimiza
lambda_max(W)/N (cuasiconvexo: lambda_max convexa, N lineal), por bloques de
número de giros k y momento q.

Extraído de holonomy.ladder (LadderDual es la subclase de la escalera
periódica 2xL); los diamantes la reutilizan.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.optimize import minimize

from .sdp import u1_sector_operators


def _translate(s, perm):
    """Estado de la base trasladado por la permutación de sitios perm (el bit
    del sitio i pasa al sitio perm[i]; qubit 0 el más significativo, como en
    holonomy.sdp)."""
    n = len(perm)
    t = 0
    for i in range(n):
        if (s >> (n - 1 - i)) & 1:
            t |= 1 << (n - 1 - perm[i])
    return t


def momentum_isometries(n_cells, basis, perm):
    """Isometrías dispersas V_q (q = 0..n_cells-1) del sector `basis` sobre
    los estados de momento 2 pi q / n_cells, con la traslación T dada por la
    permutación de sitios perm (de orden n_cells):
    |s, q> = sum_m e^{-2 pi i q m / n_cells} T^m |s> normalizado, uno por
    órbita de traslaciones compatible con q."""
    idx = {s: m for m, s in enumerate(basis)}
    seen, orbits = set(), []
    for s in basis:
        if s in seen:
            continue
        orb = [s]
        t = _translate(s, perm)
        while t != s:
            orb.append(t)
            t = _translate(t, perm)
        seen.update(orb)
        orbits.append(orb)
    out = []
    for q in range(n_cells):
        rows, cols, vals = [], [], []
        col = 0
        for orb in orbits:
            ell = len(orb)
            if (q * ell) % n_cells:
                continue
            for m, t in enumerate(orb):
                rows.append(idx[t])
                cols.append(col)
                vals.append(np.exp(-2j * np.pi * q * m / n_cells) / np.sqrt(ell))
            col += 1
        if col:
            out.append((q, sp.csr_matrix((vals, (rows, cols)), shape=(len(basis), col))))
    return out


def symmetric_blocks(n, edges, kinds, perm, n_cells, z_groups, dense_max=None, half=False,
                     keys=None):
    """Bloques (k giros, momento q) del grafo periódico, con los operadores
    sumados por tipo de arista, ("A"|"B"|"C", kind) (tipos en el orden de su
    primera aparición en kinds), y ("Z", grupo) (Z total de los sitios de
    z_groups[grupo]), como matrices densas o, en los bloques de dimensión
    mayor que dense_max, dispersas (CSR). Con half, solo los sectores
    k <= n // 2; con keys, solo esas claves. Devuelve [(k, q, {clave: matriz})]."""
    order = list(dict.fromkeys(kinds))
    out = []
    for k in range(n // 2 + 1 if half else n + 1):
        basis, ops = u1_sector_operators(n, edges, k)
        summed = {}
        for kind in order:
            es = [e for e, kk in enumerate(kinds) if kk == kind]
            for name in "ABC":
                summed[(name, kind)] = sum(ops[(name, e)] for e in es)
        for grp, sites in z_groups.items():
            summed[("Z", grp)] = sum(ops[("Z", i)] for i in sites)
        del ops
        if keys is not None:
            summed = {key: summed[key] for key in keys}
        for q, V in momentum_isometries(n_cells, basis, perm):
            Vh = V.conj().T
            dense = dense_max is None or V.shape[1] <= dense_max
            out.append((k, q, {key: (Vh @ (M @ V)).toarray() if dense else (Vh @ (M @ V)).tocsr()
                               for key, M in summed.items()}))
    return out


class PeriodicDual:
    """lambda_max del testigo W(y) = sum_m y_m O_{KEYS[m]} de un grafo
    periódico, por bloques (k, q) de symmetric_blocks.

    Con dense_max, los bloques de dimensión mayor se guardan dispersos y sus
    autovalores superiores se calculan por Lanczos (eigsh). Solo lam_max,
    top_blocks, block_gaps y v_dual_cg admiten bloques dispersos.

    Con half, solo los bloques k <= n // 2 y las claves de KEYS: válido si una
    simetría antiunitaria que conserve las claves lleva (k, q) a (n - k, -q)
    (en la escalera, el volteo global prod X compuesto con la conjugación
    compleja; lo comprueba cada subclase). labels y top_blocks solo dan
    entonces los bloques k <= n // 2."""

    def __init__(self, n, edges, kinds, perm, n_cells, z_groups, keys, dense_max=None,
                 half=False):
        self.n = n
        self.edges = list(edges)
        self.kinds = list(kinds)
        self.perm = list(perm)
        self.n_cells = n_cells
        self.KEYS = tuple(keys)
        sb = symmetric_blocks(n, self.edges, self.kinds, self.perm, n_cells, z_groups,
                              dense_max, half, self.KEYS if half else None)
        self.labels = [(k, q) for k, q, _ in sb]
        self.blocks = [red for _, _, red in sb]

    def witness_terms(self, y):
        """Coeficiente de cada clave de KEYS (las demás valen 0)."""
        return dict(zip(self.KEYS, y))

    def heisenberg_y0(self, phis):
        """Arranque: Heisenberg en todas las aristas, con la fase de los datos
        de la primera arista de cada tipo (alpha = cos, beta = sin, c = 1;
        Z = 0)."""
        first = {}
        for e, kind in enumerate(self.kinds):
            first.setdefault(kind, phis[e])
        val = {"A": np.cos, "B": np.sin}
        return np.array([0.0 if name == "Z" else (1.0 if name == "C" else val[name](first[kd]))
                         for name, kd in self.KEYS])

    @staticmethod
    def top_eig(W, r):
        """Los r autovalores mayores de W (ascendentes) y sus autovectores:
        eigh si W es densa o pequeña, Lanczos si es dispersa. Si Lanczos no
        converge (visto con n = 7 en la cadena de 2 diamantes, flujos (0, pi):
        autovalores degenerados), eigh denso."""
        d = W.shape[0]
        r = min(r, d)
        if not sp.issparse(W) or d <= max(4 * r, 64):
            w, V = np.linalg.eigh(W.toarray() if sp.issparse(W) else W)
            return w[-r:], V[:, -r:]
        try:
            w, V = eigsh(W, k=r, which="LA")
        except ArpackNoConvergence:
            w, V = np.linalg.eigh(W.toarray())
            return w[-r:], V[:, -r:]
        o = np.argsort(w)
        return w[o], V[:, o]

    def lam_max(self, y):
        best = -np.inf
        for red in self.blocks:
            W = sum(c * red[key] for key, c in zip(self.KEYS, y))
            best = max(best, self.top_eig(W, 1)[0][-1])
        return float(best)

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
        """N es lineal en y: devuelve el vector g con N(y) = g . y. Cada arista
        suma (2 cos phi_e, 2 sin phi_e, 1) en las posiciones de ("A", kind),
        ("B", kind) y ("C", kind) que estén en KEYS."""
        pos = {key: m for m, key in enumerate(self.KEYS)}
        g = np.zeros(len(self.KEYS))
        for e, kind in enumerate(self.kinds):
            for name, val in (("A", 2 * np.cos(phis[e])), ("B", 2 * np.sin(phis[e])), ("C", 1)):
                m = pos.get((name, kind))
                if m is not None:
                    g[m] += val
        return g

    @staticmethod
    def _dim(red):
        return next(iter(red.values())).shape[0]

    def v_dual_sdp(self, phis, solver="CLARABEL", **kw):
        """Dual reducido resuelto como SDP: min t con t I - W_blk(y) >= 0 en
        cada bloque (k, q) y N(y) = 1. Es exacto aunque lambda_max sea
        degenerado en el óptimo. Devuelve (v, y, estado)."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        y = cp.Variable(len(self.KEYS))
        t = cp.Variable()
        cons = [g @ y == 1]
        for red in self.blocks:
            d = self._dim(red)
            W = sum(y[m] * red[key] for m, key in enumerate(self.KEYS))
            cons.append(t * np.eye(d) - W >> 0)
        prob = cp.Problem(cp.Minimize(t), cons)
        prob.solve(solver=solver, **kw)
        return float(t.value), np.asarray(y.value), prob.status

    def v_dual_cg(self, phis, y0=None, tol=1e-8, r0=6, add=4, box=50.0, maxiter=200,
                  rmax=40, solver="CLARABEL", **kw):
        """El SDP de v_dual_sdp por generación de columnas, para bloques en los
        que el SDP completo no cabe en memoria (escalera L >= 6: Clarabel forma
        bloques densos de tamaño (n(n+1)/2)^2 por cono).

        Cada bloque se comprime a un subespacio U: min t con
        t I - U^H W_blk(y) U >= 0, N(y) = 1 y |y_m| <= box. Comprimir no sube
        lambda_max, así que t es una cota inferior de v_max (con la caja, si
        el testigo óptimo está dentro de ella); lambda_max(W(y)) en el y
        obtenido es una cota superior. Se añaden a cada U los `add`
        autovectores de mayor autovalor por encima de t y se repite hasta que
        las cotas difieran menos de tol. Arranca en y0 (o en heisenberg_y0)
        con los r0 autovectores superiores de cada bloque. Si un subespacio
        supera rmax columnas, se poda a los rmax/2 vectores de Ritz de mayor
        autovalor de U^H W(y) U (la memoria de Clarabel crece como la cuarta
        potencia del rango); la cota inferior sigue siendo válida con
        cualquier subespacio.

        Devuelve (v, y, estado, diferencia de cotas): v es la cota superior;
        estado (como en GraphDual.v_dual_cg) 'en_caja' si el y del último SDP
        restringido (el que da la cota inferior) toca la caja;
        'cota_invalida' si la diferencia es menor que -GAP_NEG (la cota
        inferior pasa de la superior más allá del ruido del solver);
        'optimal' si -GAP_NEG <= diferencia < tol; 'no_converge' en otro
        caso. Si un SDP restringido no da solución, su estado de CVXPY."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        if y0 is None:
            y0 = self.heisenberg_y0(phis)
        y0 = np.asarray(y0, float) / (g @ y0)

        def block_at(red, y):
            # Un W(y) cada vez: con L >= 10 la lista de todos no cabe en memoria.
            return sum(c * red[key] for key, c in zip(self.KEYS, y))

        U = [self.top_eig(block_at(red, y0), r0)[1] for red in self.blocks]
        best = (np.inf, y0)
        lower = -np.inf
        yv = None
        for _ in range(maxiter):
            y = cp.Variable(len(self.KEYS))
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
            eig = [self.top_eig(block_at(red, yv), add + 1) for red in self.blocks]
            upper = max(w[-1] for w, _ in eig)
            if upper < best[0]:
                best = (upper, yv)
            if best[0] - lower < tol:
                break
            for b, (w, V) in enumerate(eig):
                new = V[:, w > lower + tol / 2][:, -add:]
                if new.shape[1]:
                    if U[b].shape[1] + new.shape[1] > rmax:
                        _, Z = np.linalg.eigh(U[b].conj().T @ (block_at(self.blocks[b], yv) @ U[b]))
                        U[b] = U[b] @ Z[:, -(rmax // 2):]
                    Q, s, _ = np.linalg.svd(np.hstack([U[b], new]), full_matrices=False)
                    U[b] = Q[:, s > 1e-10 * s[0]]
        self.last_ranks = [Ub.shape[1] for Ub in U]
        gap = best[0] - lower
        # La caja se mira en el y del último SDP restringido, el que da `lower`.
        on_box = yv is not None and np.max(np.abs(yv)) > 0.99 * box
        if on_box:
            st = "en_caja"
        elif gap < -self.GAP_NEG:
            st = "cota_invalida"
        elif gap < tol:
            st = "optimal"
        else:
            st = "no_converge"
        return best[0], best[1], st, gap

    # Diferencia de cotas negativa tolerada (ruido de Clarabel, ~1e-8): más
    # negativa, la cota inferior no es válida y el estado es "cota_invalida".
    GAP_NEG = 1e-7

    def witness_spread(self, phis, v_opt, eps, solver="CLARABEL", **kw):
        """Rango de cada coeficiente del testigo dentro del conjunto casi óptimo
        {y : N(y) = 1, lambda_max(W(y)) <= v_opt + eps}: mínimo y máximo de
        cada y_m (2 len(KEYS) SDP). Si el testigo óptimo es único, el rango
        tiende a 0 con eps; si hay una cara de óptimos, queda finito."""
        import cvxpy as cp

        g = self.norm_grad(phis)
        y = cp.Variable(len(self.KEYS))
        cons = [g @ y == 1]
        for red in self.blocks:
            d = self._dim(red)
            W = sum(y[m] * red[key] for m, key in enumerate(self.KEYS))
            cons.append((v_opt + eps) * np.eye(d) - W >> 0)
        out = []
        for m in range(len(self.KEYS)):
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
            N += (2 * coef.get(("A", kind), 0.0) * np.cos(phis[e])
                  + 2 * coef.get(("B", kind), 0.0) * np.sin(phis[e]) + coef.get(("C", kind), 0.0))
        return N

    def ratio(self, y, phis):
        N = self.norm(y, phis)
        if N <= 1e-12:
            return np.inf
        return self.lam_max(y) / N

    def v_dual(self, phis, y0=None, restarts=3, seed=0):
        """min lambda_max(W)/N por Nelder-Mead con reinicios.

        Devuelve (v, y). Es una cota superior de v_max para cualquier y; la
        igualdad se contrasta con el primal.
        """
        rng = np.random.default_rng(seed)
        if y0 is None:
            y0 = self.heisenberg_y0(phis)
        best = (self.ratio(y0, phis), np.asarray(y0, float))
        starts = [best[1]] + [best[1] + 0.3 * rng.standard_normal(len(self.KEYS))
                              for _ in range(restarts)]
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

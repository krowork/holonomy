"""Dual general de v_max en un grafo cualquiera con datos R_e en SO(3).

Primal: max v con rho >= 0, tr rho = 1, <sigma_a^i> = 0 y
<sigma_a^i sigma_b^j> = v T_ab en cada arista orientada e = (i, j), con
T = -R_e. Dual: testigo

    W(y) = sum_e sum_ab y_{e,ab} sigma_a^i sigma_b^j

(9 coeficientes por arista) con normalización N(y) = sum_e sum_ab
y_{e,ab} (-R_e)_ab. Para todo rho factible tr(rho W) = v N(y) <= lambda_max(W),
así que v_max <= lambda_max(W)/N si N > 0, y v_max = min lambda_max(W)/N
(dualidad fuerte: el estado maximalmente mezclado es estrictamente factible
con v = 0). Con N(y) = 1 es el SDP min t con t I - W(y) >= 0.

Sin términos de un sitio: la inversión temporal Theta = (iY)^{⊗n} K
(antiunitaria) cambia el signo de todo Pauli de un sitio y deja fijo todo
producto de dos, sigma_a^i sigma_b^j. Si W = W_2 + sum_i z_i . sigma^i es un
testigo óptimo, Theta W Theta^-1 = W_2 - sum_i z_i . sigma^i tiene el mismo
espectro y la misma N (que solo depende de los términos de dos cuerpos), y el
promedio de los dos es W_2: lambda_max es convexa, así que
lambda_max(W_2) <= lambda_max(W) y W_2 también es óptimo. Los multiplicadores
de <sigma_a^i> = 0 se pueden tomar nulos.

Los 9|E| operadores sigma_a^i sigma_b^j se construyen una vez como matrices
dispersas de 2^n. v_dual_sdp resuelve el SDP completo (solo n <= 6);
v_dual_cg lo resuelve por generación de columnas, con conos PSD pequeños
(n = 7 con datos no abelianos: el primal completo pide más de 20 GB).
"""
import numpy as np
import scipy.sparse as sp

from .periodic import PeriodicDual

_I2 = sp.identity(2, dtype=complex, format="csr")
_PAULI = (sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
          sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
          sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)))


def _two_site(n, i, a, j, b):
    """sigma_a^i sigma_b^j en n qubits (qubit 0 el más significativo, como en
    holonomy.paulis), dispersa."""
    m = sp.identity(1, dtype=complex, format="csr")
    for k in range(n):
        f = _PAULI[a] if k == i else (_PAULI[b] if k == j else _I2)
        m = sp.kron(m, f, format="csr")
    return m


class GraphDual:
    """Testigo W(y) = sum_e sum_ab y_{e,ab} sigma_a^i sigma_b^j en el grafo
    (n, edges); y es un vector de 9|E| componentes, y[9e + 3a + b]."""

    top_eig = staticmethod(PeriodicDual.top_eig)

    def __init__(self, n, edges):
        self.n = n
        self.edges = list(edges)
        self.d = 2 ** n
        self.ops = [_two_site(n, i, a, j, b)
                    for i, j in self.edges for a in range(3) for b in range(3)]
        self.K = len(self.ops)
        self.cuts = self.product_cuts(4 * self.K)
        self.last_U = None

    def product_cuts(self, m, seed=0):
        """Filas <psi|sigma_a^i sigma_b^j|psi> = n_i^a n_j^b de m estados
        producto psi con vectores de Bloch n_i aleatorios (semilla fija):
        cada una da la restricción lineal t >= <psi|W(y)|psi>, la compresión
        a un subespacio de dimensión 1 (válida como cualquier otra)."""
        rng = np.random.default_rng(seed)
        nb = rng.standard_normal((m, self.n, 3))
        nb /= np.linalg.norm(nb, axis=2, keepdims=True)
        return np.concatenate([np.einsum("ma,mb->mab", nb[:, i], nb[:, j]).reshape(m, 9)
                               for i, j in self.edges], axis=1)

    def norm_vec(self, Rs):
        """N es lineal en y: g con N(y) = g . y, g = (-R_e) aplanado."""
        if len(Rs) != len(self.edges):
            raise ValueError("hace falta una R por arista")
        return np.concatenate([-np.asarray(R, float).reshape(9) for R in Rs])

    def witness(self, y):
        """W(y) como matriz dispersa de 2^n."""
        return sum(c * P for c, P in zip(y, self.ops) if c != 0) + sp.csr_matrix((self.d, self.d))

    def lam_max(self, y):
        return float(self.top_eig(self.witness(y), 1)[0][-1])

    @staticmethod
    def _compress(P, Ub):
        """U^H P U simetrizada (hermítica exacta) y sin el ruido de redondeo."""
        C = Ub.conj().T @ (P @ Ub)
        C = (C + C.conj().T) / 2
        C.real[np.abs(C.real) < 1e-13] = 0
        C.imag[np.abs(C.imag) < 1e-13] = 0
        return C

    @staticmethod
    def _psd(y, t, Cs):
        """t I - sum_m y_m C_m >= 0 con la inclusión real de matrices
        hermíticas, C -> [[Re C, -Im C], [Im C, Re C]] (mismo espectro, cada
        autovalor doble), como un único producto matriz-vector para CVXPY."""
        import cvxpy as cp

        r = Cs[0].shape[0]
        A = np.stack([np.block([[C.real, -C.imag], [C.imag, C.real]]).reshape(-1, order="F")
                      for C in Cs], axis=1)
        return t * np.eye(2 * r) - cp.reshape(A @ y, (2 * r, 2 * r), order="F") >> 0

    def _cons(self, y, t, U, cuts=False):
        """t I - W(y) >= 0 (U = None), o t I - U^H W(y) U >= 0 en cada
        subespacio de U y, con cuts, t >= <psi|W(y)|psi> en los estados
        producto de self.cuts. Devuelve (restricciones, restricciones PSD)."""
        if U is None:
            psd = [self._psd(y, t, [P.toarray() for P in self.ops])]
        else:
            psd = [self._psd(y, t, [self._compress(P, Ub) for P in self.ops]) for Ub in U]
        return ([self.cuts @ y <= t] if cuts else []) + psd, psd

    @staticmethod
    def _support(Zr, rel=1e-6):
        """Vectores complejos del soporte de la variable dual Z de una
        restricción PSD en la inclusión real: el autovector [a; b] de Zr
        corresponde a a + i b (los pares [a; b], [-b; a] dan el mismo rayo)."""
        Zr = (Zr + Zr.T) / 2
        w, V = np.linalg.eigh(Zr)
        keep = w > rel * max(w[-1], 1e-300)
        r = Zr.shape[0] // 2
        return V[:r, keep] + 1j * V[r:, keep]

    def _prune(self, Ub, W, Zr, keep):
        """Poda de U: el soporte U S de la variable dual (el estado
        comprimido rho = U Z U^H, que así sigue siendo factible y la cota
        inferior no baja) más los vectores de Ritz de mayor autovalor de
        U^H W U hasta `keep` columnas."""
        S = self._support(Zr) if Zr is not None else np.zeros((Ub.shape[1], 0))
        _, Z = np.linalg.eigh(Ub.conj().T @ (W @ Ub))
        fill = Z[:, -max(keep - S.shape[1], 0):] if keep > S.shape[1] else Z[:, :0]
        Q, s, _ = np.linalg.svd(np.hstack([S, fill]), full_matrices=False)
        return Ub @ Q[:, s > 1e-10 * s[0]]

    def v_dual_sdp(self, Rs, solver="CLARABEL", **kw):
        """min t con t I - W(y) >= 0 y N(y) = 1, con el cono PSD completo de
        2^n (solo n <= 6; con n = 6 el test_cg_igual_sdp_n6 entero, con el
        primal, pica en 3.8 GB). Devuelve (v, y, estado)."""
        import cvxpy as cp

        g = self.norm_vec(Rs)
        y = cp.Variable(self.K)
        t = cp.Variable()
        prob = cp.Problem(cp.Minimize(t), [g @ y == 1] + self._cons(y, t, None)[0])
        prob.solve(solver=solver, **kw)
        if y.value is None:
            return np.nan, None, prob.status
        return float(t.value), np.asarray(y.value), prob.status

    def v_dual_cg(self, Rs, y0=None, tol=1e-8, r0=6, add=4, box=50.0, maxiter=200, rmax=40,
                  cuts=True, solver="CLARABEL", **kw):
        """El SDP de v_dual_sdp por generación de columnas, como
        LadderDual.v_dual_cg pero con un único bloque de 2^n (sin simetrías
        en los datos no abelianos).

        Se comprime W a un subespacio U: min t con t I - U^H W(y) U >= 0,
        N(y) = 1 y |y_m| <= box. Comprimir no sube lambda_max, así que t es una
        cota inferior de v_max (con la caja, si el testigo óptimo está dentro
        de ella); lambda_max(W(y))/N(y) en el y obtenido es una cota superior
        (válida para cualquier y con N > 0). Se añaden a U los `add`
        autovectores de mayor autovalor por encima de t y se repite hasta que
        las cotas difieran menos de tol. Arranca en y0 o en Heisenberg rotado,
        y_e = -R_e aplanado (N = 3|E|, normalizado a 1), con sus r0
        autovectores superiores.

        Diferencias con LadderDual.v_dual_cg, necesarias con 9|E| coeficientes
        y un solo bloque (con n = 7, 90 coeficientes y r0 = 6 el SDP
        restringido no está acotado, el testigo se va a la caja y Clarabel
        falla con NumericalError a las pocas iteraciones):
        - con cuts, además, t >= <psi|W(y)|psi> en los 4|y| estados producto
          de self.cuts (compresiones a dimensión 1, lineales en y): acotan el
          problema desde la primera iteración y no cambian la cota inferior;
        - la poda (U con más de rmax columnas) conserva el soporte de la
          variable dual Z del cono comprimido (el estado rho = U Z U^H del
          SDP restringido sigue siendo factible, así que la cota inferior no
          baja) y completa con vectores de Ritz de mayor autovalor de
          U^H W(y) U hasta rmax/2 columnas; podar solo por Ritz, como en la
          escalera, avanza mucho más despacio en la cadena de 2 (medido: tras
          más de 150 iteraciones la cota inferior seguía ~1e-7 por debajo);
        - las restricciones PSD van en la inclusión real 2r x 2r como un
          único producto matriz-vector (construir la suma de 9|E| términos
          en CVXPY era el coste dominante).
        Los subespacios finales quedan en self.last_U (lista de matrices) y
        sus rangos en self.last_ranks.

        Devuelve (v, y, estado, diferencia de cotas): v es la cota superior;
        estado 'en_caja' si el y del último SDP restringido (el que da la
        cota inferior) toca la caja, o si el SDP restringido no da solución
        (infactible o fallo del solver) con la caja puesta: un fallo por la
        caja nunca parece un resultado; 'cota_invalida' si la diferencia es
        menor que -GAP_NEG (= -1e-7: la cota inferior pasa de la superior más
        allá del ruido del solver); 'optimal' si -GAP_NEG <= diferencia < tol;
        'no_converge' en otro caso. Clarabel resuelve los SDP restringidos
        con precisión ~1e-8 (estado optimal_inaccurate con tolerancias
        1e-10), así que la diferencia puede salir ligeramente negativa. El
        peor estado de CVXPY visto en las iteraciones queda en
        self.last_solver_status ('optimal', 'optimal_inaccurate', ...).

        Medido (/usr/bin/time, test_cadena2_abeliana: n = 7, cadena de 2 con
        flujos pi/2, 90 coeficientes): pico de 952 MB y 236 s, unas 70
        iteraciones, rango final 32; v - primal_u1_graph = 2e-8."""
        import cvxpy as cp

        g = self.norm_vec(Rs)
        if y0 is None:
            y0 = g.copy()
        y0 = np.asarray(y0, float) / (g @ y0)
        U = [self.top_eig(self.witness(y0), r0)[1]]
        best = (np.inf, y0)
        lower = -np.inf
        yv = None
        failed = None
        self.last_solver_status = None
        for _ in range(maxiter):
            y = cp.Variable(self.K)
            t = cp.Variable()
            cons, psd = self._cons(y, t, U, cuts)
            cons = [g @ y == 1] + ([cp.abs(y) <= box] if box is not None else []) + cons
            prob = cp.Problem(cp.Minimize(t), cons)
            try:
                prob.solve(solver=solver, **kw)
            except cp.error.SolverError:
                pass
            self.last_solver_status = self._worse(self.last_solver_status, prob.status)
            if y.value is None or t.value is None:
                failed = prob.status
                break
            lower, yv = float(t.value), np.asarray(y.value)
            W = self.witness(yv)
            w, V = self.top_eig(W, add + 1)
            upper = w[-1] / (g @ yv)
            if upper < best[0]:
                best = (upper, yv)
            if best[0] - lower < tol:
                break
            new = V[:, w > lower + tol / 2][:, -add:]
            if new.shape[1]:
                if U[0].shape[1] + new.shape[1] > rmax:
                    U[0] = self._prune(U[0], W, psd[0].dual_value, rmax // 2)
                Q, s, _ = np.linalg.svd(np.hstack([U[0], new]), full_matrices=False)
                U[0] = Q[:, s > 1e-10 * s[0]]
        self.last_U = U
        self.last_ranks = [Ub.shape[1] for Ub in U]
        gap = best[0] - lower
        if failed is not None:
            return best[0], best[1], ("en_caja" if box is not None else "no_converge"), gap
        # La caja se mira en el y del último SDP restringido, el que da `lower`.
        on_box = box is not None and yv is not None and np.max(np.abs(yv)) > 0.99 * box
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
    _STATUS_RANK = {"optimal": 0, "optimal_inaccurate": 1}

    @classmethod
    def _worse(cls, a, b):
        """El peor de dos estados de CVXPY (optimal < optimal_inaccurate <
        cualquier otro; None es "ninguno todavía")."""
        if a is None:
            return b
        return a if cls._STATUS_RANK.get(a, 2) >= cls._STATUS_RANK.get(b, 2) else b

    def witness_spread(self, Rs, v_opt, eps, U=None, solver="CLARABEL", **kw):
        """Rango de cada coeficiente del testigo dentro del conjunto casi
        óptimo {y : N(y) = 1, lambda_max(W(y)) <= v_opt + eps}: mínimo y
        máximo de cada y_m (2 * 9|E| SDP). Si el testigo óptimo es único, el
        rango tiende a 0 con eps; si hay una cara de óptimos, queda finito.

        Con U (p. ej. U = self.last_U, los subespacios finales de v_dual_cg)
        se imponen solo las restricciones comprimidas
        (v_opt + eps) I - U^H W(y) U >= 0 y los cortes de estados producto
        de self.cuts (los mismos que en v_dual_cg): el conjunto factible es mayor que
        el verdadero, así que cada rango obtenido es una cota superior del
        verdadero. Un rango que tiende a 0 prueba unicidad; uno finito no
        prueba lo contrario. Devuelve [(min, max)] en el orden de y. Si un
        SDP es infactible, no acotado o falla, lanza RuntimeError; los que
        acaban en optimal_inaccurate se marcan en self.last_spread_inaccurate
        (lista de pares de booleanos (min, max) en el mismo orden)."""
        import cvxpy as cp

        g = self.norm_vec(Rs)
        y = cp.Variable(self.K)
        cons = [g @ y == 1] + self._cons(y, v_opt + eps, U, U is not None)[0]
        out, inacc = [], []
        for m in range(self.K):
            lo_hi, flags = [], []
            for sense in (cp.Minimize, cp.Maximize):
                prob = cp.Problem(sense(y[m]), cons)
                try:
                    prob.solve(solver=solver, **kw)
                except cp.error.SolverError as exc:
                    raise RuntimeError(f"witness_spread: fallo del solver en y[{m}]") from exc
                if prob.status not in self._STATUS_RANK or y.value is None:
                    raise RuntimeError(f"witness_spread: estado {prob.status} en y[{m}]")
                lo_hi.append(float(y.value[m]))
                flags.append(prob.status == "optimal_inaccurate")
            out.append(tuple(lo_hi))
            inacc.append(tuple(flags))
        self.last_spread_inaccurate = inacc
        return out

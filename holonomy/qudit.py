"""Pegado de estados de Werner de qudits SU(d) en anillos: marco, primal y dual.

Marco
-----
Un qudit de dimensión d por vértice. El estado de Werner es

    ρ(v) = (1 − v) I/d² + v P_A/binom(d, 2),     P_A = (I − P)/2,

con P el intercambio de los dos qudits; sus marginales son I/d y
tr(ρ(v) P) = 1/d − v(1 + 1/d). A la arista orientada e = (i → j) se le asigna
ρ_e(v) = (1 ⊗ U_e) ρ(v) (1 ⊗ U_e)†, U_e ∈ SU(d) (`edge_state`). Como ρ(v)
conmuta con U ⊗ U, el gauge U_e ↦ U_j U_e U_i† no cambia v_max, y una fase
global de U_e no cambia ρ_e: los datos viven en PU(d). En d = 2, P_A es el
singlete y (1 ⊗ U)ρ(v)(1 ⊗ U)† es el ρ_e(v) de holonomia_werner.tex con
T = −R(U)^T, es decir R_e = R(U)^T = R(U†), con R(U)_ab = tr(σ_a U σ_b U†)/2
(test `test_werner_d2_es_qubit`); el gauge U_e ↦ U_j U_e U_i† es el
R_e ↦ O_i R_e O_j^T de anillos y el ángulo de holonomía es el mismo.

Holonomía. En el anillo 0 → 1 → … → n−1 → 0, con U_i el dato de la arista
(i → i+1), el producto H = U_{n−1} ⋯ U_1 U_0 pasa a V_0 H V_0† bajo el gauge;
sus autofases θ_a (`ring_holonomy`) son el invariante, módulo un desplazamiento
común (fase global; en SU(d), el centro Z_d) y módulo θ ↦ −θ (la conjugación
compleja ρ ↦ ρ* lleva los datos U_e a U_e* y conserva la factibilidad). Como en
el Lema 1 de los qubits, todo dato es gauge-equivalente al gauge uniforme
U_e = diag(e^{iθ_a/n}) en las n aristas.

Primal
------
maximizar v con ρ ⪰ 0, tr ρ = 1 y marginal de cada arista igual a ρ_e(v)
(`primal_ring`, SDP completo de d^n; se imponen tr ρ = 1, ⟨G_a⟩ = 0 en cada
sitio y ⟨G_a ⊗ G_b⟩ en cada arista, con G las matrices de Gell-Mann
generalizadas, sin restricciones redundantes). En el gauge uniforme el problema
es invariante bajo el toro diagonal V^{⊗n}; promediando, ρ se puede tomar
diagonal por bloques de números de color (n_0, …, n_{d−1}) (`primal_ring_toro`),
lo que permite d = 3, n = 5 (el completo de 243 estados no cabe en memoria).

Dual y familia de testigos
--------------------------
Dualidad débil. Sea Y un operador hermítico de dos qudits, W = Σ_e Y_e − c0 I
(Y en cada arista (i, i+1)) y c0 = n tr(Y)/d². Para todo ρ factible,
tr(ρ W) = Σ_e tr(ρ_e(v) Y) − c0, y como ρ_e(v) es afín en v,

    tr(ρ W) = v N,   N = n [tr(Y ρ_e(1)) − tr(Y ρ_e(0))]       (`ring_norm`),

sin término independiente (c0 = n tr(Y ρ_e(0)) porque ρ_e(0) = I/d²). Por tanto
v ≤ λ_max(W)/N si N > 0. Con Y = −P girado por los datos y θ = 0 se recupera la
cota d/(d+1)(1/d − e_P), e_P = E_0(Σ_i P_{i,i+1})/n (`test_testigo_su_d`),
que es v_max(n, 0) (Proposición en scripts/rigidez_qudit_limite.py; e_P por
`uls_ground_energy`).
La dualidad es fuerte (ρ = I/d^n es estrictamente factible en v = 0, Slater).

Simetrías. En el gauge uniforme los datos son invariantes bajo V^{⊗n} con V
diagonal (V ⊗ V conmuta con 1 ⊗ U_e y con ρ(v)) y bajo la traslación. Promediar
un testigo óptimo sobre el grupo compacto (toro × Z_n; Lema 6 de escaleras) da
otro óptimo invariante. Un término de dos sitios que conmuta con V ⊗ V para todo
V diagonal solo puede contener |ab⟩⟨ab| y |ab⟩⟨ba| (el carácter v_a v_b ⊗ v̄_c v̄_d
es trivial solo si {a, b} = {c, d}). Queda

    Y = Σ_ab D_ab |ab⟩⟨ab| + Σ_{a<b} (z_ab |ab⟩⟨ba| + z̄_ab |ba⟩⟨ab|),

D real y z complejo. La reflexión i ↦ −i lleva la arista (j → j+1) a
(1 ⊗ U)ρ(v)(1 ⊗ U)† con los qudits intercambiados, que es (U ⊗ 1)ρ(v)(U ⊗ 1)† =
(1 ⊗ U*)ρ(v)(1 ⊗ U*)† (U diagonal), es decir, θ ↦ −θ; la conjugación compleja K
deshace el cambio. Así r∘K es una simetría (antiunitaria) del problema para toda
holonomía; sobre Y actúa como D_ab ↦ D_ba y deja z_ab igual. Promediando, D se
toma simétrico: d(d+1)/2 + d(d−1) parámetros reales (12 en d = 3, 5 en d = 2).
K sola (inversión temporal) es simetría solo si existe un gauge con datos reales,
U_e ∈ SO(d) (salvo fase), lo que exige un espectro {θ_a} simétrico bajo θ ↦ −θ
módulo el centro: siempre en d = 2 (por eso en qubits queda el testigo de 3
coeficientes de la Prop. 4 tras quitar los campos), pero no en general en d ≥ 3;
no se usa. Los términos de un sitio (D_ab = h_a + h_b) no aportan a N y son
constantes en cada sector de color; no se pueden eliminar en general.

Orden de x (`n_params(d)` = d(d+1)/2 + d(d−1); `param_index`):
    x = [D_ab para a ≤ b en orden lexicográfico]
        + [Re z_ab, Im z_ab para a < b en orden lexicográfico],
con el término diagonal |aa⟩⟨aa| (a = b) o |ab⟩⟨ab| + |ba⟩⟨ba| (a < b).
El testigo no depende de θ (parametrización absoluta: la fase de z es la del
testigo, no relativa a los datos); θ solo entra en N, como en la Prop. 4. En
d = 2 con θ = (Θ/2, −Θ/2), x = `qubit_witness_params(α, β, c)` da exactamente
αA + βB + cC (A = −Σ(XX+YY), B = Σ(XY−YX), C = −ΣZZ) con
N = n(2α cos φ + 2β sin φ + c), φ = Θ/n.

Sectores. W conserva los números de color y la traslación: λ_max se calcula
por bloques (n_0, …, n_{d−1}) y momento k (`lam_max_sectors`), con
diagonalización densa y tomando el máximo sobre todos los bloques (el máximo
puede estar en un sector no balanceado).

`v_dual_ring` minimiza λ_max(W(x)) en el corte N(x) = 1 (con D_00 = 0 para
quitar la dirección de la identidad). El problema es convexo (λ_max es convexa
y el corte es afín) pero no diferenciable donde varios bloques empatan, que es
lo habitual en el óptimo. Por defecto usa planos de corte espectrales
(`v_dual_ring_cg`): da una cota superior (λ_max/N de un testigo explícito, con
λ_max exacta en todos los bloques) y una inferior (SDP comprimido), que en
d ≤ 3, n ≤ 7 coinciden a ≤ 1e-9. La alternativa del plan, Nelder–Mead desde
varios arranques (`v_dual_ring_nm`, uno en el testigo SU(d) con la fase de los
datos), da también una cota superior rigurosa pero se atasca en los pliegues
(2,4e-4 por encima del primal en d = 3, n = 3). Con cualquiera de los dos, el
valor se contrasta con el primal donde este existe (d = 3, n ≤ 5, reducido por
el toro en n = 4, 5); sin primal, el valor es una cota superior, y solo la
horquilla de `v_dual_ring_cg` dice cuánto le falta para ser v_max.
"""
from functools import lru_cache
from itertools import product

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

DEFAULT_SOLVER = "CLARABEL"


# ---------------------------------------------------------------------------
# Marco
# ---------------------------------------------------------------------------

def swap(d):
    """Intercambio P|ab⟩ = |ba⟩ (d² × d²)."""
    P = np.zeros((d * d, d * d))
    for a, b in product(range(d), repeat=2):
        P[b * d + a, a * d + b] = 1.0
    return P


def werner(d, v):
    """ρ(v) = (1 − v) I/d² + v P_A/binom(d, 2), P_A = (I − P)/2."""
    I = np.eye(d * d)
    PA = (I - swap(d)) / 2
    return ((1 - v) * I / d**2 + v * PA / (d * (d - 1) / 2)).astype(complex)


def edge_state(d, v, U):
    """ρ_e(v) = (1 ⊗ U) ρ(v) (1 ⊗ U)†."""
    K = np.kron(np.eye(d), np.asarray(U))
    return K @ werner(d, v) @ K.conj().T


def ring_holonomy(Us):
    """Autofases ordenadas de H = U_{n−1} ⋯ U_1 U_0 (Us[i] es la arista i → i+1).

    Son invariantes bajo el gauge (H ↦ V_0 H V_0†), pero no están en forma
    canónica: el problema solo depende de ellas módulo un desplazamiento común
    (fase global; en SU(d), el centro 2πk/d), módulo θ ↦ −θ y módulo 2π en cada
    una, y np.angle las da en (−π, π], con el corte de rama en ±π (una autofase
    cerca de π puede salir como −π + ε según el redondeo). Compárense como
    puntos e^{iθ_a} del círculo, no como números. Para el gauge uniforme da
    igual: diag(e^{iθ_a/n}) tiene producto diag(e^{iθ_a}) con cualquier rama, y
    dos ramas dan datos gauge-equivalentes (test `test_holonomia_rama_pi`).
    """
    H = np.eye(np.asarray(Us[0]).shape[0], dtype=complex)
    for U in Us:
        H = np.asarray(U) @ H
    return np.sort(np.angle(np.linalg.eigvals(H)))


def uniform_gauge(d, n, thetas):
    """U_e = diag(e^{iθ_a/n}) en las n aristas."""
    return [np.diag(np.exp(1j * np.asarray(thetas, dtype=float) / n))] * n


def gell_mann(d):
    """Las d² − 1 matrices de Gell-Mann generalizadas (hermíticas, sin traza)."""
    G = []
    for a in range(d):
        for b in range(a + 1, d):
            M = np.zeros((d, d), dtype=complex)
            M[a, b] = M[b, a] = 1
            G.append(M)
            M = np.zeros((d, d), dtype=complex)
            M[a, b], M[b, a] = -1j, 1j
            G.append(M)
    for k in range(1, d):
        M = np.zeros((d, d), dtype=complex)
        M[:k, :k] = np.eye(k)
        M[k, k] = -k
        G.append(M * np.sqrt(2 / (k * (k + 1))))
    return G


# ---------------------------------------------------------------------------
# Primal
# ---------------------------------------------------------------------------

def _digits(d, n, codes):
    """Colores (c_0, …, c_{n−1}) de los códigos s = Σ c_i d^{n−1−i} (qudit 0 el
    más significativo, como np.kron)."""
    codes = np.asarray(codes)
    return np.stack([(codes // d ** (n - 1 - i)) % d for i in range(n)], axis=1)


def _marginal_map(d, n, i, j):
    """Matriz dispersa M (d⁴ × D²) con vec_F(tr_{resto} ρ) = M vec_F(ρ), D = d^n,
    con el qudit i como primer factor del marginal y j como segundo."""
    D = d**n
    rest = [k for k in range(n) if k not in (i, j)]
    rows, cols = [], []
    for r in product(range(d), repeat=len(rest)):
        base = sum(c * d ** (n - 1 - k) for c, k in zip(r, rest))
        for a, b, c, e in product(range(d), repeat=4):
            p = base + a * d ** (n - 1 - i) + b * d ** (n - 1 - j)
            q = base + c * d ** (n - 1 - i) + e * d ** (n - 1 - j)
            rows.append((a * d + b) + (c * d + e) * d * d)
            cols.append(p + q * D)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(d**4, D * D))


def _site_map(d, n, i):
    """Igual que `_marginal_map` para el marginal de un qudit (d² × D²)."""
    D = d**n
    rest = [k for k in range(n) if k != i]
    rows, cols = [], []
    for r in product(range(d), repeat=len(rest)):
        base = sum(c * d ** (n - 1 - k) for c, k in zip(r, rest))
        for a, c in product(range(d), repeat=2):
            rows.append(a + c * d)
            cols.append(base + a * d ** (n - 1 - i) + (base + c * d ** (n - 1 - i)) * D)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(d * d, D * D))


def _vecT(M):
    """Fila r con r · vec_F(X) = tr(X M)."""
    return np.asarray(M).T.reshape(-1, order="F")


def _diagnostico(prob, mats):
    """λ_min de las matrices del estado y residuo máximo de las igualdades."""
    res = max((float(np.max(np.abs(c.violation()))) for c in prob.constraints
               if isinstance(c, cp.constraints.Equality)), default=0.0)
    lam = min(float(np.linalg.eigvalsh((M + M.conj().T) / 2)[0]) for M in mats)
    return {"lam_min": lam, "residuo": res}


def primal_ring(d, n, Us, solver=DEFAULT_SOLVER, return_state=False, **kw):
    """v_max del anillo con datos Us (Us[i] en la arista i → i+1), SDP completo.

    Devuelve (v, estado del solver). La matriz es de d^n × d^n: d = 2 hasta
    n = 6, d = 3 hasta n = 4 (n = 5 no cabe; véase `primal_ring_toro`).
    Con return_state, (v, estado, ρ, diagnóstico), con diagnóstico =
    {'lam_min': λ_min(ρ), 'residuo': máximo |residuo| de las restricciones}:
    Clarabel suele terminar en 'optimal_inaccurate' con λ_min ~ −1e-8.
    """
    D = d**n
    G = gell_mann(d)
    rows, b0, b1 = [], [], []
    # tr ρ = 1
    rows.append(sp.csr_matrix(_vecT(np.eye(D))))
    b0.append(1.0)
    b1.append(0.0)
    for i in range(n):
        S = _site_map(d, n, i)
        for g in G:
            rows.append(sp.csr_matrix(_vecT(g)) @ S)
            b0.append(0.0)
            b1.append(0.0)
    for i in range(n):
        j = (i + 1) % n
        M = _marginal_map(d, n, i, j)
        r0 = edge_state(d, 0.0, Us[i])
        r1 = edge_state(d, 1.0, Us[i])
        for ga, gb in product(G, repeat=2):
            O = np.kron(ga, gb)
            rows.append(sp.csr_matrix(_vecT(O)) @ M)
            t0 = np.trace(r0 @ O).real
            b0.append(t0)
            b1.append(np.trace(r1 @ O).real - t0)
    A = sp.vstack(rows).tocsr()
    rho = cp.Variable((D, D), hermitian=True)
    v = cp.Variable()
    cons = [rho >> 0, cp.real(A @ cp.vec(rho, order="F")) == np.array(b0) + v * np.array(b1)]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_state:
        R = np.asarray(rho.value)
        return float(v.value), prob.status, R, _diagnostico(prob, [R])
    return float(v.value), prob.status


@lru_cache(maxsize=None)
def color_sectors(d, n):
    """Lista de (números de color, códigos de los estados) en orden creciente."""
    codes = np.arange(d**n)
    dig = _digits(d, n, codes)
    counts = np.stack([(dig == a).sum(axis=1) for a in range(d)], axis=1)
    out = {}
    for s, c in zip(codes, map(tuple, counts)):
        out.setdefault(c, []).append(int(s))
    return [(c, np.array(v)) for c, v in sorted(out.items(), reverse=True)]


@lru_cache(maxsize=None)
def _toro_ops(d, n):
    """Restricciones de `primal_ring_toro` como operadores por bloque.

    Devuelve una lista, por sector (en el orden de `color_sectors`), de
    diccionarios {clave: S} con S denso (dim × dim) tal que el valor de la
    clave es Σ_sectores Σ_ij S_ij ρ_ij = Σ tr(S^T ρ_b). Claves:
      'tr';  ('s', i, a): P_i(a), a < d−1;
      ('p', i, a, b): ⟨ab|marg_{i,i+1}|ab⟩, a, b < d−1;
      ('x', i, a, b): ⟨ab|marg_{i,i+1}|ba⟩, a < b (compleja).
    """
    out = []
    for counts, states in color_sectors(d, n):
        dim = len(states)
        idx = {int(s): m for m, s in enumerate(states)}
        dig = _digits(d, n, states)
        ops = {"tr": np.eye(dim)}
        for i in range(n):
            for a in range(d - 1):
                w = (dig[:, i] == a).astype(float)
                if w.any():
                    ops[("s", i, a)] = np.diag(w)
        for i in range(n):
            j = (i + 1) % n
            for a in range(d - 1):
                for b in range(d - 1):
                    w = ((dig[:, i] == a) & (dig[:, j] == b)).astype(float)
                    if w.any():
                        ops[("p", i, a, b)] = np.diag(w)
            for a in range(d):
                for b in range(a + 1, d):
                    # ⟨ab|marg|ba⟩ = Σ_resto ρ[t, t'], t con (a, b) en (i, j) y
                    # t' con (b, a): S[t, t'] = 1 (fila t, columna t').
                    sel = np.nonzero((dig[:, i] == a) & (dig[:, j] == b))[0]
                    if len(sel) == 0:
                        continue
                    S = np.zeros((dim, dim))
                    for m in sel:
                        s = int(states[m])
                        t = s + (b - a) * d ** (n - 1 - i) + (a - b) * d ** (n - 1 - j)
                        S[m, idx[t]] = 1.0
                    ops[("x", i, a, b)] = S
        out.append(ops)
    return out


def _toro_targets(d, n, thetas):
    """Valores de las claves de `_toro_ops` en los datos: {clave: (b0, b1)}, con
    valor b0 + v b1 (de ρ_e(v) en el gauge uniforme)."""
    U = uniform_gauge(d, n, thetas)[0]
    r0 = edge_state(d, 0.0, U)
    r1 = edge_state(d, 1.0, U)
    t = {"tr": (1.0, 0.0)}
    for i in range(n):
        for a in range(d - 1):
            t[("s", i, a)] = (1 / d, 0.0)
        for a in range(d - 1):
            for b in range(d - 1):
                k = a * d + b
                t[("p", i, a, b)] = (r0[k, k].real, (r1[k, k] - r0[k, k]).real)
        for a in range(d):
            for b in range(a + 1, d):
                p, q = a * d + b, b * d + a
                t[("x", i, a, b)] = (r0[p, q], r1[p, q] - r0[p, q])
    return t


def primal_ring_toro(d, n, thetas, solver=DEFAULT_SOLVER, return_state=False, **kw):
    """v_max en el gauge uniforme con ρ diagonal por bloques de números de color.

    Exacto (promedio sobre el toro diagonal, que deja invariantes los datos).
    En cada arista el marginal solo tiene las entradas ⟨ab|·|ab⟩ y ⟨ab|·|ba⟩; se
    imponen sin redundancia (`_toro_ops`): tr ρ = 1, P_i(a) = 1/d (a < d−1) en
    cada sitio, P_e(a, b) para a, b < d−1 y las entradas ⟨ab|·|ba⟩, a < b, en
    cada arista.

    Devuelve (v, estado). Con return_state, (v, estado, bloques, diagnóstico):
    bloques es una lista de (números de color, códigos de los estados, ρ_b),
    y diagnóstico {'lam_min', 'residuo'} como en `primal_ring`.
    """
    targets = _toro_targets(d, n, thetas)
    v = cp.Variable()
    blocks, cons = [], []
    lin = {}
    for (counts, states), ops in zip(color_sectors(d, n), _toro_ops(d, n)):
        dim = len(states)
        if dim == 1:
            rb = cp.Variable((1, 1), symmetric=True)
        else:
            rb = cp.Variable((dim, dim), hermitian=True)
        cons.append(rb >> 0)
        blocks.append(rb)
        for key, S in ops.items():
            if key == "tr" or key[0] in ("s", "p"):
                expr = cp.real(cp.sum(cp.multiply(np.diag(S), cp.diag(rb))))
            else:
                expr = cp.sum(cp.multiply(S, rb))
            lin[key] = lin.get(key, 0) + expr
    for key, (b0, b1) in targets.items():
        cons.append(lin[key] == b0 + v * b1)
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_state:
        mats = [np.asarray(rb.value, dtype=complex) for rb in blocks]
        out = [(c, st, M) for (c, st), M in zip(color_sectors(d, n), mats)]
        return float(v.value), prob.status, out, _diagnostico(prob, mats)
    return float(v.value), prob.status


# ---------------------------------------------------------------------------
# Familia de testigos
# ---------------------------------------------------------------------------

def _pairs(d):
    return [(a, b) for a in range(d) for b in range(a, d)]


def _offpairs(d):
    return [(a, b) for a in range(d) for b in range(a + 1, d)]


def n_params(d):
    return d * (d + 1) // 2 + d * (d - 1)


def param_index(d, kind, a, b):
    """Posición en x de D_ab ("D", a ≤ b), Re z_ab ("Re", a < b) o Im z_ab ("Im")."""
    if kind == "D":
        return _pairs(d).index((min(a, b), max(a, b)))
    k = _offpairs(d).index((a, b))
    return d * (d + 1) // 2 + 2 * k + (0 if kind == "Re" else 1)


def edge_term(d, x):
    """Y (d² × d²) del parámetro x (orden en el docstring del módulo)."""
    x = np.asarray(x, dtype=float)
    Y = np.zeros((d * d, d * d), dtype=complex)
    for p, (a, b) in enumerate(_pairs(d)):
        Y[a * d + b, a * d + b] += x[p]
        if a != b:
            Y[b * d + a, b * d + a] += x[p]
    for (a, b) in _offpairs(d):
        z = x[param_index(d, "Re", a, b)] + 1j * x[param_index(d, "Im", a, b)]
        Y[a * d + b, b * d + a] += z
        Y[b * d + a, a * d + b] += np.conj(z)
    return Y


def _trace_coeffs(d):
    """tr(Y) es lineal en x: tr Y = t · x."""
    return np.array([np.trace(edge_term(d, e)).real for e in np.eye(n_params(d))])


def ring_norm(d, n, x, thetas):
    """N(x) = n [tr(Y ρ_e(1)) − tr(Y ρ_e(0))] en el gauge uniforme de holonomía θ.

    Todo ρ factible cumple tr(ρ W) = v N, de modo que v ≤ λ_max(W)/N si N > 0.
    """
    U = uniform_gauge(d, n, thetas)[0]
    Y = edge_term(d, x)
    return float(n * (np.trace(Y @ edge_state(d, 1.0, U)) - np.trace(Y) / d**2).real)


def _param_ops(d, n, states):
    """Operadores Σ_e (término unidad del parámetro p) en la base `states`
    (cerrada bajo el testigo), como lista de matrices dispersas."""
    states = np.asarray(states)
    idx = {int(s): m for m, s in enumerate(states)}
    dig = _digits(d, n, states)
    dim = len(states)
    m = n_params(d)
    data = [([], [], []) for _ in range(m)]
    for col, s in enumerate(states):
        s = int(s)
        for i in range(n):
            j = (i + 1) % n
            a, b = int(dig[col, i]), int(dig[col, j])
            p = param_index(d, "D", a, b)
            data[p][0].append(col)
            data[p][1].append(col)
            data[p][2].append(1.0)
            if a != b:
                t = s + (b - a) * d ** (n - 1 - i) + (a - b) * d ** (n - 1 - j)
                row = idx[t]
                lo, hi = min(a, b), max(a, b)
                # |lo hi⟩⟨hi lo| tiene z; su conjugado |hi lo⟩⟨lo hi| tiene z̄.
                pr, pi = param_index(d, "Re", lo, hi), param_index(d, "Im", lo, hi)
                data[pr][0].append(row)
                data[pr][1].append(col)
                data[pr][2].append(1.0)
                data[pi][0].append(row)
                data[pi][1].append(col)
                data[pi][2].append(1j if a > b else -1j)
    return [sp.csr_matrix((v, (r, c)), shape=(dim, dim), dtype=complex) for r, c, v in data]


def ring_witness(d, n, x, thetas=None):
    """W = Σ_e Y_e − c0 I en los d^n estados (csr). `thetas` no interviene: en
    la parametrización absoluta el testigo no depende de la holonomía (solo N)."""
    ops = _param_ops(d, n, np.arange(d**n))
    x = np.asarray(x, dtype=float)
    c0 = n * (_trace_coeffs(d) @ x) / d**2
    W = -c0 * sp.identity(d**n, dtype=complex, format="csr")
    for xp, O in zip(x, ops):
        W = W + xp * O
    return sp.csr_matrix(W)


def qubit_witness_params(alpha, beta, c):
    """x de d = 2 para el testigo α A + β B + c C de la Prop. 4 de anillos."""
    x = np.zeros(n_params(2))
    x[param_index(2, "D", 0, 0)] = -c
    x[param_index(2, "D", 1, 1)] = -c
    x[param_index(2, "D", 0, 1)] = c
    x[param_index(2, "Re", 0, 1)] = -2 * alpha
    x[param_index(2, "Im", 0, 1)] = 2 * beta
    return x


def su_d_witness(d, thetas, n=1):
    """x del testigo −Σ P_{i,i+1} girado por los datos del gauge uniforme:
    Y = −(1 ⊗ U) P (1 ⊗ U)†, U = diag(e^{iθ_a/n})."""
    phi = np.asarray(thetas, dtype=float) / n
    x = np.zeros(n_params(d))
    for a in range(d):
        x[param_index(d, "D", a, a)] = -1.0
    for (a, b) in _offpairs(d):
        # (1⊗U)|ab⟩⟨ba|(1⊗U)† = e^{i(φ_b − φ_a)} |ab⟩⟨ba|.
        z = -np.exp(1j * (phi[b] - phi[a]))
        x[param_index(d, "Re", a, b)] = z.real
        x[param_index(d, "Im", a, b)] = z.imag
    return x


@lru_cache(maxsize=None)
def sector_blocks(d, n):
    """Bloques de números de color y momento.

    Devuelve (etiquetas, isometrías, operadores): etiquetas[b] = (números de
    color, k) con momento 2πk/n; isometrías[b] (dim del sector × dim del
    bloque) con columnas de Bloch ortonormales en la base del sector;
    operadores[b] de forma (n_params, dim, dim), los operadores de los
    parámetros proyectados al bloque.
    """
    labels, isos, ops = [], [], []
    for counts, states in color_sectors(d, n):
        idx = {int(s): m for m, s in enumerate(states)}
        dig = _digits(d, n, states)
        # Traslación: (c_0, …, c_{n−1}) ↦ (c_{n−1}, c_0, …, c_{n−2}).
        shifted = np.roll(dig, 1, axis=1)
        tr = [idx[int(sum(int(c) * d ** (n - 1 - i) for i, c in enumerate(row)))] for row in shifted]
        seen, reps = set(), []
        for m in range(len(states)):
            if m in seen:
                continue
            orb, cur = [], m
            for _ in range(n):
                orb.append(cur)
                cur = tr[cur]
            seen.update(orb)
            reps.append(orb)
        P = _param_ops(d, n, states)
        dimS = len(states)
        for k in range(n):
            cols = []
            for orb in reps:
                vec = np.zeros(dimS, dtype=complex)
                for jj, mm in enumerate(orb):
                    vec[mm] += np.exp(-2j * np.pi * k * jj / n)
                nr = np.linalg.norm(vec)
                if nr > 1e-9:
                    cols.append(vec / nr)
            if not cols:
                continue
            B = np.array(cols).T
            labels.append((tuple(int(c) for c in counts), k))
            isos.append(B)
            ops.append(np.array([B.conj().T @ (O @ B) for O in P]))
    return labels, isos, ops


def lam_max_sectors(d, n, x):
    """(λ_max(W), números de color y k del bloque del máximo), por bloques."""
    x = np.asarray(x, dtype=float)
    labels, _, ops = sector_blocks(d, n)
    best, arg = -np.inf, None
    for lab, O in zip(labels, ops):
        w = np.linalg.eigvalsh(np.tensordot(x, O, axes=1))[-1]
        if w > best:
            best, arg = w, lab
    c0 = n * (_trace_coeffs(d) @ x) / d**2
    return float(best - c0), arg[0], arg[1]


class _Corte:
    """Corte afín N(x) = 1, D_00 = 0 (sin la dirección de la identidad, que da
    W = 0 y N = 0), parametrizado por las coordenadas libres y."""

    def __init__(self, d, n, thetas):
        m = n_params(d)
        self.m = m
        self.Nv = np.array([ring_norm(d, n, e, thetas) for e in np.eye(m)])
        self.i00 = param_index(d, "D", 0, 0)
        others = [p for p in range(m) if p != self.i00]
        self.s = max(others, key=lambda p: abs(self.Nv[p]))
        self.free = [p for p in others if p != self.s]
        self.ident = np.zeros(m)
        self.ident[:len(_pairs(d))] = 1.0

    def x_of(self, y):
        x = np.zeros(self.m)
        x[self.free] = y
        x[self.s] = (1.0 - self.Nv[self.free] @ y) / self.Nv[self.s]
        return x

    def normalize(self, x):
        """x en el corte (quita la identidad y divide por N), o None si N <= 0."""
        x = np.asarray(x, dtype=float)
        x = x - x[self.i00] * self.ident
        N = self.Nv @ x
        return None if N <= 0 else x / N

    def y_of(self, x):
        x = self.normalize(x)
        return None if x is None else x[self.free]


def v_dual_ring_nm(d, n, thetas, x0=None, starts=8, seed=0, maxrestart=5):
    """Mínimo de λ_max(W)/N por Nelder–Mead en el corte N = 1, D_00 = 0, desde
    el testigo SU(d) con la fase de los datos, x0 si se da, y perturbaciones
    aleatorias (semilla `seed`) hasta `starts` arranques; cada búsqueda se
    relanza desde su resultado hasta `maxrestart` veces mientras mejora.

    Devuelve (v, x), v = λ_max(W(x))/N(x): cota superior rigurosa de v_max,
    pero Nelder–Mead se atasca en los pliegues de λ_max (bloques que empatan):
    medido en d = 3, n = 3, θ = (1, 0,3, −1,3), 2,4e-4 por encima del primal
    tras 440 s (con maxrestart = 20). Úsese `v_dual_ring` (cortes).
    """
    cut = _Corte(d, n, thetas)

    def f(y):
        return lam_max_sectors(d, n, cut.x_of(y))[0]

    rng = np.random.default_rng(seed)
    y_su = cut.y_of(su_d_witness(d, thetas, n))
    inits = [y_su]
    if x0 is not None and cut.y_of(x0) is not None:
        inits.append(cut.y_of(x0))
    scale = max(1.0, np.linalg.norm(y_su))
    while len(inits) < starts:
        inits.append(y_su + 0.3 * scale * rng.normal(size=len(cut.free)))
    best_v, best_y = np.inf, None
    for y in inits:
        val = f(y)
        for _ in range(maxrestart):
            r = minimize(f, y, method="Nelder-Mead",
                         options=dict(xatol=1e-11, fatol=1e-14, maxiter=4000 * len(cut.free),
                                      maxfev=8000 * len(cut.free), adaptive=True))
            if r.fun < val - 1e-13:
                y, val = r.x, r.fun
            else:
                break
        if val < best_v:
            best_v, best_y = val, y
    x = cut.x_of(best_y)
    return float(lam_max_sectors(d, n, x)[0] / ring_norm(d, n, x, thetas)), x


def v_dual_ring_cg(d, n, thetas, x0=None, tol=1e-9, r0=2, add=2, box=50.0, maxiter=300,
                   solver=DEFAULT_SOLVER, momentos=None, **kw):
    """Mínimo de λ_max(W) en el corte N = 1, D_00 = 0, por planos de corte
    espectrales (generación de columnas, como `gdual.v_dual_cg`).

    El problema es convexo. Se comprime cada bloque b (números de color y
    momento) a un subespacio U_b y se resuelve el SDP pequeño
        mín t  con  t I − U_b† W_b(x) U_b ⪰ 0 (todo b),  N(x) = 1,  |x_p| ≤ box;
    comprimir no sube λ_max, así que t es una cota inferior del mínimo de la
    familia (si el óptimo está dentro de la caja), que es v_max (simetría y
    dualidad fuerte). λ_max(W(x)) del x obtenido, exacta en todos los bloques,
    es una cota superior. Se añaden a cada U_b los `add` autovectores de mayor
    autovalor por encima de t y se repite hasta que las cotas difieren menos de
    tol. Arranca en x0 o en el testigo SU(d) con la fase de los datos, con r0
    autovectores por bloque. Con Clarabel y sin opciones, tolerancias de 1e-10
    (con las de defecto, 1e-8, la cota inferior pasaba ~1e-8 de una superior
    válida en d = 3, n = 5, θ = 0); aun así inf es tan precisa como el solver
    (~1e-9) y puede quedar ligeramente por encima de sup.

    Devuelve (sup, inf, x, estado): estado 'optimal' si sup − inf < tol,
    'en_caja' si el x de la cota inferior toca la caja, 'no_converge' si se
    agotan las iteraciones; 'cota_invalida' si inf > sup + 1e-7.

    Con `momentos` (iterable de k), λ_max se toma solo sobre los bloques de esos
    momentos (todos los sectores de color): sup e inf acotan entonces el mínimo
    de esa λ_max restringida, que es ≤ v_max. Sirve para certificar que el
    óptimo no puede ser liso con el multiplete superior en un solo momento
    (scripts/rigidez_qudit_flujo.py). ValueError si `momentos` está vacío o
    tiene algún k fuera de 0..n−1.
    """
    cut = _Corte(d, n, thetas)
    m = cut.m
    labels, _, ops = sector_blocks(d, n)
    if momentos is not None:
        ks = set(int(k) for k in momentos)
        if not ks:
            raise ValueError("v_dual_ring_cg: 'momentos' vacío (se esperaba algún k en 0..n−1)")
        fuera = sorted(k for k in ks if not 0 <= k < n)
        if fuera:
            raise ValueError(f"v_dual_ring_cg: momentos fuera de 0..{n - 1}: {fuera}")
        ops = [O for lab, O in zip(labels, ops) if lab[1] in ks]
        if not ops:
            raise ValueError(f"v_dual_ring_cg: ningún bloque con momentos {sorted(ks)}")
    c0v = n * _trace_coeffs(d) / d**2
    xs = None if x0 is None else cut.normalize(x0)
    if xs is None:
        xs = cut.normalize(su_d_witness(d, thetas, n))

    def spectra(x):
        return [np.linalg.eigh(np.tensordot(x, O, axes=1)) for O in ops]

    spec = spectra(xs)
    U = [V[:, -min(r0, V.shape[1]):] for _, V in spec]
    # El arranque también es un testigo: su λ_max/N es una cota superior.
    upper = float((max(w[-1] for w, _ in spec) - c0v @ xs) / (cut.Nv @ xs))
    best_x, lower = xs, -np.inf
    if solver == "CLARABEL" and not kw:
        kw = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    status = "no_converge"
    for _ in range(maxiter):
        x = cp.Variable(m)
        t = cp.Variable()
        cons = [cut.Nv @ x == 1, x[cut.i00] == 0, cp.abs(x) <= box]
        for O, Ub in zip(ops, U):
            r = Ub.shape[1]
            C = np.einsum("ia,pij,jb->pab", Ub.conj(), O, Ub)
            C = (C + C.conj().transpose(0, 2, 1)) / 2
            Wc = cp.reshape(C.reshape(m, r * r).T @ x, (r, r), order="C")
            M = (t + c0v @ x) * np.eye(r) - Wc
            cons.append((M + M.H) / 2 >> 0)
        prob = cp.Problem(cp.Minimize(t), cons)
        prob.solve(solver=solver, **kw)
        if x.value is None:
            status = prob.status
            break
        xv = np.asarray(x.value)
        lower = max(lower, float(t.value))
        spec = spectra(xv)
        lam = max(w[-1] for w, _ in spec) - c0v @ xv
        val = lam / (cut.Nv @ xv)
        if val < upper:
            upper, best_x = float(val), xv
        if np.max(np.abs(xv)) > box * (1 - 1e-6):
            status = "en_caja"
        if lower > upper + 1e-7:
            status = "cota_invalida"
            break
        if upper - lower < tol:
            status = "optimal" if status != "en_caja" else status
            break
        status = "no_converge"
        for b, (w, V) in enumerate(spec):
            sel = np.nonzero(w - c0v @ xv > float(t.value) + 1e-12)[0][::-1][:add]
            if len(sel):
                Q, _ = np.linalg.qr(np.hstack([U[b], V[:, sel]]))
                U[b] = Q[:, :min(Q.shape[1], V.shape[0])]
    return upper, lower, best_x, status


def v_dual_ring(d, n, thetas, x0=None, starts=8, seed=0, metodo="cortes", tol=1e-9):
    """Cota superior de v_max por la familia invariante: mín. de λ_max(W)/N.

    metodo='cortes' (por defecto): `v_dual_ring_cg` desde x0 o el testigo SU(d);
    da además una cota inferior (la diferencia de cotas es < tol si el estado es
    'optimal'; si no, se avisa). metodo='nm': `v_dual_ring_nm` (Nelder–Mead
    desde `starts` arranques, semilla `seed`), sin certificado.

    Devuelve (v, x) con v = λ_max(W(x))/N(x), λ_max exacta por bloques: siempre
    una cota superior rigurosa de v_max.
    """
    if metodo == "nm":
        return v_dual_ring_nm(d, n, thetas, x0=x0, starts=starts, seed=seed)
    sup, inf, x, st = v_dual_ring_cg(d, n, thetas, x0=x0, tol=tol)
    if st != "optimal":
        import warnings
        warnings.warn(f"v_dual_ring: generación de columnas '{st}', sup - inf = {sup - inf:.2e}")
    return sup, x


# ---------------------------------------------------------------------------
# Sin holonomía: anillo SU(d) de intercambio (Uimin–Lai–Sutherland)
# ---------------------------------------------------------------------------
#
# Con holonomía trivial (θ = 0, U_e = I tras el gauge),
#     v_max(n, 0) = d/(d+1) (1/d − e_P(n)),   e_P(n) = E_0(Σ_i P_{i,i+1})/n
# (Proposición; demostración en scripts/rigidez_qudit_limite.py). Σ P conserva
# los números de color y conmuta con las permutaciones de los colores, así que
# su espectro en el sector (n_0, …, n_{d−1}) solo depende de los números
# ordenados. Como cada irrep de SU(d) en (C^d)^{⊗n} tiene un peso en el sector
# más equilibrado, el mínimo sin campo está siempre también ahí; aun así se
# toma el mínimo sobre todos los sectores (el campo de color lo desplaza).

def _multiset_perms(counts):
    """Configuraciones de colores con los números dados, en orden lexicográfico
    (= orden creciente del código s = Σ c_i d^{n−1−i}); array (dim, n)."""
    counts = list(counts)
    n = sum(counts)
    out, cur = [], []

    def rec():
        if len(cur) == n:
            out.append(tuple(cur))
            return
        for a in range(len(counts)):
            if counts[a]:
                counts[a] -= 1
                cur.append(a)
                rec()
                cur.pop()
                counts[a] += 1

    rec()
    return np.array(out, dtype=np.int64).reshape(-1, n)


def _compositions(n, d):
    """Vectores (n_0, …, n_{d−1}) de enteros ≥ 0 con suma n."""
    if d == 1:
        return [(n,)]
    return [(a,) + r for a in range(n, -1, -1) for r in _compositions(n - a, d - 1)]


def uls_sector_matrix(d, n, counts):
    """Σ_i P_{i,i+1} (anillo de n sitios) en el sector de números de color
    `counts`, dispersa y real, en la base de `_multiset_perms(counts)`."""
    st = _multiset_perms(counts)
    dim = len(st)
    w = d ** np.arange(n - 1, -1, -1, dtype=np.int64)
    codes = st @ w
    rows, cols = [], []
    for i in range(n):
        j = (i + 1) % n
        t = st.copy()
        t[:, [i, j]] = t[:, [j, i]]
        rows.append(np.searchsorted(codes, t @ w))
        cols.append(np.arange(dim))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))


def _lowest_eig(H, dense_max, rng, k=4, ncv=40):
    """Menor autovalor: denso si dim ≤ dense_max; si no, Lanczos (eigsh) con
    arranque aleatorio y k autovalores pedidos (Review Focus 1)."""
    dim = H.shape[0]
    if dim <= max(dense_max, 2 * ncv):
        return float(np.linalg.eigvalsh(H.toarray())[0])
    from scipy.sparse.linalg import eigsh
    w = eigsh(H, k=k, which="SA", v0=rng.standard_normal(dim), ncv=ncv, tol=0,
              return_eigenvectors=False)
    return float(np.min(w))


def uls_sector_energies(d, n, dense_max=2000, seed=0):
    """{números de color: menor autovalor de Σ_i P_{i,i+1} en ese sector}, para
    todos los sectores (se diagonaliza uno por cada vector ordenado)."""
    rng = np.random.default_rng(seed)
    cache = {}
    out = {}
    for c in _compositions(n, d):
        key = tuple(sorted(c, reverse=True))
        if key not in cache:
            cache[key] = _lowest_eig(uls_sector_matrix(d, n, key), dense_max, rng)
        out[c] = cache[key]
    return out


def uls_ground_energy(d, n, field=None, dense_max=2000, seed=0, return_sector=False):
    """E_0 de Σ_i P_{i,i+1} + Σ_i h_{c_i} (campo de color h = `field`, por
    defecto nulo) en el anillo de n qudits: mínimo sobre todos los sectores de
    números de color. Con return_sector, (E_0, sector del mínimo)."""
    E = uls_sector_energies(d, n, dense_max=dense_max, seed=seed)
    h = np.zeros(d) if field is None else np.asarray(field, dtype=float)
    vals = {c: e + float(h @ np.array(c)) for c, e in E.items()}
    c = min(vals, key=vals.get)
    return (vals[c], c) if return_sector else vals[c]


def uls_e_inf(d):
    """e_∞(d) = 1 − (2/d)[ψ(1) − ψ(1/d)], energía por sitio de Σ P en el límite
    termodinámico (Sutherland 1975; Messio–Mila 2012, ec. 4)."""
    from scipy.special import digamma
    return float(1 - 2 / d * (digamma(1.0) - digamma(1.0 / d)))

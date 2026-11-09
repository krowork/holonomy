"""Asintótica de qudits con holonomía: teoría de campos SU(d)_1 y anillo SU(d) retorcido.

Notas de la derivación (ecuaciones, referencias y comparación numérica):
docs/superpowers/notes/2026-11-09-rigidez-asintotica.md. Guion:
scripts/rigidez_qudit_asint.py. Marco y familia de testigos: holonomy/qudit.py.

Familia «isótropa con campo» (exacta para todo n)
-------------------------------------------------
Testigo Y = −(1 ⊗ U_s) P (1 ⊗ U_s)† (el −Σ P girado por un flujo de Cartan s,
U_s = diag(e^{i s_a/n})) más un campo de color de un sitio h (sin traza,
D_ab −= h_a + h_b, es decir, −2 h·Q en W). Con datos de holonomía t en el gauge uniforme y
V = U_s† U_t = diag(e^{i(t_a − s_a)/n}):
    N   = n (|tr V|² − 1)/(d(d − 1))                                  (`norma_iso`),
    λ_max(W) = n/d − mín_sectores [E_N(s) + 2 h·Q_N],
con E_N(s) la energía fundamental de Σ_i P^s_{i,i+1} en el sector de números de
color N y Q_N = N − (n/d)(1, …, 1) su carga (peso). El campo no cambia N (las
marginales de ρ_e son I/d). Así
    v_max(n, t) ≤ (n/d − G)/N,   G = máx_h mín_N [E_N(s) + 2 h·Q_N]    (`campo_lp`),
y G es, por dualidad lineal, el mínimo de Σ μ_N E_N(s) sobre las mezclas μ de
sectores con carga media nula. Con h = 0 queda la cota isótropa
(n/d − E_0(s))/N. En s = t, N = n(d + 1)/d.

Teoría de orden dominante (`H_cero`, Resultado, no teorema)
-----------------------------------------------------------
Modos cero de SU(d)_1 (d − 1 bosones, v = 2π/d, normalización de Σ P): un
estado de carga Q (peso de la clase r = n mód d) y corriente λ ∈ Λ_w tiene
    E = n e_∞ − π²(d − 1)/(3 d n) + [π²|Q|² + |s − πQ − 2πλ|²]/(d n) + o(1/n).
Desarrollando la cota de la familia hasta O(1/n²) con A = d ε_d/(d² − 1),
ε_d = 1/d − e_∞(d), B = 1/d:
    v(n, t) = v_∞ + d/(d + 1) · n^{−2} [π²(d − 1)/(3d) + H_r(t)] + o(n^{−2}),
    H_r(t) = mín_{s,h} máx_{Q,λ} {A|t − s|² − B[π²|Q|² + |s − πQ − 2πλ|²] − B h·Q},
problema convexo (A > B). La hipótesis es la de \\cite{anillos} sección 6.2: las
anisotropías de dos sitios son estacionarias en el punto isótropo y no
contribuyen a O(1/n²); el campo de color sí, porque para n ≢ 0 (mód d) todos los
sectores tienen carga y el campo los equilibra a primer orden.
"""
from functools import lru_cache
from itertools import product

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import eigsh

from holonomy import qudit

# ---------------------------------------------------------------------------
# Orden dominante
# ---------------------------------------------------------------------------


def proj(m):
    """Parte sin traza de un vector de R^d."""
    m = np.asarray(m, dtype=float)
    return m - m.mean()


def constantes(d):
    """ε_d, A, B, κ = 1 − B/A y v_∞ = d ε_d/(d + 1) (ε_d = 1/d − e_∞(d))."""
    eps = 1.0 / d - qudit.uls_e_inf(d)
    A = eps * d / (d * d - 1)
    B = 1.0 / d
    return dict(eps=eps, A=A, B=B, kappa=1 - B / A, v_inf=d * eps / (d + 1))


@lru_cache(maxsize=None)
def _cargas(d, r, qmax=2, extra=2.1):
    """Pesos Q = proj(m), m ∈ {−qmax..qmax}^d con Σm ≡ r (mód d), sin repetir,
    con |Q|² ≤ |ω_r|² + extra (la órbita mínima y la capa siguiente)."""
    Qs = {}
    for m in product(range(-qmax, qmax + 1), repeat=d):
        if (sum(m) - r) % d:
            continue
        Q = proj(m)
        Qs.setdefault(tuple(np.round(Q, 9)), Q)
    w2 = min(Q @ Q for Q in Qs.values())
    return tuple(tuple(Q) for Q in Qs.values() if Q @ Q <= w2 + extra + 1e-9)


@lru_cache(maxsize=None)
def _red(d, lmax=2):
    """Puntos 2πλ, λ = proj(l), l ∈ {−lmax..lmax}^d, sin repetir."""
    ps = {}
    for l in product(range(-lmax, lmax + 1), repeat=d):
        p = 2 * np.pi * proj(l)
        ps.setdefault(tuple(np.round(p, 9)), p)
    return np.array(list(ps.values()))


def estados_cero(d, r, centro, radio=2.5 * np.pi):
    """Pares (Q, p = πQ + 2πλ) de modos cero con |p − centro| < radio."""
    red = _red(d)
    out = []
    for Q in _cargas(d, r):
        Q = np.array(Q)
        P = np.pi * Q + red
        for p in P[np.linalg.norm(P - proj(centro), axis=1) < radio]:
            out.append((Q, p))
    return out


def H_cero(d, r, t, campo=True, tol=1e-10):
    """H_r(t) del orden dominante (docstring del módulo), con o sin campo de
    color. SOCP pequeño con Clarabel.

    Devuelve (H, s, h, estado, activas): s y h óptimos (sin traza) y las cargas
    Q distintas de los modos cero activos (multiplicador > 1e-7): son los
    sectores de color que el óptimo deja degenerados.
    """
    c = constantes(d)
    A, B = c["A"], c["B"]
    t = proj(t)
    s = cp.Variable(d)
    h = cp.Variable(d)
    u = cp.Variable()
    cons = [cp.sum(s) == 0, cp.sum(h) == 0]
    if not campo:
        cons.append(h == 0)
    est = estados_cero(d, r, t)
    cest = [u >= (A - B) * cp.sum_squares(s) - 2 * (A * t - B * p) @ s
            + A * t @ t - B * p @ p - B * np.pi ** 2 * (Q @ Q) - B * (Q @ h) for Q, p in est]
    prob = cp.Problem(cp.Minimize(u), cons + cest)
    prob.solve(solver="CLARABEL", tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    act = {}
    for (Q, _), k in zip(est, cest):
        if k.dual_value is not None and float(np.max(k.dual_value)) > 1e-7:
            act.setdefault(tuple(np.round(Q, 6)), Q)
    return (float(prob.value), np.asarray(s.value), np.asarray(h.value), prob.status,
            list(act.values()))


def coef_cero(d, r, t, campo=True):
    """Límite de n²[v(n, 0) − v(n, t)] para n ≡ r (mód d): d/(d+1)·[H_r(0) − H_r(t)]."""
    return d / (d + 1) * (H_cero(d, r, np.zeros(d), campo)[0] - H_cero(d, r, t, campo)[0])


# ---------------------------------------------------------------------------
# Anillo SU(d) retorcido por sectores de color
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _estructura(d, n, counts):
    """Estados del sector (`qudit._multiset_perms`) y, por arista (i, i+1) del
    anillo, los elementos de Σ P: filas, columnas y colores (c_i, c_{i+1}) del
    estado de partida; los diagonales (c_i = c_{i+1}) aparte."""
    st = qudit._multiset_perms(counts)
    dim = len(st)
    w = d ** np.arange(n - 1, -1, -1, dtype=np.int64)
    codes = st @ w
    rows, cols, ca, cb = [], [], [], []
    diag = np.zeros(dim)
    for i in range(n):
        j = (i + 1) % n
        a, b = st[:, i], st[:, j]
        eq = a == b
        diag += eq
        idx = np.nonzero(~eq)[0]
        tt = st[idx].copy()
        tt[:, [i, j]] = tt[:, [j, i]]
        rows.append(np.searchsorted(codes, tt @ w))
        cols.append(idx)
        ca.append(a[idx])
        cb.append(b[idx])
    return (dim, np.concatenate(rows), np.concatenate(cols), np.concatenate(ca),
            np.concatenate(cb), diag)


def uls_torcido(d, n, counts, s):
    """Σ_i P^s_{i,i+1} en el sector `counts` (gauge uniforme, flujo s por color):
    P^s|b a⟩ = e^{i(s_b − s_a)/n}|a b⟩ en cada arista, el convenio de
    `qudit.su_d_witness` (Y = −P^s)."""
    dim, rows, cols, ca, cb, diag = _estructura(d, n, tuple(int(c) for c in counts))
    phi = np.asarray(s, dtype=float) / n
    data = np.exp(1j * (phi[ca] - phi[cb]))
    H = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    return H + sp.diags(diag.astype(complex))


def energia_sector(d, n, counts, s, dense_max=300, k=4, ncv=40, seed=0, vector=False):
    """Menor autovalor de `uls_torcido`: denso hasta `dense_max`; si no, Lanczos
    con arranque aleatorio, k autovalores y ncv vectores (Review Focus 1)."""
    H = uls_torcido(d, n, counts, s)
    dim = H.shape[0]
    if dim <= max(dense_max, 2 * ncv):
        w, V = np.linalg.eigh(H.toarray())
        return (float(w[0]), V[:, 0]) if vector else float(w[0])
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    w, V = eigsh(H, k=k, which="SA", v0=v0, ncv=ncv, tol=0)
    i = int(np.argmin(w))
    return (float(w[i]), V[:, i]) if vector else float(w[i])


def energias_torcidas(d, n, s, perm=None, **kw):
    """{números de color: E_N(s)} para todos los sectores (Review Focus 2: el
    mínimo se toma siempre sobre todos).

    perm (opcional): permutación σ de los colores con E_N(s) = E_{N∘σ}(s), que
    el llamante garantiza (en la recta raíz, σ = (2, 1, 0): la simetría
    K∘(0 2) fija s; en la recta peso, σ = (1, 0, 2)); entonces se diagonaliza un
    sector de cada par y el otro se copia."""
    out = {}
    for c in qudit._compositions(n, d):
        if perm is not None:
            cs = tuple(c[perm[a]] for a in range(d))
            if cs in out:
                out[c] = out[cs]
                continue
        out[c] = energia_sector(d, n, c, s, **kw)
    return out


def carga(counts):
    """Q_N = N − (n/d)(1, …, 1)."""
    return proj(counts)


def campo_lp(E):
    """G = máx_h mín_N [E_N + 2 h·Q_N] por programación lineal (HiGHS).

    Devuelve (G, h, activos): h sin traza; activos, los sectores a menos de
    1e-9 de G. Con h = 0, mín_N E_N ≤ G.
    """
    keys = list(E)
    d = len(keys[0])
    Qs = np.array([carga(c) for c in keys])
    Ev = np.array([E[c] for c in keys])
    # Variables (u, h_0..h_{d−1}); máx u  ⇔  mín −u;  u − 2 h·Q_N ≤ E_N;  Σ h = 0.
    cobj = np.zeros(d + 1)
    cobj[0] = -1.0
    A_ub = np.hstack([np.ones((len(keys), 1)), -2 * Qs])
    A_eq = np.zeros((1, d + 1))
    A_eq[0, 1:] = 1.0
    res = linprog(cobj, A_ub=A_ub, b_ub=Ev, A_eq=A_eq, b_eq=[0.0],
                  bounds=[(None, None)] * (d + 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"campo_lp: {res.message}")
    G, h = float(res.x[0]), res.x[1:]
    val = Ev + 2 * Qs @ h
    activos = [keys[i] for i in np.nonzero(val < G + 1e-9)[0]]
    return float(np.min(val)), h, activos


def norma_iso(d, n, t, s):
    """N del testigo −P^s con datos t: n(|Σ_a e^{i(t_a − s_a)/n}|² − 1)/(d(d − 1))."""
    z = np.sum(np.exp(1j * (np.asarray(t, float) - np.asarray(s, float)) / n))
    return float(n * (abs(z) ** 2 - 1) / (d * (d - 1)))


def v_familia(d, n, t, s, E=None, campo=True, **kw):
    """Cota superior de v_max(n, t) con el testigo −P^s más el campo óptimo
    (campo=True) o sin campo. Devuelve (v, G, h, activos)."""
    if E is None:
        E = energias_torcidas(d, n, s, **kw)
    if campo:
        G, h, act = campo_lp(E)
    else:
        G = min(E.values())
        h, act = np.zeros(d), [min(E, key=E.get)]
    return (n / d - G) / norma_iso(d, n, t, s), G, h, act


def energia_min(d, n, s, **kw):
    """E_0(s) = mín sobre todos los sectores de `uls_torcido`."""
    return min(energias_torcidas(d, n, s, **kw).values())


def pendiente_lateral(d, n, p, c, delta=None, **kw):
    """Derivada lateral de E_0 en la cúspide c a lo largo de la recta desde el
    punto liso p (lado de p): (3E(c) − 4E(c − δû) + E(c − 2δû))/(2δ), O(δ²).
    Devuelve (E(c), E', |c − p|)."""
    p, c = np.asarray(p, float), np.asarray(c, float)
    R = float(np.linalg.norm(c - p))
    u = (c - p) / R
    if delta is None:
        delta = 1e-3 * R
    E = [energia_min(d, n, c - j * delta * u, **kw) for j in range(3)]
    return E[0], (3 * E[0] - 4 * E[1] + E[2]) / (2 * delta), R


def borde_ventana_iso(d, n, u, R, Ec, dE):
    """Borde de la ventana de la familia isótropa (sin campo) en una recta de
    dirección unitaria u, con el punto liso en 0 y la cúspide a distancia R
    (energía Ec, pendiente lateral dE): el testigo deja la cúspide cuando
        g(z) = (n/d − Ec) F'(z) − dE (F(z) − 1) = 0,   z = y − R ∈ (−R, 0),
    F(z) = |Σ_a e^{i z u_a/n}|² (análogo de la Proposición del criterio de borde
    de \\cite{anillos} con Δ = 1). Devuelve la anchura y (distancia desde el
    punto liso), 0 si no hay ventana."""
    from scipy.optimize import brentq
    u = np.asarray(u, float)
    dif = (u[:, None] - u[None, :]) / n

    def F(z):
        return float(np.sum(np.cos(z * dif)))

    def dF(z):
        return float(-np.sum(dif * np.sin(z * dif)))

    def g(z):
        return (n / d - Ec) * dF(z) - dE * (F(z) - 1)

    if g(-R) <= 0:
        return 0.0
    return R + brentq(g, -R, -1e-12 * R, xtol=1e-14)

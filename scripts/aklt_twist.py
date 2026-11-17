"""Teorema MPS con AKLT: robustez del pegado frente a la holonomía.

Datos: en cada arista del anillo de n espines 1, el estado de dos sitios

    rho_j(p) = I/9 + p * tau_j,   tau_j = (1 ⊗ V) (sigma - I/9) (1 ⊗ V)^†,

con V = exp(-i Theta S^z / n) (gauge uniforme) y sigma la marginal de dos
sitios del estado AKLT en el límite termodinámico,
sigma = P0/3 + (2/3) P1/3 (peso nulo en J = 2).

Afirmación (sin hipótesis espectrales):

    1 - s_n <= p(n, Theta) <= 1   para todo Theta,
    s_n = n eps_n / (r + n eps_n),  r = 1/(6 d^2) = 1/54,

con eps_n = max_j || marginal_j(psi_Theta) - rho_j(1) ||_1 y psi_Theta el
MPS de AKLT con el giro v = exp(-i Theta sigma^z / (2n)) en cada enlace
virtual. La cota superior es porque sigma tiene rango 4 < 9: con p > 1,
rho_j(p) tiene autovalor (1 - p)/9 < 0. La inferior, por el lema de
dilución. eps_n = O(3^-n) porque v ⊗ conj(v) fija los puntos fijos de la
matriz de transferencia.

Comprobaciones:
  1. simetría U(1) del MPS: e^{-i theta m} A^m = v A^m v^†;
  2. marginales por fuerza bruta (n <= 9) = por matriz de transferencia;
  3. peso nulo en J = 2 en los marcos girados (libre de frustración);
  4. eps_n * 3^n acotado, para todo Theta, hasta n = 30;
  5. p(n, Theta) exacto por SDP (n = 4, 5, 6) dentro de [1 - s_n, 1].
"""
import sys
import time

import numpy as np

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg)
    OK &= bool(cond)


# --- espín 1, base m = +1, 0, -1 -------------------------------------------
Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
Sx = (Sp + Sp.conj().T) / 2
Sy = (Sp - Sp.conj().T) / (2j)
I3 = np.eye(3)
SS = sum(np.kron(a, a) for a in (Sx, Sy, Sz)).real  # S1.S2
# proyectores sobre J = 0, 1, 2 (autovalores de S1.S2: -2, -1, 1)
P0 = (SS + np.eye(9)) @ (SS - np.eye(9)) / ((-2 + 1) * (-2 - 1))
P1 = (SS + 2 * np.eye(9)) @ (SS - np.eye(9)) / ((-1 + 2) * (-1 - 1))
P2 = (SS + 2 * np.eye(9)) @ (SS + np.eye(9)) / ((1 + 2) * (1 + 1))
check(np.allclose(P0 + P1 + P2, np.eye(9)) and np.isclose(np.trace(P0), 1)
      and np.isclose(np.trace(P1), 3) and np.isclose(np.trace(P2), 5),
      "proyectores J = 0, 1, 2")

SIGMA = P0 / 3 + (2 / 3) * P1 / 3
TAU = SIGMA - np.eye(9) / 9
check(np.linalg.matrix_rank(SIGMA) == 4, "sigma tiene rango 4 < 9 (luego p <= 1)")

# --- MPS de AKLT ------------------------------------------------------------
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sz = np.diag([1.0, -1.0]).astype(complex)
A = np.array([np.sqrt(2 / 3) * sp, -np.sqrt(1 / 3) * sz, -np.sqrt(2 / 3) * sp.conj().T])
M_OF = np.array([1, 0, -1])


def vz(theta):
    return np.diag(np.exp(-1j * theta * np.array([0.5, -0.5])))


def Vphys(theta):
    return np.diag(np.exp(-1j * theta * M_OF))


th = 0.731
check(all(np.allclose(np.exp(-1j * th * M_OF[s]) * A[s], vz(th) @ A[s] @ vz(th).conj().T)
          for s in range(3)), "simetría U(1): e^{-i th m} A^m = v A^m v^†")


def target(n, Theta):
    V = np.kron(I3, Vphys(Theta / n))
    return np.eye(9) / 9 + V @ TAU @ V.conj().T


def marg_tm(n, Theta):
    """Marginal de dos sitios (cualquier arista, por invariancia) de
    psi = tr(B^{s0} ... B^{s_{n-1}}), B^s = A^s v, por matriz de transferencia."""
    B = np.array([a @ vz(Theta / n) for a in A])
    T = sum(np.kron(b, b.conj()) for b in B)
    Z = np.trace(np.linalg.matrix_power(T, n)).real
    Tn2 = np.linalg.matrix_power(T, n - 2)
    rho = np.zeros((9, 9), dtype=complex)
    for s in range(3):
        for t in range(3):
            for s2 in range(3):
                for t2 in range(3):
                    X = np.kron(B[s] @ B[t], (B[s2] @ B[t2]).conj())
                    rho[3 * s + t, 3 * s2 + t2] = np.trace(X @ Tn2)
    return rho / Z


def psi_bf(n, Theta):
    B = np.array([a @ vz(Theta / n) for a in A])
    psi = np.zeros(3 ** n, dtype=complex)
    for idx in range(3 ** n):
        digs = np.unravel_index(idx, [3] * n)
        M = np.eye(2, dtype=complex)
        for s in digs:
            M = M @ B[s]
        psi[idx] = np.trace(M)
    return psi / np.linalg.norm(psi)


def marg_bf(psi, n, j):
    t = psi.reshape([3] * n)
    k = (j + 1) % n
    rest = [a for a in range(n) if a not in (j, k)]
    t = np.transpose(t, [j, k] + rest).reshape(9, -1)
    return t @ t.conj().T


def trnorm(X):
    return np.abs(np.linalg.eigvalsh((X + X.conj().T) / 2)).sum()


# --- 2 y 3: fuerza bruta frente a matriz de transferencia -------------------
err_bf, err_p2 = 0.0, 0.0
for n in range(4, 10):
    for Theta in (0.0, np.pi / 3, np.pi):
        psi = psi_bf(n, Theta)
        rt = marg_tm(n, Theta)
        V = np.kron(I3, Vphys(Theta / n))
        for j in range(n):
            rj = marg_bf(psi, n, j)
            err_bf = max(err_bf, np.abs(rj - rt).max())
            # peso en J = 2 en el marco girado de la arista
            err_p2 = max(err_p2, abs(np.trace(V @ P2 @ V.conj().T @ rj)))
check(err_bf < 1e-12, f"marginales: fuerza bruta = matriz de transferencia, n = 4..9 "
      f"(todas las aristas iguales; error {err_bf:.1e})")
check(err_p2 < 1e-12, f"peso nulo en J = 2 en los marcos girados (máx {err_p2:.1e})")

# --- 4: eps_n ---------------------------------------------------------------
THETAS = np.linspace(0, np.pi, 13)
print("\n  n    max_Theta eps_n    eps_n*3^n   cota inferior 1 - s_n")
eps = {}
for n in list(range(4, 21)) + [25, 30]:
    e = max(trnorm(marg_tm(n, Th) - target(n, Th)) for Th in THETAS)
    eps[n] = e
    s = n * e / (1 / 54 + n * e)
    print(f"  {n:2d}   {e:.4e}       {e * 3**n:8.4f}     {1 - s:.10f}")
ratio = [eps[n] * 3 ** n for n in range(10, 21)]
check(max(ratio) < 17, f"eps_n * 3^n acotado (máx {max(ratio):.3f} en n = 10..20)")
check(eps[20] < 1e-8, f"eps_20 = {eps[20]:.2e}")

# --- 4b: fórmula exacta de eps_n --------------------------------------------
# El canal retorcido es normal, con autovectores I, sigma^z, sigma^± y
# autovalores 1, -1/3, -e^{∓i phi}/3. Tras conjugar por 1 ⊗ V la corrección no
# depende de n, y
#   eps_n(Theta) = f(Theta) / (3^n + (-1)^n (1 + 2 cos Theta)),
#   f(Theta) = (4/9) [2 (5 + c) + sqrt(2) sqrt(245 + 44 c - c^2)],  c = cos Theta,
# con máximo f(0) = 16. Luego eps_n <= 16 / (3^n - 3) y
#   p(n, Theta) >= 1 / (1 + 54 n eps_n) >= 1 / (1 + 864 n / (3^n - 3)).


def f_exacta(Theta):
    c = np.cos(Theta)
    return 4 / 9 * (2 * (5 + c) + np.sqrt(2) * np.sqrt(245 + 44 * c - c * c))


err_f = 0.0
for n in range(3, 21):  # más allá, el redondeo (~3^n 1e-16) domina
    for Th in np.linspace(0, np.pi, 25):
        e = trnorm(marg_tm(n, Th) - target(n, Th))
        e_ex = f_exacta(Th) / (3 ** n + (-1) ** n * (1 + 2 * np.cos(Th)))
        err_f = max(err_f, abs(e - e_ex) / e_ex)
check(err_f < 1e-6, f"eps_n = f(Theta)/(3^n + (-1)^n (1 + 2 cos Theta)), n = 3..20, "
      f"25 ángulos (error relativo {err_f:.1e})")
cs = np.linspace(-1, 1, 2001)
check(np.all(np.diff(f_exacta(np.arccos(cs))) > 0) and np.isclose(f_exacta(0), 16)
      and np.isclose(f_exacta(np.pi), 112 / 9),
      "f crece con cos Theta: máximo f(0) = 16, mínimo f(pi) = 112/9")

# --- 5: SDP exacto para n pequeño -------------------------------------------
try:
    import cvxpy as cp
    import scipy.sparse as sps
    if "--sin-sdp" in sys.argv:
        raise ImportError
except ImportError:
    print("(sin cvxpy: omito el SDP)")
    sys.exit(0 if OK else 1)


def sdp_p(n, Theta, solver="CLARABEL"):
    """max p con marginales exactas I/9 + p tau_j en las n aristas. Por la
    simetría U(1) de los datos hay un óptimo diagonal por bloques de M total.
    Restricciones: <k| rho_j |i> para i <= k con carga conservada (las demás
    son conjugadas o triviales); la traza la implican las diagonales."""
    dims = [3] * n
    digits = np.array([np.unravel_index(i, dims) for i in range(3 ** n)])
    Mtot = M_OF[digits].sum(axis=1)
    sectors = [np.where(Mtot == m)[0] for m in range(-n, n + 1)]
    pos = {}
    for b, sec in enumerate(sectors):
        for r, g in enumerate(sec):
            pos[g] = (b, r)
    X = [cp.Variable((len(sc), len(sc)), hermitian=True) for sc in sectors]
    p = cp.Variable()
    cons = [x >> 0 for x in X]
    V = np.kron(I3, Vphys(Theta / n))
    tauj = V @ TAU @ V.conj().T
    pair = 3 ** np.arange(n - 1, -1, -1)
    for j in range(n):
        k = (j + 1) % n
        loc = 3 * digits[:, j] + digits[:, k]
        for i in range(9):
            for q in range(i, 9):
                a, b = divmod(i, 3); c, d = divmod(q, 3)
                if M_OF[a] + M_OF[b] != M_OF[c] + M_OF[d]:
                    continue
                cols = np.where(loc == i)[0]
                rows = cols + (c - a) * pair[j] + (d - b) * pair[k]
                terms = {}
                for r, cc in zip(rows, cols):
                    br, rr = pos[r]; bc, rc = pos[cc]
                    assert br == bc
                    terms.setdefault(br, ([], []))
                    terms[br][0].append(rr); terms[br][1].append(rc)
                val = 0
                for br, (rr, rc) in terms.items():
                    m = len(sectors[br])
                    Msel = np.zeros((m, m)); Msel[rr, rc] = 1
                    val = val + cp.sum(cp.multiply(Msel, X[br]))
                rhs = np.eye(9)[q, i] / 9
                if i == q:
                    cons.append(cp.real(val) == rhs + p * tauj[q, i].real)
                else:
                    cons.append(val == rhs + p * tauj[q, i])
    prob = cp.Problem(cp.Maximize(p), cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
    return p.value, prob.status


print("\n  SDP exacto: p(n, Theta)")
sdp_ok = True
for n in (4, 5):  # n = 6 pide > 8 GB con esta formulación
    e = max(trnorm(marg_tm(n, Th) - target(n, Th)) for Th in THETAS)
    low = 1 - n * e / (1 / 54 + n * e)
    vals = []
    for Th in (0.0, np.pi / 2, np.pi):
        t0 = time.time()
        pv, st = sdp_p(n, Th)
        vals.append(pv)
        print(f"  n = {n}  Theta = {Th:.4f}  p = {pv:.8f}  ({st}, {time.time() - t0:.1f} s)")
        sdp_ok &= (low - 1e-6 <= pv <= 1 + 1e-6)
    print(f"         variación max-min = {max(vals) - min(vals):.3e}   cota 1 - s_n = {low:.4f}")
check(sdp_ok, "p(n, Theta) del SDP dentro de [1 - s_n, 1] en n = 4, 5")

sys.exit(0 if OK else 1)

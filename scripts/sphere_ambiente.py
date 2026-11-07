"""Contraste de sphere_series.py y sphere_casquetes.py sin pasar por holonomy.sphere: geometría
propia, gauge ambiente y testigos escritos con los vectores de la esfera, sin fases, sin
orientaciones de cara y sin argumentos de simetría.

En la arista x-y (vectores unitarios), n es la normal del arco, t_x = n × x, t_y = n × y y G la
rotación que lleva y a x. El testigo es W = -sum_e (a P + b Q + c Z), como forma bilineal en
las sigmas de los dos qubits, con
  P = t_x t_y^T + n n^T,  Q = t_x n^T - n t_y^T,  Z = x y^T,
y N(W) = sum_e tr(G^T (a P + b Q + c Z)) = sum_e (2 a + c).

Comprobaciones:
  1. N(W) = 2 a + c en cada arista (P, Q, Z y G están bien construidos);
  2. con (a, b, c) = (t, 0, 1/E - 2 t), el mínimo en t coincide con v_curvado de
     data/sphere_series.csv (en el cuboctaedro, con v_curvado_b0);
  3. dodecaedro, 20 qubits, a base de puertas: lambda_max en el t de la serie coincide con v_curvado,
     y a t +- 1e-3 es mayor;
  4. hemisferios del cubo, testigo por arista sin reducción por órbitas (pero con la forma U(1)), planos de
     corte con cotas a < 1e-6: el intervalo contiene el v de data/sphere_casquetes.csv, con y sin curvatura;
  5. cuboctaedro con b, orientando cada arista con su triángulo a la izquierda: lambda_max en (t, b) de la serie
     coincide con v_curvado;
  6. casquete G1 del dodecaedro: el testigo de data/sphere_casquetes.csv, reescrito en esta geometría, da su v:
     es una cota superior;
  7. T_1u del cubo (flujos +-s en dos caras opuestas), con fases propias: SDP para el testigo por órbitas y
     lambda_max densa, cota superior; en s = 0,025 queda por debajo de v0.

Las cotas superiores no dependen de la reducción por simetría; salvo en 4, estos contrastes no comprueban que el
testigo reducido sea óptimo.

Uso: (ulimit -v 25000000; python scripts/sphere_ambiente.py)
"""
import _common  # noqa: F401
from _common import Checker

import csv
import itertools
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import LinearOperator, eigsh

DATA = Path(__file__).resolve().parent.parent / "data"
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]
PHI = (1 + 5**0.5) / 2
TOL = 1e-6
# Nombre de la serie -> nombre de la columna de contraste.
SERIE = {"tetraedro": "v_curvado", "octaedro": "v_curvado", "cubo": "v_curvado",
         "cuboctaedro": "v_curvado_b0", "icosaedro": "v_curvado", "rombododecaedro": "v_curvado"}


def vertices(nombre):
    cubo = list(itertools.product((1, -1), repeat=3))
    octa = [s * np.eye(3)[q] for q in range(3) for s in (1, -1)]
    if nombre == "tetraedro":
        return np.array([p for p in cubo if p[0] * p[1] * p[2] == 1], float)
    if nombre == "octaedro":
        return np.array(octa)
    if nombre == "cubo":
        return np.array(cubo, float)
    if nombre == "cuboctaedro":
        return np.array([p for p in itertools.product((-1, 0, 1), repeat=3)
                         if sum(map(abs, p)) == 2], float)
    if nombre == "rombododecaedro":
        return np.vstack([np.array(cubo, float), 2 * np.array(octa)])
    pares = list(itertools.product((1, -1), repeat=2))
    if nombre == "icosaedro":
        return np.array([w for a, b in pares for w in ([0, a, b * PHI], [a, b * PHI, 0], [b * PHI, 0, a])])
    if nombre == "dodecaedro":
        return np.array([list(p) for p in cubo]
                        + [w for a, b in pares for w in ([0, a / PHI, b * PHI], [a / PHI, b * PHI, 0],
                                                         [b * PHI, 0, a / PHI])], float)
    raise ValueError(nombre)


def geometria(nombre):
    """Vectores unitarios y aristas (pares a distancia mínima, sin normalizar)."""
    V = vertices(nombre)
    d = np.linalg.norm(V[:, None] - V[None], axis=2)
    m = d[d > 1e-9].min()
    E = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V)) if abs(d[i, j] - m) < 1e-9]
    return V / np.linalg.norm(V, axis=1, keepdims=True), E


def marco(x, y):
    """(P, Q, Z, G) de la arista x-y."""
    n = np.cross(x, y)
    n /= np.linalg.norm(n)
    tx, ty = np.cross(n, x), np.cross(n, y)
    P = np.outer(tx, ty) + np.outer(n, n)
    Q = np.outer(tx, n) - np.outer(n, ty)
    Z = np.outer(x, y)
    return P, Q, Z, P + Z


MARCO_TRIVIAL = (np.diag([1.0, 1, 0]), np.array([[0, -1.0, 0], [1, 0, 0], [0, 0, 0]]),
                 np.diag([0, 0, 1.0]), np.eye(3))


class Base:
    """Operadores -sum_ab Y_ab sigma_a^i sigma_b^j de cada (arista, P | Q | Z), dispersos."""

    def __init__(self, n, E, marcos):
        S = {(q, a): sp.kron(sp.kron(sp.identity(2**q, format="csr", dtype=complex),
                                     sp.csr_matrix(PAULI[a])),
                             sp.identity(2**(n - q - 1), format="csr", dtype=complex), format="csr")
             for q in range(n) for a in range(3)}
        self.M, self.norma = [], []
        for (i, j), m in zip(E, marcos, strict=True):
            pre = {(a, b): S[i, a] @ S[j, b] for a in range(3) for b in range(3)}
            for Y, nv in zip(m[:3], (2.0, 0.0, 1.0), strict=True):
                self.M.append(sp.csr_matrix(sum(-Y[a, b] * pre[a, b] for a in range(3) for b in range(3)
                                                if abs(Y[a, b]) > 1e-14)))
                self.norma.append(nv)
        self.norma = np.array(self.norma)


def tope(W, k=4):
    """Autovalores y autovectores mayores, en orden creciente. Sin arranque en caliente: W
    conserva simetrías y Lanczos no saldría del sector del vector anterior."""
    if W.shape[0] <= 256:
        w, U = np.linalg.eigh(W.toarray())
        return w[-k:], U[:, -k:]
    w, U = eigsh(W, k=k, which="LA", tol=1e-10, ncv=40)
    o = np.argsort(w)
    return w[o], U[:, o]


def planos_de_corte(base, it_max=4000):
    """(inf, sup): el LP de los cortes y el mejor lambda_max con N = 1."""
    nv = len(base.M)
    y = base.norma / (base.norma @ base.norma)
    A, inf, sup, y_mejor = [], -np.inf, np.inf, y
    for it in range(it_max):
        w, U = tope(sum(c * m for c, m in zip(y, base.M, strict=True) if c != 0))
        if w[-1] < sup:
            sup, y_mejor = w[-1], y
        A += [[np.real(np.vdot(u, m @ u)) for m in base.M] + [-1.0] for u in U.T]
        res = linprog(np.r_[np.zeros(nv), 1.0], A_ub=np.array(A), b_ub=np.zeros(len(A)),
                      A_eq=[np.r_[base.norma, 0.0]], b_eq=[1.0],
                      bounds=[(-10, 10)] * nv + [(None, None)], method="highs")
        inf = max(inf, res.x[-1])
        if sup - inf < TOL / 2:
            break
        y = (y_mejor + res.x[:nv]) / 2 if it % 2 == 0 else res.x[:nv]
    return inf, sup


def dorada(f, a, b, tol=1e-7):
    g = (5**0.5 - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return min(fc, fd)


def lmax_puertas(n, E, Ys):
    """lambda_max de sum_e Y_ab sigma_a sigma_b, aplicando cada arista como puerta de dos qubits."""
    G = [(i, j, sum(Y[a, b] * np.kron(PAULI[a], PAULI[b]) for a in range(3)
                    for b in range(3)).reshape(2, 2, 2, 2))
         for (i, j), Y in zip(E, Ys, strict=True)]

    def mv(psi):
        psi = psi.reshape((2,) * n)
        out = np.zeros_like(psi)
        for i, j, g in G:
            out += np.moveaxis(np.tensordot(g, psi, axes=([2, 3], [i, j])), [0, 1], [i, j])
        return out.ravel()

    return eigsh(LinearOperator((2**n, 2**n), matvec=mv, dtype=complex), k=1, which="LA", tol=1e-9,
                 ncv=30)[0][0]


def cuboctaedro_b(t, b):
    X, E = geometria("cuboctaedro")
    aristas = {frozenset(e) for e in E}
    ori = []
    for i, j, k in itertools.combinations(range(12), 3):
        if {frozenset(e) for e in ((i, j), (j, k), (i, k))} <= aristas:
            c = X[i] + X[j] + X[k]
            tri = (i, j, k) if np.cross(X[j] - X[i], X[k] - X[i]) @ c > 0 else (i, k, j)
            ori += [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
    return lmax_puertas(12, ori, [-(t * P + b * Q + (1 / 24 - 2 * t) * Z)
                                  for P, Q, Z, _ in (marco(X[i], X[j]) for i, j in ori)])


def casquete_dodecaedro(y):
    """lambda_max y N del testigo por órbitas (pentágono a0 b0 c0, radial a1 c1, decágono a2 b2 c2)."""
    X, E = geometria("dodecaedro")
    adj = {frozenset(e) for e in E}
    cara = next(c for c in itertools.permutations(range(20), 5)
                if all(frozenset((c[q], c[(q + 1) % 5])) in adj for q in range(5)))
    eje = X[list(cara)].mean(axis=0)
    eje /= np.linalg.norm(eje)
    orden = np.argsort(-(X @ eje))
    r = [list(orden[5 * q:5 * q + 5]) for q in range(4)]
    e1 = np.cross(eje, [1, 0, 0]) if abs(eje[0]) < 0.9 else np.cross(eje, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(eje, e1)
    ang = lambda v: np.arctan2(X[v] @ e2, X[v] @ e1)
    ciclico = lambda a, b: (a, b) if np.mod(ang(b) - ang(a), 2 * np.pi) < np.pi else (b, a)
    pent = [ciclico(a, b) for a, b in itertools.combinations(r[0], 2) if frozenset((a, b)) in adj]
    radial = [(a, b) for a in r[0] for b in r[1] if frozenset((a, b)) in adj]
    deca = [ciclico(a, b) for a in r[1] for b in r[2] if frozenset((a, b)) in adj]
    pos = {v: k for k, v in enumerate(r[0] + r[1] + r[2])}
    aristas = pent + radial + deca
    Ys = []
    for q, (i, j) in enumerate(aristas):
        P, Q, Z, _ = marco(X[i], X[j])
        o = 0 if q < 5 else 1 if q < 10 else 2
        Ys.append(-(y[f"a{o}"] * P + (0.0 if o == 1 else y[f"b{o}"]) * Q + y[f"c{o}"] * Z))
    N = sum(2 * y[f"a{o}"] + y[f"c{o}"] for o in [0] * 5 + [1] * 5 + [2] * 10)
    return lmax_puertas(15, [(pos[i], pos[j]) for i, j in aristas], Ys), N


def t1u_cubo(chk, ref_inf):
    """Cubo con fases propias: flujos 2 pi/3 +- s en las caras z = +-1."""
    V = vertices("cubo")
    idx = {tuple(v): q for q, v in enumerate(V)}
    giro = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    arr, aba = [idx[x, y, 1.0] for x, y in giro], [idx[x, y, -1.0] for x, y in giro]
    E = ([(arr[q], arr[(q + 1) % 4]) for q in range(4)] + [(aba[q], aba[(q + 1) % 4]) for q in range(4)]
         + [(aba[q], arr[q]) for q in range(4)])
    pos = {e: k for k, e in enumerate(E)}
    caras = [arr, aba[::-1]] + [[aba[q], aba[(q + 1) % 4], arr[(q + 1) % 4], arr[q]] for q in range(4)]
    D = np.zeros((6, 12))
    for c, cara in enumerate(caras):
        for u, w in zip(cara, cara[1:] + cara[:1], strict=True):
            if (u, w) in pos:
                D[c, pos[u, w]] += 1
            else:
                D[c, pos[w, u]] -= 1
    sect = {}
    for q in range(256):
        sect.setdefault(bin(q).count("1"), []).append(q)

    def operadores(s):
        f = np.full(6, 2 * np.pi / 3)
        f[0] += s
        f[1] -= s
        f[2] -= 2 * np.pi
        f[4] -= 2 * np.pi
        ph = np.linalg.lstsq(D, f, rcond=None)[0]
        J = np.array([[0, -1.0, 0], [1, 0, 0], [0, 0, 0]])
        marcos = []
        for x in ph:
            R = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 0]])
            marcos.append((R, R @ J, np.diag([0, 0, 1.0])))
        B = Base(8, E, marcos)
        M, N = [], []
        for o, ar in enumerate((range(4), range(4, 8), range(8, 12))):
            for q, nv in enumerate((2.0, 0.0, 1.0)):
                if o == 2 and q == 1:
                    continue
                M.append(sum(B.M[3 * e + q] for e in ar).toarray())
                N.append(nv * 4)
        return M, np.array(N)

    def cota(s):
        M, N = operadores(s)
        y, lam = cp.Variable(len(M)), cp.Variable()
        cons = [N @ y == 1]
        for ind in sect.values():
            Bl = [m[np.ix_(ind, ind)] for m in M]
            X_ = sum(y[q] * np.block([[b.real, -b.imag], [b.imag, b.real]]) for q, b in enumerate(Bl))
            cons.append(lam * np.eye(2 * len(ind)) - (X_ + X_.T) / 2 >> 0)
        cp.Problem(cp.Minimize(lam), cons).solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                                                 tol_feas=1e-10, max_iter=200)
        yv = y.value / (N @ y.value)
        return np.linalg.eigvalsh(sum(c * m for c, m in zip(yv, M, strict=True)))[-1]

    for s in (0.01, 0.025):
        v = cota(s)
        chk.info(f"T_1u, s = {s}: v <= {v:.11f}")
    chk.check(v < ref_inf, f"T_1u en s = 0,025: v <= {v:.11f} < {ref_inf:.11f} <= v0 (uniforme-j2 de los flujos)")
    v1 = cota(0.1)
    chk.info(f"T_1u, s = 0,1: v <= {v1:.11f}")


def main():
    chk = Checker("sphere_ambiente")
    serie = {r["nombre"]: r for r in csv.DictReader(open(DATA / "sphere_series.csv"))}
    casq = {(r["poliedro"], r["subgrafo"], r["caso"]): float(r["v"])
            for r in csv.DictReader(open(DATA / "sphere_casquetes.csv"))}

    X, E = geometria("dodecaedro")
    err = 0.0
    for i, j in E:
        P, Q, Z, G = marco(X[i], X[j])
        err = max(err, abs(np.trace(G.T @ P) - 2), abs(np.trace(G.T @ Q)), abs(np.trace(G.T @ Z) - 1),
                  np.linalg.norm(G @ X[j] - X[i]), np.linalg.norm(G.T @ G - np.eye(3)))
    chk.check(err < 1e-12, f"G lleva y a x, G ortogonal y N = 2 a + c en el dodecaedro, |dif| = {err:.1e}")

    for nombre, col in SERIE.items():
        X, E = geometria(nombre)
        base = Base(len(X), E, [marco(X[i], X[j]) for i, j in E])
        H, C = sum(base.M[3 * k] for k in range(len(E))), sum(base.M[3 * k + 2] for k in range(len(E)))
        v = dorada(lambda t, H=H, C=C, e=len(E): tope(t * H + (1 / e - 2 * t) * C, k=1)[0][-1], -0.1, 0.2)
        ref = float(serie[nombre][col])
        chk.check(abs(v - ref) < TOL, f"{nombre}: v ambiente {v:.8f}, {col} {ref:.8f}, |dif| = {abs(v - ref):.1e}")

    X, E = geometria("dodecaedro")
    marcos = [marco(X[i], X[j]) for i, j in E]
    t0, ref = float(serie["dodecaedro"]["t_curvado"]), float(serie["dodecaedro"]["v_curvado"])
    vals = [lmax_puertas(20, E, [-(t * m[0] + (1 / 30 - 2 * t) * m[2]) for m in marcos])
            for t in (t0, t0 - 1e-3, t0 + 1e-3)]
    chk.check(abs(vals[0] - ref) < TOL and min(vals[1:]) > vals[0],
              f"dodecaedro, 20 qubits: lambda_max {vals[0]:.8f} (serie {ref:.8f}), a t +- 1e-3 "
              f"{vals[1]:.8f} y {vals[2]:.8f}")

    X, E = geometria("cubo")
    for g, quita in (("H1", (-1, -1, -1)), ("H2", (1, 1, 1))):
        q = next(k for k, p in enumerate(vertices("cubo")) if tuple(p) == quita)
        pos = {v: k for k, v in enumerate(v for v in range(8) if v != q)}
        Es = [(i, j) for i, j in E if q not in (i, j)]
        Er = [(pos[i], pos[j]) for i, j in Es]
        for caso, marcos in (("trivial", [MARCO_TRIVIAL] * 9),
                             ("curvado", [marco(X[i], X[j]) for i, j in Es])):
            inf, sup = planos_de_corte(Base(7, Er, marcos))
            ref = casq["cubo", g, caso]
            chk.check(inf - 1e-7 < ref < sup + 1e-7 and sup - inf < TOL,
                      f"cubo {g} {caso}: [{inf:.8f}, {sup:.8f}], data/sphere_casquetes.csv {ref:.8f}")

    t, b = float(serie["cuboctaedro"]["t_curvado"]), float(serie["cuboctaedro"]["b_curvado"])
    ref = float(serie["cuboctaedro"]["v_curvado"])
    v = cuboctaedro_b(t, b)
    chk.check(abs(v - ref) < TOL, f"cuboctaedro con b: lambda_max {v:.8f}, v_curvado {ref:.8f}")

    fila = next(r for r in csv.DictReader(open(DATA / "sphere_casquetes.csv"))
                if (r["poliedro"], r["subgrafo"], r["caso"]) == ("dodecaedro", "G1", "curvado"))
    y = dict(zip(fila["parametros"].split(";"), map(float, fila["y"].split(";")), strict=True))
    v, N = casquete_dodecaedro(y)
    chk.check(abs(N - 1) < 1e-9 and abs(v - float(fila["v"])) < TOL,
              f"casquete del dodecaedro: testigo del CSV, N = {N:.10f}, lambda_max {v:.8f}, CSV {float(fila['v']):.8f}")

    resumen = {(r["poliedro"], r["fase"]): r for r in csv.DictReader(open(DATA / "sphere_flujos_resumen.csv"))}
    t1u_cubo(chk, float(resumen["cubo", "uniforme-j2"]["v_inf"]))
    chk.finish()


if __name__ == "__main__":
    main()

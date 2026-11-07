"""Reparto del defecto de pegado entre lo que se ve en el solape y lo que no.

v_O(G1, G2) es el mayor v para el que hay estados de G1 y de G2 con sus marginales de Werner y la
misma matriz densidad en todo el solape O. Se cumple v(G1 unión G2) <= v_O <= min(v(G1), v(G2)), y
  delta = [min(v(G1), v(G2)) - v_O] + [v_O - v(G1 unión G2)]:
el primer término se ve al comparar el solape, y el segundo no.

Cotas. Inferior: SDP primal en bloques de Sz, reparado a un par exacto de estados (se fuerza cada
restricción, que son Paulis ortogonales, y se mezcla con I/d, lo que da v' = (1 - p) v). Superior: para
W1, W2 testigos de G1, G2 y Z hermítico en O, v_O (N1 + N2) <= lambda_max(W1 + Z) + lambda_max(W2 - Z);
(W1, W2, Z) sale de un SDP y lambda_max se reevalúa por diagonalización densa, así que vale aunque el solver
termine en 'optimal_inaccurate'. Si la cota inferior de v_O supera v(unión), no hay estado global: G1 unión G2
cubre todas las aristas.

Comprobaciones:
  1. anillo de 4 con dos cadenas de 3, solape de los dos vértices opuestos: v_O = 2/(3 + sin(th/2)),
     el Teorema 3 de anillos, a 1e-5 (th = 0, 1, 2, pi);
  2. cubo: sin ligar el solape se reproduce v de data/sphere_casquetes.csv a 1e-6;
  3. v(unión) <= v_O <= min(v(G1), v(G2)), a 1e-6;
  4. residuos de la solución, recalculados con matrices completas: traza, menor autovalor, diferencia
     de las marginales del solape y correlaciones de arista frente a -v R_z(fase);
  5. par reparado con residuo < 1e-13 y cotas inferior y superior a menos de 2e-6;
  6. hemisferios curvados: el estado óptimo de H1 y su imagen por la antípoda son factibles a v(H1) y sus marginales
     en el solape tienen el mismo espectro, aunque difieren como matrices: con la igualdad salvo un unitario
     lo visible desaparece.

El dodecaedro (15 qubits por lado) queda fuera de alcance. Escribe data/sphere_solape.csv.

Uso: (ulimit -v 22000000; python scripts/sphere_solape.py)
"""
import _common  # noqa: F401
from _common import Checker

import csv
import itertools
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from holonomy import sphere
from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
PILOTO_XYZ = {0: (1, 1, 1), 1: (1, -1, 1), 2: (-1, -1, 1), 3: (-1, 1, 1),
              4: (1, 1, -1), 5: (1, -1, -1), 6: (-1, -1, -1), 7: (-1, 1, -1)}
PAULI = P = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]], complex)]
TOL = 1e-6


def sectores(n):
    out = {}
    for s in range(2**n):
        out.setdefault(bin(s).count("1"), []).append(s)
    return out


class Lado:
    """Un estado de n qubits en bloques de Sz (el bit q de la base es el qubit q)."""

    def __init__(self, n, aristas, fases):
        self.n, self.aristas, self.fases = n, aristas, fases
        self.secs = sectores(n)
        self.rho = {k: cp.Variable((len(b), len(b)), hermitian=True) for k, b in self.secs.items()}
        self.pos = {k: {s: q for q, s in enumerate(b)} for k, b in self.secs.items()}

    def diag(self, f):
        return sum(cp.real(cp.sum(cp.multiply(np.array([f(s) for s in b], float), cp.diag(self.rho[k]))))
                   for k, b in self.secs.items())

    def restricciones(self, v, fijos=()):
        """Sin las redundantes: los qubits de `fijos` ya están fijados por el otro lado."""
        cons = [r >> 0 for r in self.rho.values()]
        if not fijos:
            cons.append(self.diag(lambda s: 1.0) == 1)
        for i in range(self.n):
            if i not in fijos:
                cons.append(self.diag(lambda s, i=i: 1.0 - 2 * ((s >> i) & 1)) == 0)
        for (i, j), f in zip(self.aristas, self.fases, strict=True):
            if i in fijos and j in fijos:
                continue
            cons.append(self.diag(lambda s, i=i, j=j: (1.0 - 2 * ((s >> i) & 1))
                                  * (1.0 - 2 * ((s >> j) & 1))) == -v)
            salto = 0
            for k, b in self.secs.items():
                for s in b:
                    if (s >> i) & 1 and not (s >> j) & 1:
                        salto = salto + self.rho[k][self.pos[k][s ^ (1 << i) ^ (1 << j)], self.pos[k][s]]
            cons += [cp.real(salto) == -v * np.cos(f) / 2, cp.imag(salto) == v * np.sin(f) / 2]
        return cons

    def marginal(self, ov):
        """{(a, b): expresión} de la marginal en los qubits ov, con a <= b en el mismo sector."""
        m = len(ov)
        fuera = [q for q in range(self.n) if q not in ov]
        out = {}
        for a, b in itertools.product(range(2**m), repeat=2):
            if a > b or bin(a).count("1") != bin(b).count("1"):
                continue
            ex = 0
            for w in range(2**len(fuera)):
                resto = sum(((w >> q) & 1) << fuera[q] for q in range(len(fuera)))
                sa = sum(((a >> q) & 1) << ov[q] for q in range(m)) + resto
                sb = sum(((b >> q) & 1) << ov[q] for q in range(m)) + resto
                k = bin(sa).count("1")
                ex = ex + self.rho[k][self.pos[k][sa], self.pos[k][sb]]
            out[a, b] = ex
        return out

    def completa(self):
        R = np.zeros((2**self.n, 2**self.n), complex)
        for k, b in self.secs.items():
            R[np.ix_(b, b)] = self.rho[k].value
        return R


def operador(n, q, a):
    """Pauli a en el qubit q (bit q de la base)."""
    M = np.eye(1, dtype=complex)
    for r in range(n - 1, -1, -1):
        M = np.kron(M, PAULI[a] if r == q else np.eye(2))
    return M


def marginal_numerica(R, n, ov):
    m = len(ov)
    fuera = [q for q in range(n) if q not in ov]
    out = np.zeros((2**m, 2**m), complex)
    for a, b in itertools.product(range(2**m), repeat=2):
        for w in range(2**len(fuera)):
            resto = sum(((w >> q) & 1) << fuera[q] for q in range(len(fuera)))
            out[a, b] += R[sum(((a >> q) & 1) << ov[q] for q in range(m)) + resto,
                           sum(((b >> q) & 1) << ov[q] for q in range(m)) + resto]
    return out


def residuos(lado, v):
    """Traza, menor autovalor y error de T^e frente a -v R_z(fase), con matrices completas."""
    R = lado.completa()
    err = 0.0
    for (i, j), f in zip(lado.aristas, lado.fases, strict=True):
        T = np.array([[np.real(np.trace(R @ operador(lado.n, i, a) @ operador(lado.n, j, b)))
                       for b in range(3)] for a in range(3)])
        c, s = np.cos(f), np.sin(f)
        err = max(err, np.abs(T + v * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])).max())
    return abs(np.trace(R) - 1), np.linalg.eigvalsh((R + R.conj().T) / 2)[0], err


def resuelve(n1, E1, f1, n2, E2, f2, ov1=None, ov2=None):
    """Devuelve (v, estado, residuos). Sin ov1, los dos lados no se ligan."""
    l1, l2 = Lado(n1, E1, f1), Lado(n2, E2, f2)
    v = cp.Variable()
    cons = l1.restricciones(v) + l2.restricciones(v, tuple(ov2) if ov1 is not None else ())
    if ov1 is not None:
        t1, t2 = l1.marginal(ov1), l2.marginal(ov2)
        for (a, b), x in t1.items():
            cons.append(cp.real(x) == cp.real(t2[a, b]) if a == b else x == t2[a, b])
    prob = cp.Problem(cp.Maximize(v), cons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7, max_iter=300)
    val = float(v.value)
    tr1, ev1, e1 = residuos(l1, val)
    tr2, ev2, e2 = residuos(l2, val)
    dm = 0.0
    if ov1 is not None:
        dm = np.abs(marginal_numerica(l1.completa(), n1, ov1) - marginal_numerica(l2.completa(), n2, ov2)).max()
    return val, prob.status, max(tr1, tr2), min(ev1, ev2), max(e1, e2), dm, l1, l2


def op(n, q, M):
    """M en el qubit q; bit q de la base = qubit q (convenio de sphere_solape)."""
    out = sp.identity(1, format="csr", dtype=complex)
    for r in range(n - 1, -1, -1):
        out = sp.kron(out, sp.csr_matrix(M) if r == q else sp.identity(2, format="csr"), format="csr")
    return out


def edge_ops(n, i, j):
    X = [op(n, i, M) for M in P]
    Y = [op(n, j, M) for M in P]
    A = -(X[0] @ Y[0] + X[1] @ Y[1])
    B = X[0] @ Y[1] - X[1] @ Y[0]
    C = -(X[2] @ Y[2])
    return A, B, C


def overlap_basis(n, ov):
    """Hermíticos U(1)-invariantes |a><b| en O, sin un elemento diagonal (quita la dirección identidad)."""
    m = len(ov)
    fuera = [q for q in range(n) if q not in ov]
    def expand(a, b):
        rows, cols = [], []
        for w in range(2**len(fuera)):
            resto = sum(((w >> q) & 1) << fuera[q] for q in range(len(fuera)))
            rows.append(sum(((a >> q) & 1) << ov[q] for q in range(m)) + resto)
            cols.append(sum(((b >> q) & 1) << ov[q] for q in range(m)) + resto)
        return sp.csr_matrix((np.ones(len(rows), complex), (rows, cols)), shape=(2**n, 2**n))
    out = []
    for a, b in itertools.product(range(2**m), repeat=2):
        if a > b or bin(a).count("1") != bin(b).count("1") or (a == 0 and b == 0):
            continue
        E = expand(a, b)
        if a == b:
            out.append(E)
        else:
            out.append(E + E.conj().T)
            out.append(1j * (E - E.conj().T))
    return out


def sectors(n):
    d = {}
    for s in range(2**n):
        d.setdefault(bin(s).count("1"), []).append(s)
    return d


def solve_dual(n1, E1, f1, n2, E2, f2, ov1, ov2):
    """Mismo qubit de O: ov1[q] en lado 1 y ov2[q] en lado 2."""
    Zs1 = overlap_basis(n1, ov1)
    Zs2 = overlap_basis(n2, ov2)
    ops = []  # (side, operator, norm); Z entra con signo opuesto
    terms1, terms2, norm = [], [], []
    for (i, j), f in zip(E1, f1):
        A, B, C = edge_ops(n1, i, j)
        for M, nv in ((A, 2 * np.cos(f)), (B, 2 * np.sin(f)), (C, 1.0)):
            terms1.append(M); terms2.append(None); norm.append(nv)
    for (i, j), f in zip(E2, f2):
        A, B, C = edge_ops(n2, i, j)
        for M, nv in ((A, 2 * np.cos(f)), (B, 2 * np.sin(f)), (C, 1.0)):
            terms1.append(None); terms2.append(M); norm.append(nv)
    for z1, z2 in zip(Zs1, Zs2, strict=True):
        terms1.append(z1); terms2.append(-z2); norm.append(0.0)
    norm = np.array(norm)
    nv = len(norm)
    x = cp.Variable(nv)
    l1, l2 = cp.Variable(), cp.Variable()
    cons = [norm @ x == 1]
    for n, terms, lam in ((n1, terms1, l1), (n2, terms2, l2)):
        for k, b in sectors(n).items():
            d = len(b)
            idx = np.array(b)
            rows, cols, vals = [], [], []
            for q, M in enumerate(terms):
                if M is None:
                    continue
                S = M[idx][:, idx].tocoo()
                for r, c, val in zip(S.row, S.col, S.data):
                    # bloque real [[Re, -Im], [Im, Re]], vec por columnas
                    for (rr, cc, vv) in ((r, c, val.real), (r + d, c + d, val.real),
                                         (r, c + d, -val.imag), (r + d, c, val.imag)):
                        if vv != 0:
                            rows.append(rr + 2 * d * cc); cols.append(q); vals.append(vv)
            if not rows:
                continue
            Am = sp.csr_matrix((vals, (rows, cols)), shape=((2 * d)**2, nv))
            Mexp = cp.reshape(Am @ x, (2 * d, 2 * d), order="F")
            cons.append(lam * np.eye(2 * d) - (Mexp + Mexp.T) / 2 >> 0)
    prob = cp.Problem(cp.Minimize(l1 + l2), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, max_iter=300)
    y = x.value
    # Reevaluación exacta (flotante): operadores completos y diagonalización densa.
    W1 = sum(c * M for c, M in zip(y, terms1) if M is not None).toarray()
    W2 = sum(c * M for c, M in zip(y, terms2) if M is not None).toarray()
    W1, W2 = (W1 + W1.conj().T) / 2, (W2 + W2.conj().T) / 2
    lam1, lam2 = np.linalg.eigvalsh(W1)[-1], np.linalg.eigvalsh(W2)[-1]
    N = float(norm @ y)
    return (lam1 + lam2) / N, prob.status, N, prob.value


def repair(l1, l2, n1, n2, ov1, ov2, v, f1, f2, E1, E2):
    """Par exacto de estados a partir de la solución del solver. Devuelve (v', residuo, p)."""
    R1, R2 = l1.completa(), l2.completa()
    R1, R2 = (R1 + R1.conj().T) / 2, (R2 + R2.conj().T) / 2
    def pauli_full(n, qs, ps):
        M = sp.identity(2**n, format="csr", dtype=complex)
        for q, a in zip(qs, ps):
            M = M @ op(n, q, P[a])
        return M
    def targets(E, F, n):
        T = {}
        for q in range(n):
            for a in range(3):
                T[((q,), (a,))] = 0.0
        for (i, j), f in zip(E, F):
            c, s = np.cos(f), np.sin(f)
            Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for a in range(3):
                for b in range(3):
                    T[((i, j), (a, b))] = -v * Rm[a, b]
        return T
    # 1) marginal común de O: media, luego correcciones de Pauli de las restricciones en O
    m = len(ov1)
    def marg(R, n, ov):
        fuera = [q for q in range(n) if q not in ov]
        out = np.zeros((2**m, 2**m), complex)
        for a, b in itertools.product(range(2**m), repeat=2):
            for w in range(2**len(fuera)):
                resto = sum(((w >> q) & 1) << fuera[q] for q in range(len(fuera)))
                out[a, b] += R[sum(((a >> q) & 1) << ov[q] for q in range(m)) + resto,
                               sum(((b >> q) & 1) << ov[q] for q in range(m)) + resto]
        return out
    M1, M2 = marg(R1, n1, ov1), marg(R2, n2, ov2)
    Mt = (M1 + M2) / 2
    Mt = Mt / np.trace(Mt)
    T1 = targets(E1, f1, n1)
    pos1 = {q: k for k, q in enumerate(ov1)}
    for (qs, ps), t in T1.items():
        if all(q in pos1 for q in qs):
            Pm = pauli_full(m, [pos1[q] for q in qs], ps).toarray()
            Mt = Mt + (t - np.trace(Mt @ Pm).real) * Pm / 2**m
    def lift(Del, n, ov):
        fuera = [q for q in range(n) if q not in ov]
        out = np.zeros((2**n, 2**n), complex)
        for a, b in itertools.product(range(2**m), repeat=2):
            if Del[a, b] == 0:
                continue
            for w in range(2**len(fuera)):
                resto = sum(((w >> q) & 1) << fuera[q] for q in range(len(fuera)))
                out[sum(((a >> q) & 1) << ov[q] for q in range(m)) + resto,
                    sum(((b >> q) & 1) << ov[q] for q in range(m)) + resto] = Del[a, b]
        return out / 2**len(fuera)
    R1 = R1 + lift(Mt - M1, n1, ov1)
    R2 = R2 + lift(Mt - M2, n2, ov2)
    # 2) restricciones con algún qubit fuera de O
    for R_, E, F, n, ov, name in ((R1, E1, f1, n1, ov1, 1), (R2, E2, f2, n2, ov2, 2)):
        T = targets(E, F, n)
        for (qs, ps), t in T.items():
            if not all(q in ov for q in qs):
                Pm = pauli_full(n, qs, ps).toarray()
                R_ += (t - np.trace(R_ @ Pm).real) * Pm / 2**n
    # 3) positividad por mezcla con I/d: conserva todas las igualdades con v -> (1-p) v
    e1 = np.linalg.eigvalsh(R1)[0]
    e2 = np.linalg.eigvalsh(R2)[0]
    d1, d2 = 2**n1, 2**n2
    p = max(0.0, -e1 * d1 / (1 - e1 * d1), -e2 * d2 / (1 - e2 * d2))
    R1f = (1 - p) * R1 + p * np.eye(d1) / d1
    R2f = (1 - p) * R2 + p * np.eye(d2) / d2
    vf = (1 - p) * v
    # residuos finales, recalculados
    res = 0.0
    for R_, E, F, n in ((R1f, E1, f1, n1), (R2f, E2, f2, n2)):
        res = max(res, abs(np.trace(R_) - 1))
        for (qs, ps), t in targets(E, F, n).items():
            Pm = pauli_full(n, qs, ps).toarray()
            res = max(res, abs(np.trace(R_ @ Pm).real - (1 - p) * t))
    res = max(res, np.abs(marg(R1f, n1, ov1) - marg(R2f, n2, ov2)).max())
    return vf, res, p, min(np.linalg.eigvalsh(R1f)[0], np.linalg.eigvalsh(R2f)[0])


def antipoda(chk, p, phis, vs1, vs2):
    """Comprobación 6: la imagen del óptimo de H1 por la antípoda, compuesta con un gauge y R_x(pi)."""
    g = next(x for x in sphere.symmetry_group(p) if np.allclose(x.M, -np.eye(3)))
    fase = {}
    for (i, j), f in zip(p.edges, phis, strict=True):
        fase[i, j], fase[j, i] = f, -f
    th, pila = {0: 0.0}, [0]
    while pila:
        i = pila.pop()
        for (a, b), f in fase.items():
            if a == i and b not in th:
                th[b] = th[a] - fase[g.perm[a], g.perm[b]] - f
                pila.append(b)
    s1, f1, _ = sphere.induced_subgraph(p, phis, vs1)
    s2, f2, _ = sphere.induced_subgraph(p, phis, vs2)
    l = Lado(s1.n, list(s1.edges), f1)
    v = cp.Variable()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp.Problem(cp.Maximize(v), l.restricciones(v)).solve(solver=cp.CLARABEL, tol_gap_abs=1e-8, tol_gap_rel=1e-8,
                                                             tol_feas=1e-8)
    val, R1 = float(v.value), l.completa()
    n, sv1, sv2 = s1.n, sorted(vs1), sorted(vs2)
    Pm = np.zeros((2**n, 2**n))
    for x in range(2**n):
        Pm[sum(1 << sv2.index(g.perm[w]) for k, w in enumerate(sv1) if (x >> k) & 1), x] = 1
    mejor = None
    for signo in (1, -1):
        U = np.eye(2**n, dtype=complex)
        for k, w in enumerate(sv1):
            Rz = np.diag([np.exp(-1j * signo * th[w] / 2), np.exp(1j * signo * th[w] / 2)])
            G = np.eye(1, dtype=complex)
            for r in range(n - 1, -1, -1):
                G = np.kron(G, PAULI[0] @ Rz if r == k else np.eye(2))
            U = G @ U
        R2 = Pm @ U @ R1 @ U.conj().T @ Pm.T
        res = 0.0
        for (i, j), f in zip(s2.edges, f2, strict=True):
            T = np.array([[np.real(np.trace(R2 @ operador(n, i, a) @ operador(n, j, b))) for b in range(3)]
                          for a in range(3)])
            c, sn = np.cos(f), np.sin(f)
            res = max(res, np.abs(T + val * np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])).max())
        if mejor is None or res < mejor[0]:
            mejor = (res, R2)
    res, R2 = mejor
    ov = sorted(set(vs1) & set(vs2))
    M1 = marginal_numerica(R1, n, [sv1.index(w) for w in ov])
    M2 = marginal_numerica(R2, n, [sv2.index(w) for w in ov])
    de, dm = np.abs(np.linalg.eigvalsh(M1) - np.linalg.eigvalsh(M2)).max(), np.abs(M1 - M2).max()
    chk.check(res < 1e-9 and de < 1e-9 and dm > 1e-3,
              f"antípoda: v = {val:.7f}, residuo de aristas en H2 {res:.1e}, |espectros| {de:.1e}, |matrices| {dm:.2e}")


def vu_serie(serie, caso):
    return float(serie["cubo"][f"v_{caso}"])


def main():
    chk = Checker("sphere_solape")
    for th in (0.0, 1.0, 2.0, np.pi):
        r = resuelve(3, [(0, 1), (1, 2)], [th / 4, th / 4], 3, [(0, 1), (1, 2)], [-th / 4, -th / 4], [0, 2], [0, 2])
        ref = 2 / (3 + np.sin(th / 2))
        sup = solve_dual(3, [(0, 1), (1, 2)], [th / 4, th / 4], 3, [(0, 1), (1, 2)], [-th / 4, -th / 4], [0, 2], [0, 2])[0]
        chk.check(abs(r[0] - ref) < 1e-5 and abs(sup - ref) < 1e-7,
                  f"anillo de 4, th = {th:.4f}: v_O = {r[0]:.7f}, cota dual {sup:.9f}, Teorema 3 {ref:.9f}")

    casq = {(r["poliedro"], r["subgrafo"], r["caso"]): float(r["v"])
            for r in csv.DictReader(open(DATA / "sphere_casquetes.csv"))}
    serie = {r["nombre"]: r for r in csv.DictReader(open(DATA / "sphere_series.csv"))}
    p = sphere.polyhedron("cubo")
    idx = {tuple(v): q for q, v in enumerate(p.verts)}
    m = {k: idx[tuple(float(c) for c in xyz)] for k, xyz in PILOTO_XYZ.items()}
    cortes = {"hemisferios": ("H1", "H2", [q for q in range(8) if q != idx[-1.0, -1.0, -1.0]],
                              [q for q in range(8) if q != idx[1.0, 1.0, 1.0]]),
              "dos caras": ("G1", "G2", [m[v] for v in range(6)], [m[v] for v in range(2, 8)])}
    filas = []
    for corte, (g1, g2, vs1, vs2) in cortes.items():
        ov = sorted(set(vs1) & set(vs2))
        for caso in ("trivial", "curvado"):
            phis = np.zeros(p.E) if caso == "trivial" else sphere.phases(p, sphere.face_fluxes(p))
            s1, f1, _ = sphere.induced_subgraph(p, phis, vs1)
            s2, f2, _ = sphere.induced_subgraph(p, phis, vs2)
            o1, o2 = [sorted(vs1).index(v) for v in ov], [sorted(vs2).index(v) for v in ov]
            E1, E2 = list(s1.edges), list(s2.edges)
            libre = resuelve(s1.n, E1, f1, s2.n, E2, f2)
            ref = min(casq["cubo", g1, caso], casq["cubo", g2, caso])
            chk.check(abs(libre[0] - ref) < TOL, f"{corte} {caso}: sin ligar {libre[0]:.8f}, "
                      f"data/sphere_casquetes.csv {ref:.8f}")
            v, estado, tr, ev, ea, dm, l1, l2 = resuelve(s1.n, E1, f1, s2.n, E2, f2, o1, o2)
            chk.status_note(estado)
            inf, res_par, p_mezcla, emin = repair(l1, l2, s1.n, s2.n, o1, o2, v, f1, f2, E1, E2)
            sup = solve_dual(s1.n, E1, f1, s2.n, E2, f2, o1, o2)[0]
            chk.check(res_par < 1e-13 and emin > -1e-13 and -1e-9 < sup - inf < 2e-6,
                      f"{corte} {caso}: v_O entre {inf:.9f} (par exacto, mezcla {p_mezcla:.1e}, residuo {res_par:.1e}) "
                      f"y {sup:.9f} (testigo); oculto entre {inf - vu_serie(serie, caso):.3e} y {sup - vu_serie(serie, caso):.3e}")
            vu = float(serie["cubo"][f"v_{caso}"])
            chk.check(vu - TOL < v < ref + TOL, f"{corte} {caso}: unión {vu:.8f} <= v_O {v:.8f} <= {ref:.8f} "
                      f"(estado {estado})")
            chk.check(tr < 1e-5 and ev > -1e-5 and ea < 1e-5 and dm < 1e-5,
                      f"{corte} {caso}: residuos: traza {tr:.1e}, menor autovalor {ev:.1e}, aristas {ea:.1e}, "
                      f"marginal del solape {dm:.1e}")
            filas.append([corte, caso, len(ov), ref, v, inf, sup, vu, ref - v, v - vu, inf - vu, sup - vu, estado])
            chk.info(f"{corte} {caso}: delta {ref - vu:.7f} = visible {ref - v:.7f} + oculto {v - vu:.7f}")
    vs1, vs2 = cortes["hemisferios"][2:]
    antipoda(chk, p, sphere.phases(p, sphere.face_fluxes(p)), vs1, vs2)
    write_csv("sphere_solape.csv", ("corte", "caso", "qubits_solape", "v_casquete", "v_solape", "v_solape_inf",
                                    "v_solape_sup", "v_union", "delta_visible", "delta_oculto", "oculto_inf",
                                    "oculto_sup", "estado"), filas)
    chk.finish()


if __name__ == "__main__":
    main()

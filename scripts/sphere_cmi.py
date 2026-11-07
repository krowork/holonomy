"""Información mutua condicional I(A:C|B) y defecto de pegado delta, en los cortes del cubo y en el anillo de 4.

I(A:C|B) solo está definida si hay estado global, es decir, para v <= v(G). En el rango conflictivo
v(G) < v <= v(G_i) no hay estado global, y se mide el fallo del pegado de Petz de las dos vistas.

Cantidades:
  * I_gibbs(v): la de la máxima entropía con las marginales de Werner, exp(sum x_k O_k)/Z. Los residuos de arista se
    recalculan con todas las componentes de T^e, sin suponer simetría. Esa familia va de I/d a la cara óptima.
  * I(v(G)): la del conjunto óptimo, soportado en el autoespacio superior de un testigo óptimo.
  * I_min(v), con --minimo anillo|cubo: mínimo local entre estados U(1)-invariantes, con varios arranques. Cada resultado
    es un estado exacto y por tanto una cota superior del mínimo; no es un mínimo global certificado.
  * Petz: con rho1 en AB y rho2 en BC, sigma = rho2_BC^(1/2) rho2_B^(-1/2) rho1 rho2_B^(-1/2) rho2_BC^(1/2), y las
    distancias de traza de sigma_AB a rho1 y de sigma_BC a rho2. rho1 y rho2 son los Gibbs de cada casquete.

Comprobaciones:
  1. controles de I: producto de pares (0), mezcla clásica de GHZ (0), anillo de 4 sin holonomía (ln(4/3)), y el estado de
     Markov de la Proposición 8 de esfera (dos Werner y sqrt(SWAP) en B, con v = w/2 e I = 0);
  2. fuerte subaditividad y, en estado puro, I(A:C|B) = I(A:C);
  3. v(G) del testigo frente al Teorema 3 de anillos y a data/sphere_series.csv;
  4. residuos del Gibbs y del óptimo frente a las marginales de Werner;
  5. --minimo: estado final exacto (residuo y autovalor), y un mínimo que no sube respecto del Gibbs.

-ln F, con F la fidelidad de rho y de su recuperación por el Petz sin rotar, se da como dato: la cota I >= -ln F de Fawzi y
Renner es para algún canal, no para este, y aquí no se comprueba.

Otros modos:
  * --umbral: en el anillo, I_min sin U(1) frente a U(1), umbrales finos en theta = 0 y pi, y escala de I_min en v.
  * --contraste: en el cubo, el estado de v(G) desde el SDP primal, sin testigo, con entropías de marginales por bucles; y los
    estados con I_min = 0: I por bucles, fallo de Petz, rango y entropía, frente al Gibbs al mismo v.

Escribe data/sphere_cmi_anillo.csv, data/sphere_cmi_cortes.csv, data/sphere_cmi_petz.csv; con --minimo,
data/sphere_cmi_minimo_{anillo,cubo}.csv; con --umbral, data/sphere_cmi_sin_u1.csv, data/sphere_cmi_umbral.csv y
data/sphere_cmi_escala.csv; con --contraste, data/sphere_cmi_primal.csv y data/sphere_cmi_markov.csv.

Uso: (ulimit -v 12000000; python scripts/sphere_cmi.py [--minimo anillo|cubo | --umbral | --contraste])
"""
import _common  # noqa: F401
from _common import Checker

import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, expm
from scipy.optimize import minimize, minimize_scalar

from holonomy import sphere
from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
PILOTO_XYZ = {0: (1, 1, 1), 1: (1, -1, 1), 2: (-1, -1, 1), 3: (-1, 1, 1),
              4: (1, 1, -1), 5: (1, -1, -1), 6: (-1, -1, -1), 7: (-1, 1, -1)}


def op(n, q, M):
    """M en el qubit q; bit q de la base = qubit q (convenio de sphere_solape)."""
    out = np.eye(1, dtype=complex)
    for r in range(n - 1, -1, -1):
        out = np.kron(out, M if r == q else np.eye(2))
    return out


def ptrace(rho, n, keep):
    """Marginal en los qubits `keep`, ordenados como en la base reducida (bit k = keep ordenado k)."""
    keep = sorted(keep)
    T = rho.reshape([2] * (2 * n))
    # El eje k de la reshape es el qubit n-1-k.
    ejes_fuera = [n - 1 - q for q in range(n) if q not in keep]
    for quitados, e in enumerate(sorted(ejes_fuera, reverse=True)):
        T = np.trace(T, axis1=e, axis2=e + n - quitados)
    d = 2**len(keep)
    return T.reshape(d, d)


def espectro_log(rho):
    p, U = eigh((rho + rho.conj().T) / 2)
    return p, U


def entropia(rho):
    p = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum())


def cmi(rho, n, A, B, C):
    return (entropia(ptrace(rho, n, A + B)) + entropia(ptrace(rho, n, B + C))
            - entropia(ptrace(rho, n, B)) - entropia(rho))


def potencia(rho, s, cero=1e-13):
    """rho^s, con pseudoinversa si s < 0."""
    p, U = espectro_log(rho)
    q = np.zeros_like(p)
    pos = p > cero
    q[pos] = p[pos]**s
    return (U * q) @ U.conj().T


def traza_dist(a, b):
    return 0.5 * float(np.abs(np.linalg.eigvalsh((a - b + (a - b).conj().T) / 2)).sum())


def fidelidad(a, b):
    s = potencia(a, 0.5)
    return float(np.sqrt(np.abs(np.linalg.eigvalsh(s @ b @ s))).sum()**2)


def embed(M, n, qs):
    """M, en los qubits ordenados qs de un total de n, extendido por la identidad."""
    qs = sorted(qs)
    d, m = 2**n, len(qs)
    fuera = [q for q in range(n) if q not in qs]
    out = np.zeros((d, d), complex)
    idx = np.array([[sum(((a >> k) & 1) << qs[k] for k in range(m)) + sum(((w >> k) & 1) << fuera[k]
                                                                          for k in range(len(fuera)))
                     for a in range(2**m)] for w in range(2**len(fuera))])
    for fila in idx:
        out[np.ix_(fila, fila)] += M
    return out


def petz(rho1, rho2, n, A, B, C):
    """sigma en ABC a partir de rho1 en AB y rho2 en BC; todos los qubits con su etiqueta en n."""
    AB, BC = sorted(A + B), sorted(B + C)
    r2B = ptrace(rho2, len(BC), [BC.index(q) for q in B])
    K = embed(potencia(rho2, 0.5), n, BC) @ embed(potencia(r2B, -0.5), n, B)
    # rho1 extendido con I en C: el canal de Petz actúa solo sobre B, y C entra por rho2_BC.
    return K @ embed(rho1, n, AB) @ K.conj().T


class Datos:
    """Grafo con fases abelianas y sus operadores de arista."""

    def __init__(self, n, aristas, fases):
        self.n, self.aristas, self.fases = n, list(aristas), list(fases)
        self.X = [[op(n, q, P) for P in PAULI] for q in range(n)]
        self.ops, self.obj = [], []
        for (i, j), f in zip(self.aristas, self.fases, strict=True):
            A = -(self.X[i][0] @ self.X[j][0] + self.X[i][1] @ self.X[j][1])
            B = self.X[i][0] @ self.X[j][1] - self.X[i][1] @ self.X[j][0]
            C = -(self.X[i][2] @ self.X[j][2])
            self.ops += [A, B, C]
            self.obj += [2 * np.cos(f), 2 * np.sin(f), 1.0]
        for q in range(n):
            self.ops.append(self.X[q][2])
            self.obj.append(0.0)
        self.obj = np.array(self.obj)
        self.pila = np.array([O.ravel() for O in self.ops])

    def objetivo(self, v):
        t = v * self.obj
        t[-self.n:] = 0.0
        return t

    def W(self, t):
        """Testigo de una variable, con b = 0: t H + (1/|E| - 2t) C, con N = 1. Vale en el cubo, no en el anillo con holonomía."""
        c = 1 / len(self.aristas) - 2 * t
        out = 0
        for k, f in enumerate(self.fases):
            out = out + t * (np.cos(f) * self.ops[3 * k] + np.sin(f) * self.ops[3 * k + 1]) + c * self.ops[3 * k + 2]
        return out

    def residuo(self, rho, v):
        """Mayor error de T^e frente a -v R_z(fase), de los vectores de Bloch y de la traza."""
        err = abs(np.trace(rho) - 1)
        for (i, j), f in zip(self.aristas, self.fases, strict=True):
            c, s = np.cos(f), np.sin(f)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for a in range(3):
                for b in range(3):
                    err = max(err, abs(np.trace(rho @ self.X[i][a] @ self.X[j][b]).real + v * R[a, b]))
        for q in range(self.n):
            for a in range(3):
                err = max(err, abs(np.trace(rho @ self.X[q][a])))
        return float(err)

    def constriñidos(self):
        """Paulis cuyo valor esperado fijan las restricciones, incluida la identidad implícita."""
        out = [self.X[q][a] for q in range(self.n) for a in range(3)]
        for i, j in self.aristas:
            out += [self.X[i][a] @ self.X[j][b] for a in range(3) for b in range(3)]
        return out


def gibbs(dat, v, x0=None, tol=1e-11, maxit=400):
    """Máxima entropía a v: minimiza ln Z - x.t, convexo, por Newton amortiguado. Devuelve (rho, x)."""
    t = dat.objetivo(v)
    K, d = len(dat.ops), 2**dat.n
    x = np.zeros(K) if x0 is None else x0.copy()

    def valor(x):
        H = np.tensordot(x, dat.pila, axes=(0, 0)).reshape(d, d)
        H = (H + H.conj().T) / 2
        E, U = eigh(H)
        m = E.max()
        lnZ = m + np.log(np.exp(E - m).sum())
        return lnZ - x @ t, E, U, lnZ

    for _ in range(maxit):
        F, E, U, lnZ = valor(x)
        p = np.exp(E - lnZ)
        Oe = np.array([(U.conj().T @ O @ U).ravel() for O in dat.ops])
        g = (Oe[:, ::d + 1] @ p).real - t
        if np.abs(g).max() < tol:
            break
        dE = E[:, None] - E[None, :]
        dp = p[:, None] - p[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(dE) > 1e-9, dp / np.where(dE == 0, 1, dE), (p[:, None] + p[None, :]) / 2)
        Hs = (Oe.conj() * w.ravel()[None, :]) @ Oe.T
        Hs = Hs.real
        medias = g + t
        Hs -= np.outer(medias, medias)
        paso = np.linalg.lstsq(Hs + 1e-13 * np.trace(Hs) / K * np.eye(K), -g, rcond=1e-14)[0]
        # Cerca de la frontera Hs es casi singular: un paso enorme hace que F se pierda en el redondeo.
        paso *= min(1.0, 1.0 / np.abs(paso).max())
        a = 1.0
        while a > 1e-8:
            if valor(x + a * paso)[0] <= F + 1e-4 * a * (g @ paso):
                break
            a /= 2
        x = x + a * paso
    F, E, U, lnZ = valor(x)
    p = np.exp(E - lnZ)
    return (U * p) @ U.conj().T, x


def testigo_optimo(dat):
    """v(G) y t* del dual de una variable; válido en grafos transitivos en aristas con datos invariantes."""
    f = lambda t: np.linalg.eigvalsh(dat.W(t))[-1]  # noqa: E731
    r = minimize_scalar(f, bounds=(-3, 3), method="bounded", options=dict(xatol=1e-11, maxiter=200))
    return float(r.fun), float(r.x)


def cara_optima(dat, W, v, tol_deg=1e-6):
    """Autoespacio superior de W y estados en él con las marginales. Devuelve (P, g, libres, rho, residuo)."""
    E, U = eigh(W)
    P = U[:, E > E[-1] - tol_deg]
    g = P.shape[1]
    if g == 1:
        rho = P @ P.conj().T
        return P, 1, 0, rho, dat.residuo(rho, v)
    # Base hermítica de sigma en el autoespacio; restricciones lineales T^e y Bloch, y traza.
    base = []
    for a in range(g):
        for b in range(a, g):
            E_ab = np.zeros((g, g), complex)
            E_ab[a, b] = 1
            if a == b:
                base.append(E_ab)
            else:
                base.append((E_ab + E_ab.T) / np.sqrt(2))
                base.append(1j * (E_ab - E_ab.T) / np.sqrt(2))
    Os = [np.eye(2**dat.n, dtype=complex)] + dat.constriñidos()
    M = np.array([[np.trace(P.conj().T @ O @ P @ s).real for s in base] for O in Os])
    bvec = []
    bvec.append(1.0)
    bvec += [0.0] * (3 * dat.n)
    for f in dat.fases:
        c, s = np.cos(f), np.sin(f)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        bvec += list((-v * R).ravel())
    bvec = np.array(bvec)
    sv = np.linalg.svd(M, compute_uv=False)
    libres = len(base) - int((sv > 1e-7 * sv[0]).sum())
    c0 = np.linalg.lstsq(M, bvec, rcond=1e-7)[0]
    sigma = sum(ci * s for ci, s in zip(c0, base))
    rho = P @ sigma @ P.conj().T
    return P, g, libres, rho, dat.residuo(rho, v)


def testigo_anillo(dat, th, ref):
    """Anillo de 4: aquí b no se anula. Testigo con flujo propio phi', en la cúspide pi/4 o, si no llega a ref, libre."""
    d = Datos(dat.n, dat.aristas, dat.fases)

    def W(r, fp):
        c = 1 / 4 - 2 * r * np.cos(fp - th / 4)
        out = 0
        for k in range(4):
            out = out + r * (np.cos(fp) * d.ops[3 * k] + np.sin(fp) * d.ops[3 * k + 1]) + c * d.ops[3 * k + 2]
        return out

    res = minimize_scalar(lambda r: np.linalg.eigvalsh(W(r, np.pi / 4))[-1], bounds=(-3, 3), method="bounded",
                          options=dict(xatol=1e-11, maxiter=200))
    if abs(res.fun - ref) < 1e-8:
        return float(res.fun), W(res.x, np.pi / 4)
    a = minimize(lambda x: np.linalg.eigvalsh(W(x[0], x[1]))[-1], [res.x, th / 4 + 0.1], method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-11, maxiter=4000))
    return float(a.fun), W(*a.x)


def anillo(th):
    return Datos(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [th / 4] * 4)


def cubo(caso):
    p = sphere.polyhedron("cubo")
    phis = np.zeros(p.E) if caso == "trivial" else sphere.phases(p, sphere.face_fluxes(p))
    return p, phis, Datos(8, p.edges, phis)


def restringe(p, phis, vs):
    vs = sorted(vs)
    E, F = [], []
    for (i, j), f in zip(p.edges, phis, strict=True):
        if i in vs and j in vs:
            E.append((vs.index(i), vs.index(j)))
            F.append(f)
    return Datos(len(vs), E, F)


def cortes_cubo():
    p = sphere.polyhedron("cubo")
    idx = {tuple(v): q for q, v in enumerate(p.verts)}
    m = {k: idx[tuple(float(c) for c in xyz)] for k, xyz in PILOTO_XYZ.items()}
    return {"hemisferios": ([q for q in range(8) if q != idx[-1.0, -1.0, -1.0]],
                            [q for q in range(8) if q != idx[1.0, 1.0, 1.0]]),
            "dos caras": ([m[v] for v in range(6)], [m[v] for v in range(2, 8)])}


def partes(vs1, vs2):
    A = sorted(set(vs1) - set(vs2))
    C = sorted(set(vs2) - set(vs1))
    B = sorted(set(vs1) & set(vs2))
    return A, B, C


def fawzi_renner(rho, n, A, B, C):
    """F(rho, R_{B->BC}(rho_AB)), con el canal de Petz de rho_BC."""
    AB, BC = sorted(A + B), sorted(B + C)
    sigma = petz(ptrace(rho, n, AB), ptrace(rho, n, BC), n, A, B, C)
    return fidelidad(rho, sigma)


def controles(chk):
    s = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    par = np.outer(s, s.conj())
    rho = np.kron(par, par)
    # Pares (0,1) y (2,3): A = {0}, B = {1,2}, C = {3}.
    chk.check(abs(cmi(rho, 4, [0], [1, 2], [3])) < 1e-12, "producto de pares: I = 0")
    ghz = np.zeros((8, 8), complex)
    ghz[0, 0] = ghz[7, 7] = 0.5
    chk.check(abs(cmi(ghz, 3, [0], [1], [2])) < 1e-12, "mezcla clásica de GHZ: I(A:C|B) = 0, I(A:C) = ln 2")
    r = anillo(0.0)
    v, t = testigo_optimo(r)
    _, g, _, rho, res = cara_optima(r, r.W(t), v)
    i = cmi(rho, 4, [1], [0, 2], [3])
    chk.check(g == 1 and abs(i - np.log(4 / 3)) < 1e-7 and res < 1e-6,
              f"anillo de 4 sin holonomía: g = {g}, I = {i:.9f}, ln(4/3) = {np.log(4 / 3):.9f}, residuo {res:.1e}")
    iac = entropia(ptrace(rho, 4, [1])) + entropia(ptrace(rho, 4, [3])) - entropia(ptrace(rho, 4, [1, 3]))
    chk.check(abs(i - iac) < 1e-7, f"estado puro: I(A:C|B) = {i:.9f} = I(A:C) = {iac:.9f}")


def estado_markov_anillo(w):
    """Proposición 8 de esfera: Werner w en (1,0) y en (3,2), y sqrt(SWAP) en B = (0,2). Devuelve (rho, V)."""
    r = anillo(0.0)
    d = 16
    producto = (np.eye(d) - w * sum(r.X[1][a] @ r.X[0][a] for a in range(3))) @ (np.eye(d) - w * sum(r.X[3][a] @ r.X[2][a] for a in range(3))) / d
    V = expm(0.25j * np.pi * (np.eye(d) + sum(r.X[0][a] @ r.X[2][a] for a in range(3))) / 2)
    return V @ producto @ V.conj().T, V


def corre_markov_explicito(chk):
    r = anillo(0.0)
    _, V = estado_markov_anillo(1.0)
    solape = max(abs(np.trace(r.X[q][b] @ V @ r.X[p][a] @ V.conj().T).real / 16 - 0.5 * (a == b))
                 for p in (0, 2) for q in (0, 2) for a in range(3) for b in range(3))
    chk.check(solape < 1e-13, f"sqrt(SWAP): tr(s_q^b V s_p^a V^+)/4 = delta_ab/2, error {solape:.1e}")
    for w in (0.2, 0.5, 0.8, 1.0):
        rho, _ = estado_markov_anillo(w)
        i, res, e = cmi(rho, 4, [1], [0, 2], [3]), r.residuo(rho, w / 2), np.linalg.eigvalsh(rho)[0]
        chk.check(abs(i) < 1e-12 and res < 1e-13 and e > -1e-13, f"estado de Markov, w = {w}: v = {w / 2}, residuo {res:.1e}, "
                  f"autovalor mínimo {e:.1e}, I = {i:.1e}")


def corre_anillo(chk):
    filas = []
    for th in np.linspace(0, np.pi, 13):
        r = anillo(th)
        ref = 2 / (3 + np.sin(th / 2))
        v, W = testigo_anillo(r, th, ref)
        delta = 2 / 3 - ref
        _, g, lib, rho, res = cara_optima(r, W, v)
        A, B, C = [1], [0, 2], [3]
        i = cmi(rho, 4, A, B, C)
        x = None
        for fr in (0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
            rho2, x = gibbs(r, fr * v, x0=x)
        i_lim = cmi(rho2, 4, A, B, C)
        F = fawzi_renner(rho, 4, A, B, C)
        chk.check(abs(v - ref) < 2e-8 and res < 1e-6 and i > -1e-10 and r.residuo(rho2, 0.999 * v) < 1e-8,
                  f"th = {th:.4f}: v = {v:.8f}, g = {g}, libres {lib}, I = {i:.8f}, delta = {delta:.6f}, "
                  f"-lnF = {-np.log(F):.6f}, Gibbs(0,999 v): I = {i_lim:.6f}, residuo {res:.1e}")
        filas.append([th, v, delta, g, lib, i, -np.log(F), i_lim, res])
    write_csv("sphere_cmi_anillo.csv", ("theta", "v", "delta", "g", "libres", "I", "menoslnF", "I_gibbs_0999", "residuo"), filas)


def corre_cortes(chk):
    solape = {(r["corte"], r["caso"]): r for r in csv.DictReader(open(DATA / "sphere_solape.csv"))}
    serie = {r["nombre"]: r for r in csv.DictReader(open(DATA / "sphere_series.csv"))}
    filas = []
    for caso in ("trivial", "curvado"):
        p, phis, dat = cubo(caso)
        v, t = testigo_optimo(dat)
        ref = float(serie["cubo"][f"v_{caso}"])
        chk.check(abs(v - ref) < 1e-8, f"cubo {caso}: v(G) = {v:.9f}, data/sphere_series.csv {ref:.9f}, t = {t:.6f}")
        _, g, lib, rho_op, res = cara_optima(dat, dat.W(t), v)
        chk.info(f"cubo {caso}: autoespacio superior g = {g}, parámetros libres en la cara {lib}, residuo {res:.1e}")
        x = None
        estados = {}
        for fr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
            rho, x = gibbs(dat, fr * v, x0=x)
            estados[fr] = rho
            chk.check(dat.residuo(rho, fr * v) < 1e-8, f"cubo {caso}: Gibbs a {fr} v(G), residuo {dat.residuo(rho, fr * v):.1e}")
        for corte, (vs1, vs2) in cortes_cubo().items():
            A, B, C = partes(vs1, vs2)
            s = solape[corte, caso]
            delta = float(s["v_casquete"]) - float(s["v_union"])
            for fr, rho in estados.items():
                i = cmi(rho, 8, A, B, C)
                F = fawzi_renner(rho, 8, A, B, C)
                chk.check(i > -1e-10,
                          f"{corte} {caso}: v/v(G) = {fr}, v = {fr * v:.5f}, I_gibbs = {i:.6f}, -lnF = {-np.log(F):.6f}")
                filas.append([corte, caso, fr, fr * v, delta, float(s["delta_oculto"]), i, -np.log(F), g, lib])
            i = cmi(rho_op, 8, A, B, C)
            F = fawzi_renner(rho_op, 8, A, B, C)
            chk.info(f"{corte} {caso}: en v(G), {'estado único' if lib == 0 else f'estado de la cara ({lib} libres)'}: "
                     f"I = {i:.6f}, -lnF = {-np.log(F):.6f}; delta = {delta:.6f}, oculto {float(s['delta_oculto']):.6f}")
            filas.append([corte, caso, 1.0, v, delta, float(s["delta_oculto"]), i, -np.log(F), g, lib])
    write_csv("sphere_cmi_cortes.csv", ("corte", "caso", "v_rel", "v", "delta", "oculto", "I", "menoslnF", "g", "libres"), filas)


def corre_petz(chk):
    """Pegado de Petz de los Gibbs de cada casquete, por debajo y por encima de v(G)."""
    solape = {(r["corte"], r["caso"]): r for r in csv.DictReader(open(DATA / "sphere_solape.csv"))}
    filas = []
    for caso in ("trivial", "curvado"):
        p, phis, _ = cubo(caso)
        for corte, (vs1, vs2) in cortes_cubo().items():
            A, B, C = partes(vs1, vs2)
            s = solape[corte, caso]
            vc, vG, vO = float(s["v_casquete"]), float(s["v_union"]), float(s["v_solape"])
            d1, d2 = restringe(p, phis, vs1), restringe(p, phis, vs2)
            AB, BC = sorted(A + B), sorted(B + C)
            x1 = x2 = None
            for fr in (0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.995):
                v = fr * vc
                rho1, x1 = gibbs(d1, v, x0=x1)
                rho2, x2 = gibbs(d2, v, x0=x2)
                r1, r2 = d1.residuo(rho1, v), d2.residuo(rho2, v)
                sig = petz(rho1, rho2, 8, A, B, C)
                sig = sig / np.trace(sig)
                eA = traza_dist(ptrace(sig, 8, AB), rho1)
                eC = traza_dist(ptrace(sig, 8, BC), rho2)
                dO = traza_dist(ptrace(rho1, len(AB), [AB.index(q) for q in B]),
                                ptrace(rho2, len(BC), [BC.index(q) for q in B]))
                chk.check(max(r1, r2) < 1e-8, f"{corte} {caso}: v = {v:.5f} ({'conflicto' if v > vG else 'pegable'}), "
                          f"D_O = {dO:.5f}, Petz: eps_AB = {eA:.5f}, eps_BC = {eC:.5f}, residuo {max(r1, r2):.1e}")
                filas.append([corte, caso, v, vG, vO, vc, dO, eA, eC])
    write_csv("sphere_cmi_petz.csv", ("corte", "caso", "v", "v_G", "v_O", "v_casquete", "D_O", "eps_AB", "eps_BC"), filas)


def sectores(n):
    out = {}
    for x in range(2**n):
        out.setdefault(bin(x).count("1"), []).append(x)
    return [np.array(b) for b in out.values()]


class Busqueda:
    """Mínimo local de I(A:C|B) entre estados U(1)-invariantes con las marginales dadas: rho = M M^+/tr por bloques de Sz
    y lagrangiano aumentado, que trabaja bien en la frontera rho >= 0. Al final se fuerza cada restricción y se mezcla con
    I/d, lo que da un estado exacto con v' = (1 - p) v. Es una cota superior del mínimo sobre esos estados."""

    def __init__(self, dat, A, B, C, u1=True):
        self.dat, self.n, self.d, self.u1 = dat, dat.n, 2**dat.n, u1
        self.A, self.B, self.C = A, B, C
        self.AB, self.BC = sorted(A + B), sorted(B + C)
        # Sin U(1), un solo bloque y todas las componentes de T^e: I no es convexa, y promediar no lo reduciría a U(1).
        self.Q = sp.csr_matrix(np.array([O.ravel() for O in (dat.ops if u1 else dat.constriñidos())]))
        self.secs = sectores(self.n) if u1 else [np.arange(self.d)]
        self.sec_red = {tuple(q): sectores(len(q)) if u1 else [np.arange(2**len(q))] for q in (self.AB, self.BC, B)}
        self.idx = {tuple(q): self._indices(q) for q in (self.AB, self.BC, B)}
        self.tam = sum(len(b)**2 for b in self.secs)

    def _indices(self, qs):
        fuera = [q for q in range(self.n) if q not in qs]
        return np.array([[sum(((a >> k) & 1) << qs[k] for k in range(len(qs)))
                          + sum(((w >> k) & 1) << fuera[k] for k in range(len(fuera)))
                          for a in range(2**len(qs))] for w in range(2**len(fuera))])

    def _sube(self, M, qs):
        out = np.zeros((self.d, self.d), complex)
        for fila in self.idx[tuple(qs)]:
            out[np.ix_(fila, fila)] += M
        return out

    @staticmethod
    def _s_y_log(rho, secs):
        S, L = 0.0, np.zeros_like(rho)
        for b in secs:
            bloque = rho[np.ix_(b, b)]
            p, U = np.linalg.eigh((bloque + bloque.conj().T) / 2)
            q = np.clip(p, 1e-13, None)
            S -= float((np.clip(p, 0, None) * np.log(q)).sum())
            L[np.ix_(b, b)] = (U * np.log(q)) @ U.conj().T
        return S, L

    def i_y_gradiente(self, rho):
        s0, l0 = self._s_y_log(rho, self.secs)
        s1, l1 = self._s_y_log(ptrace(rho, self.n, self.AB), self.sec_red[tuple(self.AB)])
        s2, l2 = self._s_y_log(ptrace(rho, self.n, self.BC), self.sec_red[tuple(self.BC)])
        s3, l3 = self._s_y_log(ptrace(rho, self.n, self.B), self.sec_red[tuple(self.B)])
        return s1 + s2 - s3 - s0, l0 - self._sube(l1, self.AB) - self._sube(l2, self.BC) + self._sube(l3, self.B)

    def _rho(self, x):
        """rho y las M por bloques, desde el vector real x (partes real e imaginaria seguidas)."""
        z = x[:self.tam] + 1j * x[self.tam:]
        rho, Ms, pos = np.zeros((self.d, self.d), complex), [], 0
        for b in self.secs:
            m = z[pos:pos + len(b)**2].reshape(len(b), len(b))
            pos += len(b)**2
            Ms.append(m)
            rho[np.ix_(b, b)] = m @ m.conj().T
        nr = np.trace(rho).real
        return rho / nr, Ms, nr

    def corre(self, v, x0, outer=14, inner=1500, tmax=900.0):
        t = self.dat.objetivo(v) if self.u1 else np.array(self._paulis(v)[1])
        lam, mu, x, t0 = np.zeros(len(t)), 10.0, x0, time.time()

        def f(x):
            rho, Ms, nr = self._rho(x)
            i, G = self.i_y_gradiente(rho)
            c = (self.Q.conj() @ rho.ravel()).real - t
            G = G + ((lam + mu * c) @ self.Q).reshape(self.d, self.d)
            G = (G + G.conj().T) / 2
            eG = np.trace(G @ rho).real
            gr = []
            for b, m in zip(self.secs, Ms, strict=True):
                gr.append((2 * (G[np.ix_(b, b)] @ m - eG * m) / nr).ravel())
            gr = np.concatenate(gr)
            return i + lam @ c + mu / 2 * c @ c, np.concatenate([gr.real, gr.imag])

        for _ in range(outer):
            x = minimize(f, x, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=inner, maxcor=30, ftol=1e-14, gtol=1e-9)).x
            rho = self._rho(x)[0]
            c = (self.Q.conj() @ rho.ravel()).real - t
            lam = lam + mu * c
            if np.abs(c).max() < 1e-9 or time.time() - t0 > tmax:
                break
            mu = min(mu * 3, 1e5)
        return self.repara(rho, v)

    def repara(self, rho, v):
        """Fuerza las restricciones con Paulis ortogonales y mezcla con I/d. Devuelve (rho, v', I, residuo)."""
        R = rho.copy()
        for P, obj in zip(*self._paulis(v), strict=True):
            R = R + (obj - np.trace(R @ P).real) * P / self.d
        R = (R + R.conj().T) / 2
        R = R / np.trace(R).real
        e = np.linalg.eigvalsh(R)[0]
        p = max(0.0, -e * self.d / (1 - e * self.d))
        R = (1 - p) * R + p * np.eye(self.d) / self.d
        vf = (1 - p) * v
        return R, vf, cmi(R, self.n, self.A, self.B, self.C), self.dat.residuo(R, vf)

    def _paulis(self, v):
        Ps = self.dat.constriñidos()
        obj = [0.0] * (3 * self.n)
        for f in self.dat.fases:
            c, s = np.cos(f), np.sin(f)
            obj += list((-v * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])).ravel())
        return Ps, obj

    def arranque(self, rng, rho=None, ruido=1.0):
        """x0: aleatorio o, con rho, su raíz más ruido."""
        z = []
        for b in self.secs:
            g = rng.normal(size=(len(b), len(b))) + 1j * rng.normal(size=(len(b), len(b)))
            m = ruido * g
            if rho is not None:
                m = potencia(rho[np.ix_(b, b)], 0.5) + ruido * g
            z.append(m.ravel())
        z = np.concatenate(z)
        return np.concatenate([z.real, z.imag])


def corre_minimo(chk, cuales):
    """I_min(v) con varios arranques. cuales: 'anillo' o 'cubo'."""
    rng = np.random.default_rng(2026)
    A4, B4, C4 = [1], [0, 2], [3]
    casos = []
    if cuales == "anillo":
        for th in (0.0, 0.25, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
            vG = 2 / (3 + np.sin(th / 2))
            casos.append(("anillo4", f"th={th:.4f}", anillo(th), A4, B4, C4, vG,
                          [v for v in (0.1, 0.2, 0.3, 1 / 3, 0.35, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6) if v < vG], 6))
    else:
        for caso in ("trivial", "curvado"):
            _, _, dat = cubo(caso)
            vG, _ = testigo_optimo(dat)
            for corte, (vs1, vs2) in cortes_cubo().items():
                A, B, C = partes(vs1, vs2)
                casos.append((corte, caso, dat, A, B, C, vG,
                              [v for v in (0.2, 0.3, 1 / 3, 0.36, 0.4, 0.45, 0.5) if v < vG] + [0.99 * vG], 2))
    filas = []
    for nombre, caso, dat, A, B, C, vG, vs, arranques in casos:
        bus = Busqueda(dat, A, B, C)
        for v in vs:
            rho0, _ = gibbs_continuo(dat, v, vG)
            i0 = cmi(rho0, dat.n, A, B, C)
            vals, mejor = [], None
            for k in range(arranques):
                x0 = bus.arranque(rng, rho0, 1e-3) if k == 0 else bus.arranque(rng)
                R, vf, i, res = bus.corre(v, x0)
                vals.append(i)
                if res < 1e-9 and vf > v * (1 - 1e-6) and (mejor is None or i < mejor[0]):
                    mejor = (i, vf, res, np.linalg.eigvalsh(R)[0])
            ok = mejor is not None and mejor[0] <= i0 + 1e-7 and mejor[3] > -1e-12
            chk.check(ok, f"{nombre} {caso}: v = {v:.5f} (v(G) = {vG:.5f}): I_gibbs {i0:.3e}, I_min {mejor[0] if mejor else np.nan:.3e}, "
                      f"arranques {np.array2string(np.sort(vals), precision=2)}, v' {mejor[1] if mejor else np.nan:.6f}")
            filas.append([nombre, caso, v, vG, i0, mejor[0] if mejor else np.nan, *np.sort(vals)[:2]])
            write_csv(f"sphere_cmi_minimo_{cuales}.csv", ("corte", "caso", "v", "v_G", "I_gibbs", "I_min", "I_1", "I_2")[:len(filas[-1])], filas)


def i_min_anillo(th, v, rng, arranques, u1=True):
    """Menor I de los arranques aleatorios en el anillo de 4 y todos sus valores; solo cuentan los estados exactos a v."""
    bus = Busqueda(anillo(th), [1], [0, 2], [3], u1=u1)
    vals = []
    for _ in range(arranques):
        _, vf, i, res = bus.corre(v, bus.arranque(rng), tmax=30.0)
        if res < 1e-9 and vf > v * (1 - 1e-6):
            vals.append(i)
    return (min(vals) if vals else np.nan), len(vals)


def corre_umbral(chk):
    rng = np.random.default_rng(2027)
    con_u1 = {(round(float(r["v"]), 4), r["caso"]): float(r["I_min"]) for r in csv.DictReader(open(DATA / "sphere_cmi_minimo_anillo.csv"))}
    filas = []
    for th, vs in ((np.pi / 2, (0.1, 0.2, 0.3, 0.35)), (np.pi / 4, (0.2, 0.3, 0.45)), (np.pi, (0.3, 0.35, 0.4)),
                   (0.0, (0.5, 0.52)), (0.25, (0.2, 0.4))):
        for v in vs:
            i, k = i_min_anillo(th, v, rng, 16, u1=False)
            ref = con_u1[round(v, 4), f"th={th:.4f}"]
            # Lo que importa es si sin U(1) se baja. Más arriba solo significa un mínimo local: hay varios cerca de th = pi/4.
            chk.check(k > 0 and i > ref - max(2e-6, 0.02 * ref), f"sin U(1): th = {th:.4f}, v = {v}: I_min {i:.3e}, con U(1) {ref:.3e}, {k}/16 estados exactos"
                      + ("  [mínimo local: más alto que con U(1)]" if i > ref + max(2e-6, 0.02 * ref) else ""))
            filas.append([th, v, i, ref])
    write_csv("sphere_cmi_sin_u1.csv", ("theta", "v", "I_min_sin_u1", "I_min_u1"), filas)

    filas = []
    for th, vs in ((0.0, (0.49, 0.495, 0.5, 0.505, 0.51)),
                   (np.pi, (0.33, 1 / 3, 0.34, 0.345, 0.346, 0.347, 0.348, 0.349, 0.35, 0.36))):
        for v in vs:
            i, k = i_min_anillo(th, v, rng, 4)
            chk.check(k > 0, f"umbral: th = {th:.4f}, v = {v:.4f}: I_min {i:.3e}, {k}/4 estados exactos")
            filas.append([th, v, i])
    write_csv("sphere_cmi_umbral.csv", ("theta", "v", "I_min"), filas)

    filas = []
    for th in (0.1, 0.25, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 2.9):
        for v in (0.03, 0.06, 0.12, 0.24):
            i, k = i_min_anillo(th, v, rng, 3)
            chk.check(k > 0, f"escala: th = {th:.4f}, v = {v}: I_min {i:.3e}, I_min/v^4 {i / v**4:.4f}, I_min/(v^4 th^2) {i / (v * v * th)**2:.3f}")
            filas.append([th, v, i, i / v**4])
    write_csv("sphere_cmi_escala.csv", ("theta", "v", "I_min", "I_min_v4"), filas)


def corre_contraste(chk):
    """Contrastes del cubo. El primal no pasa por el testigo; las marginales por bucles, no por ptrace."""
    import warnings

    import cvxpy as cp
    import sphere_solape as so

    def S(R, cero):
        p = np.linalg.eigvalsh((R + R.conj().T) / 2)
        p = p[p > cero]
        return float(-(p * np.log(p)).sum())

    def i_bucles(R, A, B, C, cero=1e-9):
        m = lambda q: so.marginal_numerica(R, 8, q)  # noqa: E731
        return S(m(sorted(A + B)), cero) + S(m(sorted(B + C)), cero) - S(m(sorted(B)), cero) - S(R, cero)

    warnings.simplefilter("ignore")
    filas = []
    for caso in ("trivial", "curvado"):
        p, phis, dat = cubo(caso)
        lado = so.Lado(8, list(p.edges), list(phis))
        v = cp.Variable()
        cp.Problem(cp.Maximize(v), lado.restricciones(v)).solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=300)
        R = lado.completa()
        R = (R + R.conj().T) / 2
        R = R / np.trace(R)
        vt, t = testigo_optimo(dat)
        _, _, _, Rw, _ = cara_optima(dat, dat.W(t), vt)
        dist = traza_dist(R, Rw)
        chk.check(dist < 2e-4 and abs(float(v.value) - vt) < 1e-6, f"cubo {caso}: primal v = {float(v.value):.8f}, testigo {vt:.8f}, distancia de traza {dist:.1e}")
        for corte, (a, b) in cortes_cubo().items():
            A, B, C = partes(a, b)
            i1, i2 = i_bucles(R, A, B, C), cmi(Rw, 8, A, B, C)
            chk.check(abs(i1 - i2) < 1e-4, f"{corte} {caso}: I primal y bucles {i1:.6f}, I testigo y ptrace {i2:.6f}")
            filas.append([corte, caso, float(v.value), vt, dist, i1, i2])
    write_csv("sphere_cmi_primal.csv", ("corte", "caso", "v_primal", "v_testigo", "dist_traza", "I_primal", "I_testigo"), filas)

    rng = np.random.default_rng(21)
    filas = []
    for corte, caso, v in (("hemisferios", "trivial", 0.50), ("hemisferios", "curvado", 0.40),
                           ("dos caras", "trivial", 0.40), ("dos caras", "curvado", 0.30)):
        _, _, dat = cubo(caso)
        a, b = cortes_cubo()[corte]
        A, B, C = partes(a, b)
        bus = Busqueda(dat, A, B, C)
        rho0, _ = gibbs_continuo(dat, v, None)
        # Hay mínimos locales: se sigue arrancando, al azar, mientras no aparezca I = 0.
        R, vf, i, res = bus.corre(v, bus.arranque(rng, rho0, 1e-3))
        for _ in range(5):
            if i < 1e-7:
                break
            cand = bus.corre(v, bus.arranque(rng))
            if cand[3] < 1e-9 and cand[1] > v * (1 - 1e-6) and cand[2] < i:
                R, vf, i, res = cand
        sig = petz(ptrace(R, 8, sorted(A + B)), ptrace(R, 8, sorted(B + C)), 8, A, B, C)
        sig = sig / np.trace(sig)
        ev = np.linalg.eigvalsh(R)
        ib, dp, s_, s0, i0 = i_bucles(R, A, B, C, 1e-14), traza_dist(R, sig), S(R, 1e-14), S(rho0, 1e-14), cmi(rho0, 8, A, B, C)
        chk.check(res < 1e-9 and vf > v * (1 - 1e-6) and abs(ib) < 1e-6 and dp < 1e-3,
                  f"{corte} {caso}, v = {v}: residuo {res:.1e}, I {i:.1e}, I por bucles {ib:.1e}, |rho - Petz| {dp:.1e}, autovalor mínimo {ev[0]:.1e}, "
                  f"rango {(ev > 1e-8).sum()}, S {s_:.4f} ({s_ / (8 * np.log(2)):.3f} del máximo); Gibbs: I {i0:.4f}, S {s0:.4f}")
        filas.append([corte, caso, v, vf, res, i, ib, dp, ev[0], int((ev > 1e-8).sum()), s_, i0, s0])
    write_csv("sphere_cmi_markov.csv", ("corte", "caso", "v", "v_final", "residuo", "I", "I_bucles", "dist_petz", "autovalor_min", "rango", "S", "I_gibbs", "S_gibbs"), filas)


def gibbs_continuo(dat, v, vG):
    x = None
    for fr in (0.25, 0.5, 0.75, 0.9, 1.0):
        rho, x = gibbs(dat, fr * v, x0=x)
    return rho, x


def main():
    chk = Checker("sphere_cmi")
    if "--minimo" in sys.argv:
        corre_minimo(chk, sys.argv[sys.argv.index("--minimo") + 1])
        chk.finish()
    if "--umbral" in sys.argv:
        corre_umbral(chk)
        chk.finish()
    if "--contraste" in sys.argv:
        corre_contraste(chk)
        chk.finish()
    controles(chk)
    corre_markov_explicito(chk)
    corre_anillo(chk)
    corre_cortes(chk)
    corre_petz(chk)
    chk.finish()


if __name__ == "__main__":
    main()

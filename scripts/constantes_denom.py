"""Radio eps_2 de Z15-Z16 con la cota de Peierls por denominadores de energía (celda_k_notas.md, Sec. 7.13,
paso 9.9). Celdas de L = 2 enlaces durante un tramo T, contornos 4-conexos, peso KP a(gamma) = a (N + theta p).

Fila: celdas del contorno en un tramo; tramo de fila: celdas consecutivas. Sitio tocado: lo mueve algún salto del
tramo. Segmento: sitios tocados consecutivos (>= 2). Hechos (9.9):
  (S1) Los sitios de los extremos de un tramo de fila son intocados (los comparte con una celda correcta).
  (S2) Un sitio intocado tiene valor constante en el tramo, así que la suma de las desviaciones de los incrementos
       en los enlaces de un segmento (de intocado a intocado) es constante. Si el segmento pasa por E = 0 en algún
       instante, es 0 mod k y entonces E(t) = 0 o E(t) >= 2; si no, E(t) >= 1 todo el tramo.
  (S3) Un salto desde E = 0 deja exactamente 3 paredes en el segmento (S+S-: +2, -1, +2), y lo mismo antes de uno
       que lleva a E = 0; no hay dos intervalos seguidos con E = 0.
  (S4) Las energías de segmentos distintos y de las celdas de pared (sin sitios tocados) se suman, y los instantes
       de los saltos de segmentos distintos se integran por separado: la integral se factoriza.
  (S5) Primera fila: cada segmento empieza con E = 0; última: acaba con E = 0; ninguna de las dos tiene celdas de
       pared. Celda de pared (fila intermedia): >= 1 pared todo el tramo; fila intermedia de una celda sin saltos:
       >= 2 paredes (la fila entera suma 0 mod k).
Peso de un segmento de lambda sitios: F = sum_m b^m Nc(m, lambda) J(m), b = A|eps|, Nc = sucesiones de m enlaces
internos que tocan todos los sitios, J = cota de la integral sobre los instantes ordenados:
  E = 0 permitido:  T^{P-1}/(P-1)! 3^{-X3} 2^{-X2} (P >= 1), e^{-2T} T^m/m! (P = 0), máximo sobre las sucesiones;
  intermedia, sin E = 0: e^{-T} T^m/m!;  siempre <= T^m/m!.
Tramo de fila: matriz de transferencia sobre los sitios; fila de r celdas: máximo sobre las particiones en tramos.
También se usa la cota de fila entera de la versión anterior (E(t) = 0 o >= 2 en la fila) y se toma el mínimo.
Contorno: prod_filas G. Hasta N = 13, perfiles exactos (scripts/perfiles.txt); más allá, árboles con h^N, h el
supremo por celda (filas hasta RL celdas y la razón G(RL)/G(RL-1) como tasa asintótica), y p = 2N + 2.
Cola por percolación (COLA = 1, paso 9.10): para la percolación de sitios con densidad x,
sum_{S ∋ 0} x^N (1-x)^p <= x; restando los términos exactos N <= 13 queda delta(x) = P_x(13 < |C| < oo), y si
h e^{3a(1+2 theta)} <= x(1-x)^2 la cola es <= (e^{3a theta}/(1-x))^2 delta(x) (usa p <= 2N + 2).
Gas de una fila resumado (VACIO = 1, paso 9.11): los contornos de una sola fila no tienen interior (z = W) y se
integran primero, con KP propio; los de varias filas ven el gas a través de un factor e^{3 sigma (N + p)}, con
sigma = sum_{gamma ∋ 0, una fila} |W| e^{b(gamma)}.
Uso: python scripts/constantes_denom.py [A] [THETA=1] [E2=2] [E3=3] [E1=1] [COLA=1] [VACIO=0]
"""
import sys
from functools import lru_cache
from math import comb, factorial, lgamma, log

import numpy as np

A = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
THETA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
E2 = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
E3 = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
E1 = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
L, q, nu = 2, 2, 5
K = 10
RMAX = 13  # perfiles exactos
RL = 40  # filas calculadas para la cola
LMAX = 2 * RL - 1  # sitios de un segmento
MMAX = LMAX // 2 + 1 + K + 2
Nmax = 400
UNA, PRIMERA, ULTIMA, MEDIA = 0, 1, 2, 3

# ---------------- combinatoria fija ----------------
S2 = [[0] * (MMAX + 1) for _ in range(MMAX + 1)]
S2[0][0] = 1
for m in range(1, MMAX + 1):
    for s in range(1, m + 1):
        S2[m][s] = s * S2[m - 1][s] + S2[m - 1][s - 1]
SURJ = [[factorial(s) * S2[m][s] for s in range(MMAX + 1)] for m in range(MMAX + 1)]


def cubiertas(lam):
    """c[k] = subconjuntos de k aristas del camino de lam sitios que tocan todos los sitios."""
    # estado: (último sitio cubierto?) recorriendo aristas
    dp = {(False, 0): 1}  # (sitio izquierdo de la arista siguiente ya cubierto?, aristas usadas)
    for e in range(lam - 1):  # arista e une sitios e, e+1; al decidirla, el sitio e queda cerrado
        nd = {}
        for (cub, k), v in dp.items():
            # no usar e: el sitio e debe estar cubierto ya
            if cub:
                nd[(False, k)] = nd.get((False, k), 0) + v
            # usar e: cubre e y e+1
            nd[(True, k + 1)] = nd.get((True, k + 1), 0) + v
        dp = nd
    c = [0] * lam
    for (cub, k), v in dp.items():
        if cub:
            c[k] += v
    return c


CUB = {lam: cubiertas(lam) for lam in range(2, LMAX + 1)}
MLO = {lam: (lam + 1) // 2 for lam in CUB}
NC = {lam: [float(sum(CUB[lam][k] * SURJ[m][k] for k in range(1, min(m, lam - 1) + 1)))
            for m in range(MLO[lam], MLO[lam] + K + 1)] for lam in CUB}


@lru_cache(None)
def sucesiones(m, tipo):
    """(P, X3, X2) alcanzables con m + 1 intervalos (0: E = 0; 1: excitado), sin dos 0 seguidos."""
    ini = (0,) if tipo in (UNA, PRIMERA) else (0, 1)
    dp = {}
    for v in ini:
        dp.setdefault((v, False), set()).add((1 if v == 0 else 0, 0, 0))
    for _ in range(m):
        nd = {}
        for (pv, padj), cs in dp.items():
            for v in (0, 1):
                if v == 0 and pv == 0:
                    continue
                for (P, X3, X2) in cs:
                    if pv == 1:
                        if padj or v == 0:
                            X3n, X2n = X3 + 1, X2
                        else:
                            X3n, X2n = X3, X2 + 1
                    else:
                        X3n, X2n = X3, X2
                    nd.setdefault((v, pv == 0), set()).add((P + (v == 0), X3n, X2n))
        dp = nd
    out = set()
    for (pv, padj), cs in dp.items():
        if tipo in (UNA, ULTIMA) and pv != 0:
            continue
        for (P, X3, X2) in cs:
            out.add((P, X3 + (pv == 1 and padj), X2 + (pv == 1 and not padj)))
    return np.array(sorted(out), dtype=float).reshape(-1, 3)


LGP = {}


def J(m, tipo, T):
    triv = T ** m / factorial(m)
    suc = sucesiones(m, tipo)
    best = 0.0
    if len(suc):
        P, X3, X2 = suc.T
        lg = np.array([lgamma(max(x, 1)) for x in P])
        v = np.where(P >= 1, np.exp((np.maximum(P, 1) - 1) * np.log(T) - lg) * E3 ** -X3 * E2 ** -X2,
                     np.exp(-E2 * T) * triv)
        best = v.max()
    if tipo == MEDIA:
        best = max(best, np.exp(-E1 * T) * triv)
    return min(triv, best)


def pesos_segmento(eps, T):
    b = A * eps
    yp = b * T
    F = np.zeros((4, LMAX + 1))
    Jt = {(m, t): J(m, t, T) for t in range(4) for m in range(1, MMAX + 1)}
    for lam in range(2, LMAX + 1):
        lo = MLO[lam]
        triv_tot = sum(CUB[lam][k] * np.expm1(yp) ** k for k in range(1, lam))
        for t in range(4):
            parcial = parcial_triv = 0.0
            for i, m in enumerate(range(lo, lo + K + 1)):
                tr = T ** m / factorial(m)
                parcial += b ** m * NC[lam][i] * Jt[(m, t)]
                parcial_triv += b ** m * NC[lam][i] * tr
            F[t, lam] = parcial + max(triv_tot - parcial_triv, 0.0)
    return F


def pesos_tramo(F, T, t, ell):
    """Matriz de transferencia sobre los sitios 0..2 ell (extremos intocados)."""
    pared = np.exp(-T) if t == MEDIA else 0.0
    # estado: longitud del segmento en curso en el último sitio procesado (0 = intocado)
    est = {0: 1.0}
    for c in range(ell):
        nuevo = {}
        ultimo = c == ell - 1
        for run, w in est.items():
            for s1 in (0, 1):
                for s2 in ((0,) if ultimo else (0, 1)):
                    ww, r = w, run
                    tocada = run > 0 or s1 or s2
                    for s in (s1, s2):
                        if s:
                            r += 1
                        else:
                            if r == 1:
                                ww = 0.0
                            elif r >= 2:
                                ww *= F[t, r]
                            r = 0
                    if not tocada:
                        ww *= pared
                    if ww > 0:
                        nuevo[r] = nuevo.get(r, 0.0) + ww
        est = nuevo
    return est.get(0, 0.0)


def pesos_fila_entera(eps, T, rmax):
    """Cota de la versión anterior: E(t) = 0 o >= 2 en toda la fila (celdas con saltos, q = 2)."""
    b = A * eps * L
    y = b * T
    Y1 = np.expm1(y)
    G = np.full((4, rmax + 1), np.inf)
    Jl = {(m, t): J(m, t, T) if t != MEDIA else min(T ** m / factorial(m), _Jfila(m, T))
          for t in range(4) for m in range(1, rmax + K + 1)}
    for t in range(4):
        for r in range(1, rmax + 1):
            tot = np.exp(-T * max(r, 2)) if t == MEDIA else 0.0
            for s in range(1, r + 1):
                libre = r <= q * s
                if t != MEDIA and not libre:
                    continue
                cp = np.exp(-T * max(max(1, r - q * s), 2)) if t == MEDIA else 0.0
                par = part = 0.0
                for m in range(s, s + K + 1):
                    tr = T ** m / factorial(m)
                    Jm = max(Jl[(m, t)] if libre else 0.0, cp)
                    par += b ** m * SURJ[m][s] * Jm
                    part += b ** m * SURJ[m][s] * tr
                tot += comb(r, s) * (par + max(Y1 ** s - part, 0.0) * (1.0 if libre else cp))
            G[t, r] = tot
    return G


def _Jfila(m, T):
    """Fila intermedia sin celdas de pared: E = 0 o >= 2 (sin la opción E >= 1)."""
    suc = sucesiones(m, MEDIA)
    P, X3, X2 = suc.T
    lg = np.array([lgamma(max(x, 1)) for x in P])
    v = np.where(P >= 1, np.exp((np.maximum(P, 1) - 1) * np.log(T) - lg) * E3 ** -X3 * E2 ** -X2,
                 np.exp(-E2 * T) * T ** m / factorial(m))
    return v.max()


def pesos_fila(eps, T):
    F = pesos_segmento(eps, T)
    Gt = np.zeros((4, RL + 1))
    for t in range(4):
        for ell in range(1, RL + 1):
            Gt[t, ell] = pesos_tramo(F, T, t, ell)
    # fila de una celda intermedia sin saltos: >= 2 paredes
    Gt[MEDIA, 1] += np.exp(-2 * T) - np.exp(-T)
    # fila de r celdas: máximo sobre las particiones en tramos
    G = np.zeros((4, RL + 1))
    for t in range(4):
        for r in range(1, RL + 1):
            G[t, r] = max([Gt[t, r]] + [Gt[t, l] * G[t, r - l] for l in range(1, r)])
    Gf = pesos_fila_entera(eps, T, RMAX)
    G[:, :RMAX + 1] = np.minimum(G[:, :RMAX + 1], np.where(np.isfinite(Gf), Gf, np.inf))
    G[:, 0] = 0
    h = 0.0
    for t in range(4):
        for r in range(1, RL + 1):
            if G[t, r] > 0:
                h = max(h, G[t, r] ** (1 / r))
        if G[t, RL - 1] > 0:
            h = max(h, G[t, RL] / G[t, RL - 1])
    return G, h


# ---------------- perfiles y KP ----------------
def leer_perfiles():
    filas = []
    for linea in open("scripts/perfiles.txt"):
        izq, der = linea.split("|")
        N, pp, rf, rl, R, c = map(int, izq.split())
        v = np.zeros(4 * (RMAX + 1))
        if R == 1:
            v[UNA * (RMAX + 1) + rf] += 1
        else:
            v[PRIMERA * (RMAX + 1) + rf] += 1
            v[ULTIMA * (RMAX + 1) + rl] += 1
            for r in map(int, der.split()):
                v[MEDIA * (RMAX + 1) + r] += 1
        filas.append((N, pp, c, v))
    return (np.array([f[0] for f in filas]), np.array([f[1] for f in filas], dtype=float),
            np.array([f[2] for f in filas], dtype=float), np.array([f[3] for f in filas]))


PN, PP, PC, PM = leer_perfiles()
NDAT = PN.max()
UV, INV = np.unique(PN + THETA * PP, return_inverse=True)


def logr(n):
    m = n - 1
    return log(4 / m) + lgamma(3 * m + 4) - lgamma(m) - lgamma(2 * m + 5)


NT = np.arange(NDAT + 1, Nmax + 1)
LRT = np.array([logr(int(n)) for n in NT])
AGRID = np.linspace(0.002, 0.6, 150)
COLA = int(sys.argv[6]) if len(sys.argv) > 6 else 1  # 1: cola por percolación; 0: árboles


def _delta_perc():
    """delta(x) = x - sum_{N <= NDAT} c_{N,p} x^N (1-x)^p = P_x(0 abierto, NDAT < |C| < oo) en una malla."""
    import mpmath
    mpmath.mp.dps = 40
    cnp = [tuple(map(int, l.split())) for l in open("scripts/perimetros.txt")]
    xs = np.linspace(0.005, 1 / 3, 1500)
    d = []
    for x in xs:
        X = mpmath.mpf(x)
        d.append(float(X - mpmath.fsum(c * X ** n * (1 - X) ** pp for n, pp, c in cnp if n <= NDAT)))
    return xs, np.array(d)


XP, DP = _delta_perc()
GP = XP * (1 - XP) ** 2


def cola_perc(hp, b):
    """Cola N > NDAT: sum c_{N,p} h^N e^{3a(N + theta p)} <= (e^b/(1-x))^2 delta(x) si h' <= x(1-x)^2."""
    i = np.searchsorted(GP, hp)
    if i >= len(XP):
        return None
    x = XP[i]
    return (np.exp(b) / (1 - x)) ** 2 * max(DP[i], 0.0)


VACIO = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 1: gas de contornos de una fila resumado antes (9.11)
UNAF = PM[:, UNA * (RMAX + 1):(UNA + 1) * (RMAX + 1)].sum(1) > 0
UVL, INVL = np.unique((PN + THETA * PP)[~UNAF], return_inverse=True)
NPL = (PN + PP)[~UNAF]
BGRID = np.linspace(0.005, 1.0, 200)


def gas_una_fila(G, h):
    """sigma = min_b S_s(b) con S_s(b) = sum_N N G_una(N) e^{b(N + theta(2N+2))} <= b r_s (KP del gas de una fila)."""
    Ns = np.arange(2, RL + 1)
    g = Ns * G[UNA, 2:RL + 1]
    rs = ((Ns + THETA * (2 * Ns + 2)) / (3 * Ns + 2)).min()
    for b in BGRID:
        cola_r = h * np.exp(b * (1 + 2 * THETA))
        if cola_r >= 0.9:
            return None
        S = (g * np.exp(b * (Ns + THETA * (2 * Ns + 2)))).sum() + (RL + 1) * cola_r ** (RL + 1) / (1 - cola_r) ** 2 * np.exp(
            2 * b * THETA)
        if S <= b * rs:
            return S
    return None


def factible_T(eps, T):
    G, h = pesos_fila(eps, T)
    lG = np.log(np.maximum(G[:, :RMAX + 1], 1e-300)).ravel()
    lw0 = PM @ lG + np.log(PC)
    res = []
    if VACIO:
        sig = gas_una_fila(G, h)
        if sig is None:
            return res
        # contornos de varias filas: actividad efectiva <= W e^{3 sigma (N + p)}
        wL = np.bincount(INVL, weights=np.exp(lw0[~UNAF] + 3 * sig * NPL), minlength=len(UVL))
        for a in AGRID:
            hp = h * np.exp(3 * a * (1 + 2 * THETA) + 9 * sig)
            cola = cola_perc(hp, 3 * a * THETA + 3 * sig)
            if cola is None:
                continue
            S = (wL * np.exp(3 * a * UVL)).sum() + cola
            if S / (1 + 4 * THETA) <= a / nu:
                res.append(a)
        return res
    w = np.bincount(INV, weights=np.exp(lw0), minlength=len(UV))
    for a in AGRID:
        cola = cola_perc(h * np.exp(3 * a * (1 + 2 * THETA)), 3 * a * THETA) if COLA else None
        if cola is None:
            lt = LRT + NT * np.log(h) + 3 * a * (NT + THETA * (2 * NT + 2))
            if lt[-1] > lt.max() - 30:
                continue
            cola = np.exp(lt).sum()
        S = (w * np.exp(3 * a * UV)).sum() + cola
        if S / (1 + 4 * THETA) <= a / nu:
            res.append(a)
    return res


TGRID = np.linspace(1.0, 6.0, 51)


def factible(eps):
    mejor = None
    for T in TGRID:
        for a in factible_T(eps, T):
            if mejor is None or a / T > mejor[2]:
                mejor = (T, a, a / T)
    return mejor


if __name__ == "__main__":
    lo, hi = 1e-6, 1e-1
    for _ in range(25):
        mid = np.sqrt(lo * hi)
        if factible(mid):
            lo = mid
        else:
            hi = mid
    T, a, g = factible(lo)
    print(f"A={A} theta={THETA} E1={E1} E2={E2} E3={E3}: eps_2 = {lo:.3e}  (T={T:.2f}, a={a:.3f}, gamma_1={g:.3f})")
    G, h = pesos_fila(lo, T)
    print(f"  h = {h:.3f};  G(r = 1..4):")
    for t, nom in enumerate(("una", "primera", "última", "media")):
        print(f"   {nom:8s}", np.array2string(G[t, 1:5], precision=3))
    for f in (3, 10):
        r = factible(lo / f)
        print(f"   eps = eps_2/{f}: gamma_1 = {r[2]:.3f} (T={r[0]:.2f}, a={r[1]:.2f})")

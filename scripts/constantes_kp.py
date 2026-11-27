"""Constantes de los Lemas Z15-Z17 (celda_k_notas.md, Sec. 7.13, pasos 9.7-9.8): radio eps_2 del desarrollo de
contornos en espacio-tiempo para la carga entera con k | n.

Celdas: L enlaces consecutivos durante un tramo T. Incorrecta si algún salto toca sus L+1 sitios en el tramo, o
si alguno de sus enlaces es pared. Un salto vuelve incorrectas a lo sumo q celdas (q = 3 si L = 1, 2 si L >= 2).

Peso de un contorno de N celdas (suma sobre historias, con el conjunto S de celdas con saltos):
    W_N <= e^{-TN} [ (1 + Y e^{qT})^N - 1 - N (Y - Y2) e^{qT} ] ,   Y = y e^y ,  Y2 = y^2 e^y / 2 ,  y = A|eps| T L,
donde las celdas que no son de S ni cubiertas por S son paredes todo el tramo (e^{-T}), y el término corregido
quita S vacío e impone >= 2 saltos (todo contorno contraíble necesita uno que crea paredes y otro que las destruye).

Conectividad: dos celdas correctas en diagonal también comparten un sitio y un instante, así que las etiquetas son
constantes en las componentes 8-conexas de celdas correctas, y los contornos pueden ser 4-conexos: Delta = 4.
Conjuntos conexos de N celdas con una dada: <= (e(Delta-1))^{N-1}.
KP (Kotecky-Preiss, a(gamma) = a|gamma|): sum_{gamma ∋ 0} |z| e^{a|gamma|} <= a/nu, nu = Delta + 1 (celda y
vecinas), con |z(gamma)| <= W e^{2a|gamma|} (Z16: el cociente de interiores cuesta e^{2a|gamma|}).
Con Delta = 4 y CUENTA = 1 se usan los c_N exactos de scripts/animales.txt (poliominós fijos, Redelmeier) y,
más allá, la cota de árboles r_N = [x^N] x(1+B)^4, B = x(1+B)^3, que crece como (27/4)^N.
KP con a(gamma) = a (N + theta p), p = celdas vecinas: la condición es S <= a (1 + 4 theta)/5 (el peor caso es N = 1),
con S = sum W_N e^{3a(N + theta p)} (THETA = 0 es la de nu = 5).
Con EXTREMOS = 1 se usa además la integral exacta del primer y el último salto: tras la creación (y antes de la
aniquilación) hay EMIN paredes, así que la integral en ese instante da <= 1/EMIN en lugar de T. Factor
kappa = r_f r_l / (EMIN T)^2 (una sola fila y N <= 4: 2N/(EMIN T)), con min(1, kappa). Datos en scripts/filas.txt.
Uso: python scripts/constantes_kp.py [A] [L] [Delta] [dos_saltos=1] [CUENTA=1] [THETA=0] [EXTREMOS=0] [EMIN=3]
"""
import sys

import numpy as np

A = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Delta = int(sys.argv[3]) if len(sys.argv) > 3 else 4
dos = int(sys.argv[4]) if len(sys.argv) > 4 else 1
q = 3 if L == 1 else 2
mu = np.e * (Delta - 1)
nu = Delta + 1
Nmax = 400
CUENTA = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # 1: c_N exactos (animales.txt) + cota de árboles (Delta=4)


def log_cuentas():
    from math import lgamma, log
    N = np.arange(1, Nmax + 1)
    if Delta != 4 or not CUENTA:
        return (N - 1) * np.log(mu)
    # cota de árboles r_N = 4/m C(3m+3, m-1), m = N-1
    def logr(n):
        if n == 1:
            return 0.0
        m = n - 1
        return log(4 / m) + lgamma(3 * m + 4) - lgamma(m) - lgamma(2 * m + 5)
    lc = np.array([logr(int(n)) for n in N])
    try:
        for linea in open("scripts/animales.txt"):
            n, c, r = map(int, linea.split())
            lc[n - 1] = log(c)
    except FileNotFoundError:
        pass
    return lc


LOGC = log_cuentas()
THETA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0  # a(gamma) = a (N + theta p)


def perimetros():
    """lista por N de (p, log c_{N,p}); exacto si está en perimetros.txt, si no p = 2N+2 con la cota de árboles."""
    from collections import defaultdict
    d = defaultdict(list)
    try:
        for linea in open("scripts/perimetros.txt"):
            n, pp, c = map(int, linea.split())
            d[n].append((pp, np.log(c)))
    except FileNotFoundError:
        pass
    out = []
    for n in range(1, Nmax + 1):
        out.append(d[n] if d[n] else [(2 * n + 2, LOGC[n - 1])])
    return out


PER = perimetros()
EXTREMOS = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 1: integral exacta de la creación y la aniquilación
EMIN = float(sys.argv[8]) if len(sys.argv) > 8 else 3.0  # paredes justo después de crear / antes de aniquilar


def datos_filas():
    """(N, p, log c, tipo, r) por clase: tipo 1 = una fila con N <= 4 (kappa = 2 N rho), tipo 2 = dos extremos
    (kappa = r rho^2, r = r_f r_l, o N^2 si es una fila con N >= 5). Más allá de 13: p = 2N+2, r = N^2/4... cota N^2."""
    filas = []
    nmax_dat = 0
    try:
        for linea in open("scripts/filas.txt"):
            n, pp, rf, rl, una, c = map(int, linea.split())
            nmax_dat = max(nmax_dat, n)
            if una and n <= 4:
                filas.append((n, pp, np.log(c), 1, 2 * n))
            elif una:
                filas.append((n, pp, np.log(c), 2, n * n))
            else:
                filas.append((n, pp, np.log(c), 2, rf * rl))
    except FileNotFoundError:
        pass
    for n in range(nmax_dat + 1, Nmax + 1):
        filas.append((n, 2 * n + 2, LOGC[n - 1], 2, n * n))
    f = np.array(filas, dtype=float)
    return f[:, 0].astype(int), f[:, 1], f[:, 2], f[:, 3].astype(int), f[:, 4]


FN, FP, FL, FT, FR = datos_filas()
PN = np.array([n for n in range(1, Nmax + 1) for pp, lc in PER[n - 1]])
PP = np.array([pp for n in range(1, Nmax + 1) for pp, lc in PER[n - 1]], dtype=float)
PL = np.array([lc for n in range(1, Nmax + 1) for pp, lc in PER[n - 1]])


def kp(eps, T, a):
    y = A * eps * T * L
    if y >= 1:
        return np.inf
    Y = y * np.exp(y)
    Y2 = y * y * np.exp(y) / 2
    N = np.arange(1, Nmax + 1, dtype=float)
    X = Y * np.exp(q * T)  # peso relativo de una celda con saltos frente a una de pared
    # (1+X)^N - 1 [- N X + N X Y2/Y]: sin S vacío (sin saltos) [y sin un único salto]
    lp = N * np.log1p(X)
    sub = 1 + (N * X * (1 - Y2 / Y) if dos else 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        directo = np.expm1(lp) - (sub - 1)
        r = sub * np.exp(-lp)
        logc = np.where(lp < 30, np.log(np.maximum(directo, 1e-300)), lp + np.log1p(-np.minimum(r, 0.999999)))
    if EXTREMOS:
        rho = 1 / (EMIN * T)
        logk = np.where(FT == 1, np.log(FR * rho), np.log(FR) + 2 * np.log(rho))
        logk = np.minimum(logk, 0.0)
        tot = FL - T * FN + logc[FN - 1] + logk + 3 * a * (FN + THETA * FP)
        ult = tot[FN == Nmax].max()
        if ult > tot.max() - 30:
            return np.inf
        return np.exp(tot).sum() / (1 + 4 * THETA)
    if THETA == 0:
        logt = LOGC - T * N + logc + 3 * a * N
        if logt[-1] > logt.max() - 30 or np.diff(logt[-10:]).max() > 0:
            return np.inf
        return np.exp(logt).sum()
    tot = PL - T * PN + logc[PN - 1] + 3 * a * (PN + THETA * PP)
    ult = tot[PN == Nmax].max()
    if ult > tot.max() - 30:
        return np.inf
    return np.exp(tot).sum() / (1 + 4 * THETA)


def factible(eps):
    mejor = None
    for T in np.linspace(1.5, 6, 150):
        for a in np.linspace(0.005, 0.6, 120):
            if kp(eps, T, a) <= a / nu:
                if mejor is None or a / T > mejor[2]:
                    mejor = (T, a, a / T)
    return mejor


lo, hi = 1e-10, 1e-1
for _ in range(40):
    mid = np.sqrt(lo * hi)
    if factible(mid):
        lo = mid
    else:
        hi = mid
T, a, g = factible(lo)
print(f"A={A} L={L} Delta={Delta} dos_saltos={dos}: eps_2 = {lo:.3e}  (T={T:.2f}, a={a:.3f}, gamma_1={g:.3f})")
for f in (3, 10):
    T, a, g = factible(lo / f)
    print(f"   eps = eps_2/{f}: gamma_1 = {g:.3f} (T={T:.2f}, a={a:.2f})")

"""Radio eps_2 con contornos en tiempo continuo (celda_k_notas.md, Sec. 7.13, pasos 9.12-9.13). S+S-, A = 4.

Contorno: componente conexa del conjunto excitado X = {(i,t): enlace i es pared en t} ∪ {(i,t_v): i ∈ {j-1,j,j+1}} para
cada salto v (enlace j, instante t_v). Marco: paredes vivas de un contorno que descienden de un mismo nacimiento.
Racha: paredes consecutivas de un contorno en un instante (una racha pertenece a un solo marco).

Hechos (9.13): cada marco suma 0 mod k (conservación), así que tiene >= 2 paredes; nace con 3 (una racha) y el
contorno muere desde E = 3, r = 1, f = 1. Una racha puede tener UNA pared (borde de un interior de otra fase), así que
se lleva r (número de rachas) en el estado. Un salto que toca un marco A cae en <= P_A = E_A + 4 r_A enlaces.
Un salto cambia E en [-3, 3] y r en [-2, 2].

V(E,r,f) = b'/((1-lam)E) [ 1_{(3,1,1)} + P max V(.,.,f) + V(E+3,r+1,f+1) + e2 max V(.,.,f-1) + e3 max V(.,.,f-2) ],
P = E + 4r,  e2 = C(f,2)(P/f)^2,  e3 = C(f,3)(P/f)^3  (Maclaurin: suma sobre pares/ternas de marcos de Π P_X),
máximos sobre E' en [E-3,E+3], r' en [r-2,r+2]. b' = A|eps| e^{3 a kappa}. W_anc = b' V(3,1,1).

KP (versión continua; a(gamma) = a(int E dt + kappa m)):  5 M1 <= a kappa,  5 M2 <= a,
M1 <= W_anc(lam + s)/(e s),  M2 <= W_anc(b' e^s)/(e s),  lam = 3a (+ c si se pide decaimiento en tiempo para Z17).

Uso: python scripts/constantes_continuo.py [EMAX=40] [modo=z16|z17]
"""
import sys
from math import comb

import numpy as np
from scipy.ndimage import maximum_filter

A = 4.0
EMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODO = sys.argv[2] if len(sys.argv) > 2 else "z16"

EX = EMAX + 6  # E <= EMAX: numérico; EMAX < E <= EX: supersolución de cola
FX = EX // 2
E = np.arange(EX + 1)[:, None, None].astype(float)
R = np.arange(EX + 1)[None, :, None].astype(float)
F = np.arange(FX + 1)[None, None, :].astype(float)
FACT = (F >= 1) & (2 * F <= E) & (R >= F) & (R <= E)  # estados posibles
NUM = FACT & (E <= EMAX)
COLA = FACT & (E > EMAX)
COSTURA = FACT & (E > EMAX) & (E <= EMAX + 3)
P = E + 4 * R
Fs = np.maximum(F, 1)
E2 = np.where(F >= 2, np.array([comb(int(f), 2) for f in range(FX + 1)])[None, None, :] * (P / Fs) ** 2, 0.0)
E3 = np.where(F >= 3, np.array([comb(int(f), 3) for f in range(FX + 1)])[None, None, :] * (P / Fs) ** 3, 0.0)
Esafe = np.maximum(E, 1.0)
MUERTE = np.zeros(FACT.shape)
MUERTE[3, 1, 1] = 1.0
HUELLA = np.ones((7, 5, 1), dtype=bool)


def _lp_clases(b1, b2, b3, Ee, r):
    """max n1 b1 + n2 b2 + n3 b3 con n >= 0, n1+n2+n3 <= N = E + 2r, n1 + 2 n2 + 3 n3 = 3E (vértices).
    n_w = posiciones con w paredes en la ventana [j-1, j+1]; su salto da Delta E en [3-2w, 3-w]."""
    N = Ee + 2 * r
    cands = [
        (0 * Ee, 0 * Ee, Ee, True),
        ((3 * N - 3 * Ee) / 2, 0 * Ee, (3 * Ee - N) / 2, True),
        (0 * Ee, 1.5 * Ee, 0 * Ee, 1.5 * Ee <= N),
        (3 * Ee, 0 * Ee, 0 * Ee, 3 * Ee <= N),
        (2 * N - 3 * Ee, 3 * Ee - N, 0 * Ee, N >= 1.5 * Ee),
        (0 * Ee, 3 * N - 3 * Ee, 3 * Ee - 2 * N, N <= 1.5 * Ee),
    ]
    out = None
    for n1, n2, n3, ok in cands:
        v = np.where(ok, n1 * b1 + n2 * b2 + n3 * b3, -np.inf)
        out = v if out is None else np.maximum(out, v)
    return out


def _crec_tail(q, x):
    """(1/E) * cota del término de crecimiento para U = K rho^E (DE)^f: razón q^{Delta E} por posición."""
    b = {w: np.max([q**d for d in range(3 - 2 * w, 4 - w)], axis=0) for w in (1, 2, 3)}
    return 2 * x * q**3 + _lp_clases(b[1], b[2], b[3], np.ones_like(x), x)


def _cola_const(E1):
    """min_{rho,D} S(rho,D): para E >= E1, T(U) <= U con U = K rho^E (D E)^f si b'/(1-lam) S <= 1.
    S = sup_{phi<=1/2, x=r/E<=1} [crec(q, x) + rho^3 D (1+3/E1) e^{3 phi} + (1+4x)^2 Mt/(2D) + (1+4x)^3 Mt/(6D^2)],
    q = rho e^phi, Mt = max(q^3, rho^-3, q^-3)."""
    ph, x = np.meshgrid(np.linspace(0, 0.5, 101), np.linspace(0, 1, 51), indexing="ij")
    best = (np.inf, 1.0, 1.0)
    for rho in np.linspace(0.35, 1.0, 66):
        q = rho * np.exp(ph)
        Mt = np.maximum(np.maximum(q**3, q**-3), rho**-3)
        cr = _crec_tail(q, x)
        for D in np.linspace(2, 40, 153):
            S = np.max(cr + rho**3 * D * (1 + 3 / E1) * np.exp(3 * ph) + (1 + 4 * x) ** 2 * Mt / (2 * D)
                       + (1 + 4 * x) ** 3 * Mt / (6 * D * D))
            if S < best[0]:
                best = (S, rho, D)
    return best


S_COLA, RHO, DCOLA = _cola_const(EMAX + 4)
FORMA = np.where(COLA, RHO**E * (DCOLA * E) ** F, 0.0)
HUELLA_R = np.ones((1, 5, 1), dtype=bool)


def _desplaza_E(A_, d):
    out = np.zeros_like(A_)
    if d > 0:
        out[:-d] = A_[d:]
    elif d < 0:
        out[-d:] = A_[:d]
    else:
        out[:] = A_
    return out


def _T(V, pref):
    M = maximum_filter(V, footprint=HUELLA, mode="constant", cval=0.0)
    Mr = maximum_filter(V, footprint=HUELLA_R, mode="constant", cval=0.0)  # max sobre r' en [r-2, r+2]
    v = {d: _desplaza_E(Mr, d) for d in range(-3, 4)}  # v[d] = max_r' V(E+d, r', f)
    b1 = np.maximum(v[1], v[2])
    b2 = np.maximum(np.maximum(v[-1], v[0]), v[1])
    b3 = np.maximum(np.maximum(np.maximum(v[-3], v[-2]), v[-1]), v[0])
    crec = 2 * R * v[3] + _lp_clases(b1, b2, b3, E, R)
    nac = np.zeros_like(V)
    nac[:-3, :-1, :-1] = V[3:, 1:, 1:]
    M1 = np.zeros_like(V)
    M1[:, :, 1:] = M[:, :, :-1]
    M2 = np.zeros_like(V)
    M2[:, :, 2:] = M[:, :, :-2]
    return pref * (MUERTE + crec + nac + E2 * M1 + E3 * M2)


def _punto_fijo(V, pref, iters, tol):
    for _ in range(iters):
        N = np.where(NUM, _T(V, pref), V)
        if not np.isfinite(N[3, 1, 1]) or N[3, 1, 1] > 1e6:
            return None
        if np.max(np.abs(N - V)) < tol * max(1.0, N[3, 1, 1]):
            return N
        V = N
    return None


def W_anclada(bp, lam, iters=600, tol=1e-12, riguroso=True):
    """b' V(3,1,1). riguroso=False: truncamiento (V = 0 para E > EMAX), solo estimación.
    riguroso=True: cota superior. U = punto fijo numérico en E <= EMAX con borde U = K rho^E (D E)^f en E > EMAX;
    se comprueba T(U) <= U en la costura (EMAX < E <= EMAX+3) y, analíticamente, b'/(1-lam) S_COLA <= 1 para
    E >= EMAX+4. Entonces U es supersolución y V* <= U. None si no se puede certificar."""
    if lam >= 1:
        return None
    pref = bp / ((1 - lam) * Esafe)
    if not riguroso:
        V = _punto_fijo(np.zeros(FACT.shape), pref, iters, tol)
        return None if V is None else bp * V[3, 1, 1]
    if bp / (1 - lam) * S_COLA > 1:
        return None
    V0 = _punto_fijo(np.zeros(FACT.shape), pref, iters, tol)
    if V0 is None:
        return None
    borde = NUM & (E >= EMAX - 2)
    K = 2 * np.max(np.where(borde, V0 / np.where(borde, RHO**E * (DCOLA * E) ** F, 1.0), 0.0))
    for _ in range(12):
        U = _punto_fijo(np.where(NUM, V0, K * FORMA), pref, iters, tol)
        if U is None:
            return None
        TU = _T(U, pref)
        if np.all(TU[COSTURA] <= U[COSTURA] * (1 + 1e-12)):
            return bp * U[3, 1, 1]
        K *= 4
    return None


def condiciones(eps, a, kap, s, c=0.0):
    bp = A * eps * np.exp(3 * a * kap)
    lam = 3 * a + c
    W1 = W_anclada(bp, lam + s)
    W2 = W_anclada(bp * np.exp(s), lam)
    if W1 is None or W2 is None:
        return None
    M1, M2 = W1 / (np.e * s), W2 / (np.e * s)
    return M1, M2, 5 * M1 <= a * kap and 5 * M2 <= a


AS = np.concatenate([np.linspace(0.001, 0.005, 5), np.linspace(0.0075, 0.2, 40)])


def factible(eps, c=0.0):
    for kap in (1.0, 2.0, 3.0, 4.0):
        for s in (0.05, 0.1, 0.2, 0.4):
            for a in AS:
                r = condiciones(eps, a, kap, s, c)
                if r is None:
                    break  # a mayor solo empeora la convergencia
                if r[2]:
                    return a, kap, s, r[0], r[1]
    return None




def z17(eps, mejor=False):
    """Condiciones de Z17 (9.13): KP con decaimiento en tiempo c (lam = 3a + c) y gas de eventos en 1D.
    Evento = contorno espacial (m >= ceil(n/3)); peso <= |W| e^{4a(gamma)} e^{nu int E} max(1, Bt), con
    nu = 3 W_anc + mu/2 + nu', Bt = (1/e)(1/(e(2c - mu)) + 1/(2 e nu')), mu = c, nu' = c/2.
    q = max(1,Bt) max_n [n e^{-sigma ceil(n/3)}] W_anc(A eps e^{4 a kappa + sigma}, 4a + nu).
    Condición 1D (KP con a(C) = (mu/2)(1 + longitud)): q/(1-q) <= e mu^2 e^{-mu/2} / 4.
    Devuelve los parámetros (o, con mejor=True, los que dan más margen)."""
    best = None
    for c in (0.02, 0.05, 0.1, 0.2, 0.5):
        for kap in (1.0, 2.0, 3.0):
            for s in (0.05, 0.1, 0.2):
                for a in AS:
                    r = condiciones(eps, a, kap, s, c)
                    if r is None:
                        break
                    if not r[2]:
                        continue
                    Wb = W_anclada(A * eps * np.exp(3 * a * kap), 3 * a)
                    mu, nup = c, c / 2
                    nu = 3 * Wb + mu / 2 + nup
                    Bt = (1 / (np.e * (2 * c - mu)) + 1 / (2 * np.e * nup)) / np.e
                    cota = np.e * mu**2 * np.exp(-mu / 2) / 4
                    for sig in (0.1, 0.2, 0.5, 1.0):
                        Wq = W_anclada(A * eps * np.exp(4 * a * kap + sig), 4 * a + nu)
                        if Wq is None:
                            break
                        nmax = max(n * np.exp(-sig * np.ceil(n / 3)) for n in range(3, 200))
                        q = max(1.0, Bt) * nmax * Wq
                        if q < 1 and q / (1 - q) <= cota:
                            res = (c, a, kap, s, sig, q, cota)
                            if not mejor:
                                return res
                            if best is None or q / cota < best[5] / best[6]:
                                best = res
    return best


def biseccion(pred, lo=1e-3, hi=0.1, pasos=14):
    for _ in range(pasos):
        mid = np.sqrt(lo * hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    eps0 = 1e-3
    b = A * eps0
    print(f"comprobación: W_anc(eps=1e-3) = {W_anclada(b, 0.0):.4e},  b^2/3 = {b * b / 3:.4e}")
    if MODO == "z16":
        e2 = biseccion(lambda e: factible(e) is not None)
        print(f"EMAX={EMAX}: eps_2 = {e2:.3e}  (a, kappa, s, M1, M2) = {factible(e2)}")
    else:
        print(f"EMAX={EMAX}: eps_2(Z17) = {biseccion(lambda e: z17(e) is not None):.3e}")
        for eps in (2e-3, 5e-3):
            print(f"  eps = {eps:.0e}: (c, a, kappa, s, sigma, q, cota) = {z17(eps, mejor=True)}")

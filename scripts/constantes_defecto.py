"""Radio eps_3 del Teorema D''' (celda_k_notas.md, 9.14): el gas unidimensional de excursiones del defecto.

Excursión: tramo de Gamma entre dos instantes libres (una sola pared). Marcos como en 9.12, más el marco del defecto
(suma no nula: >= 1 pared, no muere). Estado (E, r, f) con f marcos (el del defecto incluido); la excursión acaba
al volver a (1, 1, 1). Recurrencia (máximos sobre vecinos, clases de posición y fusiones como en 9.12-9.13):

  V(E,r,f) = b'/den(E) [ crec + V(E+3, r+1, f+1) + e2 max V(f-1) + e3 max V(f-2) ],   V(1,1,1) = 1 (fin),
  den(E) = (1 - lam) E - (1 + eta) - mu ,

  lam = 3a + c (pesos de KP del vacío y de los interiores, como en Z16/Z17), eta = 2a (racimos del vacío que tocan
  los tramos libres: densidad b_Q y enlaces), mu = peso de KP del gas 1D, a(exc) = mu (duración + 1).

Densidad de excursiones por unidad de tiempo (desde un instante libre, con E' >= 2 tras el primer salto):
  m0(mu) = e^mu b' [ 5 max_{E' in [2,4], r', f = 1} V + V(4, 2, 2) ] ,   m1 <= m0(mu + s1)/(e s1) .
KP 1D (solapamiento en tiempo): m0 <= mu y m1 <= mu.

Uso: python scripts/constantes_defecto.py [EMAX=40]
"""
import sys

import numpy as np
from scipy.ndimage import maximum_filter

EMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
A = 4.0
FX = (EMAX + 1) // 2 + 1
E = np.arange(EMAX + 1)[:, None, None].astype(float)
R = np.arange(EMAX + 1)[None, :, None].astype(float)
F = np.arange(FX + 1)[None, None, :].astype(float)
# defecto >= 1 pared, los demás marcos >= 2; una racha por marco como mínimo
FACT = (F >= 1) & (E >= 2 * F - 1) & (R >= F) & (R <= E)
FIN = np.zeros(FACT.shape, dtype=bool)
FIN[1, 1, 1] = True
LIBRE = FACT & ~FIN
P = E + 4 * R
Fs = np.maximum(F, 1)
from math import comb  # noqa: E402

E2 = np.where(F >= 2, np.array([comb(int(f), 2) for f in range(FX + 1)])[None, None, :] * (P / Fs) ** 2, 0.0)
E3 = np.where(F >= 3, np.array([comb(int(f), 3) for f in range(FX + 1)])[None, None, :] * (P / Fs) ** 3, 0.0)
HUELLA = np.ones((7, 5, 1), dtype=bool)
HUELLA_R = np.ones((1, 5, 1), dtype=bool)


def _lp_clases(b1, b2, b3, Ee, r):
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


def _desplaza(A_, d):
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
    Mr = maximum_filter(V, footprint=HUELLA_R, mode="constant", cval=0.0)
    v = {d: _desplaza(Mr, d) for d in range(-3, 4)}
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
    return pref * (crec + nac + E2 * M1 + E3 * M2)


def m0(bp, lam, eta, mu, iters=3000, tol=1e-13):
    den = np.broadcast_to((1 - lam) * E - (1 + eta) - mu, FACT.shape)
    if np.any(den[LIBRE] <= 0):
        return None
    pref = np.where(LIBRE, bp / np.where(LIBRE, den, 1.0), 0.0)
    V = np.where(FIN, 1.0, 0.0)
    for _ in range(iters):
        N = np.where(LIBRE, _T(V, pref), V)
        if not np.isfinite(N).all() or N.max() > 1e250:
            return None
        if np.all(np.abs(N - V) <= tol * np.abs(N)):
            V = N
            break
        V = N
    else:
        return None
    primer = max(V[e, r, 1] for e in (2, 3, 4) for r in range(1, 4))
    return np.exp(mu) * bp * (5 * primer + V[4, 2, 2])


def gas_ok(bp, lam, eta):
    """Busca mu, s1 con m0(mu) <= mu y m0(mu+s1)/(e s1) <= mu."""
    for mu in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        M0 = m0(bp, lam, eta, mu)
        if M0 is None or M0 > mu:
            continue
        for s1 in (0.05, 0.1, 0.2, 0.3):
            M = m0(bp, lam, eta, mu + s1)
            if M is not None and M / (np.e * s1) <= mu:
                return mu, s1, M0, M / (np.e * s1)
    return None


# Parámetros del vacío certificados en 9.13 e) para Z17 (valen para todo |eps| <= 8,198e-3).
VAC = dict(c=0.2, a=0.012435897435897436, kap=1.0)

if __name__ == "__main__" and len(sys.argv) <= 2:
    a, c, kap = VAC["a"], VAC["c"], VAC["kap"]
    lam, eta = 3 * a + c, 2 * a

    def ok(eps):
        return gas_ok(A * eps * np.exp(3 * a * kap), lam, eta)

    lo, hi = 1e-4, 8.198e-3
    if ok(hi):
        lo = hi
    else:
        for _ in range(16):
            mid = np.sqrt(lo * hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
    print(f"EMAX={EMAX}: eps_3 = {lo:.4e}  (mu, s1, m0, m1) = {ok(lo)}")
    for eps in (1e-3, 2e-3, 5e-3):
        print(f"  eps = {eps:.0e}: {ok(eps)}")


# ---------------------------------------------------------------------------------------------------------------
# Versión con número de saltos y momento en la duración (9.14, «ε_4 sin Cauchy»).
# V^(j): completar la excursión tras j saltos, contando solo los finales que cumplen el criterio en el número total
# de saltos (j = 3 significa >= 3). U^(j): lo mismo pesado con la duración, U = V/den + T(U) (derivar 1/den).

def _resuelve(V0, pref, src=None, iters=4000, tol=1e-13):
    V = V0.copy()
    for _ in range(iters):
        N = np.where(LIBRE, _T(V, pref) + (0 if src is None else src), V0)
        if not np.isfinite(N).all() or N.max() > 1e250:
            return None
        if np.all(np.abs(N - V) <= tol * np.abs(N) + 1e-300):
            return N
        V = N
    return None


def excursiones(bp, lamE, eta, mu1, crit=lambda j: 1.0):
    """Densidad por unidad de tiempo (sin e^{mu0}) de excursiones, y su momento en la duración, con peso
    e^{mu1 * duración}. crit(j) en {0,1}: qué números totales de saltos se cuentan (j = 3 es >= 3)."""
    den = np.broadcast_to((1 - lamE) * E - (1 + eta) - mu1, FACT.shape)
    if np.any(den[LIBRE] <= 0):
        return None
    pref = np.where(LIBRE, bp / np.where(LIBRE, den, 1.0), 0.0)
    inv = np.where(LIBRE, 1 / np.where(LIBRE, den, 1.0), 0.0)
    V, U = {}, {}
    V[3] = _resuelve(np.where(FIN, crit(3), 0.0), pref)
    if V[3] is None:
        return None
    U[3] = _resuelve(np.zeros(FACT.shape), pref, src=V[3] * inv)
    if U[3] is None:
        return None
    for j in (2, 1):
        V[j] = np.where(LIBRE, _T(V[j + 1], pref), np.where(FIN, crit(j), 0.0))
        U[j] = np.where(LIBRE, V[j] * inv + _T(U[j + 1], pref), 0.0)

    def inicio(X):
        return bp * (5 * max(X[e, r, 1] for e in (2, 3, 4) for r in range(1, 4)) + X[4, 2, 2])

    return inicio(V[1]), inicio(U[1])


def gas_kp(bp, lamE, eta, mus=np.geomspace(1e-5, 0.4, 160)):
    """KP 1D con a(exc) = mu0 + mu1 * duración: basta m0 <= mu1 y m1 <= mu0, con
    m0 = e^{mu0} * densidad y m1 = e^{mu0} * momento (ambos con peso e^{mu1 dur})."""
    for mu1 in mus:
        r = excursiones(bp, lamE, eta, mu1)
        if r is None:
            return None
        d0, d1 = r
        for mu0 in mus:
            if np.exp(mu0) * d0 <= mu1 and np.exp(mu0) * d1 <= mu0:
                return mu0, mu1, np.exp(mu0) * d0, np.exp(mu0) * d1
    return None


def resto(eps, gas=None):
    """Cota de |omega_Q - 1 - eps^2 w_Q| (todo lo de >= 3 saltos) para S+S-. Vestido por el vacío con las
    densidades de KP de 9.12: un racimo del vacío que toca un objeto con m saltos y pared integrada int E pesa en
    total <= 5 m M1 + 5 int E M2; el vestido (interiores x2 + contacto) es <= e^{3(...)}; el tramo libre, e^{5 M2}."""
    import constantes_continuo as cc  # noqa: E402

    a, kap, s = VAC["a"], VAC["kap"], 0.1

    def Ms(w):
        bpw = A * eps * np.exp(w * a * kap)
        W1 = cc.W_anclada(bpw, w * a + s, riguroso=False)
        x = bpw * np.exp(s)
        W2 = cc.W_anclada(x, w * a, riguroso=False)
        W2_3 = W2 - x * x / (3 * (1 - w * a))  # quita el par de dos saltos, b'^2/(3(1-lam))
        return W1 / (np.e * s), W2 / (np.e * s), W2_3 / (np.e * s)

    M1, M2, M2_3 = Ms(3)
    _, M2b, _ = Ms(2)
    am, aE, eta = 15 * M1, 15 * M2, 5 * M2
    bpd = A * eps * np.exp(am)
    ge3 = excursiones(bpd, aE, eta, 0.0, lambda j: 1.0 if j == 3 else 0.0)
    eq2 = excursiones(bpd, aE, eta, 0.0, lambda j: 1.0 if j == 2 else 0.0)
    if ge3 is None or eq2 is None:
        return None
    R1 = ge3[0]
    R1v = 2 * am * eq2[0] + (4 * aE + eta) * eq2[1]  # X e^X, X <= am m + aE int E + eta dur, int E <= 4 dur
    R2 = 5 * (M2_3 + (M2 - M2b))
    g = gas_kp(bpd, aE, eta)
    if g is None:
        return None
    mu0, mu1, m0, m1 = g
    R3 = mu0 * m0 + mu1 * m1  # racimos de >= 2 excursiones: |K|(e^{a}-1) <= a |K| e^{a}
    return dict(R1=R1, R1v=R1v, R2=R2, R3=R3, tot=R1 + R1v + R2 + R3, M1=M1, M2=M2, gas=g)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "eps4":
    a, c = VAC["a"], VAC["c"]
    lam, eta = 3 * a + c, 2 * a
    ok3 = lambda e: gas_kp(A * e * np.exp(3 * a), lam, eta)  # noqa: E731
    lo, hi = 1e-3, 8.198e-3
    if ok3(hi):
        lo = hi
    else:
        for _ in range(12):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if ok3(mid) else (lo, mid)
    print(f"eps_3 (KP 1D con mu0 + mu1 dur, momento exacto) = {lo:.4e}: {ok3(lo)}", flush=True)
    for g2, nom in ((56 / 3, "n%3=1"), (38 / 3, "n%3=2")):
        lo4, hi4 = 1e-5, min(lo, 8.198e-3)
        for _ in range(14):
            mid = np.sqrt(lo4 * hi4)
            r = resto(mid)
            lo4, hi4 = (mid, hi4) if (r is not None and g2 * mid**2 > 2 * r["tot"]) else (lo4, mid)
        r = resto(lo4)
        print(f"eps_4 ({nom}) = {lo4:.3e}: g2 eps^2 = {g2 * lo4**2:.3e}, 2*resto = {2 * r['tot']:.3e}, "
              f"R1={r['R1']:.2e} R1v={r['R1v']:.2e} R2={r['R2']:.2e} R3={r['R3']:.2e}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Resto desde 4 saltos (los términos de orden 2 y 3 son exactos: w_Q y w3_Q de orbitas_segundo_orden.py).

def excursiones_J(bp, lamE, eta, mu1, crit, J=4):
    """Como excursiones(), con J cubos de saltos (j = J significa >= J). crit(j) en {0,1}."""
    den = np.broadcast_to((1 - lamE) * E - (1 + eta) - mu1, FACT.shape)
    if np.any(den[LIBRE] <= 0):
        return None
    pref = np.where(LIBRE, bp / np.where(LIBRE, den, 1.0), 0.0)
    inv = np.where(LIBRE, 1 / np.where(LIBRE, den, 1.0), 0.0)
    V, U = {}, {}
    V[J] = _resuelve(np.where(FIN, crit(J), 0.0), pref)
    if V[J] is None:
        return None
    U[J] = _resuelve(np.zeros(FACT.shape), pref, src=V[J] * inv)
    for j in range(J - 1, 0, -1):
        V[j] = np.where(LIBRE, _T(V[j + 1], pref), np.where(FIN, crit(j), 0.0))
        U[j] = np.where(LIBRE, V[j] * inv + _T(U[j + 1], pref), 0.0)

    def inicio(X):
        return bp * (5 * max(X[e, r, 1] for e in (2, 3, 4) for r in range(1, 4)) + X[4, 2, 2])

    return inicio(V[1]), inicio(U[1])


def _vacio_cubos(bp, lam, J=4):
    """W_anc por cubos de número de saltos (j = J: >= J), con la recurrencia de 9.12 (truncada en E_max)."""
    import constantes_continuo as cc

    pref = bp / ((1 - lam) * cc.Esafe)
    L = lambda X: cc._T(X, pref) / pref - cc.MUERTE  # noqa: E731
    out = {}
    for crit_j in range(2, J + 1):
        cr = lambda j: 1.0 if (j == crit_j or (crit_j == J and j >= J)) else 0.0  # noqa: E731
        # V^(j): tras j saltos; morir es el salto j+1
        Vt = np.zeros(cc.FACT.shape)
        for _ in range(4000):
            N = np.where(cc.NUM, pref * (cc.MUERTE * cr(J + 1) + L(Vt)), 0.0)
            if np.all(np.abs(N - Vt) <= 1e-13 * np.abs(N) + 1e-300):
                break
            Vt = N
        Vj = Vt
        for j in range(J - 1, 0, -1):
            Vj = np.where(cc.NUM, pref * (cc.MUERTE * cr(j + 1) + L(Vj)), 0.0)
        out[crit_j] = bp * Vj[3, 1, 1]
    return out  # {2: par, 3: tres saltos, J: >= J}


def resto4(eps, s=0.1):
    import constantes_continuo as cc

    kap = 1.0
    # a mínimo de la rejilla con las condiciones de KP del vacío (Z16) en este eps
    a = None
    for aa in np.concatenate([np.linspace(2e-5, 1e-3, 50), np.linspace(1e-3, 0.02, 39)]):
        r = cc.condiciones(eps, aa, kap, s, 0.0)
        if r is not None and r[2]:
            a = aa
            break
    if a is None:
        return None
    M1, M2 = cc.condiciones(eps, a, kap, s, 0.0)[:2]
    am, aE, eta = 15 * M1, 15 * M2, 5 * M2
    bpd = A * eps * np.exp(am)
    cubo = lambda k: (lambda j: 1.0 if j == k else 0.0)  # noqa: E731
    ge4 = excursiones_J(bpd, aE, eta, 0.0, cubo(4))
    eq2 = excursiones_J(bpd, aE, eta, 0.0, cubo(2))
    eq3 = excursiones_J(bpd, aE, eta, 0.0, cubo(3))
    if ge4 is None or eq2 is None or eq3 is None:
        return None
    R1 = ge4[0]
    R1v = (2 * am * eq2[0] + (4 * aE + eta) * eq2[1]) + (3 * am * eq3[0] + (7 * aE + eta) * eq3[1])
    # borde: contornos del vacío de >= 4 saltos que tocan la pared (peso e^{2a}), y racimos de >= 2 contornos
    x2 = A * eps * np.exp(2 * a * kap) * np.exp(s)
    W4 = _vacio_cubos(x2, 2 * a)[4]
    R2a = 5 * W4 / (np.e * s)
    x3 = A * eps * np.exp(3 * a * kap)
    Wm2 = cc.W_anclada(x3 * np.exp(2 * s), 3 * a, riguroso=False)  # m^2 <= e^{2sm}/(es)^2
    WmE = cc.W_anclada(x3 * np.exp(s), 3 * a + s, riguroso=False)  # m int E <= e^{s m + s int E}/(es)^2
    R2b = 5 * a * (WmE + kap * Wm2) / (np.e * s) ** 2
    g = gas_kp(bpd, aE, eta)
    if g is None:
        return None
    mu0, mu1, m0, m1 = g
    R3 = mu0 * m0 + mu1 * m1
    tot = R1 + R1v + R2a + R2b + R3
    return dict(a=a, R1=R1, R1v=R1v, R2a=R2a, R2b=R2b, R3=R3, tot=tot, gas=g)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "eps4b":
    C3 = {1: 32 / 3, 2: 40 / 9}
    G2 = {1: 56 / 3, 2: 38 / 3}
    for nm in (1, 2):
        cond = lambda e: (lambda r: r is not None and G2[nm] * e**2 - C3[nm] * e**3 > 2 * r["tot"])(resto4(e))  # noqa
        lo, hi = 1e-4, 8.198e-3
        if cond(hi):
            lo = hi
        else:
            for _ in range(14):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if cond(mid) else (lo, mid)
        r = resto4(lo)
        print(f"n%3={nm}: eps_4 = {lo:.3e}  hueco = {G2[nm] * lo**2 - C3[nm] * lo**3:.3e}  2*resto = {2 * r['tot']:.3e}  "
              + "  ".join(f"{k}={v:.2e}" for k, v in r.items() if k not in ("gas", "tot")), flush=True)

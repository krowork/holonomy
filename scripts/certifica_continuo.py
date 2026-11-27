"""Certificado con aritmética de intervalos de los radios de 9.12-9.13 (celda_k_notas.md).

Todo lo que entra en T es >= 0 y T es monótono, así que basta el extremo superior de cada intervalo:
- operaciones de arreglos en coma flotante, cada una seguida de nextafter(., +inf) (redondeo hacia arriba
  riguroso: el resultado redondeado al más cercano está a <= 1/2 ulp del exacto);
- escalares (exp, e, 1 - lam, condiciones de KP y de Z17) con mpmath.iv.

Qué se certifica, para parámetros fijos (eps, a, kappa, s[, c, sigma]):
1. U = punto fijo numérico de (1 + eta) T en E <= EMAX, con borde exacto U = K rho^E (D E)^f en E > EMAX
   (rho, D, K son números en coma flotante, es decir racionales exactos).
2. T_sup(U) <= U en E <= EMAX y en la costura EMAX < E <= EMAX+3 (allí contra una cota inferior de U).
3. Cola E >= EMAX+4: b'/(1-lam) S_sup <= 1, con S_sup cota superior de S(rho, D) por subintervalos en phi
   (S es creciente en x = r/E, así que basta x = 1).
Entonces U es supersolución, V <= U y W_anc <= b' U(3,1,1). Todas las condiciones son monótonas en |eps|,
así que el certificado en eps vale para todo |eps| menor con los mismos parámetros.

Uso: python scripts/certifica_continuo.py [z16|z17] [eps] [EMAX=40]
"""
import sys

MODO = sys.argv[1] if len(sys.argv) > 1 else "z16"
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else None
_EMAX = sys.argv[3] if len(sys.argv) > 3 else "40"
sys.argv = [sys.argv[0], _EMAX, MODO]

import numpy as np  # noqa: E402
from mpmath import iv, mpf  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

import constantes_continuo as cc  # noqa: E402

iv.prec = 80
INF = np.inf


def up(x):
    return np.nextafter(x, INF)


def dn(x):
    return np.nextafter(x, -INF)


def f_up(x):
    """mpf o intervalo -> float >= x."""
    v = x.b if hasattr(x, "b") else x
    return float(np.nextafter(float(v), INF))


def f_dn(x):
    v = x.a if hasattr(x, "a") else x
    return float(np.nextafter(float(v), -INF))


# ---------- coeficientes de T con redondeo hacia arriba ----------
E, R, F = cc.E, cc.R, cc.F
_Pf = up(cc.P / cc.Fs)
_c2 = np.array([float(f * (f - 1) // 2) for f in range(cc.FX + 1)])[None, None, :]
_c3 = np.array([float(f * (f - 1) * (f - 2) // 6) for f in range(cc.FX + 1)])[None, None, :]
E2U = np.where(F >= 2, up(_c2 * up(_Pf * _Pf)), 0.0)
E3U = np.where(F >= 3, up(_c3 * up(up(_Pf * _Pf) * _Pf)), 0.0)


def _lp_up(b1, b2, b3, Ee, r):
    """Como cc._lp_clases, redondeando hacia arriba. Los n_i de los vértices son enteros o medios
    enteros pequeños: exactos en coma flotante."""
    N = Ee + 2 * r
    cands = [
        (0 * Ee, 0 * Ee, Ee, np.ones_like(Ee, dtype=bool)),
        ((3 * N - 3 * Ee) / 2, 0 * Ee, (3 * Ee - N) / 2, np.ones_like(Ee, dtype=bool)),
        (0 * Ee, 1.5 * Ee, 0 * Ee, 1.5 * Ee <= N),
        (3 * Ee, 0 * Ee, 0 * Ee, 3 * Ee <= N),
        (2 * N - 3 * Ee, 3 * Ee - N, 0 * Ee, N >= 1.5 * Ee),
        (0 * Ee, 3 * N - 3 * Ee, 3 * Ee - 2 * N, N <= 1.5 * Ee),
    ]
    out = None
    for n1, n2, n3, ok in cands:
        v = up(up(up(n1 * b1) + up(n2 * b2)) + up(n3 * b3))
        v = np.where(ok, v, 0.0)
        out = v if out is None else np.maximum(out, v)
    return out


def T_up(V, pref):
    """Cota superior de T(V) (V >= 0, pref >= el prefactor exacto)."""
    M = maximum_filter(V, footprint=cc.HUELLA, mode="constant", cval=0.0)
    Mr = maximum_filter(V, footprint=cc.HUELLA_R, mode="constant", cval=0.0)
    v = {d: cc._desplaza_E(Mr, d) for d in range(-3, 4)}
    b1 = np.maximum(v[1], v[2])
    b2 = np.maximum(np.maximum(v[-1], v[0]), v[1])
    b3 = np.maximum(np.maximum(np.maximum(v[-3], v[-2]), v[-1]), v[0])
    crec = up(up(2 * R * v[3]) + _lp_up(b1, b2, b3, E, R))
    nac = np.zeros_like(V)
    nac[:-3, :-1, :-1] = V[3:, 1:, 1:]
    M1 = np.zeros_like(V)
    M1[:, :, 1:] = M[:, :, :-1]
    M2 = np.zeros_like(V)
    M2[:, :, 2:] = M[:, :, :-2]
    tot = up(up(up(up(cc.MUERTE + crec) + nac) + up(E2U * M1)) + up(E3U * M2))
    return up(pref * tot)


def _pot(base, n, rd):
    """base^n (n entero >= 0, arreglo) con redondeo rd en cada producto."""
    out = np.ones(np.broadcast(base, n).shape)
    for k in range(int(np.max(n))):
        out = np.where(k < n, rd(out * base), out)
    return out


RHO, DC = float(cc.RHO), float(cc.DCOLA)
_Ei = np.broadcast_to(E, cc.FACT.shape)
_Fi = np.broadcast_to(F, cc.FACT.shape)
_DE = DC * _Ei  # D = 20.25 = 81/4: producto exacto
FORMA_UP = np.where(cc.COLA, up(_pot(RHO, _Ei, up) * _pot(_DE, _Fi, up)), 0.0)
FORMA_DN = np.where(cc.COLA, dn(_pot(RHO, _Ei, dn) * _pot(_DE, _Fi, dn)), 0.0)
assert float(_DE.max()) * 4 == float(81 * _Ei.max())  # D E exacto


# ---------- constante de la cola: cota superior rigurosa ----------
def _lp_iv(b1, b2, b3, x):
    """L(q, x) por unidad de E (E = 1, r = x), con intervalos."""
    Ee, N = iv.mpf(1), 1 + 2 * x
    vals = [b3, (3 * N - 3) / 2 * b1 + (3 - N) / 2 * b3]
    if 1.5 <= N:
        vals.append(1.5 * b2)
    if 3 <= N:
        vals.append(3 * b1)
    if N >= 1.5:
        vals.append((2 * N - 3) * b1 + (3 - N) * b2)
    if N <= 1.5:
        vals.append((3 * N - 3) * b2 + (3 - 2 * N) * b3)
    return max(v.b for v in vals)


def S_sup(rho, D, E1, partes=4000):
    """sup_{phi in [0,1/2], x in [0,1]} del integrando de cc._cola_const. Crece con x (el LP gana vértices
    y b >= 0; los demás términos son crecientes), así que x = 1. En cada trozo [p0,p1] de phi:
    q^d <= max(q(p0)^d, q(p1)^d), e^{3 phi} <= e^{3 p1}."""
    rho, D = iv.mpf(rho), iv.mpf(D)
    x = 1
    peor = mpf(0)
    for i in range(partes):
        p0 = iv.mpf(i) / (2 * partes)
        p1 = iv.mpf(i + 1) / (2 * partes)
        q0, q1 = rho * iv.exp(p0), rho * iv.exp(p1)

        def qp(d):
            return iv.mpf([0, max((q0**d).b, (q1**d).b)])

        b = {w: iv.mpf([0, max(qp(d).b for d in range(3 - 2 * w, 4 - w))]) for w in (1, 2, 3)}
        Mt = iv.mpf([0, max(qp(3).b, qp(-3).b, (rho**-3).b)])
        S = (2 * x * qp(3) + _lp_iv(b[1], b[2], b[3], x) + rho**3 * D * (1 + iv.mpf(3) / E1) * iv.exp(3 * p1)
             + (1 + 4 * x) ** 2 * Mt / (2 * D) + (1 + 4 * x) ** 3 * Mt / (6 * D * D))
        peor = max(peor, S.b)
    return peor


S_UP = S_sup(RHO, DC, cc.EMAX + 4)


# ---------- W_anc certificado ----------
ETA = 1e-9


def _fijo(V, pref, iters=4000):
    for _ in range(iters):
        N = np.where(cc.NUM, cc._T(V, pref), V)
        if not np.isfinite(N[3, 1, 1]) or N[3, 1, 1] > 1e6:
            return None
        pos = cc.NUM & (N > 0)
        if np.all(np.abs(N - V)[pos] <= 1e-13 * N[pos]):
            return N
        V = N
    return None


def W_cert(bp, lam):
    """bp, lam: intervalos de mpmath. Devuelve una cota superior (float) de W_anc o None."""
    bp_up = f_up(bp)
    oml = f_dn(1 - lam)
    if oml <= 0:
        return None
    if (bp / (1 - lam) * S_UP).b > 1:
        return None
    pref_up = up(bp_up * up(1.0 / dn(oml * cc.Esafe)))  # >= b'/((1-lam) E)
    pref_num = pref_up * (1 + ETA)
    V0 = _fijo(np.zeros(cc.FACT.shape), pref_num)
    if V0 is None:
        return None
    borde = cc.NUM & (E >= cc.EMAX - 2)
    K = 2 * float(np.max(np.where(borde, V0 / np.where(borde, RHO ** E * (DC * E) ** F, 1.0), 0.0)))
    for _ in range(12):
        V = _fijo(np.where(cc.NUM, V0, K * FORMA_UP), pref_num)
        if V is None:
            return None
        U_up = np.where(cc.NUM, V, up(K * FORMA_UP))
        U_dn = np.where(cc.NUM, V, dn(K * FORMA_DN))
        TU = T_up(U_up, pref_up)
        chk = cc.NUM | cc.COSTURA
        if np.all(TU[chk] <= U_dn[chk]):
            return up(bp_up * U_up[3, 1, 1])
        K *= 4
    return None


def cond_cert(eps, a, kap, s, c=0.0):
    eps, a, kap, s, c = (iv.mpf(float(t)) for t in (eps, a, kap, s, c))
    bp = 4 * eps * iv.exp(3 * a * kap)
    lam = 3 * a + c
    W1 = W_cert(bp, lam + s)
    W2 = W_cert(bp * iv.exp(s), lam)
    if W1 is None or W2 is None:
        return None
    M1 = iv.mpf(W1) / (iv.e * s)
    M2 = iv.mpf(W2) / (iv.e * s)
    ok = (5 * M1).b <= (a * kap).a and (5 * M2).b <= a.a
    return f_up(M1), f_up(M2), ok


def z17_cert(eps, c, a, kap, s, sig):
    r = cond_cert(eps, a, kap, s, c)
    if r is None or not r[2]:
        return None
    eps, c, a, kap, sig = (iv.mpf(float(t)) for t in (eps, c, a, kap, sig))
    Wb = W_cert(4 * eps * iv.exp(3 * a * kap), 3 * a)
    if Wb is None:
        return None
    mu, nup = c, c / 2
    nu = 3 * iv.mpf(Wb) + mu / 2 + nup
    Bt = (1 / (iv.e * (2 * c - mu)) + 1 / (2 * iv.e * nup)) / iv.e
    cota = iv.e * mu**2 * iv.exp(-mu / 2) / 4
    Wq = W_cert(4 * eps * iv.exp(4 * a * kap + sig), 4 * a + nu)
    if Wq is None:
        return None
    # n e^{-sig ceil(n/3)} <= 3m e^{-sig m}, m = ceil(n/3) >= 1, decreciente para m > 1/sig: basta m <= 2/sig + 2
    mmax = int(2 / float(sig)) + 3
    nmax = max((3 * m * iv.exp(-sig * m)).b for m in range(1, mmax))
    q = iv.mpf([0, max(mpf(1), Bt.b)]) * nmax * iv.mpf(Wq)
    ok = q.b < 1 and (q / (1 - q)).b <= cota.a
    return (f_up(q), f_dn(cota)) if ok else None


if __name__ == "__main__":
    print(f"EMAX = {cc.EMAX}: S_cola (rejilla) = {cc.S_COLA:.4f},  S_cola certificada = {float(S_UP):.4f}"
          f"  (rho = {RHO}, D = {DC})")
    eps = EPS
    if MODO == "z16":
        par = cc.factible(eps)
        print(f"parámetros (búsqueda en coma flotante) en eps = {eps:.4e}: {par}")
        a, kap, s = par[:3]
        while True:
            r = cond_cert(eps, a, kap, s)
            print(f"  eps = {eps:.5e}: certificado = {r}")
            if r is not None and r[2]:
                break
            eps *= 0.999
        print(f"CERTIFICADO Z16: eps_2 >= {eps:.5e}  (a, kappa, s) = ({a}, {kap}, {s})")
    else:
        par = cc.z17(eps)
        print(f"parámetros (búsqueda en coma flotante) en eps = {eps:.4e}: {par}")
        c, a, kap, s, sig = par[:5]
        while True:
            r = z17_cert(eps, c, a, kap, s, sig)
            print(f"  eps = {eps:.5e}: certificado (q, cota) = {r}")
            if r is not None:
                break
            eps *= 0.999
        print(f"CERTIFICADO Z17: eps_2 >= {eps:.5e}  (c, a, kappa, s, sigma) = ({c}, {a}, {kap}, {s}, {sig})")

"""Certificado con intervalos de eps_3 y eps_4 del Teorema D''' (celda_k_notas.md, 9.14).

Igual que certifica_continuo.py (9.13 e), ahora para la recurrencia de las excursiones del defecto:
- supersolución U: punto fijo numérico de (1 + eta) T en E <= EMAX, con borde exacto K rho^E (D E)^f en E > EMAX;
  T_sup(U) <= U con redondeo hacia arriba en E <= EMAX y en la costura; en la cola, b' S_sup / ((1-lam) - c0/E1) <= 1,
  con c0 = 1 + eta + mu1 (den(E) >= ((1-lam) - c0/E1) E) y phi = f/E <= 1/2 + 1/(2 E1) (el marco del defecto
  puede tener una sola pared);
- los cubos de número de saltos son composiciones finitas de T aplicadas a U o al indicador del final, y en la cola
  se acotan por U (V^(j) <= V_todas <= U);
- el momento en la duración, Y = V/den + T(Y), con su propia supersolución K_Y rho^E (D E)^f;
- el vacío: W_cert y cond_cert de certifica_continuo.py, y W_{>=4} por cubos sobre su supersolución.

Uso: python scripts/certifica_defecto.py [EMAX=40]
"""
import sys

_EMAX = sys.argv[1] if len(sys.argv) > 1 else "40"
ARGS = list(sys.argv)
sys.argv = [sys.argv[0], "z16", "1e-3", _EMAX]

import numpy as np  # noqa: E402
from mpmath import iv, mpf  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

import certifica_continuo as cv  # noqa: E402
import constantes_continuo as cc  # noqa: E402
import constantes_defecto as cd  # noqa: E402

up, dn, f_up, f_dn = cv.up, cv.dn, cv.f_up, cv.f_dn
EMAX = cc.EMAX
EX = EMAX + 6
FX = (EX + 1) // 2 + 1
E = np.arange(EX + 1)[:, None, None].astype(float)
R = np.arange(EX + 1)[None, :, None].astype(float)
F = np.arange(FX + 1)[None, None, :].astype(float)
FACT = (F >= 1) & (E >= 2 * F - 1) & (R >= F) & (R <= E)
FIN = np.zeros(FACT.shape, dtype=bool)
FIN[1, 1, 1] = True
LIBRE = FACT & ~FIN
NUM = LIBRE & (E <= EMAX)
COLA = FACT & (E > EMAX)
COSTURA = COLA & (E <= EMAX + 3)
P = E + 4 * R
Fs = np.maximum(F, 1)
_c2 = np.array([float(f * (f - 1) // 2) for f in range(FX + 1)])[None, None, :]
_c3 = np.array([float(f * (f - 1) * (f - 2) // 6) for f in range(FX + 1)])[None, None, :]
_Pf = up(P / Fs)
E2U = np.where(F >= 2, up(_c2 * up(_Pf * _Pf)), 0.0)
E3U = np.where(F >= 3, up(_c3 * up(up(_Pf * _Pf) * _Pf)), 0.0)
E2F = np.where(F >= 2, _c2 * (P / Fs) ** 2, 0.0)
E3F = np.where(F >= 3, _c3 * (P / Fs) ** 3, 0.0)
H, HR = cc.HUELLA, cc.HUELLA_R


def _desp(A_, d):
    return cc._desplaza_E(A_, d)


def _partes(V):
    M = maximum_filter(V, footprint=H, mode="constant", cval=0.0)
    Mr = maximum_filter(V, footprint=HR, mode="constant", cval=0.0)
    v = {d: _desp(Mr, d) for d in range(-3, 4)}
    b1 = np.maximum(v[1], v[2])
    b2 = np.maximum(np.maximum(v[-1], v[0]), v[1])
    b3 = np.maximum(np.maximum(np.maximum(v[-3], v[-2]), v[-1]), v[0])
    nac = np.zeros_like(V)
    nac[:-3, :-1, :-1] = V[3:, 1:, 1:]
    M1 = np.zeros_like(V)
    M1[:, :, 1:] = M[:, :, :-1]
    M2 = np.zeros_like(V)
    M2[:, :, 2:] = M[:, :, :-2]
    return v, b1, b2, b3, nac, M1, M2


def T_f(V, pref):
    v, b1, b2, b3, nac, M1, M2 = _partes(V)
    crec = 2 * R * v[3] + cc._lp_clases(b1, b2, b3, E, R)
    return pref * (crec + nac + E2F * M1 + E3F * M2)


def T_up(V, pref):
    v, b1, b2, b3, nac, M1, M2 = _partes(V)
    crec = up(up(2 * R * v[3]) + cv._lp_up(b1, b2, b3, E, R))
    tot = up(up(up(crec + nac) + up(E2U * M1)) + up(E3U * M2))
    return up(pref * tot)


def _pot(base, n, rd):
    return cv._pot(base, n, rd)


RHO, DC = cv.RHO, cv.DC
_Ei = np.broadcast_to(E, FACT.shape)
_Fi = np.broadcast_to(F, FACT.shape)
FORMA_UP = np.where(COLA, up(_pot(RHO, _Ei, up) * _pot(DC * _Ei, _Fi, up)), 0.0)
FORMA_DN = np.where(COLA, dn(_pot(RHO, _Ei, dn) * _pot(DC * _Ei, _Fi, dn)), 0.0)
FORMA_F = np.where(COLA, RHO ** E * (DC * E) ** F, 0.0)
E1 = EMAX + 4


def S_sup_def(partes=4000):
    """Como cv.S_sup, con phi en [0, 1/2 + 1/(2 E1)]."""
    rho, D = iv.mpf(RHO), iv.mpf(DC)
    phimax = iv.mpf(1) / 2 + iv.mpf(1) / (2 * E1)
    peor = mpf(0)
    for i in range(partes):
        p0, p1 = phimax * i / partes, phimax * (i + 1) / partes
        q0, q1 = rho * iv.exp(p0), rho * iv.exp(p1)

        def qp(d):
            return iv.mpf([0, max((q0**d).b, (q1**d).b)])

        b = {w: iv.mpf([0, max(qp(d).b for d in range(3 - 2 * w, 4 - w))]) for w in (1, 2, 3)}
        Mt = iv.mpf([0, max(qp(3).b, qp(-3).b, (rho**-3).b)])
        S = (2 * qp(3) + cv._lp_iv(b[1], b[2], b[3], 1) + rho**3 * D * (1 + iv.mpf(3) / E1) * iv.exp(3 * p1)
             + 25 * Mt / (2 * D) + 125 * Mt / (6 * D * D))
        peor = max(peor, S.b)
    return peor


S_DEF = S_sup_def()
ETA = 1e-9


def _fijo(V0, pref, src=None, mask=NUM, iters=6000):
    V = V0.copy()
    for _ in range(iters):
        N = np.where(mask, T_f(V, pref) + (0 if src is None else src), V0)
        if not np.isfinite(N).all() or N.max() > 1e250:
            return None
        if np.all(np.abs(N - V) <= 1e-13 * np.abs(N) + 1e-300):
            return N
        V = N
    return None


class Exc:
    """Recurrencia certificada de las excursiones con parámetros (bp, lamE, eta, mu1) en intervalos."""

    def __init__(self, bp, lamE, eta, mu1):
        self.ok = False
        oml = 1 - lamE
        c0 = 1 + eta + mu1
        tasa = oml - c0 / E1
        if f_dn(tasa) <= 0 or (bp * S_DEF / tasa).b >= 1:
            return
        self.theta = f_up(bp * S_DEF / tasa)  # T(Y)/Y <= theta en la cola
        den_dn = np.broadcast_to(dn(dn(f_dn(oml) * E) - f_up(c0)), FACT.shape)
        if np.any(den_dn[LIBRE] <= 0):
            return
        self.inv_up = np.where(LIBRE, up(1.0 / np.where(LIBRE, den_dn, 1.0)), 0.0)
        self.bp_up = f_up(bp)
        self.pref = np.where(LIBRE, up(self.bp_up * self.inv_up), 0.0)
        self.c0, self.oml = c0, oml
        self.U = self._super(np.where(FIN, 1.0, 0.0))
        if self.U is None:
            return
        self.Y = self._momento(self.U)
        self.ok = self.Y is not None

    def _super(self, fin):
        pn = self.pref * (1 + ETA)
        V0 = _fijo(fin, pn)
        if V0 is None:
            return None
        borde = NUM & (E >= EMAX - 2)
        K = 2 * float(np.max(np.where(borde, V0 / np.where(borde, RHO ** E * (DC * E) ** F, 1.0), 0.0)))
        for _ in range(12):
            V = _fijo(np.where(NUM, V0, np.where(FIN, fin, K * FORMA_F)), pn)
            if V is None:
                return None
            Uu = np.where(NUM, V, np.where(FIN, fin, up(K * FORMA_UP)))
            Ud = np.where(NUM, V, np.where(FIN, fin, dn(K * FORMA_DN)))
            TU = T_up(Uu, self.pref)
            chk = NUM | COSTURA
            if np.all(TU[chk] <= Ud[chk]):
                self.K = K
                return Uu
            K *= 4
        return None

    def _momento(self, U):
        """Y >= V/den + T(Y). Cola: Y = K_Y rho^E (DE)^f con K_Y (1 - theta) >= K / ((1-lam) E1 - c0)."""
        src = up(U * self.inv_up)
        pn = self.pref * (1 + ETA)
        KY = f_up(iv.mpf(self.K) / (iv.mpf(f_dn(self.oml)) * E1 - self.c0) / (1 - iv.mpf(self.theta)))
        for _ in range(12):
            Y = _fijo(np.where(COLA, KY * FORMA_F, 0.0), pn, src=src * (1 + ETA))
            if Y is None:
                return None
            Yu = np.where(NUM, Y, np.where(COLA, up(KY * FORMA_UP), 0.0))
            Yd = np.where(NUM, Y, np.where(COLA, dn(KY * FORMA_DN), 0.0))
            TY = up(T_up(Yu, self.pref) + src)
            chk = NUM | COSTURA
            if np.all(TY[chk] <= Yd[chk]):
                return Yu
            KY *= 4
        return None

    def cubos(self, crit, J=4):
        """Densidad y momento (cotas superiores) de las excursiones con número total de saltos según crit."""
        if crit(J):  # >= J cuenta: V^(J) = todas = U
            VJ, YJ = self.U, self.Y
        else:
            VJ, YJ = np.zeros(FACT.shape), np.zeros(FACT.shape)
        V, Y = VJ, YJ
        for j in range(J - 1, 0, -1):
            Vn = np.where(LIBRE, T_up(V, self.pref), np.where(FIN, float(crit(j)), 0.0))
            Yn = np.where(LIBRE, up(up(Vn * self.inv_up) + T_up(Y, self.pref)), 0.0)
            # en la cola, V^(j) <= U e Y^(j) <= Y (T aplicado allí leería fuera de la malla)
            V = np.where(COLA, self.U, Vn)
            Y = np.where(COLA, self.Y, Yn)
        return self._inicio(V), self._inicio(Y)

    def _inicio(self, X):
        m = max(X[e, r, 1] for e in (2, 3, 4) for r in range(1, 4))
        return up(self.bp_up * up(up(5 * m) + X[4, 2, 2]))

    def m0m1(self):
        return self._inicio(self.U), self._inicio(self.Y)


def gas_cert(bp, lamE, eta, mu0, mu1):
    ex = Exc(bp, lamE, eta, iv.mpf(mu1))
    if not ex.ok:
        return None
    d0, d1 = ex.m0m1()
    m0 = iv.exp(iv.mpf(mu0)) * d0
    m1 = iv.exp(iv.mpf(mu0)) * d1
    return f_up(m0), f_up(m1), (m0.b <= mu1 and m1.b <= mu0)


# ---------- vacío por cubos ----------
def _T_up_vac(V, pref, cm):
    """T de 9.12 con redondeo hacia arriba y el término de muerte multiplicado por cm."""
    v, b1, b2, b3, nac, M1, M2 = None, None, None, None, None, None, None
    M = maximum_filter(V, footprint=cc.HUELLA, mode="constant", cval=0.0)
    Mr = maximum_filter(V, footprint=cc.HUELLA_R, mode="constant", cval=0.0)
    v = {d: cc._desplaza_E(Mr, d) for d in range(-3, 4)}
    b1 = np.maximum(v[1], v[2])
    b2 = np.maximum(np.maximum(v[-1], v[0]), v[1])
    b3 = np.maximum(np.maximum(np.maximum(v[-3], v[-2]), v[-1]), v[0])
    crec = up(up(2 * cc.R * v[3]) + cv._lp_up(b1, b2, b3, cc.E, cc.R))
    nac = np.zeros_like(V)
    nac[:-3, :-1, :-1] = V[3:, 1:, 1:]
    M1 = np.zeros_like(V)
    M1[:, :, 1:] = M[:, :, :-1]
    M2 = np.zeros_like(V)
    M2[:, :, 2:] = M[:, :, :-2]
    tot = up(up(up(up(cm * cc.MUERTE + crec) + nac) + up(cv.E2U * M1)) + up(cv.E3U * M2))
    return up(pref * tot)


def vac_U(bp, lam):
    """Supersolución certificada del vacío (como cv.W_cert), devolviendo U y el prefactor."""
    bp_up = f_up(bp)
    oml = f_dn(1 - lam)
    if oml <= 0 or (bp / (1 - lam) * cv.S_UP).b > 1:
        return None
    pref_up = up(bp_up * up(1.0 / dn(oml * cc.Esafe)))
    pn = pref_up * (1 + cv.ETA)
    V0 = cv._fijo(np.zeros(cc.FACT.shape), pn)
    borde = cc.NUM & (cc.E >= EMAX - 2)
    K = 2 * float(np.max(np.where(borde, V0 / np.where(borde, RHO ** cc.E * (DC * cc.E) ** cc.F, 1.0), 0.0)))
    for _ in range(12):
        V = cv._fijo(np.where(cc.NUM, V0, K * cv.FORMA_UP), pn)
        Uu = np.where(cc.NUM, V, up(K * cv.FORMA_UP))
        Ud = np.where(cc.NUM, V, dn(K * cv.FORMA_DN))
        if np.all(cv.T_up(Uu, pref_up)[cc.NUM | cc.COSTURA] <= Ud[cc.NUM | cc.COSTURA]):
            return Uu, pref_up, bp_up
        K *= 4
    return None


def W_ge4(bp, lam):
    """Cota de los contornos anclados del vacío con >= 4 saltos: V^(4) = todas <= U; V^(j) = T con muerte
    contada solo si el total j+1 >= 4."""
    r = vac_U(bp, lam)
    if r is None:
        return None
    U, pref, bp_up = r
    V = U
    for j in (3, 2, 1):
        cm = 1.0 if j + 1 >= 4 else 0.0
        Vn = np.where(cc.NUM, _T_up_vac(V, pref, cm), 0.0)
        V = np.where(cc.COLA, U, Vn)
    return up(bp_up * V[3, 1, 1])


# ---------- resto de orden >= 4 ----------
def L_req(m0, m1, c):
    """Factor de los enlaces (lema de 9.14): G(e) <= e^{L X_e} si L >= e^{m1}(m1 + m0/(e(2c - m0))) e^{0,05}.
    m0, m1: densidad y momento del gas (con e^{a(e)} y el propio vestido); 2c > m0."""
    m0, m1, c = iv.mpf(m0), iv.mpf(m1), iv.mpf(c)
    if (2 * c - m0).a <= 0:
        return None
    return f_up(iv.exp(m1) * (m1 + m0 / (iv.e * (2 * c - m0))) * iv.exp(iv.mpf("0.05")))


def resto_cert(eps, a, mu0, mu1, s=0.1, L=1.0, c4=0.02):
    """Cota certificada del resto (>= 4 saltos) en eps, con a del vacío (KP con decaimiento c4 en tiempo) y
    (mu0, mu1) del gas vestido. Vestido por excursión: (3 + L)(5 m M1 + 5 int E M2) + (2 + L) 5 dur M2."""
    eps_i, a_i, s_i = iv.mpf(eps), iv.mpf(a), iv.mpf(s)
    kap = 1
    c = cv.cond_cert(eps, a, kap, s, c4)
    if c is None or not c[2]:
        return None
    M1, M2 = iv.mpf(c[0]), iv.mpf(c[1])
    am, aE, eta = (3 + L) * 5 * M1, (3 + L) * 5 * M2, (2 + L) * 5 * M2
    bpd = 4 * eps_i * iv.exp(am)
    ex = Exc(bpd, aE, eta, iv.mpf(0))
    if not ex.ok:
        return None
    cubo = lambda k: (lambda j: 1.0 if j == k else 0.0)  # noqa: E731
    R1 = iv.mpf(ex.cubos(cubo(4))[0])
    d2, y2 = (iv.mpf(x) for x in ex.cubos(cubo(2)))
    d3, y3 = (iv.mpf(x) for x in ex.cubos(cubo(3)))
    R1v = 2 * am * d2 + (4 * aE + eta) * y2 + 3 * am * d3 + (7 * aE + eta) * y3
    x2 = 4 * eps_i * iv.exp(2 * a_i * kap) * iv.exp(s_i)
    W4 = W_ge4(x2, 2 * a_i)
    if W4 is None:
        return None
    R2a = 5 * iv.mpf(W4) / (iv.e * s_i)
    x3 = 4 * eps_i * iv.exp(3 * a_i * kap)
    Wm2 = cv.W_cert(x3 * iv.exp(2 * s_i), 3 * a_i)
    WmE = cv.W_cert(x3 * iv.exp(s_i), 3 * a_i + s_i)
    if Wm2 is None or WmE is None:
        return None
    R2b = 5 * a_i * (iv.mpf(WmE) + kap * iv.mpf(Wm2)) / (iv.e * s_i) ** 2
    g = gas_cert(bpd, aE, eta, mu0, mu1)
    if g is None or not g[2]:
        return None
    R3 = iv.mpf(mu0) * g[0] + iv.mpf(mu1) * g[1]
    Lr = L_req(g[0], g[1], c4)
    if Lr is None or Lr > L:
        return None
    tot = R1 + R1v + R2a + R2b + R3
    return dict(tot=f_up(tot), R1=f_up(R1), R1v=f_up(R1v), R2a=f_up(R2a), R2b=f_up(R2b), R3=f_up(R3), L_req=Lr)


def _a_min(eps, s=0.1, c=0.0):
    for aa in np.concatenate([np.linspace(2e-5, 1e-3, 50), np.linspace(1e-3, 0.02, 39)]):
        r = cc.condiciones(eps, aa, 1.0, s, c)
        if r is not None and r[2]:
            return aa
    return None


if __name__ == "__main__":
    print(f"EMAX = {EMAX}: S de la cola del defecto (phi <= 1/2 + 1/(2E1)) <= {float(S_DEF):.4f}", flush=True)
    # eps_3: gas con los parámetros del vacío de Z17 (9.13 e)
    a, c = cd.VAC["a"], cd.VAC["c"]
    L = float(ARGS[2]) if len(ARGS) > 2 else 1.0
    eps3 = float(ARGS[3]) if len(ARGS) > 3 else 7.8236e-3
    while True:
        lamE = (3 + iv.mpf(L)) * iv.mpf(a) + iv.mpf(c)
        eta = (2 + iv.mpf(L)) * iv.mpf(a)
        bp = 4 * iv.mpf(eps3) * iv.exp((3 + iv.mpf(L)) * iv.mpf(a))
        par = cd.gas_kp(f_up(bp), float(lamE.b), float(eta.b))
        g = None if par is None else gas_cert(bp, lamE, eta, par[0], par[1])
        Lr = None if g is None else L_req(g[0], g[1], c)
        print(f"  eps_3 = {eps3:.5e}: (mu0, mu1) = {None if par is None else par[:2]}, (m0, m1, ok) = {g},"
              f" L_req = {Lr} <= L = {L}", flush=True)
        if g is not None and g[2] and Lr is not None and Lr <= L:
            break
        eps3 *= 0.99
    print(f"CERTIFICADO eps_3 >= {eps3:.5e}", flush=True)
    # eps_4
    C3 = {1: iv.mpf(32) / 3, 2: iv.mpf(40) / 9}
    G2 = {1: iv.mpf(56) / 3, 2: iv.mpf(38) / 3}
    for nm, e4 in ((1, 2.29e-3), (2, 1.91e-3)):
        while True:
            C4 = 0.02
            a4 = _a_min(e4, c=C4)
            r = None
            if a4 is not None:
                M1, M2 = cc.condiciones(e4, a4, 1.0, 0.1, C4)[:2]
                gp = cd.gas_kp(cd.A * e4 * np.exp((3 + L) * 5 * M1), (3 + L) * 5 * M2, (2 + L) * 5 * M2)
                if gp is not None:
                    # un poco de holgura en mu sobre los valores en coma flotante
                    r = resto_cert(e4, a4, gp[0] * 1.02, gp[1] * 1.02, L=L, c4=C4)
            ok = False
            if r is not None:
                ei = iv.mpf(e4)
                hueco = G2[nm] * ei**2 - C3[nm] * ei**3
                ok = hueco.a > 2 * r["tot"]
            print(f"  n%3={nm}, eps = {e4:.4e}: a = {a4}, resto = {r}, ok = {ok}", flush=True)
            if ok:
                break
            e4 *= 0.99
        print(f"CERTIFICADO eps_4 (n%3={nm}) >= {e4:.4e}", flush=True)

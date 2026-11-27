"""Newton-Kantorovich graduado con series en eps y colas de Cauchy (9.26).

Sustituye las cotas de Cauchy globales de kantorovich_graduado.py por:
  * Ĵ, d y la tercera derivada en 0 sumados con sus series (serie_respuesta.py) hasta orden K, con la
    dependencia en Theta por Fourier (frecuencias enteras |w| <= (k + 2 grado)/n) y la cola de orden > K
    acotada por Cauchy en el polidisco P (|eps| <= eps_a, |y_i| <= R_i, |F + 1| <= A, Bauer-Fike);
  * la variación de la tercera derivada en la bola por la cuarta derivada (Cauchy con el Schwarz
    ponderado del Lema Z30: |G| <= 4 A lambda^4) y polarización de Banach (norma = sup diagonal);
  * norma euclídea con pesos por bloques, v = u / w.
Las cotas se dan en intervalos [eps_l, eps_r] (cada potencia en el extremo que la maximiza; lambda_min por
su valor en eps_l menos la derivada), de modo que cubren el continuo de eps y todo Theta real.
Uso: python scripts/kantorovich_series.py serie.npz
"""
import sys
import numpy as np

sys.path.insert(0, '/home/alex/fisica/holonomy/scripts')
_argv, sys.argv = sys.argv, sys.argv[:1]   # k1_racional lee sys.argv al importarse
import newton_graduado as NG
sys.argv = _argv

EXPO = np.array([0] * 8 + [1] * 2 + [2] * 4)        # y = S^{-1} u, S = diag(1, eps, eps^2)
BLOQ = np.array([0] * 8 + [1] * 2 + [2] * 4)
# corriente de cada dirección y: 2 para el salto doble (02<->20), 1 para los demás saltos, 0 para las blandas
CUR = np.array([1, 1, 2, 1, 1, 1, 2, 1] + [1, 1] + [0] * 4)
TAUH = 0.6                                           # |tau_h| >= 0,6  (tau_h = e_R - ebar ~ -2/3)


def carga(fn):
    z = np.load(fn)
    n, K, M = int(z['n']), int(z['K']), int(z['M'])
    B0 = NG.base_explicita(0.0); B1 = NG.base_explicita(1.0) - B0
    def a_y(X, grado):                                # x -> y con B(eps) = B0 + eps B1, por ordenes
        out = np.zeros_like(X)
        letras = 'abcd'[:grado]
        for sel in range(2 ** grado):
            mats = [(B1 if (sel >> i) & 1 else B0) for i in range(grado)]
            sh = bin(sel).count('1')
            sub = ','.join(f'{l}{l.upper()}' for l in letras)
            ein = f"m{letras}k,{sub}->m{letras.upper()}k"
            Y = np.einsum(ein, X, *mats, optimize=True)
            out[..., sh:] += Y[..., :K + 1 - sh]
        return out
    g = a_y(z['grad'], 1); h = a_y(z['hess'], 2); c = a_y(z['cub'], 3)
    F = lambda X: np.fft.fft(X, axis=0) / M           # componente de frecuencia w (w > M/2: negativa)
    out = dict(n=n, K=K, M=M, E=F(z['E']), g=F(g), h=F(h), c=F(c), g0=g[0], h0=h[0], c0=c[0], E0=z['E'][0])
    if 'cua' in z.files and z['cua'].shape[0] == M:
        q = a_y(z['cua'], 4); out['q'] = F(q); out['q0'] = q[0]
    return out


def suma_ind(v, grado):
    """Tensor t[i1..ig] = v[i1] + ... + v[ig]."""
    t = np.zeros((14,) * grado, dtype=np.asarray(v).dtype)
    for ax in range(grado):
        sh = [1] * grado; sh[ax] = 14
        t = t + np.asarray(v).reshape(sh)
    return t


def exponentes(grado):
    return suma_ind(EXPO, grado)


def multiplicidad(grado):
    """prod m_i! (cuántas veces se repite cada índice): la derivada es coeficiente por este factor."""
    import itertools, math
    m = np.ones((14,) * grado)
    for idx in itertools.product(range(14), repeat=grado):
        c = np.bincount(idx)
        m[idx] = np.prod([math.factorial(x) for x in c])
    return m


def preparar(S):
    """Separa la parte en w = 0 (Theta = 0 completa) y la parte de Theta; comprueba los ceros graduados."""
    n, K = S['n'], S['K']
    out = {}
    for key, gr in (('g', 1), ('h', 2), ('c', 3)) + ((('q', 4),) if 'q' in S else ()):
        X = S[key]
        X0 = np.real(S[key + '0'])                    # serie a Theta = 0 (real)
        e = exponentes(gr)
        k = np.arange(K + 1)
        # ceros graduados (Z30 / positividad graduada): Ĵ tiene k >= e_i + e_j; la cubica k >= 4 - 2a - b
        if gr == 2:
            minimo = e
        elif gr >= 3:                                # Z30: peso k + 2a + b >= 4 en grado >= 3
            nh = suma_ind((BLOQ == 0).astype(int), gr)
            nm = suma_ind((BLOQ == 1).astype(int), gr)
            minimo = np.maximum(0, 4 - 2 * nh - nm)
        else:
            minimo = np.zeros_like(e)
        mask = k[(None,) * gr] < minimo[..., None]
        esc = np.abs(X0).max(axis=tuple(range(gr)))   # escala por orden
        viol = (np.abs(X0) * mask / (1e-300 + esc)).max()
        out[key + '_viol'] = viol
        X0 = np.where(mask, 0, X0)
        Xt = X.copy(); Xt[0] = 0                     # parte de Theta: frecuencias w != 0
        Xt = np.where(mask[None], 0, Xt)
        # estructura de corrientes (9.24): la frecuencia w aparece en el orden k solo si |w| n <= k + sum cur
        M = X.shape[0]
        wabs = np.abs(np.where(np.arange(M) > M // 2, np.arange(M) - M, np.arange(M)))
        sc = suma_ind(CUR, gr)
        perm = wabs[(slice(None),) + (None,) * gr + (None,)] * n <= k[(None,) * (gr + 1)] + sc[None, ..., None]
        out[key + '_ruido'] = (np.abs(Xt) * ~perm).max() / (1e-300 + np.abs(Xt).max())
        Xt = np.where(perm, Xt, 0)
        out[key + '0'] = X0
        out[key + 't'] = 2 * np.abs(Xt).sum(axis=0)   # sup_Theta |sum_w c_w (e^{iwTheta} - 1)| <= 2 sum |c_w|
        out[key + 'e'] = e
        if gr == 4:
            out['m4'] = multiplicidad(4)
    M = S['E'].shape[0]
    wabs = np.abs(np.where(np.arange(M) > M // 2, np.arange(M) - M, np.arange(M)))
    Et = np.where(wabs[:, None] * n <= np.arange(K + 1)[None, :], S['E'], 0)
    out['Et'] = 2 * np.abs(Et[1:]).sum(axis=0)
    return out


def sup_pot(el, er, p):
    """sup de eps^p en [el, er] (p puede ser negativo)."""
    return np.where(p < 0, el ** p.astype(float), er ** p.astype(float))


def cotas(S, P, el, er, eps_a, R, wv):
    """Cotas en el intervalo [el, er] para el polidisco (eps_a, R por coordenada y) y pesos wv."""
    n, K = S['n'], S['K']
    Rv = np.array([R[0]] * 8 + [R[1]] * 2 + [R[2]] * 4)
    A = 2 * 2 ** 0.5 * eps_a + 8 * R[0] / 2 ** 0.5 + 2 * R[1] / 2 ** 0.5 + eps_a * R[2] * (2 + 2 / 3)
    if n * A >= 1 - R[2]:
        return None
    rr = er / eps_a
    if rr >= 1:
        return None
    k = np.arange(K + 1)
    cola = rr ** (K + 1) / (1 - rr)                 # sum_{k > K} r^k
    W = wv[BLOQ]
    # --- Ĵ = -Hess_u, en v: W Ĵ W
    e2 = P['he']
    p2 = k[None, None, :] - e2[..., None]
    J0 = -(P['h0'] * el ** p2.clip(0).astype(float)).sum(-1)      # en el
    J0 = (J0 + J0.T) / 2
    lam_l = np.linalg.eigvalsh(W[:, None] * J0 * W[None, :]).min()
    dJ = (np.abs(P['h0']) * np.where(p2 >= 1, p2 * er ** (p2 - 1).clip(0).astype(float), 0)).sum(-1)
    Jt = (P['ht'] * sup_pot(el, er, p2)).sum(-1)
    mult2 = 1 + np.eye(14)
    colaJ = mult2 * A / np.outer(Rv, Rv) * cola * sup_pot(el, er, -e2)
    pert = np.linalg.norm(W[:, None] * ((er - el) * dJ + Jt + colaJ) * W[None, :])   # Frobenius
    lam = lam_l - pert
    if lam <= 0:
        return dict(h=np.inf, lam=lam, motivo='lambda')
    beta = 1 / lam
    # --- d = S^{-1}(grad(Theta) - grad(0)) + (p-1) S^{-1} tau
    e1 = P['ge']
    p1 = k[None, :] - e1[:, None]
    d = (P['gt'] * sup_pot(el, er, p1)).sum(-1) + 2 * A / Rv * cola * sup_pot(el, er, -e1)
    dE = (P['Et'] * er ** k).sum() + 2 * A * cola
    pm1 = dE / TAUH
    # tau_y = B^T (sigma_x - I/9) : sigma = grad a Theta = 0; |tau_y,i| <= serie + cola
    tau0 = P['g0'].copy()
    diag9 = np.zeros(14); diag9[[10, 11, 12, 13]] = 1 / 9          # o_d de las blandas
    tau = np.abs(tau0[:, 0] - diag9) + (np.abs(tau0[:, 1:]) * er ** k[1:]).sum(-1) + A / Rv * cola
    d = d + pm1 * tau * sup_pot(el, er, -e1)
    gv = W * d
    eta = beta * np.linalg.norm(gv)
    # --- tercera derivada en 0 (Frobenius >= norma trilineal), en v
    e3 = P['ce']
    p3 = k[None, None, None, :] - e3[..., None]
    absC = np.abs(P['c0']) + P['ct']
    T3 = (absC * sup_pot(el, er, p3)).sum(-1)
    ii = np.arange(14)
    m3 = np.ones((14, 14, 14))
    eq = (ii[:, None, None] == ii[None, :, None]).astype(int) + (ii[None, :, None] == ii[None, None, :]) + \
         (ii[:, None, None] == ii[None, None, :])
    m3 = np.where(eq == 3, 6, np.where(eq == 1, 2, 1))
    colaT = m3 * A / (Rv[:, None, None] * Rv[None, :, None] * Rv[None, None, :]) * cola * sup_pot(el, er, -e3)
    L0 = np.linalg.norm(W[:, None, None] * W[None, :, None] * W[None, None, :] * (T3 + colaT))
    # --- cuarta derivada en la bola de radio 2 eta (Cauchy, Schwarz ponderado)
    lam_s = el / eps_a                                # lambda^4 / tau^4 ~ eps^{-4}: peor en el
    rho = np.array([lam_s ** 2 * R[0] * 1.0, el * lam_s * R[1], el ** 2 * R[2]])
    rv = (rho / wv).min()
    tau_c = rv - 2 * eta
    if tau_c <= 0:
        return dict(h=np.inf, lam=lam, motivo='bola')
    C4 = 96 * A * lam_s ** 4 / tau_c ** 4
    # lambda^4 se evalua en er (crece), tau en el (decrece): cota valida en todo el intervalo
    C4 *= (er / el) ** 4
    if 'q0' in P:
        # cuarta derivada en 0 por su serie (+ Theta + cola de Cauchy) y quinta por Cauchy en la bola
        e4 = P['qe']
        p4 = k[(None,) * 4] - e4[..., None]
        Q = ((np.abs(P['q0']) + P['qt']) * sup_pot(el, er, p4)).sum(-1)
        Rprod = suma_ind(np.log(Rv), 4)
        Q = Q + P['m4'] * A * np.exp(-Rprod) * cola * sup_pot(el, er, -e4)
        W4 = np.exp(suma_ind(np.log(W), 4))
        Q4 = np.linalg.norm(W4 * Q)
        C5 = 120 * 4 * A * lam_s ** 4 / tau_c ** 5 * (er / el) ** 4
        L = L0 + Q4 * 2 * eta + C5 * (2 * eta) ** 2 / 2
        C4 = (Q4 * 2 * eta + C5 * (2 * eta) ** 2 / 2) / (2 * eta)     # término efectivo, para el informe
    else:
        L = L0 + C4 * 2 * eta
    h = beta * L * eta
    u = 2 * eta
    deficit = (0.5 * u * np.linalg.norm(gv) + 0.6 * L * u ** 3) / TAUH
    return dict(h=h, lam=lam, beta=beta, eta=eta, L0=L0, C4term=C4 * 2 * eta, deficit=deficit, A=A,
                motivo='h' if h > 0.5 else 'ok')


def rango(S, P, eps_a, R, wv, e0=1e-30, e1=0.2, paso=1.05):
    """Mayor eps_max tal que todos los intervalos de [e0, eps_max] cumplen h <= 1/2."""
    el = e0
    ult = None
    while el < e1:
        er = el * paso
        c = cotas(S, P, el, er, eps_a, R, wv)
        if c is None or c['h'] > 0.5:
            return el, ult, c
        ult = c
        el = er
    return el, ult, None


if __name__ == "__main__":
    import itertools
    S = carga(sys.argv[1])
    P = preparar(S)
    print(f"n={S['n']} K={S['K']} M={S['M']}: ceros graduados violados (relativo) g {P['g_viol']:.1e}, "
          f"h {P['h_viol']:.1e}, c {P['c_viol']:.1e}; ruido fuera de la regla de corrientes g {P['g_ruido']:.1e}, "
          f"h {P['h_ruido']:.1e}, c {P['c_ruido']:.1e}" + (f"; cuarta: ceros {P['q_viol']:.1e}, ruido {P['q_ruido']:.1e}" if 'q_viol' in P else ''), flush=True)
    n = S['n']
    pruebas = np.geomspace(1e-7, 2e-2, 45)

    def emax_grueso(par):
        e_ok = 0.0
        for e in pruebas:
            c = cotas(S, P, e, e * 1.02, par[0], par[1:4], np.array([1, par[4], par[5]]))
            if c is None or c['h'] > 0.5:
                break
            e_ok = e
        return e_ok

    best = (0.0, None)
    for eps_a in np.arange(0.005, 0.06, 0.005):
        for Rh, Rm, Rs in itertools.product((2e-5, 5e-5, 1e-4, 2e-4, 5e-4), (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
                                            (0.1, 0.15, 0.2, 0.3, 0.4)):
            if n * (2 * 2 ** 0.5 * eps_a + 8 * Rh / 2 ** 0.5 + 2 * Rm / 2 ** 0.5 + eps_a * Rs * 8 / 3) >= 1 - Rs:
                continue
            for wm, ws in itertools.product((0.3, 1, 3), (0.3, 1, 3, 10)):
                par = (eps_a, Rh, Rm, Rs, wm, ws)
                em = emax_grueso(par)
                if em > best[0]:
                    best = (em, par)
    em, par = best
    print(f"  rejilla gruesa: eps_max ~ {em:.3e} con (eps_a, Rh, Rm, Rs, wm, ws) = {par}", flush=True)
    emax, ult, fallo = rango(S, P, par[0], par[1:4], np.array([1, par[4], par[5]]), paso=1.01)
    print(f"  intervalos finos (desde 1e-30, razón 1,01): VALE en [1e-30, {emax:.3e}]; en el borde h = {ult['h']:.3f}, "
          f"lambda = {ult['lam']:.3g}, eta = {ult['eta']:.2e}, L0 = {ult['L0']:.2e}, C4*2eta = {ult['C4term']:.2e}, "
          f"deficit <= {ult['deficit']:.1e}; falla por {fallo['motivo'] if fallo else '-'}")

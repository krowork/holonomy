"""Newton-Kantorovich graduado con datos de otro anillo (sigma_N) o de volumen infinito (sigma_inf).

Generaliza kantorovich_series.py (datos autoconsistentes sigma_n) a datos sigma != sigma_n. Cambia solo el
término independiente d del sistema g(y) = grad e(y) - sigma_W - (p - 1) tau_W:
  d = [grad_n(Theta) - grad_n(0)]  +  [grad_n(0) - sigma_W]  +  (p - 1) tau_W,
con la parte nueva d_fs = grad_n(0) - sigma_W (corrección de tamaño finito, independiente de Theta) por su serie
(serie_datos.py) más colas de Cauchy, y |p - 1| <= |e_n(Theta) - e_R| / |tau_h|, con e_R = e del anillo de los datos.

Además (paso 2), con la igualdad p(n,Theta) = p*(Theta) que da el testigo dual w* = h + sum x*_a o_a (véase
sensibilidad_celda3.tex, Lema del testigo), acota la variación en Theta del déficit,
  |delta(Theta) - delta(0)| <= [ (|J|+2 eta L) 2 eta |dx| + (2 eta)^2 |dJ|/2 ] / |tau_h|,
  |dx| <= beta' (|Delta d| + 2 eta |dJ| + |Delta p| |tau|),   beta' = beta / (1 - 2 h),
y la compara con el término principal 4|b_0| eps^n / (n (ebar - e_R)).

Modos de datos:
  N              sigma_N por su serie (serie_datos.py N K), colas de Bauer-Fike en el anillo N;
  inf:FICHERO    sigma_inf: los coeficientes universales (los de FICHERO hasta el orden k0 = ceil(N/2) - 2, por el
                 Corolario de universalidad) y una cola uniforme en N de Z26 (celda_k_notas.md, 9.22):
                 |coef_k(grad_inf)| <= Minf / eu^k, |coef_k(e_inf)| <= Winf / eu^k, k > k0.
Uso: python scripts/kantorovich_datos.py serie_n.npz datos.npz [inf] [eps_a Rh Rm Rs wm ws]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import kantorovich_series as KS  # noqa: E402
import newton_graduado as NG  # noqa: E402
sys.argv = _argv

B4 = {9: 237.278, 12: 1001.08, 15: 4476.36, 18: 20789.16}     # 4|b_0| (Corolario de b_0)
# Z26 (9.22): eps_a = 7e-3, t' = 3e-3, Wbar = 4,1e-4 -> |d_t c_inf| <= 2 Wbar / t' ; densidad |c_inf| <= Wbar
EU, MINF, WINF = 7e-3, 2 * 4.1e-4 / 3e-3, 4.1e-4


def datos(fn, K, n, inf=False):
    z = np.load(fn)
    N = int(z['N'])
    B0 = NG.base_explicita(0.0); B1 = NG.base_explicita(1.0) - B0
    g = np.real(z['grad'][:, :K + 1])
    gy = np.einsum('ak,aA->Ak', g, B0)
    gy[:, 1:] += np.einsum('ak,aA->Ak', g[:, :-1], B1)
    E = np.real(z['E'][:K + 1]).copy()
    k0 = int(np.ceil(N / 2)) - 2 if inf else K
    return dict(N=N, gy=gy, E=E, k0=k0, inf=inf, nB=np.linalg.norm(B0, 2), nB1=np.linalg.norm(B1, 2))


def cotas(S, P, D, el, er, eps_a, R, wv):
    n, K = S['n'], S['K']
    Rv = np.array([R[0]] * 8 + [R[1]] * 2 + [R[2]] * 4)
    A = 2 * 2 ** 0.5 * eps_a + 8 * R[0] / 2 ** 0.5 + 2 * R[1] / 2 ** 0.5 + eps_a * R[2] * (2 + 2 / 3)
    if n * A >= 1 - R[2]:
        return None
    rr = er / eps_a
    if rr >= 1:
        return None
    k = np.arange(K + 1)
    cola = rr ** (K + 1) / (1 - rr)
    W = KS.BLOQ.choose([1.0, wv[1], wv[2]])
    # --- Ĵ (igual que kantorovich_series)
    e2 = P['he']
    p2 = k[None, None, :] - e2[..., None]
    J0 = -(P['h0'] * el ** p2.clip(0).astype(float)).sum(-1)
    J0 = (J0 + J0.T) / 2
    WJW = W[:, None] * J0 * W[None, :]
    ev = np.linalg.eigvalsh(WJW)
    lam_l, jmax = ev.min(), np.abs(ev).max()
    dJ = (np.abs(P['h0']) * np.where(p2 >= 1, p2 * er ** (p2 - 1).clip(0).astype(float), 0)).sum(-1)
    Jt = (P['ht'] * KS.sup_pot(el, er, p2)).sum(-1)
    mult2 = 1 + np.eye(14)
    colaJ = mult2 * A / np.outer(Rv, Rv) * cola * KS.sup_pot(el, er, -e2)
    pert = np.linalg.norm(W[:, None] * ((er - el) * dJ + Jt + colaJ) * W[None, :])
    lam = lam_l - pert
    if lam <= 0:
        return dict(h=np.inf, motivo='lambda')
    beta = 1 / lam
    normJ = jmax + pert
    dJth = np.linalg.norm(W[:, None] * (Jt + 2 * colaJ) * W[None, :])       # parte de Theta de Ĵ en 0
    # --- d: parte de Theta (n), corrección de tamaño finito (datos), y (p - 1) tau
    e1 = P['ge']
    p1 = k[None, :] - e1[:, None]
    dth = (P['gt'] * KS.sup_pot(el, er, p1)).sum(-1) + 2 * A / Rv * cola * KS.sup_pot(el, er, -e1)
    Kd = min(K, D['gy'].shape[1] - 1)
    cfs = np.real(P['g0'][:, :Kd + 1]) - D['gy'][:, :Kd + 1]
    cEs = np.real(S['E0'][:Kd + 1]) - D['E'][:Kd + 1]
    kz = int(np.ceil(n / 2)) - 2                          # Corolario de universalidad: iguales para k <= kz
    cfs[:, :kz + 1] = 0.0
    cEs[:kz + 2] = 0.0
    k0 = min(Kd, D['k0'])
    kk = np.arange(k0 + 1)
    dfs = (np.abs(cfs[:, :k0 + 1]) * KS.sup_pot(el, er, kk[None, :] - e1[:, None])).sum(-1)
    dEs = (np.abs(cEs[:k0 + 1]) * er ** kk).sum()
    # colas: anillo n (Bauer-Fike) y datos
    dfs = dfs + A / Rv * cola * KS.sup_pot(el, er, -e1)
    dEs = dEs + A * cola
    if D['inf']:
        ru = er / EU
        if ru >= 1:
            return None
        cu = ru ** (k0 + 1) / (1 - ru)
        nb = D['nB'] + er * D['nB1']
        dfs = dfs + nb * np.sqrt(14) * MINF * cu * KS.sup_pot(el, er, -e1)
        dEs = dEs + WINF * cu
    else:
        N = D['N']
        rN = er * N / (n * eps_a)                         # polidisco del anillo N: eps_a n/N, R n/N
        if rN >= 1:
            return None
        cN = rN ** (Kd + 1) / (1 - rN)
        dfs = dfs + A / Rv * cN * KS.sup_pot(el, er, -e1)
        dEs = dEs + A * (n / N) * cN
    dEth = (P['Et'] * er ** k).sum() + 2 * A * cola
    pm1 = (dEs + dEth) / KS.TAUH
    tau0 = D['gy'][:, :Kd + 1]
    diag9 = np.zeros(14); diag9[[10, 11, 12, 13]] = 1 / 9
    tau = np.abs(tau0[:, 0] - diag9) + (np.abs(tau0[:, 1:]) * er ** np.arange(1, Kd + 1)).sum(-1) + A / Rv * cola
    taug = tau * KS.sup_pot(el, er, -e1)
    d = dth + dfs + pm1 * taug
    gv = W * d
    eta = beta * np.linalg.norm(gv)
    # --- tercera y cuarta (igual que kantorovich_series, sin la serie cuártica)
    e3 = P['ce']
    p3 = k[None, None, None, :] - e3[..., None]
    T3 = ((np.abs(P['c0']) + P['ct']) * KS.sup_pot(el, er, p3)).sum(-1)
    ii = np.arange(14)
    eq = (ii[:, None, None] == ii[None, :, None]).astype(int) + (ii[None, :, None] == ii[None, None, :]) + \
         (ii[:, None, None] == ii[None, None, :])
    m3 = np.where(eq == 3, 6, np.where(eq == 1, 2, 1))
    colaT = m3 * A / (Rv[:, None, None] * Rv[None, :, None] * Rv[None, None, :]) * cola * KS.sup_pot(el, er, -e3)
    W3 = W[:, None, None] * W[None, :, None] * W[None, None, :]
    L0 = np.linalg.norm(W3 * (T3 + colaT))
    T3th = np.linalg.norm(W3 * ((P['ct'] * KS.sup_pot(el, er, p3)).sum(-1) + 2 * colaT))   # parte de Theta
    lam_s = el / eps_a
    rho = np.array([lam_s ** 2 * R[0], el * lam_s * R[1], el ** 2 * R[2]])
    tau_c = (rho / wv).min() - 2 * eta
    if tau_c <= 0:
        return dict(h=np.inf, motivo='bola')
    C4 = 96 * A * lam_s ** 4 / tau_c ** 4 * (er / el) ** 4
    L = L0 + C4 * 2 * eta
    if 'q0' in P:
        # cuarta derivada en 0 por su serie y quinta por Cauchy en la bola (como kantorovich_series)
        e4 = P['qe']
        p4 = k[(None,) * 4] - e4[..., None]
        Q = ((np.abs(P['q0']) + P['qt']) * KS.sup_pot(el, er, p4)).sum(-1)
        Rprod = KS.suma_ind(np.log(Rv), 4)
        Q = Q + P['m4'] * A * np.exp(-Rprod) * cola * KS.sup_pot(el, er, -e4)
        W4 = np.exp(KS.suma_ind(np.log(W), 4))
        Q4 = np.linalg.norm(W4 * Q)
        C5 = 120 * 4 * A * lam_s ** 4 / tau_c ** 5 * (er / el) ** 4
        L = L0 + Q4 * 2 * eta + C5 * (2 * eta) ** 2 / 2
    h = beta * L * eta
    u = 2 * eta
    deficit = (0.5 * u * np.linalg.norm(gv) + 0.6 * L * u ** 3) / KS.TAUH
    out = dict(h=h, lam=lam, beta=beta, eta=eta, L=L, deficit=deficit, motivo='h' if h > 0.5 else 'ok',
               dfs=np.linalg.norm(W * dfs), dth=np.linalg.norm(W * dth))
    # --- paso 2: variación en Theta del déficit (requiere h < 1/2 estricto)
    if h < 0.5:
        bp = beta / (1 - 2 * h)
        # |Ĵ_Theta - Ĵ_0| en la bola: Taylor con la parte de Theta de la tercera en 0 y la cuarta (L - L0 >= u sup|F4|)
        dJb = dJth + u * T3th + 2 * u * (L - L0)
        Jb = normJ + L * u
        dpE = dEth / KS.TAUH
        ntau = np.linalg.norm(W * taug)
        ddel = 0.0
        for _ in range(4):
            G = np.linalg.norm(W * dth) + u * dJb + (dpE + ddel) * ntau
            dx = bp * G
            ddel = (Jb * u * dx + 0.5 * u ** 2 * dJb) / KS.TAUH
        out['ddelta'] = ddel
        if n in B4:
            out['principal'] = B4[n] * el ** n / (n * 0.7)  # ebar - e_R = 2/3 + O(eps^2) <= 0,7
    return out


def rango(S, P, D, par, e0=1e-30, e1=0.2, paso=1.01, sens=False):
    eps_a, R, wv = par[0], par[1:4], np.array([1, par[4], par[5]])
    el, ult, sens_ok = e0, None, None
    while el < e1:
        er = el * paso
        c = cotas(S, P, D, el, er, eps_a, R, wv)
        if c is None or c['h'] > 0.5:
            return el, ult, c, sens_ok
        if 'principal' in c and c['ddelta'] < 0.5 * c['principal']:
            sens_ok = er
        ult = c
        el = er
    return el, ult, None, sens_ok


if __name__ == "__main__":
    S = KS.carga(sys.argv[1])
    P = KS.preparar(S)
    inf = 'inf' in sys.argv[3:]
    D = datos(sys.argv[2], S['K'], S['n'], inf=inf)
    nums = [float(x) for x in sys.argv[3:] if x != 'inf']
    n = S['n']
    print(f"n={n}, datos {'sigma_inf (de N=' + str(D['N']) + ', k0=' + str(D['k0']) + ')' if inf else 'sigma_' + str(D['N'])}")
    # estructura: los coeficientes de grad_n(0) - sigma_N deben anularse por debajo del orden 2n/3 - 1 (túnel)
    Kd = min(S['K'], D['gy'].shape[1] - 1)
    cfs = np.real(P['g0'][:, :Kd + 1]) - D['gy'][:, :Kd + 1]
    esc = np.abs(D['gy'][:, :Kd + 1]).max(axis=0) + 1e-300
    rel = np.abs(cfs).max(axis=0) / esc
    print("  |grad_n - sigma_N|/escala por orden:", " ".join(f"{x:.0e}" for x in rel[:2 * n // 3 + 2]))
    if nums:
        cands = [tuple(nums)]
    else:
        base = {9: (0.03, 1e-4, 4e-3, 0.1, 1, 1), 12: (0.02, 2e-5, 1e-3, 0.1, 0.3, 0.3)}[n]
        cands = []
        for fa in (0.5, 0.75, 1, 1.25):
            for fh in (0.5, 1, 2):
                for fm in (0.5, 1, 2):
                    for fs in (0.5, 1, 1.5):
                        for fw in (0.3, 1, 3):
                            cands.append((base[0] * fa, base[1] * fh, base[2] * fm, base[3] * fs,
                                          base[4] * fw, base[5] * fw))
    pruebas = np.geomspace(1e-9, 2e-2, 60)
    best = (0.0, None)
    for par in cands:
        e_ok = 0.0
        for e in pruebas:
            c = cotas(S, P, D, e, e * 1.02, par[0], par[1:4], np.array([1, par[4], par[5]]))
            if c is None or c['h'] > 0.5:
                break
            e_ok = e
        if e_ok > best[0]:
            best = (e_ok, par)
    print(f"  rejilla gruesa: eps_max ~ {best[0]:.3e} con {best[1]}", flush=True)
    if best[1] is None:
        raise SystemExit("  no hay rango")
    emax, ult, fallo, sens = rango(S, P, D, best[1])
    print(f"  intervalos finos: VALE en [1e-30, {emax:.3e}]; en el borde h = {ult['h']:.3f}, eta = {ult['eta']:.2e}, "
          f"|d_fs| = {ult['dfs']:.2e}, |d_Theta| = {ult['dth']:.2e}, deficit <= {ult['deficit']:.1e}; "
          f"falla por {fallo['motivo'] if fallo else '-'}")
    if 'principal' in ult:
        print(f"  paso 2: en el borde |delta(Theta) - delta(0)| <= {ult['ddelta']:.1e}, principal >= {ult['principal']:.1e}; "
              f"la variación del déficit queda por debajo de la mitad del principal hasta eps = "
              f"{sens if sens else 0:.3e}")

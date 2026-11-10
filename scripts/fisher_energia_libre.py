"""Energía libre modular: la máxima entropía como transformada de Legendre de un vmax a temperatura finita.

El dual de la máxima entropía es S_G(v, phi) = min_y [ln Z(y) - v y.t1(phi)]. Con y.t1 = beta y W = y.O/beta, de
modo que N_phi(W) = 1,
    S_G(v, phi) = min_beta [Phi(beta, phi) - beta v],    Phi(beta, phi) = min_{N_phi(W) = 1} ln tr exp(beta W),
y f_beta = Phi/beta es una energía libre modular: vmax <= f_beta <= vmax + n ln 2/beta, no creciente en beta, con
límite vmax. Phi es convexa en beta, con Phi' = v(beta), y Phi - beta vmax decrece hacia la entropía residual
lim_{v -> vmax} S_G. Por la paridad (Phi par en el flujo), en t = 0 no hay término cruzado y
    S''(0) = Phi_tt(beta*, 0) = beta* f_beta''(0):
el signo de la parte de holonomía de la entropía es el de la curvatura de f_beta frente al flujo. Y el testigo girado
de la cota de Kohn, a temperatura finita, da una cota exacta para todo flujo:
    S_G(v, phi + delta) - S_G(v, phi) <= ln tr exp((x.O)[delta]) - ln tr exp(x.O),
con x los multiplicadores en phi: (x.O)[delta] cumple N_{phi+delta} = beta y entra en el dual.

Phi se calcula sin pasar por el resolvedor de máxima entropía: Newton sobre el sistema KKT de min ln Z(y) con
y.t1 = beta (multiplicador mu = v). Phi_tt, por diferencias centradas de Phi_t = -mu y.t1'(t) (envolvente).

Comprobaciones, en las escaleras periódicas 2 x 3 y 2 x 4 de fisher_kubo_mori.py:
  1. Legendre: min_beta [Phi - beta v] = S_G y el argmin es beta* = x.t1, con y sin flujo.
  2. Cotas de f_beta, monotonía y Phi - beta vmax decreciente.
  3. S''(0) = Phi_tt(beta*, 0) frente a la fórmula de Kubo--Mori (v = 0,1, 0,2, 0,3).
  4. Cota de Kohn a temperatura finita, con flujo finito (F = pi/8 a 2 pi) y a segundo orden.
  5. Cerca de vmax: f_beta''(0) = S''/beta frente a vmax''(0) (SDP) y a la curvatura del testigo girado. Con gap
     (2 x 4), f_beta''(0) -> vmax''(0), la curvatura del testigo reoptimizado, unas 18 veces la del girado; degenerado
     (2 x 3), vmax'' = 0 pero f_beta''(0) tiende a una constante positiva: los límites beta -> oo y t -> 0 no conmutan.

Escribe data/fisher_energia_libre{,_curvatura,_kohn,_T0,_resumen}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/fisher_energia_libre.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from fisher_kubo_mori import MaxEnt, escalera, derivadas, testigo_optimo, girado
from holonomy.report import write_csv

VS = (0.1, 0.2, 0.3)
H = 2e-3
RESUMEN = []


def ln_z(me, y):
    return float(logsumexp(eigh(np.einsum("k,kab->ab", y, me.O), eigvals_only=True)))


def dt1(u, t):
    """d t1(t u)/dt con v = 1."""
    return np.array([[-2 * np.sin(t * ue) * ue, 2 * np.cos(t * ue) * ue, 0.0] for ue in u]).ravel()


def phi_beta(me, beta, t1, y0, tol=1e-13):
    """Phi = min ln Z(y) con y.t1 = beta, por Newton sobre el sistema KKT m(y) = mu t1, y.t1 = beta.
    Devuelve Phi, y, mu y el residuo."""
    y = y0 * beta / (y0 @ t1)
    m, g, S = me.gibbs(y)
    mu = float(m @ t1 / (t1 @ t1))
    K = len(y)

    def resid(y, m, mu):
        return np.concatenate([m - mu * t1, [y @ t1 - beta]])

    r = resid(y, m, mu)
    for _ in range(80):
        if np.abs(r).max() < tol:
            break
        J = np.zeros((K + 1, K + 1))
        J[:K, :K], J[:K, K], J[K, :K] = g, -t1, t1
        paso = np.linalg.solve(J, -r)
        a = 1.0
        while a > 1e-8:
            yn, mun = y + a * paso[:K], mu + a * paso[K]
            mn, gn, Sn = me.gibbs(yn)
            rn = resid(yn, mn, mun)
            if np.abs(rn).max() < np.abs(r).max():
                break
            a /= 2
        y, mu, m, g, S, r = yn, mun, mn, gn, Sn, rn
    return S + y @ m, y, mu, float(np.abs(r).max())


def phi_tt(me, beta, u, y0):
    """Phi_tt(beta, 0) por diferencias centradas de cuarto orden de Phi_t = -mu y.t1'(t)."""
    d = {}
    res = 0.0
    for k in (-2, -1, 1, 2):
        t = k * H
        _, y, mu, r = phi_beta(me, beta, me.objetivo(1.0, t * u), y0)
        d[k] = -mu * y @ dt1(u, t)
        res = max(res, r)
    return (-d[2] + 8 * d[1] - 8 * d[-1] + d[-2]) / (12 * H), res


def testigo_simetrico(me, L):
    """Testigo óptimo simétrico sin flujo (fisher_kubo_mori.limite_T0): Heisenberg uniforme en la 2 x 4, los dos
    triángulos en la 2 x 3. Da vmax exacto (lambda_max), sin el error del SDP."""
    t1 = me.objetivo(1.0, np.zeros(3 * L))
    peldano = 1.0 if L == 4 else 0.0
    ys = np.array([[1.0 if e < 2 * L else peldano, 0.0, 1.0 if e < 2 * L else peldano] for e in range(3 * L)]).ravel()
    ys /= ys @ t1
    return ys, float(eigh(np.einsum("k,kab->ab", ys, me.O), eigvals_only=True)[-1])


def legendre(chk):
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        n = 2 * L
        _, vmax = testigo_simetrico(me, L)
        for F in (0.0, np.pi / 2):
            t = F / L
            t1 = me.objetivo(1.0, t * u)
            for v in VS:
                x, g, S, res = me.resuelve(me.objetivo(v, t * u))
                bstar = float(x @ t1)
                cache = {"y": x}

                def dual(b):
                    P, y, mu, r = phi_beta(me, b, t1, cache["y"])
                    cache["y"] = y
                    return P - b * v

                opt = minimize_scalar(dual, bracket=(0.5 * bstar, bstar, 2 * bstar), tol=1e-12)
                P, y, mu, r = phi_beta(me, bstar, t1, x)
                chk.check(abs(opt.fun - S) < 1e-10 and abs(opt.x - bstar) / bstar < 1e-5 and abs(mu - v) < 1e-10,
                          f"2x{L}, F = {F:.4f}, v = {v}: min_beta [Phi - beta v] = {opt.fun:.12f} = S_G = {S:.12f} "
                          f"(dif. {opt.fun - S:+.1e}); argmin {opt.x:.6f}, beta* = x.t1 = {bstar:.6f}; Phi'(beta*) = mu "
                          f"= {mu:.12f}; residuos {res:.0e}, {r:.0e}")
                filas.append([L, F, v, S, opt.fun, bstar, opt.x, mu, P])
        # Cotas y monotonía de f_beta, sin flujo, por continuación en beta.
        t1 = me.objetivo(1.0, 0 * u)
        y = t1.copy()
        fs, exc = [], []
        for b in (0.3, 1, 3, 10, 30, 100, 300):
            P, y, mu, r = phi_beta(me, b, t1, y)
            fs.append(P / b)
            exc.append(P - b * vmax)
            chk.info(f"2x{L}, beta = {b:6.1f}: f_beta = {P / b:.10f}, f_beta - vmax = {P / b - vmax:.3e}, "
                     f"Phi - beta vmax = {P - b * vmax:.6f}, v(beta) = mu = {mu:.10f}, residuo {r:.0e}")
            filas.append([L, 0.0, mu, np.nan, np.nan, b, b, mu, P])
        bs = np.array((0.3, 1, 3, 10, 30, 100, 300))
        fs = np.array(fs)
        chk.check(np.all(fs >= vmax - 1e-12) and np.all(fs <= vmax + n * np.log(2) / bs) and np.all(np.diff(fs) < 0)
                  and np.all(np.diff(exc) < 0),
                  f"2x{L}: vmax <= f_beta <= vmax + n ln 2/beta, f_beta decreciente y Phi - beta vmax decreciente "
                  f"(de {exc[0]:.4f} a {exc[-1]:.4f} en beta = 300)")
        # Entropía residual = lim (Phi - beta vmax) <= ln(multiplicidad del máximo del testigo óptimo).
        if L == 4:
            chk.check(0 <= exc[-1] < 1e-8, f"2x4: Phi - beta vmax = {exc[-1]:.1e} en beta = 300: entropía residual nula "
                                           "(máximo simple)")
        else:
            chk.check(1.5 < exc[-1] < np.log(16), f"2x3: Phi - beta vmax = {exc[-1]:.4f} en beta = 300, entre 1,5 y "
                                                  f"ln 16 = {np.log(16):.4f}: entropía residual")
    write_csv("fisher_energia_libre.csv", ("L", "F", "v", "S_G", "min_dual", "beta_star", "argmin", "mu", "Phi"), filas)


def curvatura(chk):
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        for v in VS:
            x, g, S, res = me.resuelve(me.objetivo(v, 0 * u))
            dm, d2m = derivadas(v, u)
            s2 = -dm @ np.linalg.solve(g, dm) - x @ d2m
            bstar = float(x @ me.objetivo(1.0, 0 * u))
            ptt, r = phi_tt(me, bstar, u, x)
            rel = abs(ptt - s2) / abs(s2)
            chk.check(rel < 1e-6, f"2x{L}, v = {v}: Phi_tt(beta*, 0) = {ptt:+.10e} = S''(0) = {s2:+.10e} "
                                  f"(dif. rel. {rel:.1e}); f_beta'' = {ptt / bstar:+.4e}, beta* = {bstar:.4f}")
            filas.append([L, v, bstar, s2, ptt, ptt / bstar])
    write_csv("fisher_energia_libre_curvatura.csv", ("L", "v", "beta", "S2", "Phi_tt", "f_tt"), filas)


def kohn_termica(chk):
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        for v in VS:
            x0, g, S0, _ = me.resuelve(me.objetivo(v, 0 * u))
            lz0 = ln_z(me, x0)
            R = {k: float(logsumexp(eigh(girado(me, edges, u, x0, k * H), eigvals_only=True))) for k in (-2, -1, 1, 2)}
            R2 = (-R[2] + 16 * R[1] - 30 * lz0 + 16 * R[-1] - R[-2]) / (12 * H**2)
            dm, d2m = derivadas(v, u)
            s2 = -dm @ np.linalg.solve(g, dm) - x0 @ d2m
            chk.check(s2 <= R2, f"2x{L}, v = {v}: S''(0) = {s2:+.6e} <= (ln Z del girado)'' = {R2:+.6e}")
            x = x0
            for F in (np.pi / 8, np.pi / 2, np.pi, 2 * np.pi):
                t = F / L
                x, _, S, res = me.resuelve(me.objetivo(v, t * u), x)
                cota = float(logsumexp(eigh(girado(me, edges, u, x0, t), eigvals_only=True))) - lz0
                chk.check(S - S0 <= cota + 1e-12,
                          f"2x{L}, v = {v}, F = {F:.4f}: Delta S = {S - S0:+.6e} <= Delta ln Z del girado = {cota:+.6e}")
                filas.append([L, v, F, S - S0, cota, s2, R2])
    write_csv("fisher_energia_libre_kohn.csv", ("L", "v", "F", "dS", "cota", "S2", "R2"), filas)


def cerca_de_vmax(chk):
    """f_beta''(0) = S''/beta cuando beta -> oo, frente a vmax''(0) (SDP, Richardson) y a lambda_max'' del girado."""
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        vmax, y0, _ = testigo_optimo(me, u)
        vh = {}
        for h in (1e-2, 5e-3):
            vh[h] = (testigo_optimo(me, u, h)[0] + testigo_optimo(me, u, -h)[0] - 2 * vmax) / h**2
        vpp = (4 * vh[5e-3] - vh[1e-2]) / 3
        lam = {k: eigh(girado(me, edges, u, y0, k * 1e-2), eigvals_only=True)[-1] for k in (-1, 0, 1)}
        kohn = (lam[1] + lam[-1] - 2 * lam[0]) / 1e-4
        chk.info(f"2x{L}: vmax''(0) = {vpp:+.5f} (SDP, Richardson de h = 1e-2 y 5e-3: {vh[1e-2]:+.5f}, {vh[5e-3]:+.5f}); "
                 f"lambda_max'' del testigo girado del SDP = {kohn:+.5f}")
        x = np.zeros(len(me.O))
        t1 = me.objetivo(1.0, 0 * u)
        ys, vsim = testigo_simetrico(me, L)
        # En la 2 x 4, x.O = beta W simétrico exactamente (Prop. prop:simetrico): más allá de eps = 1e-4 se fija beta.
        puntos = [(eps, None) for eps in (1e-1, 1e-2, 1e-3, 1e-4)] + ([(None, b) for b in (150, 200, 300)] if L == 4 else [])
        fpp = []
        for eps, b in puntos:
            if b is None:
                v = vmax * (1 - eps)
                x, g, S, res = me.resuelve(me.objetivo(v, 0 * u), x)
            else:
                x = b * ys
                m, g, S = me.gibbs(x)
                v, res = float(ys @ m), 0.0
                eps = 1 - v / vsim
            beta = float(x @ t1)
            dm, d2m = derivadas(v, u)
            s2 = -dm @ np.linalg.solve(g, dm) - x @ d2m
            ptt, r = phi_tt(me, beta, u, x)
            R = {k: float(logsumexp(eigh(girado(me, edges, u, x, k * H), eigvals_only=True))) for k in (-1, 0, 1)}
            R2 = (R[1] + R[-1] - 2 * R[0]) / H**2
            chk.info(f"2x{L}, 1 - v/vmax = {eps:.0e}: beta = {beta:8.3f}, f_beta'' = S''/beta = {s2 / beta:+.6f} "
                     f"(Phi_tt/beta = {ptt / beta:+.6f}), cota girada R''/beta = {R2 / beta:+.6f}, "
                     f"residuos {res:.0e}, {r:.0e}")
            filas.append([L, eps, beta, s2, ptt, s2 / beta, R2 / beta, vpp, kohn])
            fpp.append((s2 / beta, R2 / beta))
        chk.check(all(f <= k for f, k in fpp), f"2x{L}: f_beta'' <= R''/beta (Kohn térmica) en todos los beta")
        if L == 4:
            chk.check(abs(fpp[-1][0] - vpp) < 1e-4 and abs(fpp[-1][0] - fpp[-2][0]) < 1e-6,
                      f"2x4: f_beta''(0) -> {fpp[-1][0]:+.6f} (beta = 300; {fpp[-2][0]:+.6f} en 200) = vmax''(0) = {vpp:+.6f} "
                      f"del SDP, dif. {fpp[-1][0] - vpp:+.1e}; el testigo girado da {kohn:+.5f}, {vpp / kohn:.1f} veces menos")
            RESUMEN.extend([["vmax_tt_4", vpp], ["f_tt_lim_4", fpp[-1][0]], ["kohn_tt_4", kohn]])
        else:
            f = [a for a, _ in fpp]
            # vmax = 1/3 para todo flujo (techo y emparejamientos): vmax'' = 0; el SDP solo lo da a su ruido, 1e-9/h^2.
            chk.check(abs(vpp) < 1e-3 and all(np.diff(f) > 0) and f[-1] > 0.05,
                      f"2x3: vmax''(0) = 0 (SDP: {vpp:+.1e}, ruido), pero f_beta''(0) = "
                      f"{', '.join(f'{z:+.4f}' for z in f)} crece hacia una constante positiva: los límites no conmutan")
    write_csv("fisher_energia_libre_T0.csv", ("L", "eps", "beta", "S2", "Phi_tt", "f_tt", "kohn_beta", "vmax_tt",
                                              "kohn_T0"), filas)


def main():
    chk = Checker("fisher_energia_libre")
    legendre(chk)
    curvatura(chk)
    kohn_termica(chk)
    cerca_de_vmax(chk)
    write_csv("fisher_energia_libre_resumen.csv", ("magnitud", "valor"), RESUMEN)
    chk.finish()


if __name__ == "__main__":
    main()

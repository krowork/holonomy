"""El régimen degenerado: entropía residual exacta y cómo se llega a ella.

La máxima entropía converge al estado óptimo de mayor entropía (Prop. prop:limite del artículo II), que vive en el
autoespacio máximo de cualquier testigo óptimo. Su entropía S* sale de un problema de máxima entropía en ese autoespacio.

2 x 3 (prisma): el autoespacio máximo de los dos triángulos desacoplados es P (dimensión 16: espín 1/2 en cada
triángulo). Por SU(2) y los automorfismos, rho* = exp(-a P Sr P)/Z en P, con Sr = suma de sigma.sigma en los peldaños,
cuyo espectro en P es -5 (1), -1 (5), 1/3 (6), 5/3 (3), 3 (1); la restricción <Sr> = -3 fija a, y con z = e^{-4a/3},
3z^6 + 7z^5 + 10z^4 + 5z^3 - 1 = 0. El peldaño no conmuta con las patas: a segundo orden, la fuga fuera de P es
6 eps = g^2 c, g = a/lambda_pata, c = tr(rho* Sr Q (6 - Wtri)^-1 Q Sr), y beta sqrt(eps) -> 18 a sqrt(c/6).

Diamantes, punto A (cadena cerrada, datos triviales): So = sigma.sigma en las aristas exteriores y Sw = sigma_u.sigma_w
conmutan, rho* es función de los dos, y S* sale de su espectro conjunto. k impar: el fundamental del testigo óptimo
(b/a = -2 - e) son los dos defectos (pentámero, w = -k + 2; espín libre, w = -k - 2), 2k estados cada uno; la
restricción da el peso p = (2 + 3c)/4 del pentámero y S* = ln(2k) + H(p). k par: el fundamental de H(a) con a = 0,185
son las dos teselas (triplete, singlete), y S* = ln 2. Como el resto conmuta con el testigo, la fuga es de Boltzmann:
beta ~ ln(1/eps)/Delta, y en k = 4 la máxima entropía tiende al testigo óptimo de mayor gap.

Comprobaciones:
  1. 2 x 3: espectro de P Sr P, a, S*, polinomio de z, factibilidad de rho* (15 correlaciones por arista); máxima
     entropía densa en eps = 1e-2..1e-4: multiplicador de peldaño -> a, beta sqrt(eps) frente a 18 a sqrt(c/6),
     S - S* ~ sqrt(eps).
  1b. 2 x 3 con flujo: rho*(t) por máxima entropía en el soporte de la Proposición 8(b) (dimensión 36); rango 16,
     parte impar derivable y parte par con un pico, S*(t) = S*(0) + s1 |t|; la SLD de la máxima entropía tiende a
     la Fisher de la parte impar, y con eps = 1e-4 el pico aparece suavizado en |t| ~ 23/beta.
  2. Diamantes k = 2, 3, 4: espectro conjunto completo, vmax por programación lineal, cara óptima y S* frente a la
     fórmula; k = 3, 5, 7: degeneración 2k del fundamental en S^z = 1/2 y sus (o, w); k = 2, 4, 6: fundamental de
     H(0,185) = las dos teselas.
  3. Diamantes k = 3, 4: máxima entropía clásica de dos parámetros en eps = 1e-2..1e-6: beta/ln(1/eps) acotado,
     beta sqrt(eps) -> 0, S - S* ~ eps; k = 4: el a de la máxima entropía frente al a de mayor gap.

Escribe data/regimen_degenerado{,_fisher,_diamantes,_aproximacion,_resumen}.csv.

Uso: (ulimit -v 14000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/regimen_degenerado.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import logsumexp
import cvxpy as cp

from fisher_kubo_mori import MaxEnt, escalera
from holonomy.diamond import chain_edges, n_sites, edge_kinds
from holonomy.paulis import pauli_string
from holonomy.report import read_csv, write_csv

E_PENT = -3.3815016433276  # menor raíz de 2x^3 + 8x^2 + 3x - 4 (Prop. del punto A)
C_A = (8 + 2 * E_PENT) / (3 * (2 - E_PENT))
RESUMEN = []


def entropia(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def prisma(chk):
    L = 3
    edges, u = escalera(L)
    me = MaxEnt(2 * L, edges)
    heis = [me.O[3 * e] + me.O[3 * e + 2] for e in range(len(edges))]  # A + C = -sigma.sigma
    Wtri = sum(heis[:2 * L])
    ev, V = eigh(Wtri)
    top = ev > ev[-1] - 1e-9
    P = V[:, top]
    Sr = -sum(heis[2 * L:])
    w, Vr = eigh(P.conj().T @ Sr @ P)
    niveles, mult = np.unique(np.round(w, 8), return_counts=True)
    chk.check(P.shape[1] == 16 and abs(ev[-1] - 6) < 1e-12 and np.allclose(niveles, [-5, -1, 1 / 3, 5 / 3, 3])
              and list(mult) == [1, 5, 6, 3, 1],
              f"2x3: autoespacio máximo de los triángulos de dimensión {P.shape[1]}; espectro de P Sr P: "
              + ", ".join(f"{l:+.4f} ({m})" for l, m in zip(niveles, mult)))
    media = lambda a: float(np.exp(-a * w - logsumexp(-a * w)) @ w)
    a = brentq(lambda a: media(a) + 3, 0, 50, xtol=1e-15)
    p = np.exp(-a * w - logsumexp(-a * w))
    Sst = entropia(p)
    z = np.exp(-4 * a / 3)
    pol = 3 * z**6 + 7 * z**5 + 10 * z**4 + 5 * z**3 - 1
    rho = (P @ Vr) @ np.diag(p) @ (P @ Vr).conj().T
    err = 0.0
    for i, j in edges:
        for s in range(3):
            err = max(err, abs(np.trace(rho @ pauli_string(6, ((i, s),))).real))
            for t in range(3):
                val = np.trace(rho @ pauli_string(6, ((i, s), (j, t)))).real
                err = max(err, abs(val - (-1 / 3 if s == t else 0)))
    chk.check(abs(pol) < 1e-13 and err < 1e-12 and Sst < np.log(16),
              f"2x3: rho* = exp(-a P Sr P)/Z, a = {a:.10f}, z = e^(-4a/3) = {z:.10f} (polinomio {pol:.0e}); "
              f"S* = {Sst:.10f} < ln 16; factible: 15 correlaciones por arista a {err:.0e}")
    RESUMEN.extend([["a_prisma", a], ["z_prisma", z], ["S_prisma", Sst]])
    # segundo orden
    R = V[:, ~top] @ np.diag(1 / (6 - ev[~top])) @ V[:, ~top].conj().T
    c2 = float(np.trace(rho @ Sr @ R @ Sr).real)
    pred = 18 * a * np.sqrt(c2 / 6)
    RESUMEN.extend([["c2_prisma", c2], ["pred_beta_sqrt_eps", pred]])
    filas = []
    x = np.zeros(len(me.O))
    t1 = me.objetivo(1.0, 0 * u)
    for eps in (1e-2, 1e-3, 1e-4):
        x, g, S, res = me.resuelve(me.objetivo((1 - eps) / 3, 0 * u), x)
        beta = float(x @ t1)
        filas.append([eps, beta, beta * np.sqrt(eps), x[3 * 2 * L::3].mean(), S, S - Sst, res])
        chk.info(f"2x3, eps = {eps:.0e}: beta sqrt(eps) = {beta * np.sqrt(eps):.5f}, multiplicador de peldaño "
                 f"{x[3 * 2 * L::3].mean():.6f}, S - S* = {S - Sst:.4e}, residuo {res:.0e}")
    f = {r[0]: r for r in filas}
    chk.check(abs(f[1e-4][2] / pred - 1) < 1e-3 and f[1e-2][3] < f[1e-3][3] < f[1e-4][3] < a,
              f"2x3: beta sqrt(eps) = {f[1e-4][2]:.5f} en 1e-4 frente a 18 a sqrt(c/6) = {pred:.5f} (c = {c2:.6f}); "
              f"el multiplicador de peldaño sube hacia a = {a:.4f}")
    q = [f[1e-2][5] / f[1e-3][5], f[1e-3][5] / f[1e-4][5]]
    chk.check(all(abs(r / np.sqrt(10) - 1) < 0.05 for r in q),
              f"2x3: S - S* cae {q[0]:.3f} y {q[1]:.3f} veces por década (sqrt(10) = {np.sqrt(10):.3f}): ~ sqrt(eps)")
    write_csv("regimen_degenerado.csv", ("eps", "beta", "beta_sqrt_eps", "x_peldano", "S", "S_menos_Sast", "residuo"),
              filas)
    return me, edges, u, rho, Sst


def maxent_soporte(Op, objetivo, tol=1e-12):
    """Máxima entropía dentro de un subespacio (operadores ya restringidos), por Newton con mínimos cuadrados: los
    operadores restringidos son dependientes, y el óptimo puede estar en el borde (los pesos que sobran caen a 0)."""
    K = len(Op)
    y = np.zeros(K)
    for _ in range(300):
        E, U = eigh(np.einsum("k,kab->ab", y, Op))
        p = np.exp(E - logsumexp(E))
        M = U.conj().T[None] @ Op @ U[None]
        m = np.einsum("i,kii->k", p, M).real
        r = m - objetivo
        if np.abs(r).max() < tol:
            break
        dE, dp = E[:, None] - E[None, :], p[:, None] - p[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(dE) > 1e-10, dp / np.where(dE == 0, 1, dE), (p[:, None] + p[None, :]) / 2)
        g = ((M * w[None]).reshape(K, -1) @ M.transpose(0, 2, 1).reshape(K, -1).T).real - np.outer(m, m)
        paso = -np.linalg.lstsq(g, r, rcond=1e-12)[0]
        a = 1.0
        while a > 1e-8:
            En, Un = eigh(np.einsum("k,kab->ab", y + a * paso, Op))
            pn = np.exp(En - logsumexp(En))
            if np.abs(np.einsum("i,kii->k", pn, Un.conj().T[None] @ Op @ Un[None]).real - objetivo).max() < np.abs(r).max():
                break
            a /= 2
        y = y + a * paso
    return (U * p) @ U.conj().T, float(np.abs(r).max())


def fisher_prisma(chk, me, edges, u, rho64, Sst):
    """La familia de estados óptimos rho*(t) de la 2 x 3 en el flujo de plaquetas. Por la Proposición 8(b) de
    holonomia, todo estado factible en vmax = 1/3 vive en (Z_a + Z_b + Z_c)^2 = 1 en cada triángulo (dimensión 36,
    sin depender de t); rho*(t) es la máxima entropía ahí. rho*(t) = rho0 + t X + |t| Y + O(t^2): la parte par tiene
    un pico, S*(t) = S*(0) + s1 |t|, y la SLD de la máxima entropía tiende a la Fisher de la parte impar."""
    s = np.arange(64)
    z = lambda i: 1 - 2 * ((s >> (5 - i)) & 1)
    idx = np.flatnonzero(((z(0) + z(1) + z(2))**2 == 1) & ((z(3) + z(4) + z(5))**2 == 1))
    Op = me.O[:, idx][:, :, idx]
    R0 = rho64[np.ix_(idx, idx)]
    e0, U0 = np.linalg.eigh(R0)
    filas = []
    res_max = 0.0
    rho_t = {}
    for t in (0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05, 0.2):
        for sg in (1, -1):
            rho_t[sg * t], r = maxent_soporte(Op, me.objetivo(1 / 3, sg * t * u))
            res_max = max(res_max, r)
    ent = lambda R: entropia(np.clip(np.linalg.eigvalsh(R), 0, None))
    rango = [int((np.linalg.eigvalsh(rho_t[t]) > 1e-7).sum()) for t in (0.01, 0.05, 0.2)]
    resto = max(np.sort(np.linalg.eigvalsh(rho_t[t]))[-17] for t in (0.01, 0.05, 0.2))
    chk.check(len(idx) == 36 and rango == [16, 16, 16] and res_max < 1e-11,
              f"2x3: rho*(t) en el soporte de dimensión {len(idx)}: rango 16 en t = 0,01, 0,05, 0,2 (los otros 20 "
              f"autovalores <= {resto:.0e}); residuo <= {res_max:.0e}")

    def sld(X):
        Xt = U0.conj().T @ X @ U0
        ss = e0[:, None] + e0[None, :]
        return float((2 * np.abs(Xt)**2 / np.where(ss > 1e-12, ss, np.inf)).sum())

    impar, par, F = {}, {}, {}
    for t in (0.005, 0.01, 0.02):
        D = rho_t[t] - rho_t[-t]
        impar[t] = np.abs(np.linalg.eigvalsh(D)).sum() / 2 / t
        par[t] = np.abs(np.linalg.eigvalsh((rho_t[t] + rho_t[-t]) / 2 - R0)).sum() / t
        F[t] = sld(D / (2 * t))
        filas.append([t, impar[t], par[t], F[t], ent(rho_t[t]) - Sst])
    F_odd = (4 * F[0.005] - F[0.01]) / 3
    chk.check(max(impar.values()) - min(impar.values()) < 1e-3 and min(par.values()) > 0.8,
              f"2x3: |parte impar|_1/t = {', '.join(f'{v:.5f}' for v in impar.values())} y |parte par|_1/t = "
              f"{', '.join(f'{v:.4f}' for v in par.values())} (t = 0,005..0,02): rho*(t) no es derivable en t = 0")
    pend = {t: (ent(rho_t[t]) - Sst) / t for t in (0.001, 0.002, 0.004)}
    s1 = 2 * pend[0.001] - pend[0.002]
    chk.check(all(abs(ent(rho_t[t]) - ent(rho_t[-t])) < 1e-9 for t in (0.01, 0.2)) and 0.6 < s1 < 0.66,
              f"2x3: S*(t) par y (S*(t) - S*(0))/t = {pend[0.004]:.6f}, {pend[0.002]:.6f}, {pend[0.001]:.6f} "
              f"(t = 0,004..0,001): pico s1 = {s1:.4f}")
    t0 = {float(r["eps"]): float(r["fisher_sld"]) for r in read_csv("fisher_kubo_mori_T0.csv") if int(r["L"]) == 3}
    ext = t0[1e-4] + (t0[1e-4] - t0[1e-3]) / (np.sqrt(10) - 1)
    chk.check(abs(ext / F_odd - 1) < 1e-3,
              f"2x3: Fisher (SLD) de la parte impar en rho0 = {F_odd:.6f} (Richardson; {F[0.02]:.6f}, {F[0.01]:.6f}, "
              f"{F[0.005]:.6f}); la SLD de la máxima entropía, {t0[1e-2]:.4f}, {t0[1e-3]:.4f}, {t0[1e-4]:.4f}, "
              f"extrapolada como sqrt(eps): {ext:.5f}")
    # cruce en la máxima entropía completa: pico suavizado en |t| ~ 2 s1/(beta f'')
    T0 = {float(r["eps"]): r for r in read_csv("fisher_energia_libre_T0.csv") if int(r["L"]) == 3}
    S2, beta = float(T0[1e-4]["S2"]), float(T0[1e-4]["beta"])
    v = (1 - 1e-4) / 3
    x, _, S0, _ = me.resuelve(me.objetivo(v, 0 * u))
    cruce = {}
    for t in (0.01, 0.2):
        x, _, St, r = me.resuelve(me.objetivo(v, t * u), x)
        cruce[t] = St - S0
    w = 2 * s1 / float(T0[1e-4]["f_tt"]) / beta
    chk.check(abs(cruce[0.01] / (S2 * 0.01**2 / 2) - 1) < 0.1 and abs(cruce[0.2] / (ent(rho_t[0.2]) - Sst) - 1) < 0.15,
              f"2x3, eps = 1e-4 (beta = {beta:.1f}): Delta S_G = {cruce[0.01]:.5f} en t = 0,01 (1/2 S'' t^2 = "
              f"{S2 * 0.01**2 / 2:.5f}) y {cruce[0.2]:.5f} en t = 0,2 (S*(0,2) - S*(0) = {ent(rho_t[0.2]) - Sst:.5f}; "
              f"1/2 S'' t^2 = {S2 * 0.2**2 / 2:.4f}); anchura del pico 2 s1/(beta f'') = {w:.4f} = {w * beta:.1f}/beta")
    RESUMEN.extend([["F_impar_prisma", F_odd], ["s1_prisma", s1], ["sld_ext_prisma", ext], ["anchura_beta", w * beta],
                    ["impar_prisma", impar[0.01]], ["par_prisma", par[0.01]], ["dS_cruce_001", cruce[0.01]],
                    ["dS_cruce_02", cruce[0.2]], ["Sast_02", ent(rho_t[0.2]) - Sst], ["S2_1e-4", S2]])
    write_csv("regimen_degenerado_fisher.csv", ("t", "impar_sobre_t", "par_sobre_t", "sld_impar", "S_menos_S0"), filas)


def ops_sector(n, M, pares):
    """Suma de sigma_i.sigma_j sobre pares, dispersa y real, en el sector con M bits a 1 (qubit 0 = bit más alto)."""
    st = np.array([s for s in range(1 << n) if bin(s).count("1") == M], dtype=np.int64)
    pos = -np.ones(1 << n, dtype=np.int64)
    pos[st] = np.arange(len(st))
    total = 0
    for i, j in pares:
        bi, bj = (st >> (n - 1 - i)) & 1, (st >> (n - 1 - j)) & 1
        dif = np.flatnonzero(bi != bj)
        t = pos[st[dif] ^ ((1 << (n - 1 - i)) | (1 << (n - 1 - j)))]
        total = total + sps.diags(np.where(bi == bj, 1.0, -1.0)) + sps.csr_matrix(
            (np.full(len(dif), 2.0), (t, dif)), shape=(len(st), len(st)))
    return total


def cadena(k, M):
    n = n_sites(k, periodic=True)
    ed, kinds = chain_edges(k, periodic=True), edge_kinds(k, periodic=True)
    return (ops_sector(n, M, [e for e, t in zip(ed, kinds) if t != "uw"]),
            ops_sector(n, M, [e for e, t in zip(ed, kinds) if t == "uw"]))


def espectro_conjunto(k):
    n = n_sites(k, periodic=True)
    lv = []
    for M in range(n + 1):
        So, Sw = (A.toarray() for A in cadena(k, M))
        _, U = np.linalg.eigh(So + np.pi * Sw)
        lv.append(np.c_[np.einsum("ij,ij->j", U, So @ U), np.einsum("ij,ij->j", U, Sw @ U)])
    return np.unique(np.round(np.vstack(lv), 7), axis=0, return_counts=True)


def S_formula(k):
    if k % 2 == 0:
        return np.log(2)
    p = (2 + 3 * C_A) / 4
    return np.log(2 * k) + entropia(np.array([p, 1 - p]))


def diamantes(chk):
    filas = []
    p = (2 + 3 * C_A) / 4
    RESUMEN.extend([["p_pentamero", p], ["H_p", entropia(np.array([p, 1 - p]))]])
    for k in (2, 3, 4):
        L, d = espectro_conjunto(k)
        m = len(L)
        Aeq = np.zeros((3, m + 1))
        Aeq[0, :m], Aeq[0, -1] = L[:, 0], 12 * k
        Aeq[1, :m], Aeq[1, -1] = L[:, 1], 3 * k
        Aeq[2, :m] = 1
        c = np.zeros(m + 1)
        c[-1] = -1
        r = linprog(c, A_eq=Aeq, b_eq=[0, 0, 1], bounds=[(0, None)] * m + [(None, None)], method="highs")
        vmax = r.x[-1]
        y = r.eqlin.marginals
        cara = np.abs(y[0] * L[:, 0] + y[1] * L[:, 1] + y[2]) < 1e-7
        q = cp.Variable(int(cara.sum()))
        F, D = L[cara], d[cara]
        pr = cp.Problem(cp.Maximize(cp.sum(cp.entr(q)) + q @ np.log(D)),
                        [q >= 0, cp.sum(q) == 1, q @ F[:, 0] == -12 * k * vmax, q @ F[:, 1] == -3 * k * vmax])
        pr.solve(solver="CLARABEL")
        vf = 1 / 3 if k % 2 == 0 else 1 / 3 - C_A / k
        chk.check(abs(vmax - vf) < 1e-9 and abs(pr.value - S_formula(k)) < 1e-7,
                  f"diamantes k = {k}: vmax = {vmax:.10f} ({vf:.10f}); cara óptima "
                  + "; ".join(f"(o, w) = ({o:+.4f}, {w:+.0f}) x {dd}" for (o, w), dd in zip(F, D))
                  + f"; S* = {pr.value:.8f} = fórmula {S_formula(k):.8f}" + chk.status_note(pr.status))
        filas.append([k, "programa lineal", vmax, pr.value, S_formula(k), int(D.sum()), np.nan])
    mu = -2 - E_PENT
    for k in (3, 5, 7):
        So, Sw = cadena(k, (3 * k + 1) // 2)
        H = (So + mu * Sw).tocsc()
        # Lanczos con pocos vectores pierde copias de un autovalor degenerado (k = 5 dio 8 en vez de 10): denso hasta
        # 7000 y, por encima, un subespacio de Krylov amplio.
        nev = 2 * k + 12
        w, V = (np.linalg.eigh(H.toarray()) if H.shape[0] < 7000
                else spla.eigsh(H, k=nev, which="SA", tol=1e-12, ncv=4 * nev))
        idx = np.argsort(w)[:nev]
        deg = int((np.abs(w[idx] - w[idx][0]) < 1e-8).sum())
        G = V[:, idx[:deg]]
        # Sw en el fundamental: separa los dos defectos (Sw conmuta con H)
        ww = np.linalg.eigvalsh(G.T @ (Sw @ G))
        n_pent = int((np.abs(ww - (2 - k)) < 1e-6).sum())
        n_libre = int((np.abs(ww - (-2 - k)) < 1e-6).sum())
        chk.check(deg == 2 * k and n_pent == k and n_libre == k and w[idx][deg] - w[idx][0] > 1,
                  f"diamantes k = {k}: fundamental de So + {mu:.4f} Sw en S^z = 1/2 con degeneración {deg} = 2k: "
                  f"{n_pent} pentámeros (w = {2 - k}) y {n_libre} espines libres (w = {-2 - k}); gap {w[idx][deg] - w[idx][0]:.4f}")
        filas.append([k, "fundamental", np.nan, S_formula(k), S_formula(k), 4 * k, w[idx][deg] - w[idx][0]])
    for k in (2, 4, 6):
        So, Sw = cadena(k, 3 * k // 2)
        H = (0.185 * So + (1 - 4 * 0.185) * Sw).tocsc()
        w = np.linalg.eigvalsh(H.toarray()) if H.shape[0] < 3000 else np.sort(spla.eigsh(H, k=6, which="SA", tol=1e-12)[0])
        deg = int((np.abs(w - w[0]) < 1e-8).sum())
        chk.check(abs(w[0] + k) < 1e-9 and deg == 2,
                  f"diamantes k = {k}: lambda_min(H(0,185)) = {w[0]:.10f} = -k, degeneración {deg} en S^z = 0 (las dos "
                  f"teselas), gap {w[deg] - w[0]:.4f}")
        filas.append([k, "teselas", 1 / 3, np.log(2), np.log(2), 2, w[deg] - w[0]])
    write_csv("regimen_degenerado_diamantes.csv", ("k", "metodo", "vmax", "S_ast", "S_formula", "dim_soporte", "gap"), filas)


def aproximacion(chk):
    filas = []
    for k in (3, 4):
        L, d = espectro_conjunto(k)
        lnd = np.log(d)
        vmax = 1 / 3 if k % 2 == 0 else 1 / 3 - C_A / k
        Sst = S_formula(k)
        lam = np.zeros(2)
        res_k = {}
        for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            tgt = np.array([-12 * k, -3 * k]) * vmax * (1 - eps)
            for _ in range(200):
                zz = lnd - L @ lam
                lz = logsumexp(zz)
                q = np.exp(zz - lz)
                m = q @ L
                r = m - tgt
                if np.abs(r).max() < 1e-12:
                    break
                C = (L * q[:, None]).T @ L - np.outer(m, m)
                paso = np.linalg.solve(C, r)
                a = 1.0
                while a > 1e-10:
                    zn = lnd - L @ (lam + a * paso)
                    if np.abs(np.exp(zn - logsumexp(zn)) @ L - tgt).max() < np.abs(r).max():
                        break
                    a /= 2
                lam = lam + a * paso
            S = lz + lam @ m
            beta = 3 * k * (4 * lam[0] + lam[1])
            ak = lam[0] / (4 * lam[0] + lam[1])
            res_k[eps] = (beta, S - Sst, ak)
            filas.append([k, eps, beta, beta / np.log(1 / eps), beta * np.sqrt(eps), ak, S, S - Sst, np.abs(r).max()])
            chk.info(f"diamantes k = {k}, eps = {eps:.0e}: beta = {beta:8.3f}, beta/ln(1/eps) = {beta / np.log(1 / eps):.4f}, "
                     f"a = {ak:.6f}, S - S* = {S - Sst:.4e}, residuo {np.abs(r).max():.0e}")
        b = [res_k[e][0] for e in (1e-4, 1e-5, 1e-6)]
        ds = [res_k[e][1] for e in (1e-4, 1e-5, 1e-6)]
        chk.check(b[0] * 1e-2 > b[1] * 10**-2.5 > b[2] * 1e-3 and all(5 < ds[i] / ds[i + 1] < 10.5 for i in range(2))
                  and max(abs(z - b[0] / np.log(1e4)) for z in (b[1] / np.log(1e5), b[2] / np.log(1e6))) < 0.2 * b[0] / np.log(1e4),
                  f"diamantes k = {k}: beta sqrt(eps) baja ({b[0] * 1e-2:.3f}, {b[1] * 10**-2.5:.3f}, {b[2] * 1e-3:.3f}) y "
                  f"beta/ln(1/eps) se estabiliza ({b[0] / np.log(1e4):.2f}, {b[1] / np.log(1e5):.2f}, {b[2] / np.log(1e6):.2f}); "
                  f"S - S* cae {ds[0] / ds[1]:.2f} y {ds[1] / ds[2]:.2f} veces por década: fuga de Boltzmann")
        if k == 4:
            So, Sw = (A.toarray() for A in cadena(4, 6))

            def gap(a):
                w = np.linalg.eigvalsh(a * So + (1 - 4 * a) * Sw)
                return w[2] - w[0]

            r = minimize_scalar(lambda a: -gap(a), bounds=(0.17, 0.2), method="bounded", options={"xatol": 1e-12})
            Dn = gap(r.x) / 12
            a_me = [res_k[e][2] for e in (1e-4, 1e-5, 1e-6)]
            chk.check(a_me[0] > a_me[1] > a_me[2] > r.x and a_me[2] - r.x < 1e-3 and b[2] / np.log(1e6) > 1 / Dn,
                      f"diamantes k = 4: la máxima entropía usa a = {a_me[0]:.6f}, {a_me[1]:.6f}, {a_me[2]:.6f} "
                      f"(1e-4..1e-6), bajando hacia el testigo de mayor gap, a = {r.x:.6f} (gap normalizado {Dn:.6f}, "
                      f"1/gap = {1 / Dn:.3f}, frente a beta/ln(1/eps) = {b[2] / np.log(1e6):.3f})")
            RESUMEN.extend([["a_gap_4", r.x], ["gap_norm_4", Dn]])
    write_csv("regimen_degenerado_aproximacion.csv", ("k", "eps", "beta", "beta_sobre_ln", "beta_sqrt_eps", "a", "S",
                                                      "S_menos_Sast", "residuo"), filas)


def main():
    chk = Checker("regimen_degenerado")
    fisher_prisma(chk, *prisma(chk))
    diamantes(chk)
    aproximacion(chk)
    write_csv("regimen_degenerado_resumen.csv", ("magnitud", "valor"), RESUMEN)
    chk.finish()


if __name__ == "__main__":
    main()

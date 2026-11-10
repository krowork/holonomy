"""Curvatura de la entropía de máxima entropía frente al flujo y métrica de Kubo--Mori (información de Fisher cuántica).

El estado de máxima entropía con marginales de Werner rotadas es rho = exp(sum_k x_k O_k)/Z, una familia exponencial:
m_k = <O_k> = d_k ln Z y g = Hess ln Z es la métrica de Kubo--Mori. S = ln Z - x.m es la transformada de Legendre, de modo
que dS/dm = -x y d2S/dm2 = -g^{-1}. A lo largo de un camino de flujo m(t), con phi_e(t) = t u_e,
    S'' = -m'^T g^{-1} m' - x.m'',
con m'_{B_e} = 2 v u_e y m''_{A_e} = -2 v u_e^2 en t = 0. El primer término (Fisher) es <= 0; el segundo, el de los
multiplicadores, compite con él, y de ahí que el signo de Delta S_hol no esté fijado (sección 7 de esfera).

Implementación densa e independiente de `sphere_entropia.Gibbs`: escalera periódica 2 x L (L = 3, 4), tres operadores
por arista, sin reducción por simetría. Camino: fase +t/2 en cada arista de la pata inferior y -t/2 en la superior
(plaqueta t, patas +-L t/2), que es `sphere_entropia_holonomia.escalera(L, F)` con psi = 0 y t = F/L.

Comprobaciones:
  1. S'' por diferencias finitas de cinco puntos frente a la fórmula, en v = 0,1, 0,2, 0,3 y en una dirección al azar;
     control: en una dirección pura de gauge (u_e = g_j - g_i) las dos dan 0.
  2. 1/2 S'' t^2 frente a Delta S_hol de data/sphere_entropia_holonomia_escaleras.csv con F = pi/8 (difiere en O(t^4)).
  3. Primera ley del entrelazamiento: con K_0 = -ln rho_0 (local, suma de términos de arista), Delta S = Delta<K_0>
     - S(rho_t || rho_0) exactamente, en F = pi/8 a 2 pi. A primer orden, delta S = delta<K_0> = 0 (multiplicadores de B
     nulos); a segundo, Delta<K_0> es el término de multiplicadores y S_rel >= 0 el de Fisher.
  4. Cerca de vmax (dual por SDP), x.O se alinea con el testigo óptimo W* y el término de multiplicadores por unidad
     de ||x.O|| tiende a -v d2N_t(W*)/dt2 / ||W*||. En 2 x 3 (patas triangulares, vmax = 1/3) el testigo es degenerado y
     la alineación es lenta; el límite del término de multiplicadores solo se comprueba en 2 x 4 (el cubo).
  5. El testigo como hamiltoniano modular a T = 0: con x.O = beta W_v y N(W_v) = 1, K_G = beta (lambda_max - W_v) + cte
     y beta = -dS/dv (envolvente, comprobada por diferencias finitas). En 2 x 4 (máximo único, gap 0,091),
     beta gap - ln(1/eps) tiende a una constante, S -> 0, la Fisher de Kubo--Mori diverge como beta y la SLD queda
     finita. En 2 x 3 (máximo 16 veces degenerado), beta ~ eps^{-1/2}, queda entropía residual y las dos Fisher son
     finitas. Las dos Fisher se calculan por Daleckii--Krein.
No se comprueba la relación con la susceptibilidad de fidelidad del fundamental del testigo.

Escribe data/fisher_kubo_mori{,_primera_ley,_vmax,_T0}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/fisher_kubo_mori.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.special import logsumexp

from holonomy.paulis import pauli_string
from holonomy.report import read_csv, write_csv

VS = (0.1, 0.2, 0.3)
H = 2e-3
EPSS = (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3)
EPSS_T0 = (0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4)


def escalera(L):
    """Aristas orientadas i -> j y fase por unidad de t: patas +-1/2, peldaños 0."""
    edges, u = [], []
    for base, s in ((0, 0.5), (L, -0.5)):
        for i in range(L):
            edges.append((base + i, base + (i + 1) % L))
            u.append(s)
    for i in range(L):
        edges.append((i, L + i))
        u.append(0.0)
    return edges, np.array(u)


class MaxEnt:
    """Estado de máxima entropía denso con operadores A_e, B_e, C_e por arista, en ese orden."""

    def __init__(self, n, edges):
        ops = []
        for i, j in edges:
            ops.append(-(pauli_string(n, ((i, 0), (j, 0))) + pauli_string(n, ((i, 1), (j, 1)))))
            ops.append(pauli_string(n, ((i, 0), (j, 1))) - pauli_string(n, ((i, 1), (j, 0))))
            ops.append(-pauli_string(n, ((i, 2), (j, 2))))
        self.O = np.array([np.asarray(o.todense() if hasattr(o, "todense") else o, dtype=complex) for o in ops])

    @staticmethod
    def objetivo(v, phis):
        return v * np.array([[2 * np.cos(f), 2 * np.sin(f), 1.0] for f in phis]).ravel()

    def gibbs(self, x):
        """Medias, métrica de Kubo--Mori y entropía."""
        E, U = eigh(np.einsum("k,kab->ab", x, self.O))
        p = np.exp(E - logsumexp(E))
        # Con np.einsum en vez de matmul, U^dag O U es O(N^4) por operador y no termina en 2 x 4.
        M = U.conj().T[None] @ self.O @ U[None]
        m = np.einsum("i,kii->k", p, M).real
        dE = E[:, None] - E[None, :]
        dp = p[:, None] - p[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(dE) > 1e-10, dp / np.where(dE == 0, 1, dE), (p[:, None] + p[None, :]) / 2)
        K = len(M)
        g = ((M * w[None]).reshape(K, -1) @ M.transpose(0, 2, 1).reshape(K, -1).T).real - np.outer(m, m)
        S = -(p[p > 0] * np.log(p[p > 0])).sum()
        return m, g, S

    def ln_rho(self, x):
        """ln rho = x.O - ln Z (con la identidad)."""
        H = np.einsum("k,kab->ab", x, self.O)
        return H - logsumexp(eigh(H, eigvals_only=True)) * np.eye(len(H))

    def rho(self, x):
        E, U = eigh(np.einsum("k,kab->ab", x, self.O))
        p = np.exp(E - logsumexp(E))
        return (U * p) @ U.conj().T

    def fisher(self, x, dx):
        """Fisher de Kubo--Mori y SLD de la familia rho(x + s dx) en s = 0, por Daleckii--Krein (sin diferencias
        finitas, que con pesos de 1e-30 dan ruido en la SLD)."""
        E, U = eigh(np.einsum("k,kab->ab", x, self.O))
        p = np.exp(E - logsumexp(E))
        dH = U.conj().T @ np.einsum("k,kab->ab", dx, self.O) @ U
        dE = E[:, None] - E[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(dE) > 1e-10, (p[:, None] - p[None, :]) / np.where(dE == 0, 1, dE),
                         (p[:, None] + p[None, :]) / 2)
        drho = w * dH - np.diag(p) * np.trace(np.diag(p) @ dH).real
        bkm = (w * np.abs(dH)**2).sum().real - np.trace(np.diag(p) @ dH).real**2
        s = p[:, None] + p[None, :]
        sld = (2 * np.abs(drho)**2 / np.where(s > 0, s, np.inf)).sum()
        return float(bkm), float(sld)

    def resuelve(self, t, x0=None, tol=1e-12):
        """Newton sobre las medias, amortiguado por el residuo (una búsqueda de línea sobre ln Z - x.t se atasca en el
        ruido cerca de la tolerancia)."""
        x = np.zeros(len(self.O)) if x0 is None else x0.copy()
        m, g, S = self.gibbs(x)
        for _ in range(60):
            r = m - t
            if np.abs(r).max() < tol:
                break
            paso = np.linalg.solve(g, -r)
            a = 1.0
            while a > 1e-6:
                mn, gn, Sn = self.gibbs(x + a * paso)
                if np.abs(mn - t).max() < np.abs(r).max():
                    break
                a /= 2
            x, m, g, S = x + a * paso, mn, gn, Sn
        return x, g, S, float(np.abs(m - t).max())


def derivadas(v, u):
    dm = v * np.array([[0.0, 2 * ue, 0.0] for ue in u]).ravel()
    d2m = v * np.array([[-2 * ue**2, 0.0, 0.0] for ue in u]).ravel()
    return dm, d2m


def curvatura(me, v, u):
    """S'' en t = 0 por diferencias finitas de cinco puntos y por la fórmula; S0 y residuo."""
    x0, g, S0, res = me.resuelve(me.objetivo(v, 0 * u))
    Ss = {}
    for k in (-2, -1, 1, 2):
        _, _, S, r = me.resuelve(me.objetivo(v, k * H * u), x0)
        res = max(res, r)
        Ss[k] = S
    fd = (-Ss[2] + 16 * Ss[1] - 30 * S0 + 16 * Ss[-1] - Ss[-2]) / (12 * H**2)
    dm, d2m = derivadas(v, u)
    fis, mul = -dm @ np.linalg.solve(g, dm), -x0 @ d2m
    return fd, fis, mul, S0, res


def formula_frente_a_fd(chk):
    filas = []
    rng = np.random.default_rng(1)
    previas = {(int(r["L"]), round(r["v"], 6)): r for r in read_csv("sphere_entropia_holonomia_escaleras.csv")
               if abs(r["F"] - np.pi / 8) < 1e-9}
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        for v in VS:
            fd, fis, mul, S0, res = curvatura(me, v, u)
            bkm = fis + mul
            rel = abs(fd - bkm) / abs(bkm)
            chk.check(res < 1e-10 and rel < 1e-5,
                      f"2x{L}, v = {v}: S'' = {fd:+.10e} (dif. finitas), {bkm:+.10e} (Kubo-Mori), dif. rel. {rel:.1e}; "
                      f"Fisher {fis:+.4e}, multiplicadores {mul:+.4e}, residuo {res:.1e}")
            # Contraste con el Gibbs por órbitas del repo: la misma S sin flujo y Delta S_hol a F = pi/8.
            r = previas[L, v]
            t = np.pi / 8 / L
            pred = bkm * t**2 / 2
            chk.check(abs(S0 - r["S_trivial"]) < 1e-8 and abs(pred - r["dS"]) / abs(r["dS"]) < 1e-2,
                      f"2x{L}, v = {v}: S0 = {S0:.10f} (CSV {r['S_trivial']:.10f}); 1/2 S'' t^2 = {pred:+.5e} frente a "
                      f"Delta S_hol = {r['dS']:+.5e} con F = pi/8, dif. rel. {abs(pred - r['dS']) / abs(r['dS']):.1e}")
            filas.append([L, 2 * L, "plaquetas", v, S0, fd, bkm, fis, mul, pred, r["dS"], res])
        gg = rng.normal(size=2 * L)
        ug = np.array([gg[j] - gg[i] for i, j in edges])
        fd, fis, mul, S0, res = curvatura(me, 0.2, ug)
        chk.check(abs(fd) < 1e-6 and abs(fis + mul) < 1e-10,
                  f"2x{L}, gauge, v = 0,2: S'' = {fd:+.1e} (dif. finitas), {fis + mul:+.1e} (Kubo-Mori)")
        filas.append([L, 2 * L, "gauge", 0.2, S0, fd, fis + mul, fis, mul, np.nan, np.nan, res])
        ur = rng.normal(size=len(edges))
        fd, fis, mul, S0, res = curvatura(me, 0.2, ur)
        rel = abs(fd - fis - mul) / abs(fis + mul)
        chk.check(rel < 1e-5, f"2x{L}, dirección al azar, v = 0,2: S'' = {fd:+.10e}, {fis + mul:+.10e}, dif. rel. {rel:.1e}")
        filas.append([L, 2 * L, "azar", 0.2, S0, fd, fis + mul, fis, mul, np.nan, np.nan, res])
    write_csv("fisher_kubo_mori.csv", ("L", "n", "direccion", "v", "S0", "S2_fd", "S2_kubo_mori", "fisher",
                                       "multiplicadores", "pred_dS_F_pi8", "dS_csv", "residuo"), filas)


def primera_ley(chk):
    """Delta S = Delta<K_0> - S(rho_t || rho_0), con K_0 = -ln rho_0 = -x_0.O + ln Z local: es exacta, y su desarrollo da
    la primera ley (delta S = -x_0.m' = 0, porque los multiplicadores de B son nulos sin flujo), Delta<K_0> ~ -t^2 x_0.m''/2
    (multiplicadores) y S_rel ~ t^2 m'^T g^{-1} m'/2 (Fisher)."""
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        v = 0.2
        x0, g, S0, _ = me.resuelve(me.objetivo(v, 0 * u))
        chk.check(np.abs(x0[1::3]).max() == 0,
                  f"2x{L}, v = 0,2: multiplicadores de B nulos sin flujo ({np.abs(x0[1::3]).max():.1e}): delta S = -x_0.m' = 0")
        lr0 = me.ln_rho(x0)
        dm, d2m = derivadas(v, u)
        for F in (np.pi / 8, np.pi / 2, np.pi, 2 * np.pi):
            t = F / L
            xt, _, St, res = me.resuelve(me.objetivo(v, t * u), x0)
            lrt = me.ln_rho(xt)
            rt = me.rho(xt)
            r0 = me.rho(x0)
            dK = -np.trace((rt - r0) @ lr0).real
            Srel = np.trace(rt @ (lrt - lr0)).real
            dK2, Srel2 = -x0 @ d2m * t**2 / 2, dm @ np.linalg.solve(g, dm) * t**2 / 2
            ok = res < 1e-10 and abs(St - S0 - (dK - Srel)) < 1e-10 and Srel > 0
            if F < 0.5:
                ok = ok and abs(dK - dK2) / dK2 < 1e-3 and abs(Srel - Srel2) / Srel2 < 1e-2
            chk.check(ok, f"2x{L}, F = {F / np.pi:.3f} pi: Delta S = {St - S0:+.8e}, Delta<K> - S_rel = {dK - Srel:+.8e}; "
                          f"Delta<K> = {dK:+.4e} (2.º orden {dK2:+.4e}), S_rel = {Srel:.4e} (Fisher {Srel2:.4e})")
            filas.append([L, v, F, t, St - S0, dK, Srel, dK2, Srel2, res])
    write_csv("fisher_kubo_mori_primera_ley.csv",
              ("L", "v", "F", "t", "dS", "dK", "S_rel", "dK_2orden", "S_rel_fisher", "residuo"), filas)


def testigo_optimo(me, u):
    """vmax = min lambda_max(y.O) con N(y.O) = 1, por SDP (parte real e imaginaria en bloques)."""
    K, N = len(me.O), me.O.shape[1]
    t1 = me.objetivo(1.0, 0 * u)
    y, lam = cp.Variable(K), cp.Variable()
    Wr = sum(y[k] * me.O[k].real for k in range(K))
    Wi = sum(y[k] * me.O[k].imag for k in range(K))
    prob = cp.Problem(cp.Minimize(lam), [lam * np.eye(2 * N) - cp.bmat([[Wr, -Wi], [Wi, Wr]]) >> 0, y @ t1 == 1])
    prob.solve(solver="CLARABEL")
    return float(lam.value), y.value, prob.status


def limite_T0(chk):
    """El testigo como hamiltoniano modular a T = 0. Con x.O = beta W_v y N(W_v) = 1, rho_G = exp(beta W_v)/Z,
    K_G = beta (lambda_max(W_v) - W_v) + cte y beta = x.t1 = -dS/dv (envolvente). Cuando v -> vmax, beta -> oo.
    Con gap (2 x 4): beta Delta - ln(1/eps) tiende a una constante, S -> 0, Fisher de Kubo--Mori ~ beta y SLD finita.
    Degenerado (2 x 3, autovalor máximo 16 veces): beta ~ eps^{-1/2}, entropía residual y las dos Fisher finitas."""
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        vmax, ys, status = testigo_optimo(me, u)
        ev, V = np.linalg.eigh(np.einsum("k,kab->ab", ys, me.O))
        deg = int((ev > ev[-1] - 1e-6).sum())
        gap = ev[-1] - ev[-1 - deg]
        P = V[:, -deg:] @ V[:, -deg:].conj().T
        t1 = me.objetivo(1.0, 0 * u)
        chk.info(f"2x{L}: vmax = {vmax:.8f}, autovalor máximo de W* {deg} veces degenerado, gap {gap:.4f}"
                 + chk.status_note(status))
        x = np.zeros(len(me.O))
        res_ = {}
        for eps in EPSS_T0:
            v = vmax * (1 - eps)
            x, g, S, res = me.resuelve(me.objetivo(v, 0 * u), x)
            beta = float(x @ t1)
            dm, _ = derivadas(v, u)
            bkm, sld = me.fisher(x, np.linalg.solve(g, dm))
            peso = np.trace(P @ me.rho(x)).real
            chk.info(f"2x{L}, 1 - v/vmax = {eps:.0e}: beta = {beta:8.3f}, beta gap - ln(1/eps) = {beta * gap - np.log(1 / eps):+8.3f}, "
                     f"peso en el autoespacio máximo {peso:.6f}, S = {S:.4f}, Fisher Kubo-Mori {bkm:9.3f}, SLD {sld:.4f}, "
                     f"residuo {res:.0e}")
            res_[eps] = (beta, S, bkm, sld, x, res)
            filas.append([L, vmax, deg, gap, eps, v, beta, beta * gap - np.log(1 / eps), peso, S, bkm, sld, res])
        # beta = -dS/dv, por diferencias finitas centradas en eps = 1e-2
        b, _, _, _, xb, _ = res_[1e-2]
        dv = 1e-5 * vmax
        Sp = me.resuelve(me.objetivo(vmax * (1 - 1e-2) + dv, 0 * u), xb)[2]
        Sm = me.resuelve(me.objetivo(vmax * (1 - 1e-2) - dv, 0 * u), xb)[2]
        dSdv = (Sp - Sm) / (2 * dv)
        chk.check(abs(b + dSdv) / b < 1e-5, f"2x{L}: beta = x.t1 = {b:.6f} = -dS/dv = {-dSdv:.6f} (envolvente)")
        # comprobación cruzada: la Fisher de Kubo--Mori de Daleckii--Krein es la m'^T g^{-1} m' de antes
        x1, g1, *_ = me.resuelve(me.objetivo(0.2, 0 * u))
        dm, _ = derivadas(0.2, u)
        chk.check(abs(me.fisher(x1, np.linalg.solve(g1, dm))[0] - dm @ np.linalg.solve(g1, dm)) < 1e-10,
                  f"2x{L}, v = 0,2: Fisher de Kubo-Mori por Daleckii-Krein = m'^T g^-1 m'")
        bs = [res_[e][0] for e in (1e-3, 3e-4, 1e-4)]
        Sf, bkmf, sldf = res_[1e-4][1:4]
        if deg == 1:
            c = [bb * gap - np.log(1 / e) for bb, e in zip(bs, (1e-3, 3e-4, 1e-4))]
            chk.check(max(c) - min(c) < 0.01, f"2x{L}: beta gap - ln(1/eps) = {c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f} "
                                              "en 1e-3, 3e-4, 1e-4: Boltzmann a través del gap")
            chk.check(Sf < 0.01, f"2x{L}: S = {Sf:.4f} en 1e-4: el estado tiende a puro")
            r = [res_[e][2] / res_[e][0] for e in (1e-3, 3e-4, 1e-4)]
            s = [res_[e][3] for e in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)]
            chk.check(bkmf > 1e2 and max(r) - min(r) < 0.1 * max(r),
                      f"2x{L}: Fisher de Kubo-Mori / beta = {r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}: diverge como beta")
            chk.check(max(s) - min(s) < 0.05 * max(s),
                      f"2x{L}: Fisher SLD = {', '.join(f'{z:.4f}' for z in s)} entre 1e-2 y 1e-4: finita")
        else:
            q = bs[2] / bs[0]
            chk.check(abs(q / np.sqrt(10) - 1) < 0.05, f"2x{L}: beta(1e-4)/beta(1e-3) = {q:.3f}, sqrt(10) = {np.sqrt(10):.3f}: "
                                                       "beta ~ eps^-1/2")
            chk.check(1.5 < Sf < np.log(deg), f"2x{L}: S = {Sf:.4f} en 1e-4, entre 1,5 y ln {deg} = {np.log(deg):.4f}: "
                                              "entropía residual en el autoespacio degenerado")
            chk.check(bkmf < 1 and sldf < 1, f"2x{L}: Fisher de Kubo-Mori {bkmf:.4f} y SLD {sldf:.4f} en 1e-4: finitas")
    write_csv("fisher_kubo_mori_T0.csv", ("L", "vmax", "degeneracion", "gap", "eps", "v", "beta", "beta_gap_menos_ln",
                                          "peso_max", "S", "fisher_kubo_mori", "fisher_sld", "residuo"), filas)


def cerca_de_vmax(chk):
    filas = []
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        K = len(me.O)
        vmax, ys, status = testigo_optimo(me, u)
        Wst = np.einsum("k,kab->ab", ys, me.O)
        ev = np.linalg.eigvalsh(Wst)
        chk.check(abs(ev[-1] - vmax) < 1e-6,
                  f"2x{L}: vmax = {vmax:.8f} = lambda_max(W*) = {ev[-1]:.8f}, gap del testigo {ev[-1] - ev[-2]:.2e}"
                  + chk.status_note(status))
        if L == 3:
            chk.check(abs(vmax - 1 / 3) < 1e-7, f"2x3: vmax = 1/3 (patas triangulares), {vmax - 1 / 3:+.1e}")
        x = np.zeros(K)
        for eps in EPSS:
            v = vmax * (1 - eps)
            x, g, S, res = me.resuelve(me.objetivo(v, 0 * u), x)
            X = np.einsum("k,kab->ab", x, me.O)
            cos = np.vdot(X, Wst).real / np.linalg.norm(X) / np.linalg.norm(Wst)
            dm, d2m = derivadas(v, u)
            fis, mul = -dm @ np.linalg.solve(g, dm), -x @ d2m
            chk.info(f"2x{L}, 1 - v/vmax = {eps:.0e}: ||x.O|| = {np.linalg.norm(X):7.2f}, cos(x.O, W*) = {cos:.6f}, "
                     f"S = {S:.4f}, Fisher {fis:+.4e}, multiplicadores {mul:+.4e}, residuo {res:.1e}")
            filas.append([L, vmax, eps, v, np.linalg.norm(X), cos, S, fis, mul, fis + mul, res])
        chk.check(res < 1e-10 and cos > 0.998, f"2x{L}: en 1 - v/vmax = 1e-3, x.O se alinea con W*: cos = {cos:.6f}")
        if L == 4:
            d2N = -2 * sum(ys[3 * i] * u[i]**2 for i in range(len(u)))
            pred, med = -vmax * d2N / np.linalg.norm(Wst), mul / np.linalg.norm(X)
            chk.check(abs(med - pred) / abs(pred) < 1e-2,
                      f"2x4: multiplicadores / ||x.O|| = {med:+.5e} frente a -v d2N(W*)/dt2 / ||W*|| = {pred:+.5e}")
    write_csv("fisher_kubo_mori_vmax.csv", ("L", "vmax", "eps", "v", "norma_xO", "cos_W", "S", "fisher",
                                            "multiplicadores", "S2", "residuo"), filas)


def main():
    chk = Checker("fisher_kubo_mori")
    formula_frente_a_fd(chk)
    primera_ley(chk)
    cerca_de_vmax(chk)
    limite_T0(chk)
    chk.finish()


if __name__ == "__main__":
    main()

"""Anillos térmicos: la máxima entropía del anillo es el Heisenberg térmico, y la holonomía decae con la longitud de
correlación.

Sin flujo, el anillo de n sitios es transitivo en aristas e invariante por SU(2), así que x.O = beta W simétrico
(Prop. prop:simetrico): rho_G = exp(-b H)/Z con H = sum sigma_i.sigma_{i+1} = 4 sum S_i.S_{i+1}, el antiferromagneto de
Heisenberg a temperatura física 1/b, b = beta/(3n); v = -<sigma_0.sigma_1>/3 y S_G es su entropía térmica.

Cota de Kohn armónica: con y' el giro de x.O (K' = y'.O), d = -x.m'' = y'.m' (término de los multiplicadores),
P = m'^T g^-1 m' (Fisher) y Q = y'^T g y', Cauchy--Schwarz da P Q >= d^2, y con R'' = Q - d (curvatura del ln Z
girado) y S'' = d - P:
    S''(0) <= d R''/(d + R''),
con igualdad si g y' es proporcional a m'. En el anillo con flujo uniforme lo es (traslaciones), así que
1/S'' = 1/R'' + 1/d, con d = 2 v b/n y R'' la curvatura frente al flujo total de ln tr exp(-b H(Theta)).

Signo (Marshall): en un grafo bipartito con multiplicadores antiferromagnéticos, girar la subred B con R_z(pi) hace
que -H(Theta) tenga elementos fuera de la diagonal de módulo igual al de -H(0) >= 0; así |tr e^{-bH(Theta)}| <=
tr e^{-bH(0)} y Delta S_hol <= ln Z(Theta) - ln Z(0) <= 0 para todo flujo. En los anillos impares el punto sin
frustración es Theta = pi y S'' > 0.

Comprobaciones:
  1. Anillos n = 3..8, máxima entropía densa (fisher_kubo_mori.MaxEnt) en v = 0,5 y 0,8 vmax(n): x.O uniforme y
     SU(2); b y S_G frente al espectro de Heisenberg (holonomy.anillo_k); S'' = d R''/(d + R''); Delta S_hol(pi/2, pi)
     <= cota; cota <= 0 con n par; Delta S_hol > 0 con n impar.
  2. Escaleras 2 x 3 y 2 x 4: igualdad de la cota armónica en el camino de plaquetas y desigualdad estricta en una
     dirección al azar.
  3. Anillos n = 4..NMAX por sectores (M, k), en v fijo: S''(0) = d R''/(d + R''), signo (-1)^n, decaimiento
     exponencial |S''| ~ e^{-n/xi(v)} por pares e impares, y xi(v) creciente con v.

Escribe data/anillo_termico{,_escaleras,_escala,_xi}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/anillo_termico.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import logsumexp

from fisher_kubo_mori import MaxEnt, escalera, derivadas, girado
from holonomy import anillo_k, xxz
from holonomy.report import write_csv

H = 2e-3
NMAX = 18
VS_ESCALA = (0.2, 0.3, 0.4, 0.5, 0.55)
HT = 0.2  # paso en el flujo total para R'' por sectores (Richardson con 2 HT)


def ln_z(K):
    return float(logsumexp(eigh(K, eigvals_only=True)))


def r2_girado(me, edges, u, x):
    """R''(0) del ln Z de x.O girado, por diferencias de cinco puntos."""
    R = {k: ln_z(girado(me, edges, u, x, k * H)) for k in (-2, -1, 0, 1, 2)}
    return (-R[2] + 16 * R[1] - 30 * R[0] + 16 * R[-1] - R[-2]) / (12 * H**2), R[0]


class Termico:
    """Heisenberg térmico del anillo a partir del espectro por sectores: energías de sum S.S (multiplicidades)."""

    def __init__(self, n, Theta):
        self.n = n
        es = anillo_k.espectro(n, Theta)
        self.E = np.concatenate([e for _, _, e in es])
        self.w = np.concatenate([np.full(len(e), np.log(mu)) for _, mu, e in es])

    def ln_z(self, b):
        """ln tr exp(-b sum sigma.sigma) = ln sum exp(-4 b E)."""
        return float(logsumexp(-4 * b * self.E + self.w))

    def v(self, b):
        p = np.exp(-4 * b * self.E + self.w - self.ln_z(b))
        return float(-4 * (p @ self.E) / (3 * self.n))

    def S(self, b):
        return self.ln_z(b) - 3 * self.n * b * self.v(b)

    def b_de_v(self, v):
        return brentq(lambda b: self.v(b) - v, 1e-9, 200.0, xtol=1e-15, rtol=1e-15)


def reduccion(chk):
    filas = []
    for n in range(3, 9):
        edges = [(i, (i + 1) % n) for i in range(n)]
        u = np.full(n, 1.0 / n)  # flujo total Theta
        me = MaxEnt(n, edges)
        vmax = 4 * xxz.eps(n, 1.0, 0.0) / 3
        th = Termico(n, 0.0)
        for f in (0.5, 0.8):
            v = f * vmax
            x, g, S, res = me.resuelve(me.objetivo(v, 0 * u))
            xa, xb, xc = x[0::3], x[1::3], x[2::3]
            b = th.b_de_v(v)
            chk.check(np.abs(xa - xc).max() < 1e-12 and np.ptp(xa) < 1e-12 and np.abs(xb).max() < 1e-12
                      and abs(xa[0] - b) < 1e-9 and abs(S - th.S(b)) < 1e-9,
                      f"anillo {n}, v = {f} vmax: x.O = b sum (A + C) con b = {xa[0]:.10f} = {b:.10f} (Heisenberg térmico); "
                      f"S_G = {S:.10f} = entropía térmica {th.S(b):.10f}")
            dm, d2m = derivadas(v, u)
            P, d = dm @ np.linalg.solve(g, dm), -x @ d2m
            s2 = d - P
            R2, lz0 = r2_girado(me, edges, u, x)
            harm = d * R2 / (d + R2)
            chk.check(abs(s2 - harm) < 1e-5 * abs(s2) and abs(d - 2 * v * b / n) < 1e-10,
                      f"anillo {n}, v = {f} vmax: S'' = {s2:+.8e} = d R''/(d + R'') = {harm:+.8e}, d = 2 v b/n = {d:.6e}, "
                      f"R'' = {R2:+.4e}")
            dS = {}
            xx = x
            for F in (np.pi / 2, np.pi):
                cota = ln_z(girado(me, edges, u, x, F)) - lz0
                try:
                    xx, _, SF, r = me.resuelve(me.objetivo(v, F * u), xx)
                    ok = r < 1e-9
                except np.linalg.LinAlgError:
                    ok = False
                if not ok:
                    chk.check(n % 2 == 0, f"anillo {n}, v = {f} vmax, Theta = {F:.4f}: no factible (vmax baja con el flujo)")
                    filas.append([n, f, v, b, S, s2, R2, d, F, np.nan, cota])
                    continue
                dS[F] = SF - S
                signo = cota <= 0 if n % 2 == 0 else SF - S > 0
                chk.check(SF - S <= cota + 1e-12 and signo,
                          f"anillo {n}, v = {f} vmax, Theta = {F:.4f}: Delta S_hol = {SF - S:+.4e} <= cota {cota:+.4e}"
                          + (" <= 0 (par, Marshall)" if n % 2 == 0 else "; > 0 (impar)"))
                filas.append([n, f, v, b, S, s2, R2, d, F, SF - S, cota])
            if len(dS) == 2:
                chk.info(f"anillo {n}, v = {f} vmax: Delta S(pi)/Delta S(pi/2) = {dS[np.pi] / dS[np.pi / 2]:.5f} "
                         "(un solo armónico: 2)")
    write_csv("anillo_termico.csv", ("n", "v_frac", "v", "b", "S_G", "S2", "R2", "d", "Theta", "dS", "cota"), filas)


def escaleras(chk):
    filas = []
    rng = np.random.default_rng(3)
    for L in (3, 4):
        edges, u = escalera(L)
        me = MaxEnt(2 * L, edges)
        for nombre, uu in (("plaquetas", u), ("azar", rng.normal(size=len(edges)))):
            v = 0.2
            x, g, S, _ = me.resuelve(me.objetivo(v, 0 * uu))
            dm, d2m = derivadas(v, uu)
            P, d = dm @ np.linalg.solve(g, dm), -x @ d2m
            R2, _ = r2_girado(me, edges, uu, x)
            Q = R2 + d
            harm = d * R2 / (d + R2)
            cs = P * Q / d**2
            if nombre == "plaquetas":
                chk.check(abs(cs - 1) < 1e-5, f"2x{L}, plaquetas, v = 0,2: P Q/d^2 = {cs:.8f}: igualdad, "
                                              f"S'' = {d - P:+.6e} = d R''/(d + R'') = {harm:+.6e}")
            else:
                chk.check(cs > 1 + 1e-3 and d - P < harm, f"2x{L}, dirección al azar, v = 0,2: P Q/d^2 = {cs:.5f} > 1, "
                                                         f"S'' = {d - P:+.6e} < d R''/(d + R'') = {harm:+.6e}")
            filas.append([L, nombre, v, d - P, R2, d, P, Q, cs])
    write_csv("anillo_termico_escaleras.csv", ("L", "direccion", "v", "S2", "R2", "d", "P", "Q", "PQ_d2"), filas)


def escala(chk):
    filas = []
    s2 = {}
    for n in range(4, NMAX + 1):
        th = {T: Termico(n, T) for T in (0.0, HT, 2 * HT)}
        vmax_n = -4 * th[0.0].E.min() / (3 * n)
        for v in VS_ESCALA:
            if v >= 0.97 * vmax_n:
                continue
            b = th[0.0].b_de_v(v)
            dR = {T: th[T].ln_z(b) - th[0.0].ln_z(b) for T in (HT, 2 * HT)}
            r1, r2 = 2 * dR[HT] / HT**2, 2 * dR[2 * HT] / (2 * HT)**2
            R2 = (4 * r1 - r2) / 3
            d = 2 * v * b / n
            S2 = d * R2 / (d + R2)
            s2[n, v] = S2
            filas.append([n, v, b, th[0.0].S(b), R2, d, S2, vmax_n])
            chk.info(f"anillo {n:2d}, v = {v}: b = {b:.6f}, S_G/n = {th[0.0].S(b) / n:.8f}, R'' = {R2:+.4e}, "
                     f"S'' = {S2:+.4e}")
    write_csv("anillo_termico_escala.csv", ("n", "v", "b", "S_G", "R2", "d", "S2", "vmax_n"), filas)
    xis = {}
    resumen = []
    for v in VS_ESCALA:
        ns = sorted(n for n, vv in s2 if vv == v)
        par = [n for n in ns if n % 2 == 0]
        imp = [n for n in ns if n % 2 == 1]
        chk.check(all(s2[n, v] < 0 for n in par) and all(s2[n, v] > 0 for n in imp),
                  f"v = {v}: S'' < 0 en los {len(par)} anillos pares y > 0 en los {len(imp)} impares")
        for nombre, grupo in (("pares", par), ("impares", imp)):
            # xi local entre n y n + 2; los que pasan del suelo de precisión (1e-11) no cuentan
            vals = [(n, 2 / np.log(abs(s2[n, v] / s2[n + 2, v]))) for n in grupo[:-1]
                    if n + 2 in grupo and abs(s2[n + 2, v]) > 1e-11]
            if len(vals) >= 3:
                xi = [z for _, z in vals]
                xis[v, nombre] = xi[-1]
                bn = [r[2] for r in filas if r[0] == vals[-1][0] + 2 and r[1] == v][0]
                resumen.append([v, nombre, xi[-1], vals[-1][0], bn, xi[-1] / bn])
                chk.info(f"v = {v}, {nombre}: xi = {xi[-1]:.4f}, b = {bn:.4f}, xi/b = {xi[-1] / bn:.4f} "
                         "(v_s/(pi T) = 2b)")
                chk.check(max(xi[-3:]) - min(xi[-3:]) < 0.1 * xi[-1],
                          f"v = {v}, {nombre}: xi local {', '.join(f'{z:.3f}' for z in xi)} (n = {vals[0][0]}..{vals[-1][0]}): "
                          "decaimiento exponencial")
    write_csv("anillo_termico_xi.csv", ("v", "paridad", "xi", "n_desde", "b", "xi_sobre_b"), resumen)
    for v in VS_ESCALA:
        if (v, "pares") in xis and (v, "impares") in xis:
            # Con v = 0,55 (xi ~ 2,2, n <= 18) b(n) aún deriva un 0,3 % entre pares e impares: tolerancia del 1 %.
            chk.check(abs(xis[v, "pares"] / xis[v, "impares"] - 1) < 1e-2,
                      f"v = {v}: la misma xi en pares ({xis[v, 'pares']:.4f}) e impares ({xis[v, 'impares']:.4f}), "
                      f"dif. rel. {abs(xis[v, 'pares'] / xis[v, 'impares'] - 1):.1e}")
    for nombre in ("pares", "impares"):
        serie = [xis[v, nombre] for v in VS_ESCALA if (v, nombre) in xis]
        chk.check(all(np.diff(serie) > 0), f"{nombre}: xi crece con v: {', '.join(f'{z:.3f}' for z in serie)}")


def main():
    chk = Checker("anillo_termico")
    reduccion(chk)
    escaleras(chk)
    escala(chk)
    chk.finish()


if __name__ == "__main__":
    main()

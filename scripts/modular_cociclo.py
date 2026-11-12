"""Nota «Tiempo modular y holonomía», sección 5: cociclo de Connes entre la máxima entropía y el giro puro.

rho_T = e^{x.O}/Z es la máxima entropía con holonomía; rhobar = e^{xbar.O}/Zbar el giro puro (los multiplicadores de
flujo nulo girados), con la misma holonomía. Los dos giran juntos con el gauge, y u_s = rho_T^{is} rhobar^{-is} es
covariante. Con h = ln Zbar - ln Z_0 y x0 los multiplicadores sin flujo:
  1. Identidad exacta: Delta S_hol = h - S(rho_T || rhobar) (el defecto de la Proposición 15, Kohn a temperatura finita, del artículo II).
  2. Reparto exacto: S(rhobar) - S_0 = h - x0.grad h y -xbar.(m_T - mbar) = x0.grad h; a orden dominante,
     -(l - 1) h y l h (homogeneidad), y h ~ término de lazos 2 (-1)^{l+1} v^l sum_{|C|=l} (1 - cos theta_C).
  3. S(rho_T || rhobar), S(rhobar || rho_T) = O(v^{2l-2}), y Delta S/h - 1 ~ v^{l-2}.
  4. F(s) = tr(rhobar^{1-is} rho_T^{is}): F(0) = 1, F'(0) = -i S(rhobar || rho_T), y 1 - |F(s)| ~ s^2 S(rho_T || rhobar)
     a v pequeño (s = 0,25, 1, 4).
Anillos de 6 y 8 con Theta = pi y escalera periódica 2 x 4 con t = pi, pi/2, en v = 0,03-0,15 (y 0,6 vmax en la 2 x 4).

Escribe data/modular_cociclo{,_F}.csv.

Uso: (ulimit -v 8000000; OMP_NUM_THREADS=8 .venv/bin/python scripts/modular_cociclo.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh

from holonomy.report import write_csv
from modular_cota import (anillo, ciclos, escalera, girar, no_triviales, objetivo_escalera, ops_anillo, ops_clases,
                          simetrico)

VS = (0.03, 0.05, 0.07, 0.1, 0.15)
V_2x4 = 0.6 * 0.53556549


def estado(O, y):
    E, U = eigh(np.einsum("k,kab->ab", y, O))
    lnZ = np.log(np.exp(E - E.max()).sum()) + E.max()
    p = np.exp(E - lnZ)
    m = np.array([np.einsum("i,ij,ji->", p, U.conj().T, o @ U).real for o in O])
    return dict(lnr=E - lnZ, U=U, p=p, m=m, S=-(p * np.log(p)).sum(), lnZ=lnZ)


def potencia(st, z):
    """rho^z, z complejo."""
    return (st["U"] * np.exp(z * st["lnr"])) @ st["U"].conj().T


def srel(a, b):
    """S(a || b) = tr a (ln a - ln b)."""
    lnb = (b["U"] * b["lnr"]) @ b["U"].conj().T
    return -a["S"] - np.einsum("i,ij,ji->", a["p"], a["U"].conj().T, lnb @ a["U"]).real


def casos():
    for n in (6, 8):
        f = np.pi / n
        yield (f"anillo {n}, Theta=pi", ops_anillo(n), anillo(n), np.full(n, f),
               lambda v, n=n, f=f: n * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]),
               lambda v, n=n: n * v * np.array([2.0, 0.0, 1.0]),
               lambda y, f=f: np.array([y[0] * np.cos(f), y[0] * np.sin(f), y[2]]), VS)
    edges, u = escalera(4)
    for t in (np.pi, np.pi / 2):
        yield (f"escalera 2x4, t={t:.4f}", ops_clases(4), edges, t * u,
               lambda v, t=t: objetivo_escalera(4, v, t), lambda v: objetivo_escalera(4, v, 0.0),
               lambda y, t=t: girar(y, t), VS + (V_2x4,))


def cociclo(rb, r1, s):
    return np.trace(potencia(rb, 1 - 1j * s) @ potencia(r1, 1j * s))


def main():
    chk = Checker("modular_cociclo")
    filas, filasF = [], []
    for nombre, O, edges, phis, obj, obj0, gir, vs in casos():
        n = int(np.log2(O.shape[1]))
        me = simetrico(O)
        ell = min(len(V) for V in no_triviales(n, edges, phis))
        lazo = sum(2 * (-1) ** (ell + 1) * (1 - np.cos(h)) for V, h in ciclos(n, edges, phis) if len(V) == ell)
        res = []
        for v in vs:
            y0 = me.resuelve(obj0(v))[0]
            yb = gir(y0)
            y1, _, _, err = me.resuelve(obj(v), x0=yb)
            r0, rb, r1 = estado(O, y0), estado(O, yb), estado(O, y1)
            h = rb["lnZ"] - r0["lnZ"]
            dS = r1["S"] - r0["S"]
            s12, s21 = srel(r1, rb), srel(rb, r1)
            giro, ren = rb["S"] - r0["S"], -yb @ (r1["m"] - rb["m"])
            ident = max(abs(dS - (h - s12)), abs(giro + ren - s12 - dS))
            d = 1e-4
            dF = (cociclo(rb, r1, d) - cociclo(rb, r1, -d)) / (2 * d)
            F0 = cociclo(rb, r1, 0.0)
            chk.check(err < 1e-10 and ident < 1e-12 and abs(F0 - 1) < 1e-12 and abs(dF + 1j * s21) < 1e-7 * max(s21, 1e-2),
                      f"{nombre}, v = {v:.4f}: Delta S = h - S_rel con error {ident:.1e}; F(0) = 1, F'(0) = -i S(rhobar||rho) "
                      f"con error {abs(dF + 1j * s21):.1e}")
            fF = [(1 - abs(cociclo(rb, r1, s))) / (s * s * s12) if s12 > 1e-13 else np.nan for s in (0.25, 1.0, 4.0)]
            res.append((v, dS, h, giro, ren, s12, s21, dS / h, *fF))
            filasF += [(nombre, v, s, x) for s, x in zip((0.25, 1.0, 4.0), fF)]
        res = np.array(res)
        filas += [(nombre, ell, lazo, *r) for r in res]
        pequeños = res[:len(VS)]
        v0 = pequeños[0]
        chk.check(abs(v0[3] / v0[1] + (ell - 1)) < 0.1 and abs(v0[4] / v0[1] - ell) < 0.1,
                  f"{nombre}: reparto en v = {v0[0]}: giro {v0[3] / v0[1]:+.3f} Delta S (-(l-1) = {-(ell - 1)}), "
                  f"renormalización {v0[4] / v0[1]:+.3f} Delta S (l = {ell})")
        chk.check(abs(v0[2] / v0[0] ** ell / lazo - 1) < 0.15,
                  f"{nombre}: h/v^l en v = {v0[0]} = {v0[2] / v0[0] ** ell:+.4f} frente al término de lazos {lazo:+.0f}")
        # pendiente entre los dos v mayores: en el anillo de 8, S_rel ~ v^14 está en el redondeo con v <= 0,07
        a, b = pequeños[-2], pequeños[-1]
        pend = lambda k, d=0.0: np.log((b[k] - d) / (a[k] - d)) / np.log(b[0] / a[0])
        p12, p21, pk = pend(5), pend(6), pend(7, 1.0)
        chk.check(abs(p12 - (2 * ell - 2)) < 0.8 and abs(p21 - (2 * ell - 2)) < 0.8 and abs(pk - (ell - 2)) < 0.8,
                  f"{nombre}: pendientes en v = 0,1-0,15: S(rho||rhobar) {p12:.2f}, S(rhobar||rho) {p21:.2f} "
                  f"(2l - 2 = {2 * ell - 2}); Delta S/h - 1 {pk:.2f} (l - 2 = {ell - 2})")
        if ell == 4:
            f05 = res[1, 8:]
            chk.check(np.all(np.abs(f05 - 1) < 0.06),
                      f"{nombre}, v = 0,05: (1 - |F(s)|)/(s^2 S_rel) en s = 0,25, 1, 4: {np.round(f05, 3)}")
            chk.info(f"{nombre}, v = {V_2x4:.4f}: Delta S/h = {res[-1, 7]:.4f}, (1 - |F(s)|)/(s^2 S_rel) = "
                     f"{np.round(res[-1, 8:], 3)}")
    write_csv("modular_cociclo.csv", ("caso", "l", "lazo", "v", "Delta_S", "h", "giro", "renormalizacion",
                                      "S_rho_rhobar", "S_rhobar_rho", "Delta_S_sobre_h", "F_0.25", "F_1", "F_4"), filas)
    write_csv("modular_cociclo_F.csv", ("caso", "v", "s", "uno_menos_F_sobre_s2_Srel"), filasF)
    chk.finish()


if __name__ == "__main__":
    main()

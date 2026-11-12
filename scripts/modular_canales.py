"""Artículo II, sección 11.6: los dos canales del correlador de máxima entropía.

G(s) = tr(rho Z_0(s) Z_j), K = -x.O. Con Gbar el giro puro (multiplicadores de flujo nulo girados) y tau = v s:
  G_phi - G_0 = D_I + D_R,  D_I = Gbar - G_0 (interferencia),  D_R = G_phi - Gbar (renormalización).
Proposición 31: |D_I| <= C max(v, tau^2)^{V*-1}, con D_I -> D_inf(tau) (giro puro a T infinita con K/v) a tau fijo, y
|D_R| <= sqrt(2 S(rho_phi || rhobar)) + 2|s| ||K_phi - Kbar|| = O(v^{l-1} + tau v^{l-2}).
Se comprueba, en el anillo de 6 (Theta = pi) y la escalera 2 x 4 (t = pi), v = 0,01-0,05:
  1. la desigualdad de D_R punto a punto (tau en 0,001-2);
  2. envolvente max_{tau' <= 1} |D_R| ~ v^{l-2} y |D_I| ~ v^0;
  3. |D_I - D_inf| baja con v (correcciones O(v));
  4. cruce de los canales tau_c ~ v^{(l-2)/(2V*-3)} con j = 0.

Escribe data/modular_canales{,_cruce}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 .venv/bin/python scripts/modular_canales.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from holonomy.report import write_csv
from modular_cociclo import estado, srel
from modular_cota import (anillo, escalera, girar, no_triviales, objetivo_escalera, ops_anillo, ops_clases, simetrico,
                          vstar, zdiag)

VS = np.array([0.01, 0.015, 0.02, 0.03, 0.05])
TAUS = np.concatenate([np.geomspace(1e-3, 0.3, 60), np.linspace(0.3, 2.0, 60)[1:]])


def corr(O, y, n, j, ss, termico=True):
    """tr(rho Z_0(s) Z_j), rho = e^{y.O}/Z (o I/N), K = -y.O."""
    E, U = eigh(np.einsum("k,kab->ab", y, O))
    p = np.exp(E - logsumexp(E)) if termico else np.full(len(E), 1 / len(E))
    dK = E[:, None] - E[None, :]
    Z0 = (U.conj().T * zdiag(n, 0)[None, :]) @ U
    Zj = (U.conj().T * zdiag(n, j)[None, :]) @ U
    W = p[:, None] * Z0 * Zj.T
    keep = np.abs(W) > 1e-17 * np.abs(W).max()
    w, om = W[keep], dK[keep]
    return np.array([(w * np.exp(-1j * s * om)).sum().real for s in ss])


def casos():
    n = 6
    f = np.pi / n
    yield ("anillo 6, Theta=pi", n, ops_anillo(n), anillo(n), np.full(n, f),
           lambda v: n * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]), lambda v: n * v * np.array([2.0, 0, 1.0]),
           lambda y: np.array([y[0] * np.cos(f), y[0] * np.sin(f), y[2]]), (0, 3))
    edges, u = escalera(4)
    yield ("escalera 2x4, t=pi", 8, ops_clases(4), edges, np.pi * u, lambda v: objetivo_escalera(4, v, np.pi),
           lambda v: objetivo_escalera(4, v, 0.0), lambda y: girar(y, np.pi), (0, 1, 2, 6))


def main():
    chk = Checker("modular_canales")
    filas, filas_c = [], []
    b1 = np.argmin(abs(TAUS - 1.0))
    for nombre, n, O, edges, phis, obj, obj0, gir, js in casos():
        me = simetrico(O)
        nt = no_triviales(n, edges, phis)
        ell = min(len(V) for V in nt)
        sol = {}
        for v in VS:
            y0 = me.resuelve(obj0(v))[0]
            yb = gir(y0)
            y1 = me.resuelve(obj(v), x0=yb)[0]
            dK = np.einsum("k,kab->ab", y1 - yb, O)
            s_rel = max(srel(estado(O, y1), estado(O, yb)), 0.0)   # en el anillo, S_rel ~ v^10 sale < 0 por redondeo
            sol[v] = (y0, yb, y1, np.sqrt(2 * s_rel), np.abs(np.linalg.eigvalsh(dK)).max())
        for j in js:
            Vs = vstar(nt, j)
            envI, envR, dinf, tc, viol = [], [], [], [], 0.0
            for v in VS:
                y0, yb, y1, pinsker, normdK = sol[v]
                ss = TAUS / v
                g0, gb, g1 = corr(O, y0, n, j, ss), corr(O, yb, n, j, ss), corr(O, y1, n, j, ss)
                DI, DR = gb - g0, g1 - gb
                cociente = (np.abs(DR) / (pinsker + 2 * ss * normdK)).max()
                viol = max(viol, cociente)
                Dinf = corr(O, yb / v, n, j, TAUS, False) - corr(O, y0 / v, n, j, TAUS, False)
                eI, eR = np.maximum.accumulate(np.abs(DI)), np.maximum.accumulate(np.abs(DR))
                envI.append(eI[b1])
                envR.append(eR[b1])
                dinf.append(np.abs(DI - Dinf)[:b1 + 1].max() / np.abs(Dinf)[:b1 + 1].max())
                k = np.argmax(eI > eR)
                tc.append(TAUS[k] if k > 0 else np.nan)
                filas.append((nombre, j, v, eI[b1], eR[b1], dinf[-1], tc[-1], cociente))
            pR = np.polyfit(np.log(VS), np.log(envR), 1)[0]
            pI = np.polyfit(np.log(VS), np.log(envI), 1)[0]
            chk.check(viol <= 1.0, f"{nombre}, j = {j}: |D_R| <= sqrt(2 S_rel) + 2|s| ||dK|| en todo tau (máximo del "
                                   f"cociente {viol:.3f})")
            chk.check(abs(pR - (ell - 2)) < 0.25 and abs(pI) < 0.4,
                      f"{nombre}, j = {j}: envolvente en tau <= 1: D_R ~ v^{pR:.2f} (l - 2 = {ell - 2}), D_I ~ v^{pI:.2f}")
            chk.check(dinf[0] < dinf[-1],
                      f"{nombre}, j = {j}: max|D_I - D_inf|/max|D_inf| (tau <= 1) = {dinf[0]:.3f} en v = 0,01 y "
                      f"{dinf[-1]:.3f} en v = 0,05")
            tc = np.array(tc)
            if j == 0:
                ptc = np.polyfit(np.log(VS), np.log(tc), 1)[0]
                pred = (ell - 2) / (2 * Vs - 3)
                chk.check(abs(ptc - pred) < 0.05, f"{nombre}, j = 0: cruce tau_c = {np.round(tc, 4)} ~ v^{ptc:.3f}, "
                                                  f"predicción (l - 2)/(2V* - 3) = {pred:.3f}")
                filas_c.append((nombre, j, ell, Vs, ptc, pred))
            else:
                chk.info(f"{nombre}, j = {j}: cruce tau_c = {np.round(tc, 4)} (la cota de D_R no ve la localidad)")
    write_csv("modular_canales.csv", ("caso", "j", "v", "env_DI_tau1", "env_DR_tau1", "rel_DI_Dinf", "tau_c",
                                         "cociente_cota_DR"), filas)
    write_csv("modular_canales_cruce.csv", ("caso", "j", "l", "Vstar", "exponente_tau_c", "prediccion"), filas_c)
    chk.finish()


if __name__ == "__main__":
    main()

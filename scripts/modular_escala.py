"""Nota «Tiempo modular y holonomía», secciones 4 y 5: escala temporal de la holonomía y renormalización a v finito.

1. Giro puro a temperatura infinita, C_j(s) = tr(Z_0(s) Z_j)/N, amplitudes del flujo nulo en v = 0,6 vmax:
   - anillos de 6, 8 y 10 con Theta = pi: primer s con |Delta C_j|/max|C_j| > 1e-2, en unidades del tiempo de vuelta
     n/(6,7 a), en j = 0 y en las antípodas;
   - escaleras periódicas 2 x 4 (v = 0,6 vmax) y 2 x 5 (v = 0,3) con flujo por plaqueta: el mismo umbral, en s a.
2. Máxima entropía a v finito: |x| de las patas de la 2 x 4 con t = 0 y t = pi/2 en v = 0,6 vmax; en el anillo de 10
   (v = 0,6 vmax), cambio relativo de |x_e| con Theta y cambio de <Z_0 Z_j> (j >= 2), relativo a su valor sin flujo.
Solo numérico; los umbrales son arbitrarios.

Escribe data/modular_escala{,_maxent}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 .venv/bin/python scripts/modular_escala.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from holonomy import xxz
from holonomy.report import write_csv
from modular_cota import girar, objetivo_escalera, ops_anillo, ops_clases, simetrico, zdiag

VMAX_2x4 = 0.53556549


def correladores(O, y, n, js, ss, termico=False):
    """C_j(s) = tr(rho Z_0(s) Z_j), rho = I/N o e^{y.O}/Z, K = -y.O; S_z se conserva y casi todos los pesos son nulos."""
    E, U = eigh(np.einsum("k,kab->ab", y, O))
    p = np.exp(E - logsumexp(E)) if termico else np.full(len(E), 1.0 / len(E))
    dK = E[:, None] - E[None, :]                         # K_n - K_m con K = -E
    Z0 = (U.conj().T * zdiag(n, 0)[None, :]) @ U
    C = {}
    for j in js:
        Zj = (U.conj().T * zdiag(n, j)[None, :]) @ U
        W = p[:, None] * Z0 * Zj.T
        keep = np.abs(W) > 1e-15 * np.abs(W).max()
        w, om = W[keep], dK[keep]
        C[j] = np.array([(w * np.exp(-1j * s * om)).sum() for s in ss]).real
    return C


def llegada(C1, C0, ss, umbral=1e-2):
    R = np.abs(C1 - C0) / np.abs(C0).max()
    return ss[np.argmax(R > umbral)] if R.max() > umbral else np.nan


def anillos(chk, filas):
    for n in (6, 8, 10):
        O = ops_anillo(n)
        v = 0.6 * 4 * xxz.eps(n, 1.0, 0.0) / 3
        y0 = simetrico(O).resuelve(n * v * np.array([2.0, 0.0, 1.0]))[0]
        a = abs(y0[0])
        vf = 6.7 * a
        ss = np.concatenate([np.geomspace(0.02, 3.0, 60), np.linspace(3.0, 3 * n / vf, 300)])
        f = np.pi / n
        js = (0, n // 2)
        C0 = correladores(O, y0, n, js, ss)
        C1 = correladores(O, np.array([y0[0] * np.cos(f), y0[0] * np.sin(f), y0[2]]), n, js, ss)
        r = {j: llegada(C1[j], C0[j], ss) / (n / vf) for j in js}
        chk.check(0.6 < r[n // 2] < 0.9 and r[n // 2] < r[0],
                  f"anillo {n}, Theta = pi: llegada (1e-2) en {r[0]:.3f} vueltas en j = 0 y {r[n // 2]:.3f} en las "
                  "antípodas (antes en las antípodas)")
        filas += [(f"anillo {n}", j, "vueltas", r[j]) for j in js]


def escaleras(chk, filas):
    for L, v, ts in ((4, 0.6 * VMAX_2x4, (np.pi / 2, np.pi)), (5, 0.3, (2 * np.pi / 5, 4 * np.pi / 5))):
        n = 2 * L
        O = ops_clases(L)
        y0 = simetrico(O).resuelve(objetivo_escalera(L, v, 0.0))[0]
        a = abs(y0[0])
        ss = np.concatenate([np.geomspace(0.02, 1.0, 40), np.linspace(1.0, 25.0 * 0.35 / a, 300)[1:]])
        js = list(range(L // 2 + 1)) + [L + k for k in range(L // 2 + 1)]
        C0 = correladores(O, y0, n, js, ss)
        for t in ts:
            C1 = correladores(O, girar(y0, t), n, js, ss)
            r = np.array([llegada(C1[j], C0[j], ss) * a for j in js])
            chk.check(np.all(np.isfinite(r)) and r.max() < 1.1,
                      f"escalera 2x{L}, t = {t:.4f}: llegada (1e-2) en s a = {r.min():.3f}..{r.max():.3f}")
            filas += [(f"escalera 2x{L}, t={t:.4f}", j, "s_a", rj) for j, rj in zip(js, r)]


def maxent(chk):
    filas = []
    me = simetrico(ops_clases(4))
    v = 0.6 * VMAX_2x4
    y0 = me.resuelve(objetivo_escalera(4, v, 0.0))[0]
    y1, _, _, res = me.resuelve(objetivo_escalera(4, v, np.pi / 2), x0=girar(y0, np.pi / 2))
    p0, p1 = np.hypot(*y0[:2]), np.hypot(*y1[:2])
    chk.check(res < 1e-10, f"escalera 2x4, v = {v:.4f}: |x| de las patas {p0:.5f} (t = 0) -> {p1:.5f} (t = pi/2)")
    filas += [("escalera 2x4, |x| pata", 0.0, p0), ("escalera 2x4, |x| pata", np.pi / 2, p1)]
    n = 10
    O = ops_anillo(n)
    me = simetrico(O)
    v = 0.6 * 4 * xxz.eps(n, 1.0, 0.0) / 3
    zz = {}
    for Th in (0.0, np.pi / 2, np.pi):
        f = Th / n
        y = me.resuelve(n * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]))[0]
        E, U = eigh(np.einsum("k,kab->ab", y, O))
        p = np.exp(E - logsumexp(E))
        prob = (np.abs(U) ** 2) @ p                      # diagonal de rho en la base computacional
        zz[Th] = (np.hypot(y[0], y[1]), [(prob * zdiag(n, 0) * zdiag(n, j)).sum() for j in range(n // 2 + 1)])
    for Th in (np.pi / 2, np.pi):
        dx = zz[Th][0] / zz[0.0][0] - 1
        dz = [abs(zz[Th][1][j] / zz[0.0][1][j] - 1) for j in range(2, n // 2 + 1)]
        chk.check(dx > 0 and max(dz) < 1e-3,
                  f"anillo 10, Theta = {Th:.4f}: |x_e| cambia {dx:.2e} relativo; <Z0 Zj> (j = 2..5) cambia "
                  f"{min(dz):.1e}..{max(dz):.1e} relativo")
        filas += [("anillo 10, cambio relativo |x_e|", Th, dx)] + \
                 [(f"anillo 10, cambio relativo <Z0 Z{j}>", Th, d) for j, d in zip(range(2, 6), dz)]
    write_csv("modular_escala_maxent.csv", ("magnitud", "flujo", "valor"), filas)


def main():
    chk = Checker("modular_escala")
    filas = []
    anillos(chk, filas)
    escaleras(chk, filas)
    write_csv("modular_escala.csv", ("caso", "j", "unidad", "llegada"), filas)
    maxent(chk)
    chk.finish()


if __name__ == "__main__":
    main()

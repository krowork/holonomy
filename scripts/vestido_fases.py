"""Vestido por fases para el modelo de carga entera (celda_k_notas.md, Sec. 7.13, pasos 1 y 2).

Paso 1 (L0 > 0): Kirkwood-Thomas restringido a los caracteres cuyos tramos constantes no nulos (cíclicos) miden
menos de L0; los demás coeficientes se fijan a 0. Se da si converge, la energía por sitio y la densidad del
residuo en los caracteres excluidos, max_x sum_{mu excl., mu ∋ x} |R^(mu)|.

Paso 2: para cada fase c, el punto fijo G_c alrededor de chi_{c.1} (H'_c = chi_{-c1} H' chi_{c1}). Para la isla
mu = c en I (longitud L, resto 0) se compara el radio de Gershgorin de la columna mu de e^{-G} H' e^{G} con
  - G = G_0 (una fase, Lema Z8);
  - G = G_mu, vestido por fases: g_c(nu) si sop nu ⊆ I, g_0(nu) en otro caso.
Vestido de pared (RPARED, por defecto 2): punto fijo en los nu cerca de las paredes con Delta >= 1; se separa la
parte del radio que mueve la pared (Delta <= 0), que es dinámica de verdad y no se puede quitar.
Uso: [RPARED=r] python scripts/vestido_fases.py modelo n eps [L0]   (L0 = 0: sin restricción)
"""
import os
import sys

import numpy as np

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def tramo_max(dig):
    """Longitud del mayor tramo cíclico constante y no nulo de cada fila (n si es constante no nula)."""
    N, n = dig.shape
    best = np.zeros(N, int)
    run = np.zeros(N, int)
    for j in list(range(n)) * 2:  # dos vueltas para los tramos cíclicos
        nz = dig[:, j] != 0
        same = dig[:, j] == dig[:, j - 1]
        run = np.where(nz, np.where(same, run + 1, 1), 0)
        best = np.maximum(best, np.minimum(run, n))
    return best


def kt(Hop, n, borde, permitido, maxit=2000, tol=1e-12):
    libre = (borde > 0) & permitido
    g = np.zeros(K**n, complex)
    for it in range(maxit):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(Hop(np.exp(G)) / np.exp(G), n)
        paso = np.where(libre, Rh / np.where(libre, borde, 1), 0)
        g -= paso
        if not np.isfinite(paso).all() or np.abs(paso).max() > 1e3:
            return None
        if np.abs(paso).max() < tol:
            return g, Rh, it
    return None


RPARED = int(os.environ.get("RPARED", 2))


def main():
    modelo, n, eps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    L0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    hcl, v, q = termino(modelo)
    H = anillo(n, K, q, hcl + eps * v, 0.0)
    dig, am, dm = marco(n)
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    tm = tramo_max(dig)
    permitido = tm < L0 if L0 else np.ones(K**n, bool)
    sop = dig != 0

    def Hc(c):
        chi = np.exp(2j * np.pi / K * c * dig.sum(1))
        return lambda x: chi.conj() * am(H @ dm(chi * x))

    res = kt(Hc(0), n, borde, permitido)
    if res is None:
        print(f"{modelo} n={n} eps={eps} L0={L0}: KT diverge")
        return
    g0, Rh, it = res
    excl = (borde > 0) & ~permitido
    dens = max(np.abs(Rh[excl & sop[:, x]]).sum() for x in range(n)) if excl.any() else 0.0
    print(f"{modelo} n={n} eps={eps} L0={L0}: KT en {it} it., E/n = {Rh[0].real / n:.10f}, "
          f"densidad del residuo excluido = {dens:.2e}", flush=True)
    if L0:
        return
    gs = {0: g0}
    for c in range(1, K):
        out = kt(Hc(c), n, borde, permitido)
        gs[c] = out[0]
        print(f"   fase {c}: E/n = {out[1][0].real / n:.10f}")
    E = Rh[0]
    Gfun = lambda g: np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
    pw = K ** np.arange(n)
    for c in range(1, K):
        fila = []
        for L in range(1, n):
            mu_dig = np.zeros(n, int)
            mu_dig[1:1 + L] = c
            mu = int((mu_dig * pw).sum())
            chi = np.exp(2j * np.pi / K * (dig @ mu_dig))
            dentro = ~(sop & (mu_dig == 0)[None, :]).any(1)  # sop nu ⊆ I
            r = []
            for g in (g0, np.where(dentro, gs[c], g0)):
                G = Gfun(g)
                col = chi_hat(np.exp(-G) * am(H @ dm(np.exp(G) * chi)), n)
                r.append((col[mu] - E).real)
                r.append(np.abs(col).sum() - abs(col[mu]))
            # vestido de pared: punto fijo en los nu de soporte a distancia <= RPARED de las paredes que suben la
            # energía clásica (Delta = |d(mu+nu)| - |d mu| >= 1); los de Delta <= 0 (mover la pared) se dejan.
            sal = ((dig + mu_dig[None, :]) % K * pw).sum(1)  # índice de mu + nu
            Delta = borde[sal] - borde[mu]
            x = np.arange(n)
            dpar = np.minimum.reduce([np.abs((x - p + n / 2) % n - n / 2) for p in (0.5, L + 0.5)])
            cerca = dpar <= RPARED
            W = (dig != 0).any(1) & ~(sop & ~cerca[None, :]).any(1) & (Delta >= 1)
            g = np.where(dentro, gs[c], g0).astype(complex)
            for _ in range(500):
                G = Gfun(g)
                col = chi_hat(np.exp(-G) * am(H @ dm(np.exp(G) * chi)), n)
                paso = np.where(W, col[sal] / np.where(W, Delta, 1), 0)
                g = g - paso
                if np.abs(paso).max() < 1e-12 or not np.isfinite(paso).all():
                    break
            mover = np.zeros(K**n, bool)
            mover[sal[(Delta <= 0) & (dig != 0).any(1)]] = True
            rad = np.abs(col).sum() - abs(col[mu])
            rad_mov = np.abs(col[mover]).sum()
            fila.append(f"L={L}: {r[1]:.2f}->{r[3]:.2f}->{rad:.2f} [mover {rad_mov:.2f}] (diag {r[2]:.3f}->"
                        f"{(col[mu] - E).real:.3f})")
        print(f"   isla c={c}, radio una fase -> por fases -> con pared: " + "  ".join(fila), flush=True)


if __name__ == "__main__":
    main()

"""Gershgorin completo en la base vestida por fases (celda_k_notas.md, Sec. 7.13, paso 3: el cambio de base).

Base: b_mu = chi_mu e^{G_mu}, con G_mu = sum_nu g_{c(D)}(nu) chi_nu si sop nu cae dentro de un dominio D de mu
(tramo constante maximal, cíclico, de valor c(D)), y g_0(nu) en otro caso. Las g_c son los vacíos completos de
Kirkwood-Thomas de cada fase (Anderson, vacio_islas.py). Se calcula la matriz M = B^{-1} H' B, con B la matriz de
los b_mu como funciones en el marco, y se compara con la base de una sola fase (G_mu = G_0):
  - eta = max sobre mu no constantes de radio_mu / |d mu|;
  - la cota de Gershgorin del hueco, min_{mu no constante} (Re M_mumu - E - radio_mu), frente al hueco exacto;
  - el radio máximo de las columnas constantes y el número de condición de B.
Uso: python scripts/base_fases.py modelo n eps [eps ...]
"""
import sys

import numpy as np
import scipy.linalg as sl

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def vacio(Hop, n, libre, mem=8, beta=0.5):
    def F(g):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(Hop(np.exp(G)) / np.exp(G), n)
        return np.where(libre, -Rh / np.where(libre, libre, 1), 0), Rh

    g = np.zeros(K**n, complex)
    X, Fs = [], []
    for _ in range(3000):
        f, Rh = F(g)
        if np.abs(f).max() < 1e-12:
            return g, Rh[0]
        X.append(g.copy())
        Fs.append(f.copy())
        X, Fs = X[-mem:], Fs[-mem:]
        if len(Fs) > 1:
            dF = np.array([Fs[i + 1] - Fs[i] for i in range(len(Fs) - 1)]).T
            dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
            gam = np.linalg.lstsq(dF, f, rcond=None)[0]
            g = g + beta * f - (dX + beta * dF) @ gam
        else:
            g = g + beta * f
    raise RuntimeError("KT no converge")


def dominios(m):
    """Etiqueta de dominio (tramo constante maximal cíclico) de cada sitio."""
    n = len(m)
    cortes = [j for j in range(n) if m[j] != m[j - 1]]
    if not cortes:
        return np.zeros(n, int)
    lab = np.empty(n, int)
    for i, a in enumerate(cortes):
        b = cortes[(i + 1) % len(cortes)]
        j = a
        while True:
            lab[j] = i
            j = (j + 1) % n
            if j == b:
                break
    return lab


def main():
    modelo, n = sys.argv[1], int(sys.argv[2])
    hcl, v, q = termino(modelo)
    dig, am, dm = marco(n)
    N = K**n
    borde = (dig != np.roll(dig, -1, axis=1)).sum(1)
    sop = dig != 0
    for eps in [float(x) for x in sys.argv[3:]]:
        H = anillo(n, K, q, hcl + eps * v, 0.0)
        # libre = borde (para dividir); se pasa borde con 0 en las constantes
        gs, E = {}, None
        for c in range(K):
            chi = np.exp(2j * np.pi / K * c * dig.sum(1))
            Hop = lambda x, chi=chi: chi.conj() * am(H @ dm(chi * x))
            gs[c], Ec = vacio(Hop, n, np.where(borde > 0, borde, 0))
            E = Ec if E is None else E
        Gf = lambda g: np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        if N <= 3000:
            ev = np.linalg.eigvalsh(H.toarray())
        else:
            from scipy.sparse.linalg import eigsh
            ev = np.sort(eigsh(H, k=K + 1, which="SA", tol=1e-10)[0])
        hueco = ev[K] - ev[0]
        dt = np.complex128 if N <= 3000 else np.complex64
        for nombre in ("una fase", "por fases"):
            B = np.empty((N, N), dt)
            for mu in range(N):
                m = dig[mu]
                if nombre == "una fase":
                    g = gs[0]
                else:
                    lab = dominios(m)
                    g = gs[0].copy()
                    for d in np.unique(lab):
                        en_d = lab == d
                        dentro = ~(sop & ~en_d[None, :]).any(1) & sop.any(1)
                        g = np.where(dentro, gs[int(m[en_d][0])], g)
                B[:, mu] = np.exp(2j * np.pi / K * (dig @ m)) * np.exp(Gf(g))
            pw = K ** np.arange(n)
            if N <= 3000:
                cols = np.arange(N)
            else:  # constantes, todas las islas y una muestra al azar
                islas = []
                for c in range(1, K):
                    for L in range(1, n):
                        m = np.zeros(n, int)
                        m[1:1 + L] = c
                        islas.append(int((m * pw).sum()))
                const_idx = [int((np.full(n, c) * pw).sum()) for c in range(K)]
                cols = np.unique(np.r_[const_idx, islas, np.random.default_rng(0).choice(N, 300, False)])
            HB = np.column_stack([am(H @ dm(B[:, j].astype(complex))) for j in cols]).astype(dt)
            cond = np.linalg.cond(B) if N <= 3000 else float("nan")
            lu = sl.lu_factor(B, overwrite_a=True, check_finite=False)
            del B
            Mm = sl.lu_solve(lu, HB, check_finite=False)
            del lu, HB
            d = Mm[cols, np.arange(cols.size)].astype(complex)
            rad = np.abs(Mm).sum(0, dtype=float) - np.abs(d)
            del Mm
            bc = borde[cols]
            nc = bc > 0
            eta = (rad[nc] / bc[nc]).max()
            cota = ((d - E).real - rad)[nc].min()
            ncol = "" if N <= 3000 else f" [{cols.size} columnas]"
            print(f"{modelo} n={n} eps={eps} {nombre}{ncol}: eta = {eta:.3f}, cota de Gershgorin del hueco = {cota:.3f} "
                  f"(exacto {hueco:.3f}), radio máx. de las constantes = {rad[~nc].max():.1e}, "
                  f"cond(B) = {cond:.1e}", flush=True)


if __name__ == "__main__":
    main()

"""Espectro completo del anillo de Heisenberg con flujo por sectores de magnetización y momento.

H(Phi) = sum_i [ 1/2 (e^{i Phi/n} S+_i S-_{i+1} + h.c.) + Sz_i Sz_{i+1} ], la misma convención que `holonomy.xxz`
con Delta = 1 (sitio i = bit (s >> i) & 1, bit 1 = espín arriba). El flujo uniforme conserva la traslación T, que
lleva el bit i al i + 1, así que H es diagonal por bloques en (M, k). Base de momento de un representante r de
periodo p: |r, k> = (sqrt(p)/n) sum_{l=0}^{n-1} e^{-ikl} T^l |r>, que existe si k p = 0 mod 2 pi. Si
H|r> = sum_j h_j T^{l_j} |r_j>, entonces <r_j, k|H|r, k> += h_j e^{i k l_j} sqrt(p/p_j).

Para el espectro térmico basta M <= n/2: el sector n - M con flujo Phi tiene el espectro del M con -Phi (inversión
de espín), que es el del M con Phi (reflexión del anillo).
"""
from functools import lru_cache

import numpy as np


def _rot(s, m, n):
    """Rota los bits de s m posiciones hacia arriba (bit i -> i + m)."""
    mask = (1 << n) - 1
    m %= n
    return ((s << m) | (s >> (n - m))) & mask if m else s


@lru_cache(maxsize=None)
def sector(n, M):
    """Representantes, periodos y saltos hacia delante S+_i S-_{i+1} del sector M.

    Devuelve (reps, per, diag, a, b, l): reps y per por representante; diag = sum Sz Sz de cada representante; y,
    por cada salto desde el representante a, el representante b de llegada y el desplazamiento l con
    estado = T^l rep_b.
    """
    todos = np.array([s for s in range(1 << n) if bin(s).count("1") == M], dtype=np.int64)
    rots = np.array([_rot(todos, m, n) for m in range(n)])  # rots[m] = T^m s
    rep = rots.min(axis=0)
    mstar = rots.argmin(axis=0)  # rep = T^{m*} s, luego s = T^{-m*} rep
    reps = np.unique(rep)
    idx = {int(r): k for k, r in enumerate(reps)}
    pos = {int(s): k for k, s in enumerate(todos)}
    per = np.array([next(m for m in range(1, n + 1) if _rot(int(r), m, n) == int(r)) for r in reps])
    diag = np.zeros(len(reps))
    a, b, l = [], [], []
    for ia, r in enumerate(reps):
        r = int(r)
        for i in range(n):
            j = (i + 1) % n
            bi, bj = (r >> i) & 1, (r >> j) & 1
            diag[ia] += 0.25 if bi == bj else -0.25
            if bi == 0 and bj == 1:
                t = (r | (1 << i)) & ~(1 << j)
                q = pos[t]
                a.append(ia)
                b.append(idx[int(rep[q])])
                l.append((-int(mstar[q])) % n)
    return reps, per, diag, np.array(a, dtype=np.int64), np.array(b, dtype=np.int64), np.array(l, dtype=np.int64)


def espectro(n, Phi, Ms=None):
    """Autovalores de H(Phi) por sectores. Devuelve lista de (M, multiplicidad del sector, autovalores), con
    multiplicidad 2 para M < n/2 (el sector n - M) y 1 para M = n/2."""
    out = []
    for M in (range(n // 2 + 1) if Ms is None else Ms):
        reps, per, diag, a, b, l = sector(n, M)
        ev = []
        for q in range(n):
            k = 2 * np.pi * q / n
            ok = (q * per) % n == 0
            if not ok.any():
                continue
            sel = np.flatnonzero(ok)
            loc = -np.ones(len(reps), dtype=np.int64)
            loc[sel] = np.arange(len(sel))
            m = ok[a] & ok[b]
            F = np.zeros((len(sel), len(sel)), dtype=complex)
            np.add.at(F, (loc[b[m]], loc[a[m]]), np.exp(1j * k * l[m]) * np.sqrt(per[a[m]] / per[b[m]]))
            Hk = np.diag(diag[sel]).astype(complex) + 0.5 * (np.exp(1j * Phi / n) * F + np.exp(-1j * Phi / n) * F.conj().T)
            ev.append(np.linalg.eigvalsh(Hk))
        mult = 1 if 2 * M == n else 2
        out.append((M, mult, np.concatenate(ev)))
    return out

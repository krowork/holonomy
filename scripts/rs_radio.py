"""Hipótesis (A) de celda_k_notas.md, Sec. 7: radio de analiticidad de cada nivel del multiplete frente a n (exploración).

Como h_cl es diagonal, la serie de Rayleigh-Schrödinger de cada nivel lambda_p (sector de momento p, donde el
fundamental clásico |p> = sum_r e^{-ipr} T^r |c_0> / sqrt(k) es no degenerado) sale de una recurrencia barata:
    psi^(m) = S [V psi^(m-1) - sum_{j=1}^m E^(j) psi^(m-j)],   E^(m) = <p|V|psi^(m-1)>,
con S = (E_0 - H_cl)^{-1} fuera de las configuraciones fundamentales (normalización intermedia). Se dan los
coeficientes c_m = E^(m)/n y la estimación del radio de convergencia por el criterio de la raíz y por Padé: si el
radio no baja con n, hay un disco uniforme, que es (A2) para la energía.

Uso: python scripts/rs_radio.py modelo theta_sobre_pi p_sobre_2pi_k mmax n [n ...]   (MU en el entorno)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

import celda_k_flujo as ent
import celda_k_llenado as ll

MU = float(os.environ.get("MU", 1.0))


def sistema(modelo, n, theta):
    if modelo == "entero":
        N, diag, r, c = ent.build(n, 0)
        codes, _ = ent.sector(n, 0)
        A = sp.coo_matrix((2 * np.exp(1j * theta / n) * np.ones(r.size), (r, c)), shape=(N, N))
    else:
        N, diag, r, c, r2, c2 = ll.build(n, n // 3)
        codes = np.arange(3**n, dtype=np.int64)
        dig = (codes[None, :] // 3 ** np.arange(n)[:, None]) % 3
        codes = codes[(dig == 0).sum(0) == n // 3]
        A = sp.coo_matrix((np.r_[np.exp(1j * theta / n) * np.ones(r.size), MU * np.ones(r2.size)],
                           (np.r_[r, r2], np.r_[c, c2])), shape=(N, N))
    A = A.tocsr()
    return codes, diag, (A + A.getH()).tocsr()


def trasladar(code, n):
    d = [(code // 3**j) % 3 for j in range(n)]
    d = d[-1:] + d[:-1]
    return sum(x * 3**j for j, x in enumerate(d))


def serie(modelo, n, theta, p, mmax):
    codes, diag, V = sistema(modelo, n, theta)
    E0 = diag.min()
    fund = np.nonzero(diag == E0)[0]
    k = fund.size
    c0 = codes[fund[0]]
    psi0 = np.zeros(codes.size, complex)
    cr = c0
    for r in range(k):
        psi0[np.searchsorted(codes, cr)] += np.exp(-1j * p * r)
        cr = trasladar(cr, n)
    assert cr == c0
    psi0 /= np.linalg.norm(psi0)
    den = E0 - diag
    S = np.where(diag == E0, 0.0, 1.0 / np.where(diag == E0, 1.0, den))
    psis, E = [psi0], [E0]
    for m in range(1, mmax + 1):
        Vpsi = V @ psis[-1]
        E.append(np.vdot(psi0, Vpsi))
        rhs = Vpsi - sum(E[j] * psis[m - j] for j in range(1, m))
        psis.append(S * rhs)
    return k, np.array(E) / n


def radio(c):
    m = np.arange(len(c))
    sel = (m >= len(c) // 2) & (np.abs(c) > 0)
    raiz = np.exp(-np.log(np.abs(c[sel])) / m[sel])
    return raiz.min(), raiz[-1]


def main():
    modelo, th, pp, mmax = sys.argv[1], float(sys.argv[2]) * np.pi, float(sys.argv[3]), int(sys.argv[4])
    ns = [int(x) for x in sys.argv[5:]]
    tab = {}
    for n in ns:
        k, c = serie(modelo, n, th, 2 * np.pi * pp / 3, mmax)
        tab[n] = c
        r_min, r_ult = radio(c)
        print(f"n={n:2d} k={k}: |Im c|max={np.abs(c.imag).max():.1e}  radio (raíz, m>={mmax // 2}): "
              f"min {r_min:.4f}, último {r_ult:.4f}", flush=True)
    print("coeficientes c_m = E^(m)/n (parte real):")
    for m in range(1, min(mmax, 16) + 1):
        print(f"m={m:2d} " + "  ".join(f"{tab[n][m].real:+.10e}" for n in ns))
    np.save(os.environ.get("SALIDA", "/dev/null") if os.environ.get("SALIDA") else "/tmp/_rs.npy",
            np.array([tab[n] for n in ns]))


if __name__ == "__main__":
    main()

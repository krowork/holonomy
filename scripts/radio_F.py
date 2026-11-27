"""Radio explícito de la hipótesis (F) (sensibilidad_celda3.tex, Proposición del túnel y su radio).

(F): el nivel de momento 0 es el fundamental de H_n(eps, Theta) para toda Theta. Se certifica así:

1. Serie. En cada sector de momento q (q = 0, 1, 2), Rayleigh-Schrödinger desde chi_q = sum_r w^{qr}|c_r>,
   con giro uniforme, sobre las configuraciones a <= j saltos de los patrones (exacto para psi_0..psi_j).
   Con el cociente de Rayleigh de Phi = sum_{a<=j} eps^a psi_a se obtienen los coeficientes de lambda_q
   hasta el orden K = 2j + 1 (regla 2n+1 de Wigner). g_q = lambda_q - lambda_0 tiene coeficientes nulos
   por debajo de m = 2n/3 (Proposición del túnel), que se ponen a cero exactamente.
2. Theta. El coeficiente de orden k es un polinomio trigonométrico con frecuencias |w| <= k/n
   (Lema de corrientes); se recupera por FFT con M > 2 floor(K/n) valores y se acota
   sup_Theta |g_k| <= sum_w |ĝ_{k,w}|.
3. Cola. Por Bauer-Fike (Lema bf, sin término diagonal), para |eps| < rho_0 = 1/(2 sqrt2 n) cada lambda_q es el
   único autovalor de su sector en |z + n| < 1, analítico, con |lambda_q + n| <= 2 sqrt2 n |eps|. Así
   |g_q| <= M(rho) = 4 sqrt2 n rho en |eps| = rho, y la cola de orden > K es <= M (eps/rho)^{K+1}/(1 - eps/rho).
4. Cota. f(eps) = c_m - sum_{m<k<=K} C_k eps^{k-m} - M eps^{K+1-m}/(rho^{K+1}(1 - eps/rho)), con
   c_m = ĝ_{m,0} - sum_{w!=0}|ĝ_{m,w}| - err_m y C_k = sum_w |ĝ_{k,w}| + err_k, es decreciente en eps:
   f(eps_F) > 0 da g_q(eps, Theta) >= eps^m f(eps) > 0 en todo (0, eps_F] y para toda Theta.
   Los sectores de carga != 0 tienen energía clásica >= -n + 2 (al menos dos paredes, Lema clásico) y,
   por Bauer-Fike, quedan por encima del multiplete con el mismo eps < 1/(2 sqrt2 n).
   err_k = TOL * (|E_k^(q)| + |E_k^(0)|) acota el redondeo de la resta.

Uso: python scripts/radio_F.py n j [M]       (por ejemplo 9 12 8)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_serie_rapida import conjunto  # noqa: E402

TOL = 1e-12


def montar(n, j):
    codes, pw = conjunto(n, j, tres=True)
    N = codes.size
    dig = (codes[:, None] // pw[None, :]) % 3
    H0 = -np.sum((np.roll(dig, -1, axis=1) - dig) % 3 == 1, axis=1).astype(float)
    filas, cols = [], []
    for s in range(n):
        k = (s + 1) % n
        a, b = dig[:, s], dig[:, k]
        msk = (a < 2) & (b > 0)                      # S+_s S-_k, fase e^{i Theta/n}
        new = codes[msk] + pw[s] - pw[k]
        idx = np.searchsorted(codes, new)
        ok = (idx < N) & (codes[np.minimum(idx, N - 1)] == new)
        filas.append(idx[ok]); cols.append(np.nonzero(msk)[0][ok])
    r, c = np.concatenate(filas), np.concatenate(cols)
    pats = [np.searchsorted(codes, int(np.dot([(x + rr) % 3 for x in range(n)], pw))) for rr in range(3)]
    return N, H0, r, c, pats


def serie_lambda(n, j, th, q, datos):
    N, H0, r, c, pats = datos
    A = sp.coo_matrix((2.0 * np.exp(1j * th / n) * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    T = (A + A.getH()).tocsr()
    E0 = -float(n)
    d = H0 - E0
    esP = np.zeros(N, bool); esP[pats] = True
    d[esP] = 1.0
    chi = np.zeros(N, complex)
    for rr, p in enumerate(pats):
        chi[p] = np.exp(2j * np.pi * q * rr / 3)
    psi = [chi]
    E = [0.0]                                          # E_k de lambda - E0
    peor = 0.0
    for k in range(1, j + 1):
        Tp = T @ psi[k - 1]
        Ek = np.vdot(chi, Tp) / 3.0
        E.append(Ek)
        resto = Tp[esP] - Ek * chi[esP]                # debe anularse (momento)
        peor = max(peor, np.abs(resto).max() / max(1.0, np.abs(Tp).max()))
        rhs = -Tp
        for l in range(1, k):
            rhs = rhs + E[l] * psi[k - l]
        v = rhs / d
        v[esP] = 0.0
        psi.append(v)
    P = np.array(psi)                                  # (j+1) x N
    G0 = P.conj() @ (d * (~esP) * P).T                 # <psi_a|(H0 - E0)|psi_b>
    G1 = P.conj() @ (T @ P.T)
    S = P.conj() @ P.T
    K = 2 * j + 1
    num = np.zeros(K + 1, complex); den = np.zeros(K + 1, complex)
    for a in range(j + 1):
        for b in range(j + 1):
            if a + b <= K:
                num[a + b] += G0[a, b]; den[a + b] += S[a, b]
            if a + b + 1 <= K:
                num[a + b + 1] += G1[a, b]
    inv = np.zeros(K + 1, complex); inv[0] = 1 / den[0]
    for k in range(1, K + 1):
        inv[k] = -sum(den[i] * inv[k - i] for i in range(1, k + 1)) / den[0]
    lam = np.array([sum(num[i] * inv[k - i] for i in range(k + 1)) for k in range(K + 1)])
    rs = np.array(E)
    acuerdo = np.abs(lam[:j + 1] - rs).max() / max(1.0, np.abs(rs).max())
    return lam.real, np.abs(lam.imag).max(), peor, acuerdo


def coeficientes(n, j, M):
    datos = montar(n, j)
    K = 2 * j + 1
    m = 2 * n // 3
    th = 2 * np.pi * np.arange(M) / M
    lam = {}
    for q in range(3):
        for s, t in enumerate(th):
            lam[q, s], im, peor, ac = serie_lambda(n, j, t, q, datos)
            if s == 0:
                print(f"  q={q}: |Im| max {im:.1e}, patrones fuera de chi {peor:.1e}, RS vs Rayleigh {ac:.1e}",
                      flush=True)
    res = {}
    for q in (1, 2):
        g = np.array([lam[q, s] - lam[0, s] for s in range(M)])          # M x (K+1)
        err = TOL * np.max([np.abs(lam[q, s]) + np.abs(lam[0, s]) for s in range(M)], axis=0)
        gh = np.fft.fft(g, axis=0) / M                                   # ĝ_{k,w}
        ruido_bajo = np.abs(g[:, :m]).max()
        g0 = gh[0].real
        resto_w = np.abs(gh[1:]).sum(axis=0)
        res[q] = dict(g0=g0, resto_w=resto_w, err=err, ruido_bajo=ruido_bajo,
                      escala_bajo=np.abs([lam[0, s][:m] for s in range(M)]).max())
    return res, K, m


def radio(n, res, K, m):
    rho0 = 1 / (2 * np.sqrt(2) * n)
    tope = rho0  # sectores de carga != 0: hueco 2, como el de carga 0
    def f(eps, q, rho):
        R = res[q]
        cm = R["g0"][m] - R["resto_w"][m] - R["err"][m]
        Ck = np.abs(R["g0"][m + 1:]) + R["resto_w"][m + 1:] + R["err"][m + 1:]
        s = cm - sum(C * eps ** (i + 1) for i, C in enumerate(Ck))
        Mr = 4 * np.sqrt(2) * n * rho
        return s - Mr * eps ** (K + 1 - m) / (rho ** (K + 1) * (1 - eps / rho))
    def ok(eps):
        rhos = eps + (rho0 * (1 - 1e-12) - eps) * np.linspace(1e-3, 1, 4000)
        return all(max(f(eps, q, rho) for rho in rhos) > 0 for q in (1, 2))
    lo, hi = 0.0, min(tope, rho0 * (1 - 1e-9))
    if ok(hi):
        return hi, True
    lo = hi * 1e-30
    if not ok(lo):
        return 0.0, False
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo, False


if __name__ == "__main__":
    n, j = int(sys.argv[1]), int(sys.argv[2])
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    K = 2 * j + 1
    if M <= 2 * (K // n):
        raise SystemExit(f"hace falta M > {2 * (K // n)}")
    res, K, m = coeficientes(n, j, M)
    for q in (1, 2):
        R = res[q]
        print(f"  q={q}: ruido en órdenes < {m}: {R['ruido_bajo']:.1e} (escala {R['escala_bajo']:.1e}); "
              f"coef. de orden {m}: {R['g0'][m]:.10g} (3|T_n|), parte de Theta {R['resto_w'][m]:.1e}")
        print("        g_k (k = m..K, media en Theta):", " ".join(f"{x:.4g}" for x in R["g0"][m:]))
    e, tope = radio(n, res, K, m)
    print(f"n={n}, j={j} (K={K}), M={M}: (F) certificada en 0 < eps <= {e:.4g}"
          + (" (tope de Bauer-Fike/carga)" if tope else ""))

"""Cuasi-modo de Kirkwood-Thomas en el marco de reloj Z_k (celda_k_notas.md, Sec. 7.7; exploración).

Marco: transformada de Fourier en cada sitio, F_ab = w^{ab}/sqrt(k), seguida del gauge diagonal D = prod_j w^{-j s_j}
(cierra en el anillo si k | n). En ese marco -P_asc = -(1/k) sum_j sum_m (X_j^dag X_{j+1})^m y su fundamental en cada
sector de la carga Z_k es el vector uniforme. Ansatz psi'(s) = exp(sum_mu g(mu) chi_mu(s)), chi_mu(s) = w^{mu.s}.
Con R = (H' psi')/psi', el punto fijo de KT es R^(mu) = 0 para mu no constante; se itera
    g(mu) <- g(mu) - R^(mu)/|d mu|        (|d mu| = número de enlaces donde mu salta).
Lo que queda en los caracteres constantes c != 0 es el residuo r_c del cuasi-modo: (H - E) psi' = (sum_c r_c chi_c) psi'.

Uso: python scripts/kt_zk.py modelo theta_sobre_pi n eps [eps ...]
  modelo: entero (S+S-, sin simetría de reloj) | llenado (salto + intercambio, mu = 1: intercambio SU(3), con ella)
"""
import sys

import numpy as np

from dual_celda import anillo

K = 3
W = np.exp(2j * np.pi / K)


def termino(modelo):
    """(h_cl, v, q) de dos sitios, índice 3a + b."""
    hcl = np.zeros((9, 9))
    for a in range(3):
        hcl[3 * a + (a + 1) % 3, 3 * a + (a + 1) % 3] = -1
    v = np.zeros((9, 9))
    if modelo == "entero":
        for a in range(2):
            for b in range(1, 3):
                v[3 * (a + 1) + (b - 1), 3 * a + b] = 2
        q = [-1, 0, 1]
    else:  # intercambio de contenidos distintos con amplitud 1 (salto de la partícula 0 y el intercambio neutro 1-2)
        for a in range(3):
            for b in range(3):
                if a != b:
                    v[3 * b + a, 3 * a + b] = 1
        q = [1, 0, 0]
    v = v + v.T if modelo == "entero" else v
    return hcl, v, q


def marco(n):
    """Devuelve funciones a_marco (original -> marco) y desde_marco, sobre vectores de 3^n (código sum s_j 3^j)."""
    dig = (np.arange(K**n)[:, None] // K ** np.arange(n)[None, :]) % K
    fase = W ** (-(np.arange(n)[None, :] * dig).sum(1) % K)  # D = prod_j w^{-j s_j}: el uniforme es c_0 = 0 1 2 ...
    forma = (K,) * n  # el eje 0 de la forma C es el sitio n-1

    def a_marco(x):
        y = np.fft.ifftn(x.reshape(forma)).ravel() * np.sqrt(K**n)  # F_ab = w^{ab}/sqrt(k) en cada sitio
        return fase * y

    def desde_marco(y):
        x = np.fft.fftn((fase.conj() * y).reshape(forma)).ravel() / np.sqrt(K**n)
        return x

    return dig, a_marco, desde_marco


def chi_hat(R, n):
    """Coeficientes R^(mu) con R(s) = sum_mu R^(mu) w^{mu.s}; mismo orden de índices que los vectores."""
    return np.fft.fftn(R.reshape((K,) * n)).ravel() / K**n


def kt(modelo, n, eps, theta, maxit=400, tol=1e-13):
    hcl, v, q = termino(modelo)
    H = anillo(n, K, q, hcl + eps * v, theta)
    dig, a_marco, desde_marco = marco(n)
    mu = dig  # los caracteres tienen el mismo índice que las configuraciones
    borde = (mu != np.roll(mu, -1, axis=1)).sum(1)
    const = borde == 0
    g = np.zeros(K**n, complex)
    hist = []
    for it in range(maxit):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n  # G(s) = sum_mu g(mu) w^{mu.s}
        psi_m = np.exp(G)
        R = a_marco(H @ desde_marco(psi_m)) / psi_m
        Rh = chi_hat(R, n)
        paso = np.where(const, 0, Rh / np.where(const, 1, borde))
        g = g - paso
        err = np.abs(paso).max()
        hist.append(err)
        if not np.isfinite(err) or err > 1e3:
            return None
        if err < tol:
            break
    E = Rh[0]
    r = np.array([Rh[np.ravel_multi_index((c,) * n, (K,) * n)] for c in range(1, K)])
    psi = desde_marco(psi_m)
    res = np.linalg.norm(H @ psi - E * psi) / np.linalg.norm(psi)
    return E, r, res, it, err


def main():
    modelo, th, n = sys.argv[1], float(sys.argv[2]) * np.pi, int(sys.argv[3])
    hcl, v, q = termino(modelo)
    for eps in [float(x) for x in sys.argv[4:]]:
        out = kt(modelo, n, eps, th)
        if out is None:
            print(f"n={n} eps={eps}: diverge")
            continue
        E, r, res, it, err = out
        # niveles exactos más bajos del anillo completo, para comparar
        Hfull = anillo(n, K, q, hcl + eps * v, th)
        ev = np.linalg.eigvalsh(Hfull.toarray())[:4] if K**n <= 1000 else None
        extra = "" if ev is None else f"  exactos {ev[0]:.10f} {ev[1]:.10f} {ev[2]:.10f} | {ev[3]:.6f}"
        print(f"n={n} eps={eps}: it={it} paso={err:.1e} E={E.real:.10f}{E.imag:+.1e}i  |r_c|={np.abs(r).max():.2e}  "
              f"residuo={res:.2e}{extra}", flush=True)


if __name__ == "__main__":
    main()

"""Comprobación por ED del Newton con datos sigma_N (sensibilidad_celda3.tex, datos de otro anillo).

Para el anillo n y datos sigma_N (marginal del fundamental del anillo N a Theta = 0), resuelve por Newton las 15
ecuaciones
    grad e(x) = sigma_W + (p - 1) tau_W,   e(x) - x . grad e(x) = sigma_h + (p - 1) tau_h,
con e(x) = lambda_0(H_n(Theta) + sum_a x_a O_a(Theta)) / n por ED (sector Sz = 0, nivel de momento 0) y la hessiana por
diferencias del gradiente de Hellmann-Feynman. Imprime p*(Theta), p_E(Theta), delta = p_E - p* y su variación en Theta,
y comprueba la hipótesis del testigo: el nivel de momento 0 de H_{x*}(Theta) es el más bajo de los tres del multiplete
(y del sector).
Uso: python scripts/newton_datos_ed.py n N eps [eps ...]
"""
import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import serie_respuesta as SR  # noqa: E402
sys.argv = _argv

EBAR = -1 / 3


class Anillo:
    def __init__(self, n, eps, th):
        H0, T, Os, P = SR.operadores(n, th)
        self.n = n
        self.H = (eps * T + __import__('scipy.sparse', fromlist=['diags']).diags(H0)).tocsr()
        self.Os = Os
        codes, dig = SR.sector(n, 0)
        pw = 3 ** np.arange(n, dtype=np.int64)
        sh = np.roll(dig, -1, axis=0)
        self.perm = np.searchsorted(codes, (sh.astype(np.int64) * pw[:, None]).sum(0))

    def nivel(self, x):
        Hx = self.H + sum(xa * O for xa, O in zip(x, self.Os))
        w, v = eigsh(Hx, k=4, which='SA', tol=1e-14)
        o = np.argsort(w); w, v = w[o], v[:, o]
        # momento de cada vector: <v|T|v>
        mom = [np.vdot(v[:, i], v[self.perm, i]) for i in range(4)]
        i0 = int(np.argmax([np.real(m) for m in mom[:3]]))     # el de momento 0 (<T> ~ 1)
        psi = v[:, i0]
        g = np.array([np.real(np.vdot(psi, O @ psi)) for O in self.Os]) / self.n
        return w[i0] / self.n, g, w, i0, mom

    def grad(self, x):
        return self.nivel(x)[1]


def datos(N, eps):
    R = Anillo(N, eps, 0.0)
    e, g, *_ = R.nivel(np.zeros(len(R.Os)))
    return e, g


def resolver(A, sig_e, sig_W, tauW, tauh, x0=None):
    m = len(A.Os)
    x = np.zeros(m) if x0 is None else x0.copy()
    p = 1.0
    for it in range(30):
        e, g, *_ = A.nivel(x)
        F = np.concatenate([g - sig_W - (p - 1) * tauW, [e - x @ g - sig_e - (p - 1) * tauh]])
        if np.abs(F).max() < 1e-14:
            break
        h = 1e-6
        Hs = np.array([(A.grad(x + h * np.eye(m)[a]) - A.grad(x - h * np.eye(m)[a])) / (2 * h) for a in range(m)])
        Hs = (Hs + Hs.T) / 2
        Jm = np.zeros((m + 1, m + 1))
        Jm[:m, :m] = Hs; Jm[:m, m] = -tauW
        Jm[m, :m] = -(Hs @ x); Jm[m, m] = -tauh
        dz = np.linalg.solve(Jm, -F)
        x += dz[:m]; p += dz[m]
    return x, p, np.abs(F).max()


if __name__ == "__main__":
    n, N = int(sys.argv[1]), int(sys.argv[2])
    for eps in map(float, sys.argv[3:]):
        sig_e, sig_W = datos(N, eps)
        tauW = sig_W.copy(); tauW[:4] -= 1 / 9                 # W = [0, 4, 7, 8] diagonales primero
        tauh = sig_e - EBAR
        res = {}
        for th in (0.0, np.pi):
            A = Anillo(n, eps, th)
            e0 = A.nivel(np.zeros(len(A.Os)))[0]
            pE = 1 + (sig_e - e0) / (EBAR - sig_e)
            x, p, err = resolver(A, sig_e, sig_W, tauW, tauh)
            _, _, w, i0, mom = A.nivel(x)
            res[th] = (pE, p, x)
            print(f"eps={eps} Theta={th:.3f}: p_E={pE:.15f}  p*={p:.15f}  delta={pE - p:.3e}  |x*|={np.abs(x).max():.2e}  "
                  f"residuo={err:.0e}  nivel de momento 0 = {i0} (de 0..3), huecos {w[1]-w[0]:.2e} {w[2]-w[0]:.2e} {w[3]-w[0]:.2e}")
        dpE = res[np.pi][0] - res[0.0][0]
        dp = res[np.pi][1] - res[0.0][1]
        dd = (res[np.pi][0] - res[np.pi][1]) - (res[0.0][0] - res[0.0][1])
        print(f"  eps={eps}: p_E(pi)-p_E(0) = {dpE:.4e}, p*(pi)-p*(0) = {dp:.4e}, delta(pi)-delta(0) = {dd:.2e}", flush=True)

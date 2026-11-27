"""Constantes de la revisión del Lema Z12' (celda_k_notas.md, Sec. 7.13, paso 8).

Para un |eps| dado, busca (Mbar, x = Mbar|eps|, tau) que maximizan el número N de escalas que admite el
esquema de una escala (Lema Z12'') con pérdida de peso lambda = theta + tau + sigma_Z por escala, y da
n_* = l_0 4^(N-1), el tamaño hasta el que la robustez sale uniforme.

Notación (S+S-, k = 3): a = A = 7,11, a_nc = 4,44, b = 22,3.
  d = (4a + b)/Mbar;  A_w(x) = sum_{j>=1} (4j+3) e^{w(2j+1)} x^j;  gamma_w = 2 d A_w(x)  (densidad del vacío,
  con un factor 2 para las correcciones);  sigma_Z: menor raíz de 2k gamma_s <= s (Lema Z11);
  theta = 2 gamma_0 (masa por sitio del vestido de b_alpha);
  eta'(w) = (2(e^{4d}-1) + 2A|eps|e^{4d})(1 + 4 d A_w(x)) e^{4w}   (cota de Z10 por pared, peso w);
  q_L = eta'/(1-eta') coth(tau/2), con eta' en w_0 = sigma_0 + tau + sigma_Z; se pide q_L <= 1/4.
  Pesos: w_0 < ln(1/x)/4 (A_w y eta' finitas), sigma_m = sigma_0 - m lambda >= sigma_min,
  sigma_min = max(4 gamma_{sigma_Z}, theta + sigma_Z) + tau.
Uso: python scripts/constantes_z12.py eps [eps ...]
"""
import sys

import numpy as np

k, a, A, anc, b = 3, 7.11, 7.11, 4.44, 22.3


def Aw(w, x):
    r = np.exp(2 * w) * x
    if r >= 1:
        return np.inf
    return np.exp(w) * (4 * r / (1 - r) ** 2 + 3 * r / (1 - r))


def esquema(eps, Mb, tau):
    x = Mb * eps
    d = (4 * a + b) / Mb
    gam = lambda w: 2 * d * Aw(w, x)
    s = 0.0
    for _ in range(200):  # punto fijo de s = 2k gamma_s
        s2 = 2 * k * gam(s)
        if not np.isfinite(s2) or s2 > 1:
            return None
        if abs(s2 - s) < 1e-12:
            break
        s = s2
    sZ = s * 1.01 + 1e-9
    theta = 2 * gam(0.0)
    lam = theta + tau + sZ
    smin = max(4 * gam(sZ), theta + sZ) + tau
    eta = lambda w: (2 * (np.exp(4 * d) - 1) + 2 * A * eps * np.exp(4 * d)) * (1 + 4 * d * Aw(w, x)) * np.exp(4 * w)
    # mayor w_0 con q_L <= 1/4
    best = None
    for w0 in np.linspace(0, np.log(1 / x) / 4, 400)[1:-1]:
        e = eta(w0)
        if not np.isfinite(e) or e >= 1:
            break
        qL = e / (1 - e) / np.tanh(tau / 2)
        if qL <= 0.25:
            best = (w0, e, qL)
    if best is None:
        return None
    w0, e, qL = best
    s0 = w0 - tau - sZ
    if s0 < smin:
        return None
    N = int((s0 - smin) // lam) + 1
    return dict(x=x, d=d, sZ=sZ, theta=theta, lam=lam, w0=w0, s0=s0, smin=smin, eta=e, qL=qL, N=N)


def main():
    for eps in [float(v) for v in sys.argv[1:]]:
        mejor = None
        for x in np.geomspace(1e-4, 0.6, 60):
            Mb = x / eps
            if Mb < 1500:
                continue
            for tau in np.geomspace(1e-5, 0.2, 60):
                r = esquema(eps, Mb, tau)
                if r and (mejor is None or r["N"] > mejor[2]["N"]):
                    mejor = (Mb, tau, r)
        if mejor is None:
            print(f"eps={eps:.1e}: ninguna elección cumple las condiciones")
            continue
        Mb, tau, r = mejor
        l0 = max(16, int(np.ceil(np.log(16) / (4 * tau))))  # tau l_1 >= ln 16
        print(f"eps={eps:.1e}: Mbar={Mb:.3g} x={r['x']:.3g} tau={tau:.2e} d={r['d']:.2e} sigma_Z={r['sZ']:.3g} "
              f"theta={r['theta']:.3g} lambda={r['lam']:.3g} w0={r['w0']:.3g} sigma0={r['s0']:.3g} "
              f"sigma_min={r['smin']:.3g} eta'={r['eta']:.2e} q_L={r['qL']:.3f} N={r['N']} l0={l0} "
              f"log10 n_* = {np.log10(l0) + (r['N'] - 1) * np.log10(4):.1f}")


if __name__ == "__main__":
    main()

"""Cota tipo Kohn y rigidez del testigo óptimo en anillos de n = 6 a 20.

Artículo: quinto trabajo (holonomía y rigidez), sección de la cota tipo Kohn;
usa la Proposición 4 de anillos (reducción al XXZ retorcido) y las secciones
5.3 (cúspide y ventana) y 6.2 (rigidez) de anillos. La demostración de la
cota está en el docstring de `holonomy.rigidez`.

En el punto liso Theta0 = Phi_s (0 para n par, pi para n impar) el testigo
óptimo tiene flujo Phi0 = Theta0 y anisotropía Delta0. Con
den0 = Delta0 + 2, e0 = eps_n(Delta0, Phi0) y la rigidez del testigo
D_W = -n^2 d^2 eps_n/dPhi^2 (Delta0, Phi0), la fórmula de la envolvente
(F_{Delta Theta} = F_{Delta Phi} = 0 por la simetría Phi -> 2 Phi_s - Phi;
n^2 F_{Theta Theta} = a, n^2 F_{Phi Theta} = -a, n^2 F_{Phi Phi} = a + b) da
    n^2 v''(Theta0) = a b/(a + b) = b dPhi*/dTheta,
    a = 8 e0/den0^2,  b = -4 D_W/den0,  dPhi*/dTheta = a/(a + b),
mientras que la cota de Kohn (el testigo girado sigue el giro con
dPhi/dTheta = 1) solo da n^2 v'' <= b. La curvatura es la de Kohn por la
velocidad con que el flujo óptimo sigue a la holonomía: a + b > 0 es la
condición de mínimo en Phi y es pequeña porque la ventana de desanclaje es
estrecha (psi/pi ~ (a + b)/a si Phi* fuese lineal en Theta hasta la cúspide;
es la relación psi/pi ~ 1 - 3 D/(2 eps) de anillos 6.3 con D = D_W).

Comprueba, para cada n:
  1. Phi0 = Theta0 es el mínimo en el flujo (rejilla de 7 flujos en [0, pi])
     y a + b > 0 (mínimo local);
  2. eps_n(Delta0, Phi0 + h) = eps_n(Delta0, Phi0 - h) (simetría) y D_W > 0;
  3. n^2 v'' <= b (la cota de Kohn a segundo orden); para n <= 10, la
     curvatura de la envolvente frente a diferencias finitas de v_free
     (Richardson con h = psi/32 y psi/64, error relativo < 1e-5);
  4. para Theta1 = Theta0 +- pi/2, +- pi (los dos signos dan la misma
     Theta1 reducida a [0, pi] y, por la simetría de eps, la misma cota): la
     cota de Kohn 4 eps_n(Delta0, Phi0 + Theta1 - Theta0)/den0 >= v(Theta1),
     con eps reevaluado sin arranque en caliente en todos los sectores de
     magnetización (diferencia con el eps de la minimización < 1e-10), y la
     cota inversa (testigo óptimo de Theta1 girado hacia Theta0) >= v(Theta0).
     Así
        v0 - cota <= v0 - v1 <= cota_inv - v1.
     v(Theta1) es el mínimo de v_at_flux sobre la rejilla de 7 flujos en
     [0, pi], que contiene la cúspide: si fuese mayor que el verdadero v, la
     comprobación v1 <= cota sería más exigente, no menos; el objetivo en el
     mínimo se reevalúa también sin arranque en caliente.

Escribe data/rigidez_anillos.csv (una fila por n y giro), con n^2 (v0 - v1)
entre n^2 (v0 - cota) y n^2 (cota_inv - v1), sus cocientes con la caída, y,
para comparar, n^2 por la cota de orden 1/n que da el Lema 5(i) de anillos
(|eps(Phi) - eps(Phi')| <= 1/n), 4n/den0.

Uso: python scripts/rigidez_anillos.py   (304 s y 0,54 GB de pico, medidos
con /usr/bin/time -f "%M KB %e s")
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import time

import numpy as np
import scipy.sparse.linalg as spla

from holonomy import dual, xxz
from holonomy.report import read_csv, write_csv

NS = range(6, 21)
XATOL = 1e-7          # en Delta: el error de v es cuadrático, ~1e-14
TOL = 1e-10
FLUJOS_REJILLA = np.linspace(0.0, np.pi, 7)
GIROS = (np.pi / 2, np.pi)  # los giros -dT dan la misma Theta1 reducida
N_FD = 10             # diferencias finitas de v_free hasta este n (diagonalización densa)


def eps_frio(n, Delta, Phi):
    """eps_n en todos los sectores, sin arranque en caliente (Lanczos con
    vector inicial aleatorio de ARPACK y dos autovalores por sector)."""
    E = n * Delta / 4  # ferromagnético (M = 0, n)
    for M in range(1, n):
        H = xxz.hamiltonian(n, M, Delta, Phi)
        if H.shape[0] <= xxz.DENSE_MAX:
            w = np.linalg.eigvalsh(H.toarray())[:1]
        else:
            w = spla.eigsh(H, k=2, which="SA", tol=1e-13)[0]
        E = min(E, float(np.min(w)))
    return -E / n


def reducir(T):
    """Theta en [0, pi] con la misma v (v es par y 2 pi-periódica)."""
    return abs((T + np.pi) % (2 * np.pi) - np.pi)


def v_rejilla(n, T, flujos):
    """min sobre los flujos dados de v_at_flux. Devuelve (v, Phi, Delta)."""
    mejor = None
    for P in flujos:
        v, D = dual.v_at_flux(xxz.eps, n, T, P, xatol=XATOL)
        if mejor is None or v < mejor[0]:
            mejor = (v, P, D)
    return mejor


def segunda_derivada(f, x0, h):
    """f'' por diferencias centradas con Richardson (h y h/2)."""
    f0 = f(x0)
    d = [(f(x0 + s) + f(x0 - s) - 2 * f0) / s**2 for s in (h, h / 2)]
    return (4 * d[1] - d[0]) / 3


def main():
    ck = Checker("rigidez_anillos")
    psi_tabla = {int(r["n"]): r["psi_over_pi"] for r in read_csv("edges_ed.csv")}
    filas = []
    for n in NS:
        t_n = time.time()
        T0 = dual.smooth_flux(n)
        P0 = T0
        v0, D0 = dual.v_at_flux(xxz.eps, n, T0, P0, xatol=XATOL)
        den0 = D0 + 2.0
        # 1. optimalidad del flujo
        v_rej = v_rejilla(n, T0, FLUJOS_REJILLA)
        ck.check(v_rej[0] >= v0 - TOL,
                 f"n={n}: Phi0 = Theta0 = {T0:.4f} mínimo en la rejilla de flujos "
                 f"(v0 = {v0:.10f}, mejor de la rejilla {v_rej[0]:.10f} en Phi = {v_rej[1]:.3f})")
        # 2. rigidez del testigo
        e0 = eps_frio(n, D0, P0)
        ck.check(abs(e0 - xxz.eps(n, D0, P0)) < TOL, f"n={n}: eps en (Delta0, Phi0) sin arranque en caliente")

        def e(P, D0=D0, n=n):
            return xxz.eps(n, D0, P)

        h = 0.1
        ck.check(abs(e(P0 + h) - e(P0 - h)) < TOL, f"n={n}: eps(Phi0 + h) = eps(Phi0 - h)")
        DW = -n**2 * segunda_derivada(e, P0, h)
        a = 8 * e0 / den0**2
        b = -4 * DW / den0
        ck.check(DW > 0 and a + b > 0,
                 f"n={n}: D_W = {DW:.6f} > 0 y a + b = {a + b:.3e} > 0 (Delta0 = {D0:.6f})")
        curv = a * b / (a + b)
        ck.check(curv <= b + TOL,
                 f"n={n}: n^2 v'' = {curv:.4f} <= cota de Kohn b = -4 D_W/den0 = {b:.4f} "
                 f"(cociente = dPhi*/dTheta = {curv / b:.2f})")
        psi_est = (a + b) / a
        ck.info(f"n={n}: psi/pi estimado (a + b)/a = {psi_est:.5f}; tabla de anillos {psi_tabla[n]:.5f}")
        curv_fd = np.nan
        if n <= N_FD:
            hT = np.pi * psi_tabla[n] / 32
            sgn = 1.0 if T0 == 0 else -1.0
            v_h = [dual.v_free(xxz.eps, n, T0 + sgn * s)[0] for s in (hT, hT / 2)]
            d = [(vv - v0) / s**2 for vv, s in zip(v_h, (hT, hT / 2))]
            curv_fd = n**2 * 2 * (4 * d[1] - d[0]) / 3
            ck.check(abs(curv_fd / curv - 1) < 1e-5,
                     f"n={n}: n^2 v'' por diferencias finitas de v_free = {curv_fd:.4f} "
                     f"(envolvente {curv:.4f}, h = {hT:.2e})")
        # 3. cota de Kohn al girar
        for dT in GIROS:
            T1 = reducir(T0 + dT)
            # +-dT: misma Theta1 reducida y, por la simetría de eps, la misma cota
            menos = 4 * xxz.eps(n, D0, P0 - dT) / den0
            cota = 4 * eps_frio(n, D0, P0 + dT) / den0
            ck.check(abs(cota - 4 * xxz.eps(n, D0, P0 + dT) / den0) < 4 * TOL
                     and abs(cota - menos) < 4 * TOL,
                     f"n={n}, giro +-{dT / np.pi:.1f} pi: eps del testigo girado sin arranque en "
                     f"caliente y cotas de +- iguales")
            v1, P1, D1 = v_rejilla(n, T1, FLUJOS_REJILLA)
            ck.check(v1 <= cota + TOL,
                     f"n={n}, Theta1 = {T1 / np.pi:.1f} pi: v1 = {v1:.10f} <= cota = {cota:.10f}")
            den1 = D1 + 2 * np.cos((P1 - T1) / n)
            # v1 sin arranque en caliente: un eps subestimado haría v1 <= cota más fácil
            ck.check(abs(4 * eps_frio(n, D1, P1) / den1 - v1) < 4 * TOL,
                     f"n={n}, Theta1 = {T1 / np.pi:.1f} pi: v1 sin arranque en caliente")
            cota_inv = 4 * eps_frio(n, D1, P1 + T0 - T1) / den1
            ck.check(v0 <= cota_inv + TOL,
                     f"n={n}, Theta1 = {T1 / np.pi:.1f} pi: cota inversa {cota_inv:.10f} >= v0 = {v0:.10f}")
            caida = v0 - v1
            fila = dict(n=n, Theta0=T0, Theta1=T1, giro=dT, Delta0=D0, Phi1=P1, Delta1=D1,
                        v0=v0, v1=v1, cota=cota, cota_inv=cota_inv,
                        D_W=DW, a=a, b=b, curv_envolvente=curv, curv_fd=curv_fd,
                        n2_caida=n**2 * caida, n2_cota_inf=n**2 * (v0 - cota),
                        n2_cota_sup=n**2 * (cota_inv - v1),
                        cociente_inf=(v0 - cota) / caida, cociente_sup=(cota_inv - v1) / caida,
                        n2_cota_lema5=4 * n / den0)
            filas.append(fila)
            ck.info(f"n={n}, giro {dT / np.pi:.1f} pi: n^2 (v0 - v1) = {fila['n2_caida']:.5f} entre "
                    f"{fila['n2_cota_inf']:.5f} (Kohn) y {fila['n2_cota_sup']:.5f} (inversa); "
                    f"cocientes {fila['cociente_inf']:.4f} y {fila['cociente_sup']:.4f}; "
                    f"Phi1 = {P1:.4f}")
        ck.info(f"n={n}: {time.time() - t_n:.1f} s")
    cab = list(filas[0])
    write_csv("rigidez_anillos.csv", cab, [[f[k] for k in cab] for f in filas])
    ck.finish()


if __name__ == "__main__":
    main()

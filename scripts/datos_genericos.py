"""Datos de llenado 1/3 con sigma bien condicionada: ¿pasa la regla del testigo a los datos? (exploración).

Término h0 = -P_asc + eps K + delta D con carga q = (1, 0, 0) (llenado 1/3), K una combinación aleatoria de TODOS
los operadores U(1)-invariantes de dos sitios fuera de la diagonal (incluida la conversión entre las especies
neutras 1 y 2, que da peso a los pares 11 y 22) y D una diagonal aleatoria. Los datos son la marginal del
fundamental del anillo de NREF sitios (3 | NREF), en el espacio completo. Para cada n se dan:
  - el autovalor mínimo de sigma (lo bien condicionada que está);
  - la sensibilidad de los datos con su horquilla certificada y la del testigo fijo w0 = -h0;
  - alpha, el peso del salto de w0 en el testigo óptimo en Theta = 0.

Uso: OMP_NUM_THREADS=1 python scripts/datos_genericos.py semilla eps delta n [n ...]   (NTHETA, NREF en el entorno)
"""
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from dual_celda import anillo, base_u1, dual, lmax

Q = [1, 0, 0]


def termino(rng, eps, delta):
    h = np.zeros((9, 9), complex)
    for a in range(3):
        h[3 * a + (a + 1) % 3, 3 * a + (a + 1) % 3] -= 1
    h += delta * np.diag(rng.normal(size=9))
    for B in base_u1(Q):
        if np.count_nonzero(np.diag(B)) == 0:  # solo los elementos fuera de la diagonal
            h += eps * rng.normal() * B / np.sqrt(2)
    return h


def datos(h, nref):
    H = anillo(nref, 3, Q, h, 0.0)
    w, v = eigsh(H, k=1, which="SA", tol=1e-13)
    psi = v[:, 0]
    carga = np.vdot(psi, (np.arange(3**nref)[:, None] // 3 ** np.arange(nref)[None, :] % 3 == 0).sum(1) * psi).real
    T = psi.reshape(3 ** (nref - 2), 3, 3)
    rho = np.einsum("rba,rdc->abcd", T, T.conj()).reshape(9, 9)
    return rho, carga


def main():
    semilla, eps, delta = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    ns = [int(x) for x in sys.argv[4:]]
    nth = int(os.environ.get("NTHETA", 8))
    nref = int(os.environ.get("NREF", 12))
    h = termino(np.random.default_rng(semilla), eps, delta)
    sigma, carga = datos(h, nref)
    tau = sigma - np.eye(9) / 9
    w0 = -h
    g = np.real(np.trace(w0 @ tau))
    w0n = w0 / g
    off0 = w0n - np.diag(np.diag(w0n))
    print(f"semilla={semilla} eps={eps} delta={delta} NREF={nref}: partículas en el fundamental {carga:.6f} "
          f"(esperado {nref // 3}); autovalores de sigma: "
          + " ".join(f"{x:.1e}" for x in np.linalg.eigvalsh(sigma)), flush=True)
    thetas = np.linspace(0, 2 * np.pi, nth + 1)[:-1]
    for n in ns:
        t0 = time.time()
        P, L, F = [], [], []
        alpha = None
        for t in thetas:
            lam_max, _ = lmax(anillo(n, 3, Q, w0, t))
            F.append((lam_max - n * np.trace(w0).real / 9) / (n * g))
            sup, w, inf = dual(n, 3, Q, tau, t, w0=w0, metodo="suave")
            P.append(sup)
            L.append(inf)
            if alpha is None:
                w = w - np.trace(w) / 9 * np.eye(9)
                off = w - np.diag(np.diag(w))
                alpha = np.real(np.vdot(off0, off)) / np.real(np.vdot(off0, off0))
        P, L, F = np.array(P), np.array(L), np.array(F)
        sP, sF = P.max() - P.min(), F.max() - F.min()
        s_inf, s_sup = max(L.max() - P.min(), 0.0), P.max() - L.min()
        print(f"n={n} (n%3={n % 3}): p en [{P.min():.8f}, {P.max():.8f}]  s_datos={sP:.3e} "
              f"(certificada en [{s_inf:.2e}, {s_sup:.2e}])  s_testigo_fijo={sF:.3e}  cociente={sP / sF:.3f}  "
              f"alpha(Theta=0)={alpha:.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

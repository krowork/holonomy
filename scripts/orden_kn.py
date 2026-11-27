"""Comprobación del teorema con k | n (celda_k_notas.md, Sec. 7): órdenes en eps (exploración).

El teorema predice, para k | n y eps -> 0 con n fijo:
  - la dependencia en Theta del multiplete (y del fundamental) empieza como pronto en el orden ceil(n/2);
  - el desdoblamiento del multiplete de k niveles, también;
  - E_0(n)/n - E_0(2n)/(2n) es O(eps^ceil(n/2)).
Se mide el exponente efectivo log(X(2 eps)/X(eps))/log 2 en los dos modelos de celda 3:
  entero: espín 1 quiral (celda_k_flujo.py), sector S^z = 0;
  llenado: llenado 1/3 (celda_k_llenado.py), con el intercambio neutro de amplitud MU*eps (MU en el entorno).

Uso: OMP_NUM_THREADS=4 python scripts/orden_kn.py modelo n [eps ...]
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import celda_k_flujo as ent
import celda_k_llenado as ll

MU = float(os.environ.get("MU", 1.0))


def hamiltoniano(modelo, n, eps, theta, cache={}):
    key = (modelo, n)
    if key not in cache:
        cache[key] = ent.build(n, 0) if modelo == "entero" else ll.build(n, n // 3)
    datos = cache[key]
    ph = np.exp(1j * theta / n)
    if modelo == "entero":
        N, diag, r, c = datos
        A = sp.coo_matrix((2 * eps * ph * np.ones(r.size), (r, c)), shape=(N, N))
    else:
        N, diag, r, c, r2, c2 = datos
        A = sp.coo_matrix((np.r_[eps * ph * np.ones(r.size), MU * eps * np.ones(r2.size)],
                           (np.r_[r, r2], np.r_[c, c2])), shape=(N, N))
    A = A.tocsr()
    return A + A.getH() + sp.diags(diag)


def bajos(H, k=4):
    if H.shape[0] <= 6000:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(eigsh(H, k=k, which="SA", tol=1e-14)[0])


def main():
    modelo, n = sys.argv[1], int(sys.argv[2])
    epss = [float(x) for x in sys.argv[3:]] or [0.05, 0.1, 0.2]
    thetas = np.linspace(0, 2 * np.pi, 17)[:-1]
    print(f"modelo={modelo} n={n} MU={MU}: predicción, orden >= ceil(n/2) = {-(-n // 2)}")
    filas = []
    for eps in epss:
        L = np.array([bajos(hamiltoniano(modelo, n, eps, t)) for t in thetas])  # (Theta, nivel)
        R = L[:, 0].max() - L[:, 0].min()          # respuesta al flujo del fundamental
        Rm = (L[:, :3].mean(1)).max() - (L[:, :3].mean(1)).min()  # de la media del multiplete
        split = (L[:, 2] - L[:, 0]).max()          # anchura del multiplete
        gap = (L[:, 3] - L[:, 2]).min()            # separación al resto
        e_n = L[0, 0] / n
        e_2n = bajos(hamiltoniano(modelo, 2 * n, eps, 0.0), 1)[0] / (2 * n) if 2 * n <= 12 else np.nan
        filas.append((eps, R, Rm, split, gap, abs(e_n - e_2n)))
        print(f"eps={eps:.3f}: R={R:.3e} R(media)={Rm:.3e} anchura={split:.3e} hueco={gap:.3f} "
              f"|e_n-e_2n|={abs(e_n - e_2n):.3e}", flush=True)
    for (e1, *x1), (e2, *x2) in zip(filas, filas[1:]):
        p = [np.log(b / a) / np.log(e2 / e1) if a > 0 and b > 0 else np.nan for a, b in zip(x1, x2)]
        print(f"exponente {e1}->{e2}: R {p[0]:.2f}  R(media) {p[1]:.2f}  anchura {p[2]:.2f}  "
              f"|e_n-e_2n| {p[4]:.2f}")


if __name__ == "__main__":
    main()

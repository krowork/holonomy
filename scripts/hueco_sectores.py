"""Hipótesis (G) para el intercambio SU(3) (celda_k_notas.md, Sec. 7.9): fundamental de cada sector de contenido.

H(0) = -P_asc + eps * sum_{a != b} |b a><a b| en el anillo de n sitios (3 | n). El intercambio conserva el número de
cada especie, así que H es diagonal por bloques en los sectores de contenido (c_0, c_1, c_2). Se da el hueco
min_{c != c*} E_c - E_{c*}, con c* = (n/3, n/3, n/3), y el sector que lo realiza.
Uso: python scripts/hueco_sectores.py n eps [eps ...]
"""
import sys
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from dual_celda import anillo
from kt_zk import termino


def main():
    n = int(sys.argv[1])
    hcl, v, q = termino("llenado")
    dig = (np.arange(3**n)[:, None] // 3 ** np.arange(n)[None, :]) % 3
    cnt = np.stack([(dig == a).sum(1) for a in range(3)], 1)
    claves = cnt @ np.array([1, n + 1, (n + 1) ** 2])
    Hcl = anillo(n, 3, q, hcl, 0.0).tocsr()
    V = anillo(n, 3, q, v, 0.0).tocsr()
    cstar = (n // 3,) * 3
    for eps in [float(x) for x in sys.argv[2:]]:
        H = (Hcl + eps * V).tocsr()
        E = {}
        for c in product(range(n + 1), repeat=3):
            if sum(c) != n or tuple(sorted(c)) != tuple(c) and False:
                continue
            idx = np.nonzero(claves == c[0] + c[1] * (n + 1) + c[2] * (n + 1) ** 2)[0]
            if idx.size == 0:
                continue
            Hs = H[idx][:, idx]
            E[c] = (np.linalg.eigvalsh(Hs.toarray())[0] if idx.size < 2500
                    else eigsh(Hs, k=1, which="SA", tol=1e-10)[0][0])
        otros = {c: e for c, e in E.items() if c != cstar}
        cmin = min(otros, key=otros.get)
        print(f"n={n} eps={eps:+.2f}: E_c*={E[cstar]:.6f}  hueco={otros[cmin] - E[cstar]:.4f} en el sector {cmin}  "
              f"(hueco/n del sector más lejano {max(otros.values()) / n - E[cstar] / n:.3f})", flush=True)


if __name__ == "__main__":
    main()

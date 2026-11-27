"""El defecto de carga 1/3 del modelo de llenado 1/3 con n = 2 mod 3 (exploración).

Sector con m+1 partículas, n = 3m+2. Calcula el cúmulo bajo del espectro, los estados clásicos de un
defecto (energía clásica -(n-1)) y el hamiltoniano efectivo de des Cloizeaux sobre ellos, con sus
amplitudes de salto agrupadas por especie y desplazamiento.

Uso: python scripts/defecto_tercio.py n eps [eps ...]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

from celda_k_llenado import LAM, build  # noqa: E402


def digits(code, n):
    return [(code // 3**i) % 3 for i in range(n)]


def main():
    n = int(sys.argv[1])
    epss = [float(x) for x in sys.argv[2:]] or [0.05, 0.1]
    Np = (n + 2) // 3
    N, diag, r, c, r2, c2 = build(n, Np)
    # códigos del sector, para identificar los estados clásicos
    codes = np.arange(3**n, dtype=np.int64)
    dig = np.array([(codes // 3**i) % 3 for i in range(n)])
    codes = codes[(dig == 0).sum(0) == Np]
    cl = np.nonzero(diag == -(n - 1))[0]
    K = cl.size
    info = []
    for a in cl:
        s = digits(codes[a], n)
        j = [i for i in range(n) if (s[(i + 1) % n] - s[i]) % 3 != 1]
        assert len(j) == 1
        j = j[0]
        info.append((s[(j + 1) % n], j))  # especie b = valor tras el defecto, enlace j
    print(f"n={n} Np={Np} dim={N} estados clásicos de un defecto: {K}  especies:",
          sorted({b for b, _ in info}))
    for eps in epss:
        A = sp.coo_matrix((np.r_[eps * np.ones(r.size), LAM * np.ones(r2.size)], (np.r_[r, r2], np.r_[c, c2])),
                          shape=(N, N)).tocsr()
        H = A + A.getH() + sp.diags(diag)
        k = min(K + 4, N - 2)
        if N < 3000:
            w, v = np.linalg.eigh(H.toarray())
            w, v = w[:k], v[:, :k]
        else:
            w, v = eigsh(H, k=k, which="SA", tol=1e-13, ncv=max(3 * k, 40), maxiter=10**6)
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        print(f" eps={eps}: E0={w[0]:.8f}  anchura del cúmulo de {K}: {w[K - 1] - w[0]:.3e}  "
              f"hueco al siguiente: {w[K] - w[K - 1]:.3f}")
        O = v[cl, :K]  # proyección de los K autovectores bajos sobre los estados clásicos
        S = O @ O.conj().T
        ev, U = np.linalg.eigh(S)
        Sm = U @ np.diag(ev**-0.5) @ U.conj().T
        Heff = Sm @ O @ np.diag(w[:K]) @ O.conj().T @ Sm
        print(f"   peso de los estados clásicos en el cúmulo: min autovalor de S = {ev.min():.4f}")
        groups = {}
        for x in range(K):
            for y in range(K):
                if x == y:
                    continue
                (b1, j1), (b2, j2) = info[x], info[y]
                d = (j2 - j1) % n
                d = min(d, n - d)
                key = (b1, b2, d)
                groups[key] = max(groups.get(key, 0), abs(Heff[x, y]))
        for key in sorted(groups, key=lambda k: -groups[k])[:12]:
            b1, b2, d = key
            print(f"   especies {b1}->{b2}, desplazamiento {d:2d}: |t| = {groups[key]:.3e}")
        dg = np.real(np.diag(Heff))
        for b in sorted({b for b, _ in info}):
            sel = [i for i in range(K) if info[i][0] == b]
            print(f"   especie {b}: energía propia {dg[sel].mean():.8f} (dispersión {np.ptp(dg[sel]):.1e})")


if __name__ == "__main__":
    main()

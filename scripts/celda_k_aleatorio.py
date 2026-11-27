"""Orden de celda 3 con carga entera y perturbación U(1)-invariante aleatoria (exploración).

Espín 1, s in {0,1,2}, carga q = s - 1 (S^z). El término de enlace es
    h = -P_asc + eps * K + delta * D,
con P_asc el proyector sobre (s, s+1 mod 3), K una matriz hermítica aleatoria de dos sitios con solo los elementos
fuera de la diagonal que conservan la carga total (5 amplitudes complejas) y D diagonal aleatoria (9 energías).
El giro va en el segundo sitio: <a'b'|h^Theta|ab> = e^{-i phi (q_b' - q_b)} <a'b'|h|ab>, phi = Theta/n.

Para cada muestra y cada n, el guion da la carga S^z del sector fundamental y R_n = max_Theta E_0 - min_Theta E_0.
Con 3 no divisor de n, S^z es la carga del defecto que fuerza el anillo (la celda tiene carga 0).

Uso: python scripts/celda_k_aleatorio.py [muestras] [eps] [delta] [semilla]   (NS en el entorno: lista de n)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

Q = np.array([-1, 0, 1])


def termino(rng, eps, delta):
    """h de dos sitios (9x9, índice 3a+b) y la carga transportada al segundo sitio por cada elemento."""
    h = np.zeros((9, 9), complex)
    for a in range(3):
        h[3 * a + (a + 1) % 3, 3 * a + (a + 1) % 3] -= 1
    h += delta * np.diag(rng.normal(size=9))
    for x in range(9):
        for y in range(x + 1, 9):
            ax, bx, ay, by = x // 3, x % 3, y // 3, y % 3
            if Q[ax] + Q[bx] == Q[ay] + Q[by]:
                z = eps * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
                h[x, y] += z
                h[y, x] += np.conj(z)
    return h


class Anillo:
    def __init__(self, n):
        self.n = n
        codes = np.arange(3**n, dtype=np.int64)
        dig = np.array([(codes // 3**i) % 3 for i in range(n)], dtype=np.int8)
        self.sec = {}
        for Sz in range(-2, 3):
            keep = dig.sum(0) - n == Sz
            if keep.any():
                self.sec[Sz] = (codes[keep], dig[:, keep])

    def hamiltoniano(self, h, Sz, theta):
        n = self.n
        codes, dig = self.sec[Sz]
        pw = 3 ** np.arange(n, dtype=np.int64)
        phi = theta / n
        rows, cols, vals = [], [], []
        for i in range(n):
            j = (i + 1) % n
            idx = 3 * dig[i].astype(np.int64) + dig[j]
            for x in range(9):  # estado de entrada (a, b) en (i, j)
                sel = np.nonzero(idx == x)[0]
                if sel.size == 0:
                    continue
                a, b = divmod(x, 3)
                for y in range(9):
                    v = h[y, x]
                    if v == 0:
                        continue
                    a2, b2 = divmod(y, 3)
                    v = v * np.exp(-1j * phi * (Q[b2] - Q[b]))
                    new = codes[sel] + (a2 - a) * pw[i] + (b2 - b) * pw[j]
                    rows.append(np.searchsorted(codes, new) if y != x else sel)
                    cols.append(sel)
                    vals.append(np.full(sel.size, v))
        N = codes.size
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, N)).tocsr()

    def e0(self, h, Sz, theta):
        H = self.hamiltoniano(h, Sz, theta)
        if H.shape[0] < 600:
            return np.linalg.eigvalsh(H.toarray())[0]
        return eigsh(H, k=1, which="SA", tol=1e-12)[0][0]


def main():
    ns_muestras = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    semilla = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ns = [int(x) for x in os.environ.get("NS", "6,7,8,9,10").split(",")]
    anillos = {n: Anillo(n) for n in ns}
    thetas = np.linspace(0, 2 * np.pi, 9)[:-1]
    rng = np.random.default_rng(semilla)
    print(f"eps={eps} delta={delta} semilla={semilla}; por n: carga del fundamental en Theta=0 | R_n | n^2 R_n")
    for s in range(ns_muestras):
        h = termino(rng, eps, delta)
        assert np.allclose(h, h.conj().T)
        partes = []
        for n in ns:
            A = anillos[n]
            E00 = {Sz: A.e0(h, Sz, 0.0) for Sz in A.sec}
            emin = min(E00.values())
            cand = [Sz for Sz in E00 if E00[Sz] < emin + 0.3]  # sectores que pueden ganar en algún Theta
            E = np.array([min(A.e0(h, Sz, t) if t else E00[Sz] for Sz in cand) for t in thetas])
            R = E.max() - E.min()
            gana = min(E00, key=E00.get)
            partes.append(f"n={n}: Sz={gana:+d} R={R:.2e} n2R={n * n * R:.4f}")
        print(f"[{s}] " + " | ".join(partes), flush=True)


if __name__ == "__main__":
    main()

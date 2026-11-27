"""Serie de Taylor de K1(eps) en coma flotante con vectores dispersos (celda_k_notas.md, 9.21).

Mismo objeto que k1_racional.py (Rayleigh-Schrödinger sobre el patrón 0,1,2,...), pero sobre el conjunto de
configuraciones a <= K+2 saltos del patrón, con H0, T y los O_a como matrices dispersas. Sirve para órdenes
altos (K = 8), donde las fracciones son demasiado lentas. Se valida contra los coeficientes exactos de orden <= 6.
Uso: python scripts/k1_serie_rapida.py n K [salida.npz]
"""
import sys

import numpy as np
import os
DEBUG = os.environ.get("DEBUG")
import scipy.sparse as sp

from k1_racional import base


def codigos_patrones(n, pw):
    return np.array(sorted(int(np.dot(np.array([(j + c) % 3 for j in range(n)]), pw)) for c in range(3)),
                    dtype=np.int64)


def conjunto(n, D, tres=False):
    """Códigos (base 3) de las configuraciones a <= D saltos del patrón (o de los tres patrones si tres)."""
    pw = 3 ** np.arange(n, dtype=np.int64)
    p = np.array([j % 3 for j in range(n)])
    c0 = codigos_patrones(n, pw) if tres else np.array([int(np.dot(p, pw))], dtype=np.int64)
    niveles = [c0]
    vistos = c0.copy()
    for d in range(D):
        cur = niveles[-1]
        dig = (cur[:, None] // pw[None, :]) % 3
        nuevos = []
        for j in range(n):
            k = (j + 1) % n
            a, b = dig[:, j], dig[:, k]
            m = (a < 2) & (b > 0)
            nuevos.append(cur[m] + pw[j] - pw[k])
            m = (a > 0) & (b < 2)
            nuevos.append(cur[m] - pw[j] + pw[k])
        nv = np.unique(np.concatenate(nuevos))
        nv = nv[~np.isin(nv, vistos, assume_unique=True)]
        niveles.append(nv)
        vistos = np.union1d(vistos, nv)
        print(f"  profundidad {d + 1}: {nv.size} nuevas, total {vistos.size}", flush=True)
    return vistos, pw


def matrices(n, codes, pw):
    N = codes.size
    dig = (codes[:, None] // pw[None, :]) % 3
    H0 = -np.sum((np.roll(dig, -1, axis=1) - dig) % 3 == 1, axis=1).astype(float)
    filas, cols = [], []
    for j in range(n):
        k = (j + 1) % n
        a, b = dig[:, j], dig[:, k]
        m = (a < 2) & (b > 0)
        new = codes[m] + pw[j] - pw[k]
        idx = np.searchsorted(codes, new)
        ok = (idx < N) & (codes[np.minimum(idx, N - 1)] == new)
        src = np.nonzero(m)[0][ok]
        filas.append(idx[ok]); cols.append(src)
    r = np.concatenate(filas); c = np.concatenate(cols)
    Tf = sp.coo_matrix((2.0 * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    T = (Tf + Tf.T).tocsr()
    B = base()
    Os = []
    for o in B:
        filas, cols, vals = [], [], []
        for j in range(n):
            k = (j + 1) % n
            par = dig[:, j] * 3 + dig[:, k]
            acciones = []
            if o[0] == "d":
                acciones.append((o[1], o[1], 1.0))
            else:
                x, y = o[1], o[2]
                sg = 1.0 if o[0] == "re" else -1.0  # 'im' se guarda sin el factor i (9.20)
                acciones += [(y, x, 1.0), (x, y, sg)]  # (par origen, par destino, coef)
            for (src_par, dst_par, coef) in acciones:
                m = par == src_par
                new = codes[m] + (dst_par // 3 - dig[m, j]) * pw[j] + (dst_par % 3 - dig[m, k]) * pw[k]
                idx = np.searchsorted(codes, new)
                ok = (idx < N) & (codes[np.minimum(idx, N - 1)] == new)
                filas.append(idx[ok]); cols.append(np.nonzero(m)[0][ok]); vals.append(np.full(ok.sum(), coef))
        Os.append(sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                                shape=(N, N)).tocsr())
    return H0, T, Os, B


def serie(n, K):
    codes, pw = conjunto(n, K + 2)
    H0, T, Os, B = matrices(n, codes, pw)
    p = np.searchsorted(codes, int(np.dot(np.array([j % 3 for j in range(n)]), pw)))
    E0 = H0[p]
    den = E0 - H0
    deg = den == 0  # configuraciones degeneradas con el patrón (a >= 2n/3 saltos): no deben alcanzarse
    den[deg] = 1.0
    psi = [np.zeros(codes.size)]
    psi[0][p] = 1.0
    E = [E0]
    for m in range(1, K + 2):
        rhs = T @ psi[m - 1]
        E.append(rhs[p])
        if m == K + 1:
            break
        for k in range(1, m + 1):
            rhs = rhs - E[k] * psi[m - k]
        v = rhs / den
        v[p] = 0.0
        fuera = np.abs(v[deg]).max() if deg.sum() > 1 else 0.0
        if fuera > 0:
            raise RuntimeError(f"orden {m}: la serie alcanza otra configuración degenerada (n demasiado pequeño)")
        psi.append(v)
    E = np.array(E[:K + 1])
    Psi = np.array(psi)  # (K+1) x N
    norma = np.array([sum(Psi[i] @ Psi[m - i] for i in range(m + 1)) for m in range(K + 1)])
    # inversa de la serie de la norma
    inv = np.zeros(K + 1); inv[0] = 1 / norma[0]
    for m in range(1, K + 1):
        inv[m] = -sum(norma[i] * inv[m - i] for i in range(1, m + 1)) / norma[0]
    V = [(O @ Psi.T).T for O in Os]  # (K+1) x N cada uno
    nb = len(B)
    Kc = np.zeros((nb, nb, K + 1))
    for a in range(nb):
        Ua = V[a]
        for b in range(nb):
            if (B[a][0] == "im") != (B[b][0] == "im"):
                continue
            Vb = V[b]
            G0 = Ua @ (H0[:, None] * Vb.T)          # <u_i, H0 v_j>
            G1 = Ua @ (T @ Vb.T)                    # <u_i, T v_j>
            S = Ua @ Vb.T                           # <u_i, v_j>
            c = np.zeros(K + 1)
            for m in range(K + 1):
                for i in range(m + 1):
                    c[m] += G0[i, m - i]
                    if m >= 1 and i <= m - 1:
                        c[m] += G1[i, m - 1 - i]
                    for k in range(m - i + 1):
                        c[m] -= E[k] * S[i, m - i - k]
            Kc[a, b] = np.array([sum(c[i] * inv[m - i] for i in range(m + 1)) for m in range(K + 1)]) / n
    return B, Kc, E / n


def serie_chi(n, K, tres=False):
    """Serie de chi_ab = (1/n) <phi_a| (H - E)^{-1} |phi_b>,  phi = (1 - P) O psi (psi normalizado).
    xi_b = (H - E)^{-1} phi_b orden a orden: (H0 - E0) xi_m = phi_m - T xi_{m-1} + sum_k E_k xi_{m-k},
    con <psi|xi> = 0 (fija la componente en el patrón).
    tres=True (n pequeño, 9.23): RS en el sector de momento 0, partiendo de la suma de los tres patrones;
    los patrones son entonces las únicas configuraciones degeneradas y no rompen la serie."""
    codes, pw = conjunto(n, K + 2, tres)
    H0, T, Os, B = matrices(n, codes, pw)
    p = np.searchsorted(codes, int(np.dot(np.array([j % 3 for j in range(n)]), pw)))
    E0 = H0[p]
    den = E0 - H0
    P = np.searchsorted(codes, codigos_patrones(n, pw)) if tres else np.array([p])
    deg = den == 0
    if tres:
        assert np.array_equal(np.sort(np.nonzero(deg)[0]), np.sort(P)), "degeneradas distintas de los patrones"
        deg[:] = False
    den[den == 0] = 1.0
    psi = [np.zeros(codes.size)]
    psi[0][P] = 1.0
    E = [E0]
    for m in range(1, K + 2):
        rhs = T @ psi[m - 1]
        if np.ptp(rhs[P]) > 1e-9 * (1 + abs(rhs[P[0]])):
            raise RuntimeError("los patrones no son simétricos")
        E.append(rhs[P[0]])
        if m == K + 1:
            break
        for k in range(1, m + 1):
            rhs = rhs - E[k] * psi[m - k]
        v = rhs / den
        v[P] = 0.0
        if deg.sum() > 1 and np.abs(v[deg]).max() > 0:
            raise RuntimeError(f"orden {m}: configuración degenerada alcanzada")
        psi.append(v)
    E = np.array(E[:K + 1])
    Psi = np.array(psi)

    def mul(a, b):  # series escalares
        return np.array([sum(a[i] * b[m - i] for i in range(m + 1)) for m in range(K + 1)])

    def inv(a):
        r = np.zeros(K + 1); r[0] = 1 / a[0]
        for m in range(1, K + 1):
            r[m] = -sum(a[i] * r[m - i] for i in range(1, m + 1)) / a[0]
        return r

    N = np.array([sum(Psi[i] @ Psi[m - i] for i in range(m + 1)) for m in range(K + 1)])
    iN = inv(N)
    def phi_de(O):
        U = (O @ Psi.T).T                                     # O psi, por órdenes
        expo = mul(np.array([sum(Psi[i] @ U[m - i] for i in range(m + 1)) for m in range(K + 1)]), iN)
        return U - np.array([sum(expo[k] * Psi[m - k] for k in range(m + 1)) for m in range(K + 1)])

    nb = len(B)
    xis = []
    for b in range(nb):
        ph = phi_de(Os[b])
        xi = []
        for m in range(K + 1):
            rhs = ph[m].copy()
            if m >= 1:
                rhs -= T @ xi[m - 1]
            for k in range(1, m + 1):
                rhs += E[k] * xi[m - k]
            if DEBUG: print("b", b, "orden", m, "rhs[P] =", rhs[P])
            if np.abs(rhs[P]).max() > 1e-9 * (1 + np.abs(rhs).max()):
                raise RuntimeError(f"chi: incompatibilidad en la componente del patrón (orden {m}): {rhs[P]}")
            otros = deg.copy(); otros[P] = False
            if otros.any() and np.abs(rhs[otros]).max() > 0:
                raise RuntimeError(f"chi: orden {m} alcanza otra configuración degenerada (n pequeño)")
            x = -rhs / den  # (H0 - E0) xi_m = r_m, y den = E0 - H0
            x[P] = 0.0
            x[P] = (-sum(Psi[i] @ xi[m - i] for i in range(1, m + 1)) if m >= 1 else 0.0) / len(P)
            xi.append(x)
        xis.append(np.array(xi))
        del ph
    Ch = np.zeros((nb, nb, K + 1))
    for a in range(nb):
        pa = phi_de(Os[a])
        for b in range(nb):
            if (B[a][0] == "im") != (B[b][0] == "im"):
                continue
            s_ = np.array([sum(pa[i] @ xis[b][m - i] for i in range(m + 1)) for m in range(K + 1)])
            Ch[a, b] = mul(s_, iN) / n
    return B, Ch, E / n


if __name__ == "__main__":
    n, K = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 4 and sys.argv[4] in ("chi", "chi3"):
        B, Kc, En = serie_chi(n, K, tres=sys.argv[4] == "chi3")
    else:
        B, Kc, En = serie(n, K)
    print("E/n =", En)
    if len(sys.argv) > 3:
        np.savez(sys.argv[3], Kc=Kc, En=En)

"""Series en eps de la respuesta del fundamental de momento 0 de H(Theta) + sum_a y_a O_a(Theta) (9.26).

Rayleigh-Schrödinger desde la suma de los tres patrones, en todo el sector Sz = 0, con giro uniforme
(fase e^{i Theta/n} por salto) y operadores girados o_a(Theta) = (1 x V) o_a (1 x V)^+. Para cada orden
k <= K da los coeficientes de (por sitio, e = E/n):
  grad_a   = d e / d y_a                       (Hellmann-Feynman, <O_a>/n)
  hess_ab  = d^2 e / d y_a d y_b               (= -2 chi_ab)
  cub_abc  = d^3 e / d y_a d y_b d y_c         (Wigner 2n+1: sum_sigma <psi1|O - <O>|psi1>)
en las coordenadas W de newton_graduado (base de Hilbert-Schmidt), con y = 0.
Todo es complejo: con eps real, la serie de Re(s) tiene coeficientes (s_k + conj(s_k))/2.
Uso: python scripts/serie_respuesta.py n K salida.npz [M] [cubica] [cuarta] [k0]   (M valores de Theta = 2 pi j / M;
k0: en el sector de momento 0, que contiene al fundamental y a toda la respuesta; mismo resultado, N/n estados)
"""
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, '/home/alex/fisica/holonomy/scripts')
from celda_k_flujo import sector
_argv, sys.argv = sys.argv, sys.argv[:1]   # k1_racional lee sys.argv al importarse
from k1_racional import base
import newton_graduado as NG
sys.argv = _argv

B = base()
W = NG.W


def momento0(n, codes):
    """Representantes de las órbitas de traslación (mínimo código) y tamaño de cada órbita.
    Devuelve (R, ridx, orb): R = índices (en codes) de los representantes, ridx[i] = órbita de codes[i],
    orb[r] = tamaño de la órbita r."""
    top = 3 ** (n - 1)
    rot = codes.copy()
    rep = codes.copy()
    per = np.zeros(codes.size, dtype=np.int64)
    for j in range(1, n + 1):
        rot = rot // 3 + (rot % 3) * top                     # traslación de un sitio
        np.minimum(rep, rot, out=rep)
        per[(per == 0) & (rot == codes)] = j
    reps = np.unique(rep)
    R = np.searchsorted(codes, reps)
    ridx = np.searchsorted(reps, rep)
    return R, ridx, per[R]


def operadores(n, th, k0=False):
    """H0 (diagonal), T y los O_a en el sector Sz = 0; con k0, en el de momento 0 (base de órbitas
    normalizadas |r> = |o_r|^{-1/2} sum_{c in o_r} |c>; <r'|A|r> = sqrt(|o_r|/|o_r'|) sum_{c' in o_r'} A_{c' r})."""
    codes, dig = sector(n, 0)
    pw = 3 ** np.arange(n, dtype=np.int64)
    if k0:
        R, ridx, orb = momento0(n, codes)
        sq = np.sqrt(orb.astype(float))
        fila = lambda new: ridx[np.searchsorted(codes_all, new)]
        peso = lambda r, c: sq[c] / sq[r]
        codes_c, dig_c = codes[R], dig[:, R]
    else:
        fila = lambda new: np.searchsorted(codes_all, new)
        peso = lambda r, c: np.ones(np.size(r))
        codes_c, dig_c = codes, dig
    N = codes_c.size
    codes_all, codes, dig = codes, codes_c, dig_c
    H0 = np.zeros(N)
    filas, cols = [], []
    for j in range(n):
        k = (j + 1) % n
        a, b = dig[j], dig[k]
        H0 -= ((b - a) % 3 == 1)
        m = (a < 2) & (b > 0)                       # S+_j S-_k
        new = codes[m] + pw[j] - pw[k]
        filas.append(fila(new)); cols.append(np.nonzero(m)[0])
    r = np.concatenate(filas); c = np.concatenate(cols)
    A = sp.coo_matrix((2.0 * np.exp(1j * th / n) * peso(r, c), (r, c)), shape=(N, N)).tocsr()
    T = (A + A.getH()).tocsr()
    Os = []
    for w in W:
        M = NG.mat9(B[w])
        rows, cs, vals = [], [], []
        for j in range(n):
            k = (j + 1) % n
            col = dig[j] * 3 + dig[k]
            for row in range(9):
                coef = M[row, col]
                m = np.abs(coef) > 0
                if not m.any():
                    continue
                q = row // 3 - dig[j][m]                # carga que entra en el sitio j
                new = codes[m] + (row // 3 - dig[j][m]) * pw[j] + (row % 3 - dig[k][m]) * pw[k]
                rr, cc = fila(new), np.nonzero(m)[0]
                rows.append(rr); cs.append(cc)
                vals.append(coef[m] * np.exp(1j * th * q / n) * peso(rr, cc))
        Os.append(sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cs))),
                                shape=(N, N)).tocsr())
    pats = np.array(sorted(int(np.dot(np.array([(j + c) % 3 for j in range(n)]), pw)) for c in range(3)))
    if k0:                                                    # los tres patrones son una órbita
        P = np.unique(ridx[np.searchsorted(codes_all, pats)])
        assert P.size == 1
    else:
        P = np.searchsorted(codes, pats)
        assert np.array_equal(codes[P], pats)
    return H0, T, Os, P


def serie(n, K, th, cubica=True, cuarta=False, verbose=False, k0=False):
    H0, T, Os, P = operadores(n, th, k0)
    N = H0.size
    E0 = H0[P[0]]
    den = E0 - H0
    deg = den == 0
    assert np.array_equal(np.nonzero(deg)[0], np.sort(P)), "degeneradas distintas de los patrones"
    den[deg] = 1.0
    psi = [np.zeros(N, complex)]
    psi[0][P] = 1.0
    E = [E0 + 0j]
    for m in range(1, K + 1):
        rhs = T @ psi[m - 1]
        if np.ptp(np.abs(rhs[P])) > 1e-9 * (1 + np.abs(rhs).max()):
            raise RuntimeError("patrones no simétricos")
        E.append(rhs[P[0]])
        for k in range(1, m + 1):
            rhs = rhs - E[k] * psi[m - k]
        v = rhs / den
        v[P] = 0.0
        psi.append(v)
    E = np.array(E)
    Psi = np.array(psi)                                      # (K+1) x N

    def mul(a, b):                                           # producto de series (ultimo eje = orden)
        out = np.zeros(np.broadcast_shapes(a.shape, b.shape), complex)
        for m in range(K + 1):
            out[..., m] = sum(a[..., i] * b[..., m - i] for i in range(m + 1))
        return out

    def inv(a):
        r = np.zeros(K + 1, complex); r[0] = 1 / a[0]
        for m in range(1, K + 1):
            r[m] = -sum(a[i] * r[m - i] for i in range(1, m + 1)) / a[0]
        return r

    def braket(X, Y):                                        # <X(eps)|Y(eps)>, X, Y: (K+1) x N
        G = X.conj() @ Y.T                                   # G[i, j] = <X_i|Y_j>
        return np.array([sum(G[i, m - i] for i in range(m + 1)) for m in range(K + 1)])

    Nn = braket(Psi, Psi)
    iN = inv(Nn)
    nb = len(W)
    grad = np.zeros((nb, K + 1), complex)
    Phi, Xi = [], []
    for a in range(nb):
        U = (Os[a] @ Psi.T).T
        ex = mul(braket(Psi, U), iN)
        grad[a] = ex / n
        ph = U - np.array([sum(ex[k] * Psi[m - k] for k in range(m + 1)) for m in range(K + 1)])
        xi = []
        for m in range(K + 1):
            rhs = ph[m].copy()
            if m >= 1:
                rhs -= T @ xi[m - 1]
            for k in range(1, m + 1):
                rhs += E[k] * xi[m - k]
            if np.abs(rhs[P]).max() > 1e-8 * (1 + np.abs(rhs).max()):
                raise RuntimeError(f"incompatibilidad en el patrón (a={a}, orden {m}): {rhs[P]}")
            x = -rhs / den
            x[P] = 0.0
            if m >= 1:
                x[P] = -sum(np.vdot(Psi[i], xi[m - i]) for i in range(1, m + 1)) / len(P)
            xi.append(x)
        Phi.append(ph); Xi.append(np.array(xi))
    Phi = np.array(Phi); Xi = np.array(Xi)                   # nb x (K+1) x N
    # hess_ab = -(<phi_a|xi_b> + conj<phi_b|xi_a>)/N/n
    S = np.zeros((nb, nb, K + 1), complex)
    for a in range(nb):
        for b in range(nb):
            S[a, b] = braket(Phi[a], Xi[b])
    S = mul(S, iN)
    hess = -(S + np.conj(S.transpose(1, 0, 2))) / n
    cub = None
    if cubica:
        # t_abc = (<xi_a|O_b|xi_c> - <O_b> <xi_a|xi_c>)/N ;  cub = 2 Re(t_abc + t_bac + t_acb)/n
        X2 = Xi.reshape(nb * (K + 1), N)
        G0 = X2.conj() @ X2.T                                # <xi_a,i|xi_c,j>
        G0 = G0.reshape(nb, K + 1, nb, K + 1)
        conv = lambda G: np.array([sum(G[:, i, :, m - i] for i in range(m + 1)) for m in range(K + 1)]
                                  ).transpose(1, 2, 0)       # nb x nb x (K+1)
        XX = conv(G0)
        t = np.zeros((nb, nb, nb, K + 1), complex)
        for b in range(nb):
            Y = (Os[b] @ X2.T).T
            Gb = (X2.conj() @ Y.T).reshape(nb, K + 1, nb, K + 1)
            t[:, b, :, :] = conv(Gb) - mul(XX, n * grad[b])
            if verbose:
                print("  cubica b =", b, flush=True)
        t = mul(t, iN)
        s = t + t.transpose(1, 0, 2, 3) + t.transpose(0, 2, 1, 3)
        cub = (s + s.conj()) / n
    cua = None
    if cuarta:
        # zeta_cd = R (O_c - <O_c>) xi_d ;  T_abcd = -1/2 <xi_a|O_b - <O_b>|zeta_cd + zeta_dc> - E2_ab <xi_c|xi_d>
        # d^4 E = sum sobre las 24 permutaciones de T  (Wigner 2n+1; verificado en un juguete con mpmath)
        # Por pares c <= d, sin guardar los zeta (para n = 12 no caben en memoria).
        ex = n * grad
        X2c = X2.conj()
        conv1 = lambda G: np.array([sum(G[:, i, m - i] for i in range(m + 1)) for m in range(K + 1)]).T

        def zeta(c, d):
            OX = (Os[c] @ Xi[d].T).T
            rhs_s = OX - np.array([sum(ex[c, k] * Xi[d][m - k] for k in range(m + 1)) for m in range(K + 1)])
            pr = mul(braket(Psi, rhs_s), iN)
            rhs_s = rhs_s - np.array([sum(pr[k] * Psi[m - k] for k in range(m + 1)) for m in range(K + 1)])
            z = []
            for m in range(K + 1):
                rhs = rhs_s[m].copy()
                if m >= 1:
                    rhs -= T @ z[m - 1]
                for k in range(1, m + 1):
                    rhs += E[k] * z[m - k]
                if np.abs(rhs[P]).max() > 1e-8 * (1 + np.abs(rhs).max()):
                    raise RuntimeError(f"cuarta: incompatibilidad en el patrón (orden {m})")
                x = -rhs / den
                x[P] = 0.0
                if m >= 1:
                    x[P] = -sum(np.vdot(Psi[i], z[m - i]) for i in range(1, m + 1)) / len(P)
                z.append(x)
            return np.array(z)

        U = np.zeros((nb, nb, nb, nb, K + 1), complex)
        for c in range(nb):
            for d in range(c, nb):
                zs = 2 * zeta(c, c) if d == c else zeta(c, d) + zeta(d, c)          # zeta_cd + zeta_dc
                xz = conv1((X2c @ zs.T).reshape(nb, K + 1, K + 1))
                for b in range(nb):
                    Gb = conv1((X2c @ (Os[b] @ zs.T)).reshape(nb, K + 1, K + 1)) - mul(xz, ex[b])
                    U[:, b, c, d] = Gb
                    U[:, b, d, c] = Gb
            if verbose:
                print("  cuarta: c =", c, flush=True)
        E2 = n * hess / 2
        Tq = mul(-0.5 * U - mul(E2[:, :, None, None, :], XX[None, None, :, :, :]), iN)
        import itertools
        s4 = sum(Tq.transpose(p + (4,)) for p in itertools.permutations(range(4)))
        cua = (s4 + s4.conj()) / 2 / n
    return dict(E=E / n, grad=grad, hess=hess, cub=cub, cua=cua, N=Nn)


if __name__ == "__main__":
    n, K = int(sys.argv[1]), int(sys.argv[2])
    M = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = {}
    for j in range(M):
        th = 2 * np.pi * j / M
        r = serie(n, K, th, cubica=(j == 0) or ('cubica' in sys.argv), cuarta='cuarta' in sys.argv,
                  k0='k0' in sys.argv, verbose='verbose' in sys.argv)
        for key, v in r.items():
            if v is not None:
                out.setdefault(key, []).append(v)
        print(f"Theta = {th:.4f}: E/n = {np.real(r['E'][:4])} ...", flush=True)
    np.savez(sys.argv[3], **{k: np.array(v) for k, v in out.items()}, M=M, K=K, n=n)

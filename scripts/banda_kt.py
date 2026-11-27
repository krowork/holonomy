"""Banda de un defecto no perturbativa con k ∤ n (celda_k_notas.md, Sec. 7.12).

Intercambio SU(3) en el anillo de n sitios con 3 ∤ n. En el marco de reloj (kt_zk.marco), el gauge
D = prod_j w^{-j s_j} no cierra: el enlace de cierre (n-1, 0) lleva un giro de reloj t = -n mod 3, y en la base de
caracteres -P_asc es diagonal con valor -n + |d_tw mu| (paredes torcidas, >= 1). El vestido G_0 es el punto fijo de
Kirkwood-Thomas del hamiltoniano "recto" H_0, que es H con el término del enlace de cierre conjugado por X_0^r (la
etiqueta del sitio 0 desplazada en r), elegido para que su marco sea invariante por traslación y su fundamental
clásico tenga paredes 0. Se calcula H~ = e^{-G_0} H'_tw e^{G_0} en la base de caracteres (densa) y se da:
  - P = {mu : |d_tw mu| = 1} (n k defectos, etiquetados por el enlace roto b y el valor c);
  - Gershgorin por columnas: radio máximo de las columnas de P y cota inferior del resto, Re(diag) - radio;
  - Feshbach: H_eff(z) = H_PP + H_PQ (z - H_QQ)^{-1} H_QP; para cada uno de los n k niveles exactos más bajos,
    el menor valor singular de H_eff(lambda) - lambda (debe ser ~0);
  - el decaimiento del salto efectivo, max |H_eff(z0)_{(b,c),(b',c')}| frente a la distancia |b - b'|.
Con GAUGE=1, el flujo se pone entero en el enlace de cierre y H_0 no lleva flujo (la versión del Lema Z7).
Uso: [GAUGE=1] python scripts/banda_kt.py theta_sobre_pi eps n [n ...]
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

from dual_celda import anillo
from kt_zk import K, chi_hat, marco, termino


def enlace_cierre(n, q, h, theta):
    """Solo el término del enlace (n-1, 0), con el mismo convenio de giro que dual_celda.anillo."""
    N = K**n
    codes = np.arange(N, dtype=np.int64)
    pw = K ** np.arange(n, dtype=np.int64)
    dig = (codes[None, :] // pw[:, None]) % K
    i, j = n - 1, 0
    rows, cols, vals = [], [], []
    idx = K * dig[i] + dig[j]
    for x in range(K * K):
        sel = np.nonzero(idx == x)[0]
        a, b = divmod(x, K)
        for y in range(K * K):
            if h[y, x] == 0:
                continue
            a2, b2 = divmod(y, K)
            rows.append(codes[sel] + (a2 - a) * pw[i] + (b2 - b) * pw[j])
            cols.append(sel)
            vals.append(np.full(sel.size, h[y, x] * np.exp(-1j * theta / n * (q[b2] - q[b]))))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()


def permuta_sitio0(n, r):
    """X_0^r: s_0 -> s_0 + r mod K, como matriz de permutación."""
    codes = np.arange(K**n)
    s0 = codes % K
    nuevo = codes - s0 + (s0 + r) % K
    return sp.csr_matrix((np.ones(K**n), (nuevo, codes)), shape=(K**n, K**n))


def kt_fijo(H, n, am, dm, borde, maxit=500, tol=1e-13):
    const = borde == 0
    g = np.zeros(K**n, complex)
    for _ in range(maxit):
        G = np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n
        Rh = chi_hat(am(H @ dm(np.exp(G))) / np.exp(G), n)
        paso = np.where(const, 0, Rh / np.where(const, 1, borde))
        g -= paso
        if not np.isfinite(paso).all() or np.abs(paso).max() > 1e3:
            raise RuntimeError("KT diverge")
        if np.abs(paso).max() < tol:
            break
    return np.fft.ifftn(g.reshape((K,) * n)).ravel() * K**n, Rh[0], np.abs(g).sum()


def main():
    th, eps = float(sys.argv[1]) * np.pi, float(sys.argv[2])
    hcl, v, q = termino("llenado")
    h = hcl + eps * v
    for n in [int(x) for x in sys.argv[3:]]:
        assert n % K
        N = K**n
        dig, am, dm = marco(n)
        t = (-n) % K
        borde = (dig != np.roll(dig, -1, axis=1)).sum(1)  # paredes rectas
        # paredes torcidas: en el enlace (n-1, 0) la relación es mu_{n-1} - mu_0 = t' (se fija abajo)
        if os.environ.get("GAUGE") == "1":
            # todo el flujo en el enlace de cierre (gauge diagonal, mismo espectro); H_0 sin flujo
            H = anillo(n, K, q, h, 0.0) - enlace_cierre(n, q, h, 0.0) + enlace_cierre(n, q, h, th * n)
            Hc = enlace_cierre(n, q, h, 0.0)
            Hb = H - enlace_cierre(n, q, h, th * n)
        else:
            H = anillo(n, K, q, h, th)
            Hc = enlace_cierre(n, q, h, th)
            Hb = H - Hc
        H0 = None
        for r in range(1, K):
            P = permuta_sitio0(n, r)
            cand = Hb + P @ Hc @ P.T
            cl = anillo(n, K, q, hcl, 0.0) - enlace_cierre(n, q, hcl, 0.0)
            Pc = enlace_cierre(n, q, hcl, 0.0)
            cl = cl + P @ Pc @ P.T
            uno = dm(np.ones(N, complex))
            if np.allclose(cl @ uno, -n * uno):
                H0 = cand
                break
        assert H0 is not None
        G, E0, gnorm = kt_fijo(H0, n, am, dm, borde)
        # paredes torcidas: diagonal de -P_asc del anillo real en la base de caracteres
        Hcl = anillo(n, K, q, hcl, 0.0)
        dtw = np.empty(N, int)
        eG, emG = np.exp(G), np.exp(-G)
        M = np.empty((N, N), complex)
        for mu in range(N):
            chi = np.exp(2j * np.pi / K * (dig @ dig[mu]))
            if mu < N:
                dtw[mu] = int(round((chi_hat(am(Hcl @ dm(chi)), n)[mu]).real + n))
            M[:, mu] = chi_hat(emG * am(H @ dm(eG * chi)), n)
        Pm = dtw == 1
        assert Pm.sum() == n * K, (Pm.sum(), np.bincount(dtw))
        off = np.abs(M).sum(0) - np.abs(np.diag(M))
        dg = np.diag(M) - E0
        radP = off[Pm].max()
        diagP = dg[Pm].real
        cotaQ = (dg[~Pm].real - off[~Pm]).min()
        ev = np.linalg.eigvalsh(H.toarray()) - E0
        band = ev[: n * K]
        print(f"n={n} eps={eps:+.2f} Theta/pi={th / np.pi}: t={t}  sum|g|={gnorm:.3f}  E0/n={E0.real / n:.6f}")
        print(f"   P: Re diag en [{diagP.min():.4f}, {diagP.max():.4f}], radio max {radP:.4f};  Q: Re diag - radio >= "
              f"{cotaQ:.4f}")
        print(f"   exacto: banda (nk = {n * K} niveles) en [{band.min():.4f}, {band.max():.4f}], siguiente "
              f"{ev[n * K]:.4f}")
        iP, iQ = np.nonzero(Pm)[0], np.nonzero(~Pm)[0]
        HPP, HPQ, HQP, HQQ = M[np.ix_(iP, iP)], M[np.ix_(iP, iQ)], M[np.ix_(iQ, iP)], M[np.ix_(iQ, iQ)]

        def heff(z):
            return HPP + HPQ @ np.linalg.solve((z + E0) * np.eye(iQ.size) - HQQ, HQP)
        # columnas de Q: theta_Q(z) = max radio_Q/|diag - z| con z en la banda
        offQ = np.abs(HQQ).sum(0) - np.abs(np.diag(HQQ))
        thQ = max((offQ / np.abs(np.diag(HQQ) - E0 - z)).max() for z in (band.min(), band.max()))
        sv = []
        for lam in band[:: max(1, band.size // 6)]:
            sv.append(np.linalg.svd(heff(lam) - (lam + E0) * np.eye(iP.size), compute_uv=False)[-1])
        print(f"   Neumann en l^1 de (z - H_QQ)^-1: theta_Q = {thQ:.4f};  Feshbach: min s.v. de H_eff(l) - l = "
              f"{max(sv):.1e} (máx sobre {len(sv)} niveles)")
        # salto efectivo frente a la distancia entre enlaces rotos
        # orden de los defectos en el anillo de longitud n k que recorre la traslación física T
        pw = K ** np.arange(n)
        Tcode = (np.roll(dig, -1, axis=1) * pw).sum(1)
        pos = {}
        for o, raiz in enumerate(iP):
            if raiz in pos:
                continue
            mu = raiz
            for x in range(n * K):
                pos[mu] = (o, x)
                chi = np.zeros(N, complex)
                chi[Tcode] = dm(np.exp(2j * np.pi / K * (dig @ dig[mu])))
                mu = int(np.abs(chi_hat(am(chi), n)).argmax())
                if mu == raiz:
                    break
        L = x + 1  # longitud de cada órbita (n: la carga Z_k de reloj separa P en k anillos de n defectos)
        oP = np.array([pos[m][0] for m in iP])
        xP = np.array([pos[m][1] for m in iP])
        He = heff(band.mean()) - E0 * np.eye(iP.size)
        dist = np.abs(xP[:, None] - xP[None, :])
        dist = np.minimum(dist, L - dist)
        misma = oP[:, None] == oP[None, :]
        cruz = np.abs(He[~misma]).max() if (~misma).any() else 0.0
        amp = [np.abs(He[misma & (dist == r)]).max() for r in range(L // 2 + 1)]
        print(f"   {K} anillos de {L} defectos; |H_eff| entre anillos distintos: {cruz:.1e}")
        print(f"   H_eff(z0) - E0: diagonal {np.diag(He).real.min():.4f}..{np.diag(He).real.max():.4f}; "
              f"max|salto| por distancia en el anillo: " + "  ".join(f"{r}:{x:.1e}" for r, x in
                                                                                enumerate(amp) if r), flush=True)


if __name__ == "__main__":
    main()

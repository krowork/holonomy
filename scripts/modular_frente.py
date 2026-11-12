"""Artículo II, sección 11.2: el flujo modular sigma_s(A) = e^{-iKs} A e^{iKs}, K = -x.O.

1. Anillo de 10 con flujo uniforme Theta (multiplicadores iguales por traslación),
   T K T^-1 = K (T = (iY)^{xn} K), y frente de ||[Z_0(s), Z_j]||_HS/sqrt(N) > 0,2 en v = 0,3, 0,6, 0,9 vmax(0):
   velocidad (pendiente de j frente a s_j, j = 2..5) proporcional a b = |x_A| (tabla tab:frente), los mismos s_j para
   Theta = 0, pi/2, pi con v <= 0,6 vmax. La fase de x es Theta/n exactamente en Theta = 0, pi y se separa con
   Theta = pi/2 (renormalización de los multiplicadores, Proposición 29).
2. Escalera periódica 2 x 4 cerca de vmax, v = vmax (1 - eps): gap de K / beta = gap del testigo (0,09112), peso
   espectral de Z_0 en frecuencia modular nula ~ 0,24 eps, frecuencia media / beta -> 0,1785, cociente de pesos de
   frecuencia negativa y positiva ~ 0,92 eps.

Escribe data/modular_frente{,_vmax}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 .venv/bin/python scripts/modular_frente.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np
from scipy.linalg import eigh

from fisher_kubo_mori import MaxEnt, escalera, testigo_optimo
from holonomy import xxz
from holonomy.report import write_csv
from modular_cota import anillo, dense, ops_anillo, simetrico, zdiag
from holonomy.paulis import pauli_string

UMBRAL = 0.2
SS = np.linspace(0, 12, 241)


def T_unitario(n):
    """Parte unitaria de T = (iY)^{x n} K."""
    iY = np.array([[0.0, 1.0], [-1.0, 0.0]])
    U = np.array([[1.0]])
    for _ in range(n):
        U = np.kron(U, iY)
    return U


def conmutador_hs(E, U, A, B, ss):
    """||[A(s), B]||_HS / sqrt(N), A(s) = e^{-iKs} A e^{iKs}, en la base propia de K."""
    At, Bt = U.conj().T @ A @ U, U.conj().T @ B @ U
    dE = E[:, None] - E[None, :]
    out = []
    for s in ss:
        As = At * np.exp(-1j * s * dE)
        out.append(np.linalg.norm(As @ Bt - Bt @ As) / np.sqrt(len(E)))
    return np.array(out)


def frente(chk):
    n = 10
    vmax = 4 * xxz.eps(n, 1.0, 0.0) / 3
    sym = simetrico(ops_anillo(n))
    T = T_unitario(n)
    Z = [np.diag(zdiag(n, j)).astype(complex) for j in range(n // 2 + 1)]
    filas, sj_por_v = [], {}
    for frac in (0.3, 0.6, 0.9):
        v = frac * vmax
        for Th in (0.0, np.pi / 2, np.pi):
            f = Th / n
            y, _, S, res = sym.resuelve(n * v * np.array([2 * np.cos(f), 2 * np.sin(f), 1.0]))
            K = -np.einsum("k,kab->ab", y, sym.O)
            trev = np.linalg.norm(T @ K.conj() @ T.T - K) / np.linalg.norm(K)
            # Theta = 0, pi: la reflexión del anillo seguida de un gauge deja el problema fijo y la máxima entropía es
            # única, así que la fase de x es exactamente Theta/n; en otro Theta se separa en O(v^{n-1}) (Prop. dx).
            dfase = np.arctan2(y[1], y[0]) - f
            exacta = Th in (0.0, np.pi)
            chk.check(res < 1e-10 and trev < 1e-14 and (abs(dfase) < 1e-12 if exacta else frac > 0.3 or abs(dfase) < 1e-5),
                      f"anillo {n}, v = {frac} vmax, Theta = {Th:.4f}: fase de x - Theta/n = {dfase:+.2e}"
                      f"{' (exacta por simetría)' if exacta else ''}, |T K T^-1 - K|/|K| = {trev:.1e}")
            E, U = eigh(K)
            sj = []
            for j in range(1, n // 2 + 1):
                c = conmutador_hs(E, U, Z[0], Z[j], SS)
                sj.append(SS[np.argmax(c > UMBRAL)])
            sj = np.array(sj)
            b = abs(y[0])
            vel = np.polyfit(sj[1:], np.arange(2, n // 2 + 1), 1)[0]
            sj_por_v.setdefault(frac, []).append(sj)
            filas.append((frac, Th, S, b, *sj, vel, vel / b, y[1] / y[0]))
            chk.info(f"v = {frac} vmax, Theta = {Th:.4f}: S = {S:.4f}, b = {b:.4f}, s_j = {sj}, velocidad = {vel:.4f}, "
                     f"velocidad/b = {vel / b:.4f}")
        if frac <= 0.6:
            igual = all(np.array_equal(sj_por_v[frac][0], s) for s in sj_por_v[frac][1:])
            chk.check(igual, f"v = {frac} vmax: los s_j no cambian con Theta")
    vb = np.array([f[-2] for f in filas if f[1] == 0.0])
    chk.check(np.all(np.abs(vb - 6.75) < 0.15), f"velocidad/b en Theta = 0: {np.round(vb, 2)} (en torno a 6,7)")
    write_csv("modular_frente.csv", ("v_sobre_vmax", "Theta", "S", "b", "s1", "s2", "s3", "s4", "s5", "velocidad",
                                     "velocidad_sobre_b", "xB_sobre_xA"), filas)


def cerca_de_vmax(chk):
    L = 4
    edges, u = escalera(L)
    me = MaxEnt(2 * L, edges)
    vmax, _, st = testigo_optimo(me, u)
    chk.check(abs(vmax - 0.53556549) < 1e-7, f"2x{L}: vmax = {vmax:.8f}" + chk.status_note(st))
    Z0 = dense(pauli_string(2 * L, ((0, 2),)))
    x, filas = None, []
    for eps in (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3):
        x, _, S, res = me.resuelve(me.objetivo(vmax * (1 - eps), 0 * u), x0=x)
        E, U = eigh(-np.einsum("k,kab->ab", x, me.O))
        p = np.exp(-(E - E.min()))
        p /= p.sum()
        beta = x @ me.objetivo(1.0, 0 * u)
        gapK = np.unique(np.round(E - E.min(), 9))[1]
        w = p[:, None] * np.abs(U.conj().T @ Z0 @ U) ** 2
        dE = E[None, :] - E[:, None]
        tot = w.sum()
        estat = w[np.abs(dE) < 1e-9].sum() / tot
        wmed = (w * np.abs(dE)).sum() / tot / beta
        neg = w[dE < -1e-9].sum() / w[dE > 1e-9].sum()
        chk.check(res < 1e-10 and abs(gapK / beta - 0.09112) < 5e-6,
                  f"eps = {eps:.0e}: beta = {beta:.4f}, gap_K/beta = {gapK / beta:.6f}, peso estático/eps = "
                  f"{estat / eps:.4f}, <|omega|>/beta = {wmed:.4f}, (w-/w+)/eps = {neg / eps:.4f}")
        filas.append((eps, S, beta, gapK, gapK / beta, estat, wmed, neg))
    ult = filas[-1]
    chk.check(abs(ult[5] / ult[0] - 0.24) < 0.01 and abs(ult[6] - 0.1785) < 5e-4 and abs(ult[7] / ult[0] - 0.92) < 0.01,
              f"eps = 1e-3: peso estático = {ult[5] / ult[0]:.4f} eps, <|omega|> = {ult[6]:.4f} beta, "
              f"w-/w+ = {ult[7] / ult[0]:.4f} eps")
    write_csv("modular_frente_vmax.csv", ("eps", "S", "beta", "gap_K", "gap_K_sobre_beta", "peso_estatico",
                                          "omega_medio_sobre_beta", "w_neg_sobre_w_pos"), filas)


def main():
    chk = Checker("modular_frente")
    cerca_de_vmax(chk)
    frente(chk)
    chk.finish()


if __name__ == "__main__":
    main()

"""Certificación de la hipótesis espectral (H) del Teorema 7 para cada n.

(H): en (Delta*, Phi_c) el nivel fundamental de H_XXZ en todo el espacio es un
doblete (n par) o dos dobletes con S^z = +-1/2 (n impar); en cada sector de S^z
sus dos estados tienen momentos distintos, y la pendiente lateral es s_n > 0.

Se certifica en todo un intervalo I_n de Delta que contiene el tramo de Delta*:
Delta*(d) crece con d (sección 5.3), así que basta con [Delta*(0), Delta*(pi/n)]
y un margen. Método:

  * Bloques. H(Delta, Phi_c) conmuta con S^z y con la traslación T (gauge
    uniforme), así que se diagonaliza por bloques (M, K) en bases de Bloch. Los
    sectores M y n - M son unitariamente equivalentes en Phi_c (inversión de
    espines lleva Phi a -Phi, y -Phi_c = Phi_c mód 2 pi por gauge): basta M <= n/2.
  * Doblete. Los dos bloques (M, K+-) con el autovalor más bajo deben estar en
    M = floor(n/2), tener el mismo autovalor (los relaciona la simetría S) y
    quedar estrictamente por debajo de todo lo demás: el segundo autovalor de esos
    bloques y el primero de todos los bloques restantes.
  * Entre puntos de la malla. dH/dDelta = sum Sz Sz tiene norma <= n/4, así que
    cada autovalor es (n/4)-lipschitziano en Delta (Weyl), y también el mínimo de
    una familia de autovalores. En [a, b], con h = b - a, la separación entre el
    doblete y el resto es >= media de los extremos - (n/4) h.
  * Pendiente. s_n = |j|/n con j = <psi|dH/dPhi|psi> en el bloque K+. Con
    ||dH/dPhi|| <= 1/2 y ||psi'|| <= (n/4)/g (g = gap dentro del bloque),
    |dj/dDelta| <= n/(4 g); en [a, b], |j| >= media de |j| - (n/(8 g)) h si j no
    cambia de signo, con g acotado por debajo de la misma forma.
  * Los subintervalos que no cumplen se bisecan (hasta MAX_DEPTH niveles).

El redondeo no se controla con aritmética de intervalos: los autovalores de
eigh tienen error relativo ~1e-15 respecto de ||H|| <= n, y todos los márgenes
exigidos son >= TOL = 1e-8 por encima de esa escala. Es una prueba asistida por
ordenador para cada n, no para todo n.

Uso: python scripts/certify_h.py [n ...]      (por defecto n = 6..18; n = 18 tarda ~10 min)
Escribe data/certify_h.csv.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)

import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from holonomy import bethe, dual, xxz
from holonomy.report import Checker, write_csv

MARGIN = 0.01   # margen alrededor de [Delta*(0), Delta*(pi/n)]
H0 = 0.02       # paso inicial de la malla
MAX_DEPTH = 8
TOL = 1e-8
COMPLETENESS_MAX = 12  # control de la base de Bloch con diagonalización completa


def bloch_blocks(n, M):
    """Bases de Bloch del sector M: {k: matriz dispersa dim x nk}, K = 2 pi k / n.

    T es la traslación de holonomy.xxz: s -> rotación de bits a la izquierda.
    """
    full = 2 ** n - 1
    states = [s for s in range(2 ** n) if bin(s).count("1") == M]
    index = {s: i for i, s in enumerate(states)}
    seen = set()
    cols = {k: ([], [], []) for k in range(n)}  # filas, columnas, valores
    ncol = dict.fromkeys(range(n), 0)
    for s in states:
        if s in seen:
            continue
        orbit = [s]
        t = ((s << 1) | (s >> (n - 1))) & full
        while t != s:
            orbit.append(t)
            t = ((t << 1) | (t >> (n - 1))) & full
        seen.update(orbit)
        p = len(orbit)
        for k in range(n):
            if (k * p) % n:
                continue  # K incompatible con el período de la órbita
            K = 2 * np.pi * k / n
            r, c, v = cols[k]
            for j, t in enumerate(orbit):
                r.append(index[t])
                c.append(ncol[k])
                v.append(np.exp(-1j * K * j) / np.sqrt(p))
            ncol[k] += 1
    dim = len(states)
    return {k: sp.csr_matrix((v, (r, c)), shape=(dim, ncol[k]))
            for k, (r, c, v) in cols.items() if ncol[k] > 0}


class Problem:
    def __init__(self, n, Phi=None):
        self.n = n
        self.Pc = dual.cusp_flux(n) if Phi is None else Phi
        self.sectors = list(range(0, n // 2 + 1))
        self.blocks = {M: bloch_blocks(n, M) for M in self.sectors}
        # Piezas de H por bloque: H = Delta * C + X, y la corriente J = dH/dPhi.
        self.parts = {}
        ph = np.exp(1j * self.Pc / n)
        for M in self.sectors:
            diag, Kop = xxz.sector_operators(n, M)
            X = 0.5 * (ph * Kop + np.conj(ph) * Kop.getH())
            J = (0.5j / n) * (ph * Kop - np.conj(ph) * Kop.getH())
            C = sp.diags(diag)
            for k, B in self.blocks[M].items():
                Bh = B.getH()
                self.parts[(M, k)] = tuple(np.asarray((Bh @ A @ B).todense()) for A in (C, X, J))
        self.check_translation()

    def check_translation(self):
        """T conmuta con H en el gauge uniforme (comprobación de la base)."""
        n, M = self.n, self.n // 2
        diag, Kop = xxz.sector_operators(n, M)
        H = xxz.hamiltonian(n, M, 0.7, self.Pc)
        states = [s for s in range(2 ** n) if bin(s).count("1") == M]
        idx = {s: i for i, s in enumerate(states)}
        full = 2 ** n - 1
        rows = [idx[((s << 1) | (s >> (n - 1))) & full] for s in states]
        T = sp.csr_matrix((np.ones(len(states)), (rows, range(len(states)))), shape=(len(states),) * 2)
        self.translation_error = abs(T @ H - H @ T).max()

    def block_completeness(self, D=0.9):
        """Control: la unión de los espectros por bloques es el espectro del sector."""
        err = 0.0
        for M in self.sectors:
            full = np.linalg.eigvalsh(xxz.hamiltonian(self.n, M, D, self.Pc).toarray())
            parts = np.sort(np.concatenate([np.linalg.eigvalsh(D * C + X)
                                            for (MM, _), (C, X, _) in self.parts.items() if MM == M]))
            err = max(err, np.abs(full - parts).max())
        return err

    def data(self, D):
        """Autovalores bajos de cada bloque y corriente del doblete en Delta = D."""
        low = {}
        vec = {}
        for key, (C, X, _) in self.parts.items():
            Hb = D * C + X
            m = Hb.shape[0]
            if m == 1:
                low[key] = (Hb[0, 0].real, np.inf)
                vec[key] = np.ones(1, complex)
                continue
            w, V = sla.eigh(Hb, subset_by_index=[0, 1])
            low[key] = (w[0], w[1])
            vec[key] = (V[:, 0], np.linalg.norm(Hb @ V[:, 0] - w[0] * V[:, 0]))
        # Doblete: los dos bloques con el autovalor más bajo.
        order = sorted(low, key=lambda key: low[key][0])
        dbl = sorted(order[:2])  # orden fijo: los dos autovalores son iguales por S
        Ed = low[dbl[0]][0]
        rest = min([low[key][0] for key in order[2:]] + [low[key][1] for key in dbl])
        kp = dbl[0]  # rama de referencia, la misma en todos los puntos de la malla
        psi, res = vec[kp]
        J = self.parts[kp][2]
        j = float(np.real(np.vdot(psi, J @ psi)))
        g = low[kp][1] - low[kp][0]
        return dict(D=D, dbl=dbl, Ed=Ed, Ed2=low[dbl[1]][0], rest=rest, j=j, g=g,
                    jerr=res / g)  # |<J>| es 1/2-acotado: error <= 2 (1/2) sin(theta) <= res/g


def certify_interval(prob, a, b, depth, out):
    """Certifica [a, b] bisecando si hace falta. Devuelve True/False."""
    n = prob.n
    L = n / 4
    h = b["D"] - a["D"]
    ok_labels = a["dbl"] == b["dbl"]
    sep = 0.5 * ((a["rest"] - a["Ed"]) + (b["rest"] - b["Ed"])) - L * h
    g_lo = 0.5 * (a["g"] + b["g"]) - L * h
    same_sign = a["j"] * b["j"] > 0
    j_lo = (0.5 * (abs(a["j"]) + abs(b["j"])) - max(a["jerr"], b["jerr"])
            - (n / (8 * g_lo)) * h) if g_lo > 0 else -np.inf
    if ok_labels and sep > TOL and g_lo > TOL and same_sign and j_lo > TOL:
        out.append((a["D"], b["D"], sep, j_lo))
        return True
    if depth >= MAX_DEPTH:
        out.append((a["D"], b["D"], sep, j_lo))
        return False
    mid = prob.data(0.5 * (a["D"] + b["D"]))
    return (certify_interval(prob, a, mid, depth + 1, out)
            and certify_interval(prob, mid, b, depth + 1, out))


def delta_range(n):
    Pc = dual.cusp_flux(n)
    lo = dual.v_cusp(bethe.eps, n, Pc, xatol=1e-10)[1]                 # d = 0
    hi = dual.v_cusp(bethe.eps, n, dual.smooth_flux(n), xatol=1e-10)[1]  # d = pi/n
    return min(lo, hi) - MARGIN, max(lo, hi) + MARGIN, lo, hi


def certify(n, ck, rows):
    t0 = time.time()
    lo, hi, d0, d1 = delta_range(n)
    prob = Problem(n)
    ck.check(prob.translation_error < 1e-12, f"n={n}: [T, H] = 0 (error {prob.translation_error:.1e})")
    if n <= COMPLETENESS_MAX:
        err = prob.block_completeness()
        ck.check(err < 1e-10, f"n={n}: los bloques de Bloch reproducen el espectro de cada sector (error {err:.1e})")
    m = max(2, int(np.ceil((hi - lo) / H0)))
    grid = [prob.data(D) for D in np.linspace(lo, hi, m + 1)]
    M = n // 2
    ks = [k for (_, k) in grid[0]["dbl"]]
    in_sector = all(key[0] == M for key in grid[0]["dbl"])
    ck.check(in_sector and ks[0] != ks[1],
             f"n={n}: doblete en el sector M={M}, momentos K/pi = {2 * ks[0] / n:.4f}, {2 * ks[1] / n:.4f}")
    # Control negativo: en el punto liso el fundamental es simple, y el mismo
    # criterio debe detectarlo (los dos bloques más bajos no son degenerados).
    smooth = Problem(n, dual.smooth_flux(n)).data(0.5 * (lo + hi))
    ck.check(abs(smooth["Ed"] - smooth["Ed2"]) > 1e-3,
             f"n={n}: control negativo, en el punto liso no hay doblete "
             f"(separación {abs(smooth['Ed'] - smooth['Ed2']):.4f})")
    sym = max(abs(g["Ed"] - g["Ed2"]) for g in grid)
    ck.check(sym < 1e-10, f"n={n}: los dos bloques del doblete tienen el mismo autovalor (dif. {sym:.1e})")
    pieces = []
    ok = all(certify_interval(prob, a, b, 0, pieces) for a, b in zip(grid[:-1], grid[1:]))
    sep_min = min(p[2] for p in pieces)
    j_min = min(p[3] for p in pieces)
    ck.check(ok, f"n={n}: (H) certificada en Delta in [{lo:.5f}, {hi:.5f}] "
                 f"(Delta* de {d0:.5f} a {d1:.5f}); {len(pieces)} subintervalos; separación >= {sep_min:.4f}, "
                 f"|j| >= {j_min:.4f} (s_n >= {j_min / n:.2e}); {time.time() - t0:.0f} s")
    rows.append((n, lo, hi, d0, d1, len(pieces), sep_min, j_min / n, int(ok)))


def main():
    ns = [int(x) for x in sys.argv[1:]] or list(range(6, 19))
    ck = Checker()
    rows = []
    for n in ns:
        certify(n, ck, rows)
    write_csv("certify_h.csv", ["n", "delta_lo", "delta_hi", "delta_star_d0", "delta_star_dmax",
                                "subintervalos", "separacion_min", "s_n_min", "certificado"], rows)
    ck.finish()


if __name__ == "__main__":
    main()

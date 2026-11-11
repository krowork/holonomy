"""Borrador «Parches causales»: boosts de Carroll en nervios pequeños.

Datos: en cada arista i < j, M_e = R_z(phi_e) B(b_e), con B(b) = [[1,0,0],[0,1,0],[b1,b2,1]] (e_a -> e_a + b_a l) y la
marginal rho_e(v) = (I - v sum_ab (M_e)_ab s_a s_b)/4 (hipótesis P2).

1. v_max por el SDP primal (CVXPY, `sdp.primal`, que impone Bloch nulo y las nueve correlaciones de cada arista) en el
   triángulo, K_4 (tetraedro) y el octaedro, con |b| igual en todas las aristas y dos elecciones de dirección: todas a lo
   largo de e_1 ("eje") y al azar ("azar"); sin datos SO(2) y con los de curvatura. Se compara con la ventana
   w/chi' <= v_max <= min(1/3, w), w = 1/(1 + |b|).
2. Paridad de la entropía de máxima entropía S_G en el octaedro con curvatura, a v = 0,15. Gauges globales:
   R_x(pi) lleva (phi, b1, b2) a (-phi, -b1, b2) y R_z(pi) lleva b a -b, así que la parte impar en phi,
   O = [S(phi, b) - S(-phi, b)]/2, es impar en cada componente de b por separado: se anula si todos los boosts son
   paralelos a un eje (b2 = 0), y si no, empieza en orden b1 b2. Se comprueba, y se mide O frente a la escala de b y de phi.

Máxima entropía: dual convexo F(x) = ln tr exp(sum x_k O_k) - x . t, con O_k las nueve s_a s_b de cada arista (el máximo es
invariante bajo la inversión temporal, que fija los productos de dos Pauli y anula Bloch); L-BFGS y Newton con el hessiano
exacto de Kubo-Mori. S = F(x*); el residuo de las marginales certifica el máximo.

Escribe data/parches_boosts_vmax.csv y data/parches_boosts_paridad.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 python scripts/parches_boosts.py)
"""
import _common  # noqa: F401
from _common import CLARABEL_OPTS

import itertools
import time
from functools import reduce

import numpy as np
from scipy.optimize import minimize

from holonomy import sdp, sphere
from holonomy.report import write_csv
import sphere_casquetes as cas

I2 = np.eye(2)
PAULI = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0 + 0j, -1.0])]


def rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def boost(b):
    return np.array([[1, 0, 0], [0, 1, 0], [b[0], b[1], 1.0]])


def grafo(nombre):
    if nombre == "triangulo":
        return 3, ((0, 1), (0, 2), (1, 2)), np.array([np.pi / 6, -np.pi / 6, np.pi / 6]), 3  # flujo pi/2
    p = sphere.polyhedron(nombre)
    return p.n, p.edges, cas.data_phases(p, "curvado"), sphere.chromatic_index(p)


def datos(phis, bs):
    return [rz(f) @ boost(b) for f, b in zip(phis, bs, strict=True)]


def v_primal(n, edges, Ms):
    return sdp.primal(n, Ms, edges=list(edges), solver="CLARABEL", **CLARABEL_OPTS)


# ---------------------------------------------------------------------------
# Máxima entropía general
# ---------------------------------------------------------------------------

class MaxEnt:
    def __init__(self, n, edges):
        self.n, self.edges = n, edges
        self.ops = []
        for i, j in edges:
            for a, b in itertools.product(range(3), repeat=2):
                o = [I2] * n
                o[i], o[j] = PAULI[a], PAULI[b]
                self.ops.append(reduce(np.kron, o))
        self.ops = np.array(self.ops)

    def objetivo(self, Ms, v):
        return np.concatenate([(-v * M).ravel() for M in Ms])

    def _estado(self, x):
        K = np.tensordot(x, self.ops, axes=1)
        w, U = np.linalg.eigh(K)
        lnZ = w.max() + np.log(np.exp(w - w.max()).sum())
        p = np.exp(w - lnZ)
        return lnZ, p, U, w

    def _medias(self, p, U):
        rho = (U * p) @ U.conj().T
        return np.einsum("kij,ji->k", self.ops, rho).real

    def resuelve(self, Ms, v, x0=None):
        t = self.objetivo(Ms, v)
        x = np.zeros(len(t)) if x0 is None else x0.copy()

        def f(x):
            lnZ, p, U, _ = self._estado(x)
            return lnZ - x @ t, self._medias(p, U) - t

        r = minimize(f, x, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10, ftol=1e-15))
        x = r.x
        for _ in range(30):  # Newton con el hessiano de Kubo-Mori
            lnZ, p, U, w = self._estado(x)
            g = self._medias(p, U) - t
            if np.abs(g).max() < 1e-11:
                break
            Ok = np.einsum("ai,kab,bj->kij", U.conj(), self.ops, U, optimize=True)
            dw = w[:, None] - w[None, :]
            dp = p[:, None] - p[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(np.abs(dw) > 1e-12, dp / np.where(dw == 0, 1, dw), p[:, None])
            H = np.einsum("kij,lji,ij->kl", Ok, Ok, k, optimize=True).real
            m = self._medias(p, U)
            H -= np.outer(m, m)
            paso = np.linalg.lstsq(H, -g, rcond=1e-12)[0]
            a = 1.0
            F0 = lnZ - x @ t
            while a > 1e-6:
                lnZ1, *_ = self._estado(x + a * paso)
                if lnZ1 - (x + a * paso) @ t <= F0 + 1e-4 * a * (g @ paso):
                    break
                a /= 2
            x = x + a * paso
        lnZ, p, U, _ = self._estado(x)
        res = np.abs(self._medias(p, U) - t).max()
        return lnZ - x @ t, res, x


# ---------------------------------------------------------------------------

def parte_vmax():
    rng = np.random.default_rng(3)
    filas = []
    for nombre in ("triangulo", "tetraedro", "octaedro"):
        n, edges, ph_curv, chi = grafo(nombre)
        E = len(edges)
        dirs_azar = rng.normal(size=(E, 2))
        dirs_azar /= np.linalg.norm(dirs_azar, axis=1, keepdims=True)
        for caso, ph in (("sin datos", np.zeros(E)), ("curvatura", ph_curv)):
            for modo in ("eje", "azar"):
                for nb in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                    if nb == 0 and modo == "azar":
                        continue
                    u = np.tile([1.0, 0.0], (E, 1)) if modo == "eje" else dirs_azar
                    t0 = time.time()
                    v = v_primal(n, edges, datos(ph, nb * u))
                    w = 1 / (1 + nb)
                    filas.append([nombre, n, caso, modo, nb, v, w / chi, min(1 / 3, w)])
                    print(f"{nombre:9s} {caso:9s} {modo:4s} |b| = {nb:3.1f}: v_max = {v:.7f}   "
                          f"ventana [{w / chi:.4f}, {min(1 / 3, w):.4f}]  ({time.time() - t0:.1f} s)", flush=True)
    write_csv("parches_boosts_vmax.csv", ("grafo", "N", "datos", "direccion", "b", "v_max", "inf", "sup"), filas)


def parte_paridad():
    n, edges, ph, _ = grafo("octaedro")
    E = len(edges)
    me = MaxEnt(n, edges)
    rng = np.random.default_rng(5)
    u = rng.normal(size=(E, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = 0.15
    filas = []

    def S(phis, bs):
        s, res, _ = me.resuelve(datos(phis, bs), v)
        return s, res

    # comprobaciones de gauge
    b0 = 0.6 * u
    s1, r1 = S(ph, b0)
    s2, r2 = S(-ph, b0 * [-1, 1])
    s3, r3 = S(ph, -b0)
    print(f"gauge R_x(pi): |S(phi,b) - S(-phi,(-b1,b2))| = {abs(s1 - s2):.1e}; R_z(pi): |S(phi,b) - S(phi,-b)| = "
          f"{abs(s1 - s3):.1e}  (residuos {max(r1, r2, r3):.1e})", flush=True)
    filas.append(["gauge", "Rx", 0.6, 1.0, s1, s2, (s1 - s2) / 2, max(r1, r2)])
    filas.append(["gauge", "Rz", 0.6, 1.0, s1, s3, (s1 - s3) / 2, max(r1, r3)])
    for modo, uu in (("eje", np.tile([1.0, 0.0], (E, 1))), ("azar", u)):
        for nb in (0.1, 0.2, 0.4, 0.8):
            for lam in (0.25, 0.5, 1.0):
                sp_, rp = S(lam * ph, nb * uu)
                sm, rm = S(-lam * ph, nb * uu)
                O = (sp_ - sm) / 2
                filas.append([modo, "impar", nb, lam, sp_, sm, O, max(rp, rm)])
                print(f"{modo:4s} |b| = {nb:3.1f}, flujo x {lam:4.2f}: S+ = {sp_:.10f}, S- = {sm:.10f}, "
                      f"O = {O: .3e}  (residuo {max(rp, rm):.1e})", flush=True)
    write_csv("parches_boosts_paridad.csv", ("direccion", "prueba", "b", "escala_flujo", "S_mas", "S_menos", "O", "residuo"),
              filas)


def control():
    """Sin boosts, el resolutor general coincide con el de sectores de sphere_entropia (octaedro con curvatura, v = 0,15)."""
    from holonomy import sphere as sph
    from sphere_entropia import Gibbs
    n, edges, ph, _ = grafo("octaedro")
    s, res, _ = MaxEnt(n, edges).resuelve(datos(ph, np.zeros((len(edges), 2))), 0.15)
    p = sph.polyhedron("octaedro")
    _, S, r = Gibbs(p, cas.data_phases(p, "curvado"), sph.edge_orbits(p, sph.symmetry_group(p))).resuelve(0.15)
    print(f"control: S general = {s:.12f}, S por sectores = {S:.12f}, diferencia {abs(s - S):.1e}", flush=True)
    assert abs(s - S) < 1e-9 and max(res, r) < 1e-9


def main():
    control()
    parte_paridad()
    parte_vmax()


if __name__ == "__main__":
    main()

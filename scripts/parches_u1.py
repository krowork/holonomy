"""Artículo III (parches): (P2) con simetría U(1) en vez de SU(2).

Si el qubit de cada parche es la helicidad de un modo sin masa a lo largo del generador l, el grupo de marcos de Carroll
es el grupo pequeño ISO(2) de Wigner: los boosts de Carroll actúan trivialmente y las rotaciones alrededor de l, como
R_z(phi). La única simetría de la vista de dos parches es entonces U(1), y la familia natural tiene dos parámetros:

    rho_e = (I - v_perp sum_{a,b in {x,y}} R(phi_e)_ab s_a s_b - v_z Z Z)/4,     marginales I/2 (P1).

El conjunto de los (v_perp, v_z) compatibles es convexo. Se describe por su función soporte
h(c) = max {c_perp v_perp + c_z v_z : compatibles}, calculada de dos maneras:

* SDP primal (CVXPY, Clarabel) en el triángulo con varias holonomías, el tetraedro y el octaedro;
* sin datos y en grafos transitivos en aristas, como autovalor máximo de un XXZ: promediando con los automorfismos, la
  rotación global U(1) y R_x(pi)^{xN} (que fija los datos triviales y anula Bloch y la parte antisimétrica de xy), todo
  estado da uno con las mismas medias y cada arista de la forma de (P2u), así que
  h(c) = lambda_max( -(1/|E|) sum_e [c_perp (XX+YY)/2 + c_z ZZ] ). Se aplica al tetraedro, octaedro, icosaedro y toros
  triangulares 3x3 y 4x4 (el 3x4 no es transitivo en aristas).

Comprueba:
1. que la recta isótropa v_perp = v_z da los v_max de la tabla del artículo (sección "El nervio");
2. que los dos métodos coinciden donde se solapan;
3. el techo longitudinal v_z <= 1/3 con un triángulo;
y mide si hay vistas de pares entrelazadas compatibles: con T = diag(-v_perp, -v_perp, -v_z) en el gauge de la arista,
el par es separable si y solo si 2|v_perp| + |v_z| <= 1, así que hay pares entrelazados si y solo si
e = max_{s, s'} h(2s, s') > 1.
4. separa e por el signo de v_perp: e_+ (v_perp >= 0, el convenio tipo singlete del artículo) y e_- (v_perp <= 0,
   xy ferromagnético), con max {2|v_perp| + |v_z| : s v_perp >= 0} = max_t min_{mu >= 0} h(s (2 + mu), t). Cambiar
   el signo de v_perp es un gauge solo en grafos bipartitos; en general equivale a sumar pi a la holonomía de cada
   ciclo impar (en el triángulo, f -> pi - f).

Escribe data/parches_u1_{resumen,soporte}.csv.

Uso: (ulimit -v 12000000; OMP_NUM_THREADS=8 python scripts/parches_u1.py)
"""
import _common  # noqa: F401
from _common import CLARABEL_OPTS

import itertools
from functools import reduce

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh

from holonomy import sphere
from holonomy.report import write_csv
import sphere_casquetes as cas
from parches_nervios import toro

FALLOS = []
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1.0 + 0j, -1.0])
PAULI = (PX, PY, PZ)
SIGNOS = [(s, t) for s in (1, -1) for t in (1, -1)]
NDIR = 72


def ok(cond, msg):
    print(("OK   " if cond else "FALLO ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


def op(n, d, sparse=False):
    if sparse:
        o = [sp.identity(2, format="csr", dtype=complex)] * n
        for k, v in d.items():
            o[k] = sp.csr_matrix(v)
        return reduce(lambda a, b: sp.kron(a, b, format="csr"), o)
    o = [np.eye(2)] * n
    for k, v in d.items():
        o[k] = v
    return reduce(np.kron, o)


# ---------------------------------------------------------------------------
# Función soporte por el SDP primal
# ---------------------------------------------------------------------------

class SDP:
    def __init__(self, n, edges, phis):
        self.R = cp.Variable((2**n, 2**n), hermitian=True)
        self.vp, self.vz = cp.Variable(), cp.Variable()
        R = self.R
        cons = [R >> 0, cp.real(cp.trace(R)) == 1]
        for i in range(n):
            cons += [cp.real(cp.trace(R @ op(n, {i: P}))) == 0 for P in PAULI]
        for (i, j), ph in zip(edges, phis):
            c, s = np.cos(ph), np.sin(ph)
            Mxy = np.array([[c, -s], [s, c]])
            for a, b in itertools.product(range(3), repeat=2):
                ev = cp.real(cp.trace(R @ op(n, {i: PAULI[a], j: PAULI[b]})))
                if a < 2 and b < 2:
                    cons.append(ev == -self.vp * Mxy[a, b])
                elif a == 2 and b == 2:
                    cons.append(ev == -self.vz)
                else:
                    cons.append(ev == 0)
        self.cons = cons
        self.cp_, self.cz_ = cp.Parameter(), cp.Parameter()
        self.prob = cp.Problem(cp.Maximize(self.cp_ * self.vp + self.cz_ * self.vz), cons)
        self.iso = cp.Problem(cp.Maximize(self.vp), cons + [self.vp == self.vz])

    def h(self, c):
        self.cp_.value, self.cz_.value = float(c[0]), float(c[1])
        self.prob.solve(solver="CLARABEL", **CLARABEL_OPTS)
        return self.prob.value, (self.vp.value, self.vz.value)

    def v_iso(self):
        self.iso.solve(solver="CLARABEL", **CLARABEL_OPTS)
        return self.iso.value


# ---------------------------------------------------------------------------
# Función soporte por el XXZ (sin datos, transitivo en aristas)
# ---------------------------------------------------------------------------

class XXZ:
    def __init__(self, n, edges):
        E = len(edges)
        self.A = sum(op(n, {i: PX, j: PX}, True) + op(n, {i: PY, j: PY}, True) for i, j in edges) * (-0.5 / E)
        self.B = sum(op(n, {i: PZ, j: PZ}, True) for i, j in edges) * (-1.0 / E)
        self.dense = 2**n <= 1024
        if self.dense:
            self.A, self.B = self.A.toarray(), self.B.toarray()

    def h(self, c):
        H = c[0] * self.A + c[1] * self.B
        if self.dense:
            w, U = np.linalg.eigh(H)
            psi = U[:, -1]
            lam = w[-1]
        else:
            w, U = eigsh(H, k=1, which="LA", tol=1e-12)
            psi, lam = U[:, 0], w[0]
        vp = np.real(psi.conj() @ (self.A @ psi))
        vz = np.real(psi.conj() @ (self.B @ psi))
        return lam, (vp, vz)

    def v_iso(self):
        # (v, v) en la región si y solo si v (c_p + c_z) <= h(c) para toda c con c_p + c_z > 0
        f = lambda a: self.h((np.cos(a), np.sin(a)))[0] / (np.cos(a) + np.sin(a))  # noqa: E731
        r = minimize_scalar(f, bounds=(-np.pi / 4 + 1e-6, 3 * np.pi / 4 - 1e-6), method="bounded",
                            options=dict(xatol=1e-10))
        return r.fun


# ---------------------------------------------------------------------------

def e_lado(solver, s):
    """max {2|v_perp| + |v_z| : s v_perp >= 0} por dualidad de Lagrange en la restricción de signo."""
    mejor = -np.inf
    for t in (1.0, -1.0):
        f = lambda m: solver.h((s * (2.0 + m), t))[0]  # noqa: E731
        r = minimize_scalar(f, bounds=(0.0, 50.0), method="bounded", options=dict(xatol=1e-9))
        mejor = max(mejor, min(r.fun, f(0.0)))
    return mejor


def analiza(nombre, metodo, solver, v_tabla, filas_r, filas_s):
    dirs = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, NDIR, endpoint=False)]
    for c in dirs:
        hv, (vp, vz) = solver.h(c)
        filas_s.append([nombre, metodo, c[0], c[1], hv, vp, vz])
    vz_max = solver.h((0.0, 1.0))[0]
    vp_max, vp_min = solver.h((1.0, 0.0))[0], -solver.h((-1.0, 0.0))[0]
    ents = []
    for s, t in SIGNOS:
        hv, punto = solver.h((2.0 * s, 1.0 * t))
        ents.append((hv, punto))
    e, punto = max(ents, key=lambda x: x[0])
    e_mas, e_menos = e_lado(solver, 1), e_lado(solver, -1)
    ok(abs(max(e_mas, e_menos) - e) < 1e-5, f"{nombre} [{metodo}]: max(e_+, e_-) = e")
    viso = solver.v_iso()
    filas_r.append([nombre, metodo, viso, v_tabla, vz_max, vp_min, vp_max, e, punto[0], punto[1], e_mas, e_menos])
    print(f"  {nombre:24s} [{metodo}] v_iso = {viso:.7f} (tabla {v_tabla}), v_z <= {vz_max:.4f}, "
          f"v_perp en [{vp_min:.4f}, {vp_max:.4f}], e = {e:.4f} en ({punto[0]:.4f}, {punto[1]:.4f})"
          f"{'  PARES ENTRELAZADOS' if e > 1 + 1e-6 else ''}; e_+ = {e_mas:.5f}, e_- = {e_menos:.5f}", flush=True)
    if v_tabla is not None:
        ok(abs(viso - v_tabla) < 2e-6, f"{nombre} [{metodo}]: recta isótropa = v_max de la tabla")
    return dict(viso=viso, vz=vz_max, e=e, punto=punto, e_mas=e_mas, e_menos=e_menos)


def main():
    filas_r, filas_s = [], []

    print("\n1. Triángulo con holonomía f (SDP)")
    tri = ((0, 1), (1, 2), (0, 2))
    res_tri = {}
    for k in (0, 1 / 3, 1 / 2, 2 / 3, 1):
        # f = k pi = phi_01 + phi_12 - phi_02
        res_tri[k] = analiza(f"triángulo f={k:.3f}pi", "SDP", SDP(3, tri, (0.0, 0.0, -k * np.pi)), 1 / 3, filas_r,
                             filas_s)
        ok(abs(res_tri[k]["vz"] - 1 / 3) < 1e-6, f"triángulo f={k:.3f}pi: v_z <= 1/3, con igualdad")
    # el ejemplo (W + Wbar)/2: (v_perp, v_z) = (-2/3, 1/3) con f = 0, e = 5/3
    ok(abs(res_tri[0]["e"] - 5 / 3) < 1e-6, f"triángulo f=0: e = 5/3 ((W + Wbar)/2), calculado {res_tri[0]['e']:.6f}")
    for k, k2 in ((0, 1), (1 / 3, 2 / 3)):
        a, b = res_tri[k], res_tri[k2]
        ok(abs(a["e_mas"] - b["e_menos"]) < 1e-5 and abs(a["e_menos"] - b["e_mas"]) < 1e-5,
           f"triángulo: cambiar el signo de v_perp es f -> pi - f (f = {k:.3f}pi)")
    ok(abs(res_tri[0]["e_mas"] - 1) < 1e-6 and abs(res_tri[1]["e_mas"] - 5 / 3) < 1e-6,
       "triángulo, convenio tipo singlete: sin pares entrelazados con f = 0, e_+ = 5/3 con f = pi")

    print("\n2. Poliedros y toros")
    TABLA = {("tetraedro", "trivial"): 1 / 3, ("tetraedro", "curvado"): 1 / 3, ("octaedro", "trivial"): 1 / 3,
             ("octaedro", "curvado"): 0.2962963, ("icosaedro", "trivial"): 0.2750173}
    comp = {}
    for nombre, caso in TABLA:
        p = sphere.polyhedron(nombre)
        ph = cas.data_phases(p, caso)
        if p.n <= 6:
            comp[nombre, caso, "SDP"] = analiza(f"{nombre} {caso}", "SDP", SDP(p.n, p.edges, ph), TABLA[nombre, caso],
                                                filas_r, filas_s)
        if caso == "trivial":
            comp[nombre, caso, "XXZ"] = analiza(f"{nombre} {caso}", "XXZ", XXZ(p.n, p.edges), TABLA[nombre, caso],
                                                filas_r, filas_s)
    for L1, L2, vt in ((3, 3, 0.2592593), (4, 4, 0.2376532)):
        g = toro(L1, L2)[0]
        comp[f"toro{L1}x{L2}", "trivial", "XXZ"] = analiza(f"toro {L1}x{L2}", "XXZ", XXZ(g.n, g.edges), vt, filas_r,
                                                          filas_s)
    for nombre in ("tetraedro", "octaedro"):
        a, b = comp[nombre, "trivial", "SDP"], comp[nombre, "trivial", "XXZ"]
        ok(abs(a["e"] - b["e"]) < 1e-5 and abs(a["vz"] - b["vz"]) < 1e-5,
           f"{nombre} trivial: SDP y XXZ coinciden (e = {a['e']:.6f} y {b['e']:.6f})")
    for clave in (("octaedro", "trivial", "SDP"), ("icosaedro", "trivial", "XXZ"), ("toro3x3", "trivial", "XXZ"),
                  ("toro4x4", "trivial", "XXZ")):
        ok(abs(comp[clave]["e_mas"] - 1) < 1e-6, f"{clave[0]} sin datos: ningún par entrelazado con v_perp >= 0")
    ok(abs(comp["octaedro", "curvado", "SDP"]["e"] - 1) < 1e-6, "octaedro curvado: ningún par entrelazado")
    for clave, r in comp.items():
        ok(r["vz"] < 1 / 3 + 1e-6, f"{clave[0]} {clave[1]} [{clave[2]}]: v_z <= 1/3 (techo longitudinal)")

    write_csv("parches_u1_resumen.csv", ("grafo", "metodo", "v_iso", "v_max_tabla", "vz_max", "vperp_min", "vperp_max",
                                         "e", "vperp_e", "vz_e", "e_mas", "e_menos"), filas_r)
    write_csv("parches_u1_soporte.csv", ("grafo", "metodo", "c_perp", "c_z", "h", "vperp", "vz"), filas_s)
    print(f"\n{'TODO OK' if not FALLOS else f'{len(FALLOS)} FALLOS'}")
    raise SystemExit(bool(FALLOS))


if __name__ == "__main__":
    main()

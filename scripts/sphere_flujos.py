"""Búsqueda de flujos de cara generales en octaedro y cubo: ¿el flujo uniforme por
ángulo sólido es el mínimo de v_max?

Octaedro: flujos f_p con suma 0 mód 2 pi (7 libres), primal U(1) en al menos 500 puntos
al azar, Nelder-Mead desde el mejor y 40 perturbaciones del uniforme de cada tamaño
0,1 y 0,3 (delta de suma nula con |delta|_inf = tamaño). En el mejor punto y en la
perturbación más baja de cada tamaño, testigo dual sin simetría (OrbitDual, un (a, b, c)
por arista): cota superior certificada lambda_max/N e inferior por planos de corte.
Cubo (--cubo): 10 perturbaciones del uniforme de tamaño 0,2 con el primal, cada una con su
certificado dual. En ambos, además, v con flujo uniforme 2 pi j/F para todo j y barridos
con el dual en cada dirección irreducible de O_h sobre las caras (octaedro: A_2u, T_1u,
T_2g; cubo: T_1u, E_g), que cubren todas las direcciones de flujo por simetría y no solo
las muestreadas. Con --cuartico: cubo, orden s^4 de T_1u con E_g relajada, en u = (1,0,0),
(1,1,0) y (1,1,1), en s = 0,2 y 0,3. Con --t1u: cubo, barrido fino de T_1u en s de 0,005 a 0,2, con testigo por
órbitas del subgrupo que conserva los flujos y cotas a 2e-10: v baja en s < 0,038, de modo que el uniforme es
una silla a escala 5e-9 y T_1u sí tiene término cuadrático, muy pequeño. Escribe data/sphere_t1u.csv.

Semillas fijas. Cada evaluación se añade a data/sphere_flujos.csv; relanzar reutiliza las
hechas (incluido el recorrido de Nelder-Mead, que es determinista). Resumen en
data/sphere_flujos_resumen.csv.

Comprobaciones: v en el uniforme reproduce 8/27 y 0,424853; primal y dual coinciden a 1e-6
en el mejor punto y en una muestra de puntos con estado no 'optimal'; v(j) = v(F - j).

Uso: (ulimit -v 25000000; python scripts/sphere_flujos.py)          # octaedro
     (ulimit -v 25000000; python scripts/sphere_flujos.py --cubo)   # ~10 GB por punto
     (ulimit -v 25000000; python scripts/sphere_flujos.py --cuartico)
     (ulimit -v 25000000; python scripts/sphere_flujos.py --t1u)
"""
import _common  # noqa: F401
from _common import Checker

import argparse
import csv
import resource
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from holonomy import sdp, sphere
from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
EVALS = DATA / "sphere_flujos.csv"
RESUMEN = "sphere_flujos_resumen.csv"
CAB = ("poliedro", "fase", "indice", "flujos", "v", "estado", "segundos", "v_sup", "v_inf")
CAB_RES = ("poliedro", "fase", "n_eval", "v_min", "n_bajo_ref", "n_bajo_ref_1e6", "n_no_optimal",
           "dv_min", "dv_max", "flujos_min", "v_sup", "v_inf")
V_OCTA = 8 / 27
SEMILLA = 20261107
N_AZAR = 500
SIZES_OCTA = (0.1, 0.3)
N_PERT_OCTA = 40
SIZE_CUBO = 0.2
N_CUBO = 10
TMAX_CUBO = 180.0
MARGEN = 1e-6


def fmt(f):
    return ";".join(repr(float(x)) for x in f)


def data_phases(p, f):
    """Fases con holonomía f_p mód 2 pi, con la suma de flujos 2 pi m en la cara 0."""
    f = np.asarray(f, dtype=float)
    m = np.zeros(p.F)
    m[0] = np.round(f.sum() / (2 * np.pi))
    if abs(f.sum() - 2 * np.pi * m[0]) > 1e-9:
        raise ValueError("los flujos no suman 0 mód 2 pi")
    return sphere.phases(p, f, m=m)


class Store:
    """Evaluaciones ya hechas (por poliedro y flujos) y escritura incremental."""

    def __init__(self):
        self.known = {}
        self.rows = []
        if EVALS.exists():
            for r in csv.DictReader(open(EVALS)):
                self.known[r["poliedro"], r["flujos"]] = r
                self.rows.append(r)

    def primal(self, p, fase, indice, f):
        key = (p.name, fmt(f))
        if key in self.known:
            r = self.known[key]
            return float(r["v"]), r["estado"], float(r["segundos"])
        t0 = time.time()
        phis = data_phases(p, f)
        v, _, estado = sdp.primal_u1_graph(p.n, list(p.edges), phis, return_blocks=True)
        dt = time.time() - t0
        self.add(p.name, fase, indice, f, v, estado, dt)
        return v, estado, dt

    def add(self, nombre, fase, indice, f, v, estado, dt, v_sup="", v_inf=""):
        new = not EVALS.exists()
        fila = dict(poliedro=nombre, fase=fase, indice=indice, flujos=fmt(f), v=v, estado=estado,
                    segundos=dt, v_sup=v_sup, v_inf=v_inf)
        with open(EVALS, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(CAB)
            w.writerow([fila[c] for c in CAB])
        fila = {k: str(x) for k, x in fila.items()}
        self.known[nombre, fila["flujos"]] = fila
        self.rows.append(fila)


def dual_certificate(p, f, target=5e-9, max_iter=4000):
    """Testigo sin simetría: (cota superior, inferior, iteraciones)."""
    phis = data_phases(p, f)
    orbs = sphere.edge_orbits(p, sphere.identity_group(p))
    res = sphere.OrbitDual(p, phis, orbs, dense_max=2000, k_eigs=8, near=0.05).minimize(
        target=target, max_iter=max_iter)
    return res.v, max(res.lower, res.lower_dual), res.n_iter


def perturbation(rng, F, size):
    """delta de suma nula con |delta|_inf = size."""
    d = rng.normal(size=F)
    d -= d.mean()
    return d * size / np.max(np.abs(d))


def summary(nombre, fase, filas, ref, certificado=("", "")):
    vs = np.array([float(r["v"]) for r in filas])
    q = int(np.nanargmin(vs))
    return [nombre, fase, len(filas), vs[q], int(np.sum(vs < ref)), int(np.sum(vs < ref - MARGEN)),
            sum(r["estado"] != "optimal" for r in filas), float(np.nanmin(vs) - ref),
            float(np.nanmax(vs) - ref), filas[q]["flujos"], *certificado]


def run_octa(chk, store):
    p = sphere.polyhedron("octaedro")
    uni = np.full(p.F, np.pi / 2)
    v0, est, _ = store.primal(p, "uniforme", 0, uni)
    chk.check(abs(v0 - V_OCTA) < 1e-6, f"octaedro uniforme: primal {v0:.9f} ({est}), 8/27 = {V_OCTA:.9f}")

    filas = []
    rng = np.random.default_rng(SEMILLA)
    azar = rng.uniform(0, 2 * np.pi, size=(N_AZAR, p.F - 1))
    t0 = time.time()
    for q, x in enumerate(azar):
        f = np.r_[x, -x.sum()]
        store.primal(p, "azar", q, f)
        filas.append(store.known[p.name, fmt(f)])
        if (q + 1) % 100 == 0:
            chk.info(f"azar: {q + 1}/{N_AZAR}, {time.time() - t0:.0f} s")
    vs = np.array([float(r["v"]) for r in filas])
    best = int(np.nanargmin(vs))
    chk.info(f"azar: mínimo {vs[best]:.9f} en el punto {best}; por debajo de 8/27: "
             f"{int(np.sum(vs < V_OCTA))} ({int(np.sum(vs < V_OCTA - MARGEN))} con margen 1e-6)")

    cont = {"n": 0}
    nm_filas = []

    def objetivo(x):
        f = np.r_[x, -x.sum()]
        v, _, _ = store.primal(p, "nelder-mead", cont["n"], f)
        nm_filas.append(store.known[p.name, fmt(f)])
        cont["n"] += 1
        return v

    x0 = azar[best]
    simplex = np.vstack([x0] + [x0 + 0.3 * e for e in np.eye(p.F - 1)])
    r = minimize(objetivo, x0, method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, maxfev=1500, xatol=1e-4, fatol=1e-8))
    chk.info(f"Nelder-Mead: {cont['n']} evaluaciones, v = {r.fun:.9f}, éxito = {r.success}, {r.message}")

    res = [summary(p.name, "azar", filas, V_OCTA), summary(p.name, "nelder-mead", nm_filas, V_OCTA)]
    rng = np.random.default_rng(SEMILLA + 1)
    for size in SIZES_OCTA:
        pert = []
        for q in range(N_PERT_OCTA):
            f = uni + perturbation(rng, p.F, size)
            store.primal(p, f"pert{size}", q, f)
            pert.append(store.known[p.name, fmt(f)])
        res.append(summary(p.name, f"pert{size}", pert, V_OCTA))
        dv = np.array([float(x["v"]) for x in pert]) - V_OCTA
        chk.info(f"perturbaciones de tamaño {size}: v - 8/27 entre {dv.min():+.3e} y {dv.max():+.3e}; "
                 f"por debajo con margen 1e-6: {int(np.sum(dv < -MARGEN))}")

    todas = filas + nm_filas
    vs = np.array([float(x["v"]) for x in todas])
    q = int(np.nanargmin(vs))
    f_best = np.array([float(x) for x in todas[q]["flujos"].split(";")])
    t0 = time.time()
    sup, inf, it = dual_certificate(p, f_best)
    chk.check(abs(sup - vs[q]) < 1e-6 and sup - inf < 1e-8,
              f"octaedro, mejor punto: primal {vs[q]:.9f}, dual: sup {sup:.10f}, inf {inf:.10f}, "
              f"dif {sup - inf:.1e} ({it} iter, {time.time() - t0:.1f} s)")
    chk.info(f"mejor flujo (mód 2pi, en unidades de pi): "
             f"{np.round(np.mod(f_best, 2 * np.pi) / np.pi, 6).tolist()}; "
             f"8/27 - sup = {V_OCTA - sup:+.3e}")
    suu, inu, _ = dual_certificate(p, uni)
    chk.check(abs(suu - V_OCTA) < 1e-8 and suu - inu < 1e-8,
              f"octaedro uniforme sin simetría: sup {suu:.10f}, inf {inu:.10f}")
    res.append(summary(p.name, "mejor", [todas[q]], V_OCTA, (sup, inf)))

    # Lo que baja de 8/27 solo está demostrado con la cota superior del testigo.
    for size in SIZES_OCTA:
        pert = [x for x in store.rows if x["poliedro"] == p.name and x["fase"] == f"pert{size}"]
        worst = min(pert, key=lambda x: float(x["v"]))
        fw = np.array([float(x) for x in worst["flujos"].split(";")])
        sup, inf, _ = dual_certificate(p, fw)
        chk.check(sup < V_OCTA - MARGEN and sup - inf < 1e-8,
                  f"octaedro, perturbación {size} más baja: primal {float(worst['v']):.9f} ({worst['estado']}), "
                  f"dual sup {sup:.10f}, inf {inf:.10f}; 8/27 - sup = {V_OCTA - sup:+.3e}")
        res.append(summary(p.name, f"pert{size}-certificada", [worst], V_OCTA, (sup, inf)))

    # Fiabilidad de los puntos con estado 'optimal_inaccurate': contraste con el dual.
    rng = np.random.default_rng(SEMILLA + 3)
    malas = [x for x in filas if x["estado"] != "optimal"]
    muestra = rng.choice(len(malas), size=min(20, len(malas)), replace=False)
    err = 0.0
    for q in muestra:
        fw = np.array([float(x) for x in malas[int(q)]["flujos"].split(";")])
        sup, inf, _ = dual_certificate(p, fw)
        err = max(err, abs(float(malas[int(q)]["v"]) - sup))
    chk.check(err < 1e-6, f"octaedro: {len(muestra)} de {len(malas)} puntos al azar con estado no 'optimal': "
              f"|primal - dual sup| máx. = {err:.1e}")

    res += uniform_table(chk, p)
    res += scan(chk, p, uni, suu, octa_directions(p))
    return res


def irreducible_directions(p):
    """Direcciones de flujo del cubo de suma nula. T_1u: +1 en una cara y -1 en la opuesta;
    E_g: dos pares opuestos con signos contrarios."""
    c = np.array([p.verts[list(f)].mean(axis=0) for f in p.faces])
    eje = [np.array([np.abs(x @ e) > 0.5 for x in c]) for e in np.eye(3)]
    t1u = np.array([np.sign(x[0]) if abs(x[0]) > 0.5 else 0.0 for x in c])
    e_g = eje[0].astype(float) - eje[1].astype(float)
    return dict(T1u=t1u, Eg=e_g)


def octa_directions(p):
    """Direcciones de flujo del octaedro de suma nula: A_2u (signo de xyz en el centro de la
    cara, las dos familias de caras alternas), T_1u (signo de x) y T_2g (signo de yz)."""
    c = np.array([p.verts[list(f)].mean(axis=0) for f in p.faces])
    return dict(A2u=np.sign(c[:, 0] * c[:, 1] * c[:, 2]), T1u=np.sign(c[:, 0]),
                T2g=np.sign(c[:, 1] * c[:, 2]))


def scan(chk, p, uni, sup0, dirs):
    """v(uniforme + s d) con el dual sin simetría, en cada dirección irreducible."""
    res = []
    for nombre, d in dirs.items():
        chk.check(abs(d.sum()) < 1e-12, f"{p.name}: dirección {nombre} de suma nula: {d.tolist()}")
        d = d / np.max(np.abs(d))
        for s_ in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
            f = uni + s_ * d
            sup, inf, _ = dual_certificate(p, f)
            chk.info(f"{p.name} barrido {nombre} s = {s_:+.2f}: sup - v0 = {sup - sup0:+.6e}, "
                     f"inf - v0 = {inf - sup0:+.6e}")
            res.append([p.name, f"barrido-{nombre}-{s_}", 1, sup, int(sup < sup0), int(sup < sup0 - MARGEN),
                        0, sup - sup0, sup - sup0, fmt(f), sup, inf])
    return res


def uniform_table(chk, p):
    """Flujo uniforme 2 pi j/F, j = 0..F-1 (suma 2 pi j), con el dual reducido de una
    órbita: la curvatura por ángulo sólido es j = 2. Comprueba v(j) = v(F - j)."""
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    res, vals = [], {}
    for j in range(p.F):
        f = np.full(p.F, 2 * np.pi * j / p.F)
        r = sphere.reduced_dual(p, data_phases(p, f), orbs)
        vals[j] = r.v
        chk.info(f"{p.name} uniforme j = {j} (f = {2 * j / p.F:.4f} pi): v = {r.v:.10f}, dif {r.gap:.1e}, "
                 f"t = {r.t:.8f}, k = {r.kmax}")
        res.append([p.name, f"uniforme-j{j}", 1, r.v, "", "", 0, "", "", fmt(f), r.v, r.lower])
    err = max(abs(vals[j] - vals[p.F - j]) for j in range(1, p.F))
    chk.check(err < 2e-8, f"{p.name}: v(j) = v(F - j) (inversión de los flujos), |dif| máx. = {err:.1e}")
    return res


def run_cubo(chk, store):
    p = sphere.polyhedron("cubo")
    uni = np.full(p.F, 2 * np.pi / 3)
    t0 = time.time()
    v0, est, dt0 = store.primal(p, "uniforme", 0, uni)
    chk.check(abs(v0 - 0.424853) < 1e-5, f"cubo uniforme: primal {v0:.9f} ({est}), {dt0:.0f} s")
    sup0, inf0, _ = dual_certificate(p, uni)
    chk.check(sup0 - inf0 < 1e-8 and abs(sup0 - v0) < 1e-6,
              f"cubo uniforme sin simetría: sup {sup0:.10f}, inf {inf0:.10f}")

    rng = np.random.default_rng(SEMILLA + 2)
    pert = [uni + perturbation(rng, p.F, SIZE_CUBO) for _ in range(N_CUBO)]
    filas, tiempos = [], []
    for q, f in enumerate(pert):
        if tiempos and max(tiempos) > TMAX_CUBO and q >= 6:
            chk.info(f"cubo: alguna evaluación pasó de {TMAX_CUBO:.0f} s; se recorta a 6")
            break
        v, est, dt = store.primal(p, f"pert{SIZE_CUBO}", q, f)
        tiempos.append(dt)
        sup, inf, it = dual_certificate(p, f)
        filas.append(dict(store.known[p.name, fmt(f)]))
        chk.check(abs(sup - v) < 1e-6 and sup - inf < 1e-8,
                  f"cubo pert. {q}: primal {v:.9f} ({est}, {dt:.0f} s), dual sup {sup:.10f}, inf {inf:.10f}, "
                  f"inf - v0 = {inf - sup0:+.3e}")
    dv = np.array([float(x["v"]) for x in filas]) - v0
    chk.info(f"cubo: v - v0 entre {dv.min():+.3e} y {dv.max():+.3e}, {len(filas)} perturbaciones")
    res = [summary(p.name, f"pert{SIZE_CUBO}", filas, v0)]

    res += uniform_table(chk, p)
    res += scan(chk, p, uni, sup0, irreducible_directions(p))
    chk.info(f"cubo: {time.time() - t0:.0f} s; pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GB")
    return res


def run_cubo_cuartico(chk):
    """En T_1u y s grande domina el término cuártico (--t1u da el cuadrático, muy pequeño). Con
    E_g relajada (el acoplo x x y puede volverlo negativo), es una forma cúbica invariante
    A|u|^4 + B sum u_i^4 + ..., cuyos extremos en la esfera están en (1,0,0) y (1,1,1); se
    miden esos dos y (1,1,0), minimizando en la componente de E_g que conserva la simetría."""
    p = sphere.polyhedron("cubo")
    uni = np.full(p.F, 2 * np.pi / 3)
    sup0, inf0, _ = dual_certificate(p, uni)
    c = np.array([p.verts[list(f)].mean(axis=0) for f in p.faces])
    res = []
    for nombre, u, e in (("100", (1, 0, 0), (1, -0.5, -0.5)), ("110", (1, 1, 0), (-0.5, -0.5, 1)),
                         ("111", (1, 1, 1), None)):
        u = np.array(u, dtype=float)
        dx = c @ u / np.max(np.abs(c @ u))
        de = None if e is None else np.abs(c) @ np.array(e)
        chk.check(abs(dx.sum()) < 1e-12 and (de is None or abs(de.sum()) < 1e-12),
                  f"cubo: u = {nombre}, T_1u {dx.tolist()}, E_g {None if de is None else de.tolist()}")
        for x in (0.2, 0.3):
            f0 = uni + x * dx
            if de is None:
                y_opt = 0.0
                sup, inf, _ = dual_certificate(p, f0)
                evals = 1
            else:
                memo = {}

                def obj(y):
                    memo[y] = dual_certificate(p, f0 + y * de)
                    return memo[y][0]

                r = minimize_scalar(obj, bounds=(-0.1, 0.1), method="bounded",
                                    options=dict(xatol=1e-3, maxiter=14))
                y_opt = float(r.x)
                evals = len(memo)
                sup, inf, _ = memo.get(y_opt, dual_certificate(p, f0 + y_opt * de))
            chk.info(f"cubo T_1u {nombre} x = {x}: y* = {y_opt:+.5f}, sup - v0 = {sup - sup0:+.6e}, "
                     f"inf - v0 = {inf - sup0:+.6e}, (sup - v0)/x^4 = {(sup - sup0) / x**4:+.5f}, {evals} eval")
            res.append([p.name, f"cuartico-{nombre}-{x}", evals, sup, int(sup < sup0), int(sup < sup0 - MARGEN),
                        0, sup - sup0, sup - sup0, fmt(f0 + y_opt * (0 if de is None else de)), sup, inf])
    return res


def run_cubo_t1u(chk):
    """Flujos +-s en dos caras opuestas del cubo. Cotas del dual por órbitas del subgrupo que conserva los
    flujos; la superior vale para cualquier testigo con N = 1."""
    p = sphere.polyhedron("cubo")
    grupo_total = sphere.symmetry_group(p)
    caras = [frozenset(c) for c in p.faces]
    base = 2 * np.pi / 3

    def cotas(f):
        grupo = [g for g in grupo_total
                 if all(abs(f[caras.index(frozenset(g.perm[v] for v in c))] - f[q]) < 1e-12
                        for q, c in enumerate(p.faces))]
        m = np.zeros(p.F)
        m[0] = np.round(f.sum() / (2 * np.pi))
        d = sphere.OrbitDual(p, sphere.phases(p, f, m=m), sphere.edge_orbits(p, grupo), dense_max=2000,
                             k_eigs=8, near=0.05)
        r = d.minimize(target=2e-10, max_iter=1500)
        return r.v, max(r.lower, r.lower_dual)

    sup0, inf0 = cotas(np.full(p.F, base))
    chk.info(f"uniforme: [{inf0:.12f}, {sup0:.12f}]")
    centros = np.array([p.verts[list(c)].mean(axis=0) for c in p.faces])
    q0 = 0
    q1 = int(np.argmax(np.linalg.norm(centros - centros[q0], axis=1)))
    filas, val = [], {}
    for x in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.14, 0.2):
        f = np.full(p.F, base)
        f[q0] -= x
        f[q1] += x
        sup, inf = cotas(f)
        val[x] = (sup, inf)
        filas.append([x, sup, inf, sup - sup0, inf - inf0])
        chk.info(f"s = {x:.3f}: [{inf:.12f}, {sup:.12f}], sup - v0 = {sup - sup0:+.3e}")
    write_csv("sphere_t1u.csv", ("s", "v_sup", "v_inf", "sup_menos_v0", "inf_menos_v0"), filas)
    chk.check(val[0.025][0] < inf0, f"silla: v(s = 0,025) <= {val[0.025][0]:.10f} < {inf0:.10f} <= v0")
    chk.check(val[0.05][1] > sup0 and val[0.1][1] > sup0, "por encima de s = 0,05 sube, también con la cota inferior")
    xs = np.array([x for x in val if x <= 0.08])
    ys = np.array([val[x][0] - sup0 for x in xs])
    a, b = np.linalg.lstsq(np.c_[xs**2, xs**4], ys, rcond=None)[0]
    chk.info(f"ajuste en s <= 0,08: v - v0 = {a:.2e} s^2 + {b:.2e} s^4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubo", action="store_true", help="cubo, en vez del octaedro")
    ap.add_argument("--cuartico", action="store_true", help="cubo: orden cuártico en T_1u con E_g relajada")
    ap.add_argument("--t1u", action="store_true", help="cubo: barrido fino de T_1u, silla a escala 5e-9")
    args = ap.parse_args()
    chk = Checker("sphere_flujos")
    if args.t1u:
        run_cubo_t1u(chk)
        chk.finish()
        return
    store = Store()
    if args.cuartico:
        res = run_cubo_cuartico(chk)
    else:
        res = run_cubo(chk, store) if args.cubo else run_octa(chk, store)
    prev = []
    path = DATA / RESUMEN
    if path.exists():
        nuevas = {(r[0], r[1]) for r in res}
        prev = [[r[c] for c in CAB_RES] for r in csv.DictReader(open(path))
                if (r["poliedro"], r["fase"]) not in nuevas]
    write_csv(RESUMEN, CAB_RES, prev + res)
    for r in res:
        chk.info(f"{r[0]} {r[1]}: {r[2]} eval, v_min = {r[3]:.9f}, por debajo de la referencia "
                 f"{r[4]} ({r[5]} con margen 1e-6), no 'optimal' {r[6]}")
    chk.check(True, f"pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GB")
    chk.finish()


if __name__ == "__main__":
    main()

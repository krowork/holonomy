"""Cadena abierta de 2 diamantes (7 qubits): mapas abelianos y orientación relativa (exploración).

Sitios y aristas de holonomy.diamond.chain_edges(2): uniones 0, 3, 6,
centrales (1, 2) y (4, 5), 10 aristas orientadas.

1. Mapa abeliano v(fA, fB) con holonomy.sdp.primal_u1_graph, los mismos
   flujos de triángulo en las dos celdas (gauge simétrico de
   holonomy.diamond.abelian_phases): malla 13x13 en [0, pi]^2 y la recta
   fA = -fB (12 puntos con fA en (0, pi]; fA = fB es la diagonal de la malla).
   Comprobaciones: v <= 1/3 + 1e-7 en todos los puntos (cota del triángulo; el primal
   U(1) acaba a menudo optimal_inaccurate y sobrepasa 1/3 en ~3e-8)
   y v = 1/3 a 1e-7 sin holonomía (fA = fB = 0).
2. Simetrías v(fA, fB) = v(fB, fA) = v(-fA, -fB) a 1e-7 en 5 puntos al azar
   (semilla fija). Son información: si alguna no vale se dice en la salida
   y no cuenta como FALLO.
3. Orientación relativa: H_A = R_z(theta_A), H_B = R_y(gamma) R_z(theta_B)
   R_y(gamma)^T en la celda 1 y el mismo par conjugado por Q en la celda 2
   (holonomy.diamond.relative_pairs), en el gauge de árbol
   (holonomy.diamond.chain_rotations), con GraphDual.v_dual_cg (el primal
   completo no abeliano de 7 qubits no cabe en memoria). En 4 puntos
   (theta_A, theta_B, gamma): (pi, pi, pi/2), el mínimo abeliano del paso 1
   (con gamma = 0), (pi/2, pi/2, pi/2) y uno genérico (0.6 pi, 0.85 pi,
   0.3 pi), min_Q v por Nelder-Mead sobre el vector de rotación de Q,
   5 arranques (v_rel). Comprobaciones: Q = I con datos abelianos reproduce
   primal_u1_graph a 1e-7; en el punto abeliano, Q = R_x(pi) invierte los
   dos flujos de la celda 2 y debe reproducir primal_u1_graph con flujos
   (fA, fB) y (-fA, -fB) a 1e-7; toda evaluación de v_dual_cg con estado
   distinto de 'optimal' es un FALLO y su valor no se usa (Nelder-Mead la ve
   como +inf).

Coste y tope (medido con /usr/bin/time en (pi, pi, pi/2), 2026-11-03): una
evaluación de v_dual_cg cuesta ~170 s en frío y ~140-150 s en caliente
(tol = 1e-8; con tol = 1e-6 solo baja a ~115 s, así que se deja 1e-8), con
pico de ~900 MB. Nelder-Mead sin tope en 3 variables, 4 puntos x 5
arranques, serían ~100 evaluaciones por arranque, unas 80 h. Por eso:
  - arranque en caliente: cada evaluación parte del testigo y de la última
    evaluación 'optimal' (y0) con un subespacio inicial de R0_WARM = 32
    autovectores (el rango final típico) en lugar de 6; ahorra ~10-15 %;
  - caché en disco reanudable, data/diamond_chain2_evals.csv (una fila por
    evaluación, con estado y diferencia de cotas; no se repite al relanzar);
  - tope de MAXFEV = 20 evaluaciones por arranque (las 4 del símplex
    inicial cuentan dentro). Efecto del tope: el Nelder-Mead se corta antes
    de converger a xatol = 1e-3 / fatol = 1e-9, así que cada min_Q es una
    cota superior del mínimo local de su arranque (sigue siendo un v
    certificado de un Q concreto), no el mínimo convergido; en la salida y
    en el CSV se dice cuántos arranques llegaron al tope.
Estimación: 4 x 5 x 20 = 400 evaluaciones, ~16 h, más ~25 min del mapa
abeliano (unos 8 s por punto con primal_u1_graph).
Medido (2026-11-04): 77 680 s (21,6 h) con la caché vacía, pico residente
0,94 GB; regenerar desde la caché (solo el mapa abeliano) 1 810 s, 0,91 GB.

Escribe data/diamond_chain2_abel.csv (fA, fB, v, estado), data/diamond_chain2_rel.csv
(theta_A, theta_B, gamma, v con Q = I (nan si su estado no es 'optimal') y su
estado, cota superior de min_Q v, vector de rotación de Q, evaluaciones,
fallos, arranques en el tope) y la caché data/diamond_chain2_evals.csv.

v_rel(tA, tB, g, starts=5, seed=0) se importa desde otros guiones (no
ejecuta nada al importar).

Uso: (ulimit -v 8000000; /usr/bin/time -f "%M KB" python scripts/diamond_chain2.py [--rapido])
--rapido (prueba de humo): malla 3x3, solo el punto abeliano, 1 arranque con tope 4;
escribe los mismos CSV con sufijo _rapido (caché aparte).
--rapido medido: 565 s y 968 MB de pico (/usr/bin/time -f "%M KB").
Ejecución completa: estimada en ~16 h (pico ~1 GB); tiempo y pico medidos
pendientes de la ejecución completa.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from holonomy import diamond
from holonomy.gdual import GraphDual
from holonomy.report import DATA, write_csv
from holonomy.sdp import primal_u1_graph

N, K = 7, 2
EDGES = diamond.chain_edges(K)
EVALS = DATA / "diamond_chain2_evals.csv"
EVAL_COLS = ["theta_A", "theta_B", "gamma", "rx", "ry", "rz", "v", "estado", "gap"]
SEED = 20261103
MAXFEV = 20
R0_WARM = 32
V_TOL = 1e-7
pi = np.pi

_cache = {}
_loaded = False
_dual = None
_warm = None
# Evaluaciones de v_dual_cg con estado distinto de 'optimal' (no se usan como valor).
fallos = []
# Datos de la última llamada a v_rel (ver su docstring).
info_v_rel = {"arranques_en_tope": 0}


def v_abel(fA, fB):
    """v de la cadena de 2 con flujos (fA, fB) en cada celda (escalares, o
    secuencias de longitud 2 para flujos distintos por celda), primal U(1).
    Devuelve (v, estado)."""
    phis = diamond.abelian_phases(K, fA, fB)
    v, _, st = primal_u1_graph(N, EDGES, phis, return_blocks=True, **CLARABEL_OPTS)
    return v, st


def _key(tA, tB, g, rv):
    return tuple(round(float(x), 10) for x in (tA, tB, g, *rv))


ESTADOS = ("optimal", "en_caja", "cota_invalida", "no_converge")


def _parse_eval(row):
    """(clave, (v, estado, gap)) de una fila de la caché, o None si la fila
    no es válida (número de campos, flotantes, estado conocido)."""
    if len(row) != len(EVAL_COLS) or row[7] not in ESTADOS:
        return None
    try:
        x = [float(c) for c in row[:7]] + [float(row[8])]
    except ValueError:
        return None
    return _key(*x[:3], x[3:6]), (x[6], row[7], x[7])


def load():
    """Carga la caché EVALS. Una última línea corrupta o truncada (corte a
    medio escribir) se descarta con un aviso y se quita del fichero, para que
    la siguiente fila no se pegue a ella; una línea corrupta en medio del
    fichero, o una cabecera distinta de EVAL_COLS, lanza ValueError."""
    global _loaded
    _loaded = True
    if not EVALS.exists():
        return
    with open(EVALS, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    if not lines:
        return
    if next(csv.reader(lines[:1])) != EVAL_COLS:
        raise ValueError(f"{EVALS}: cabecera inesperada")
    parsed = []
    for i, line in enumerate(lines[1:], start=2):
        row = next(csv.reader([line]), [])
        if not row:
            continue
        ev = _parse_eval(row) if line.endswith("\n") else None
        if ev is None:
            if i == len(lines):
                warnings.warn(f"{EVALS}: última línea {i} corrupta o truncada, se descarta: {line!r}")
                with open(EVALS, "w", newline="") as fh:
                    fh.writelines(lines[:-1])
                break
            raise ValueError(f"{EVALS}: línea {i} corrupta en medio del fichero: {line!r}")
        parsed.append(ev)
    _cache.update(parsed)


def v_dual(tA, tB, g, rv):
    """v_dual_cg en (theta_A, theta_B, gamma) con Q = exp(rv), con caché en
    disco. Devuelve (v, estado, diferencia de cotas); solo 'optimal' es un
    resultado."""
    global _dual, _warm
    if not _loaded:
        load()
    k = _key(tA, tB, g, rv)
    if k not in _cache:
        if _dual is None:
            _dual = GraphDual(N, EDGES)
        Q = Rotation.from_rotvec(k[3:]).as_matrix()
        Rs = diamond.chain_rotations(K, diamond.relative_pairs(*k[:3], Q))
        warm = dict(y0=_warm, r0=R0_WARM) if _warm is not None else {}
        v, y, st, gap = _dual.v_dual_cg(Rs, **warm, **CLARABEL_OPTS)
        if st == "optimal":
            _warm = y
        _cache[k] = (float(v), st, float(gap))
        new = not EVALS.exists()
        with open(EVALS, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(EVAL_COLS)
            w.writerow([*k, *_cache[k]])
    v, st, gap = _cache[k]
    if st != "optimal" and k not in {f[0] for f in fallos}:
        fallos.append((k, st, gap))
    return v, st, gap


def _objective(tA, tB, g):
    def f(rv):
        v, st, _ = v_dual(tA, tB, g, rv)
        return v if st == "optimal" else np.inf
    return f


def v_rel(tA, tB, g, starts=5, seed=0, maxfev=MAXFEV):
    """min_Q v de la cadena de 2 con el par (H_A, H_B) de (tA, tB, g) en la
    celda 1 y conjugado por Q en la celda 2: Nelder-Mead sobre el vector de
    rotación de Q desde `starts` arranques (el primero Q = I, los demás
    rotaciones uniformes de semilla `seed`), cada uno con tope de `maxfev`
    evaluaciones de v_dual_cg (ver el docstring del módulo). Solo cuentan las
    evaluaciones con estado 'optimal'; las demás quedan en `fallos`.
    Devuelve (v mínimo, vector de rotación de Q); (inf, nan) si ninguna
    evaluación fue 'optimal'. Por el tope, v es una cota superior de min_Q v
    (el v certificado de un Q concreto), no el mínimo convergido. Tras la
    llamada, info_v_rel["arranques_en_tope"] (diccionario del módulo) dice
    cuántos arranques acabaron en el tope."""
    rng = np.random.default_rng(seed)
    f = _objective(tA, tB, g)
    best = (f(np.zeros(3)), np.zeros(3))
    capped = 0
    for s in range(starts):
        x0 = np.zeros(3) if s == 0 else Rotation.random(random_state=rng.integers(2 ** 31)).as_rotvec()
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-9, maxfev=maxfev,
                                    initial_simplex=x0 + np.vstack([np.zeros(3), 0.3 * np.eye(3)])))
        capped += res.nfev >= maxfev
        if res.fun < best[0]:
            best = (float(res.fun), np.asarray(res.x))
    info_v_rel["arranques_en_tope"] = int(capped)
    if not np.isfinite(best[0]):
        return np.inf, np.full(3, np.nan)
    return best


def main():
    global EVALS
    rapido = "--rapido" in sys.argv
    chk = Checker("diamond_chain2" + (" --rapido" if rapido else ""))
    suf = "_rapido" if rapido else ""
    EVALS = DATA / f"diamond_chain2_evals{suf}.csv"

    # 1. Mapa abeliano.
    fs = np.linspace(0, pi, 3 if rapido else 13)
    rows, vals = [], {}
    for a in fs:
        for b in fs:
            v, st = v_abel(a, b)
            vals[(a, b)] = v
            rows.append(["malla", a, b, f"{v:.12f}", st])
    for a in fs[1:]:
        v, st = v_abel(a, -a)
        vals[(a, -a)] = v
        rows.append(["fA=-fB", a, -a, f"{v:.12f}", st])
    write_csv(f"diamond_chain2_abel{suf}.csv", ["caso", "fA", "fB", "v", "estado"], rows)
    vmax = max(vals.values())
    chk.check(vmax <= 1 / 3 + V_TOL, f"mapa abeliano ({len(vals)} puntos): v <= 1/3 + 1e-7, máx v - 1/3 = {vmax - 1 / 3:+.1e}")
    v0 = vals[(0.0, 0.0)]
    chk.check(abs(v0 - 1 / 3) < V_TOL, f"sin holonomía: v = {v0:.12f}, |v - 1/3| = {abs(v0 - 1 / 3):.1e}")
    nopt = sum(r[4] != "optimal" for r in rows)
    chk.info(f"primal U(1): {nopt} de {len(rows)} puntos con estado no 'optimal' (optimal_inaccurate de Clarabel)")
    # v(0, π) = v(π, 0) por simetría: se redondea para que el empate no lo decida el ruido del solver
    (amin, bmin), vmin = min(vals.items(), key=lambda t: (round(t[1], 8), t[0]))
    chk.info(f"mínimo de la malla: (fA, fB)/pi = ({amin / pi:.4f}, {bmin / pi:.4f}), v = {vmin:.10f}")
    for a in fs[1:]:
        chk.info(f"fA = {a / pi:.4f} pi: v(fA, fA) = {vals[(a, a)]:.10f}, v(fA, -fA) = {vals[(a, -a)]:.10f}")

    # 2. Simetrías (información).
    rng = np.random.default_rng(SEED)
    for a, b in rng.uniform(-pi, pi, (2 if rapido else 5, 2)):
        v1, v2, v3 = v_abel(a, b)[0], v_abel(b, a)[0], v_abel(-a, -b)[0]
        d12, d13 = abs(v1 - v2), abs(v1 - v3)
        msg = (f"simetrías en (fA, fB)/pi = ({a / pi:+.4f}, {b / pi:+.4f}): v = {v1:.10f}, "
               f"|v(fB,fA) - v| = {d12:.1e}, |v(-fA,-fB) - v| = {d13:.1e}")
        if d12 < V_TOL and d13 < V_TOL:
            chk.info("cumple " + msg)
        else:
            chk.info("NO CUMPLE (información, no fallo) " + msg)

    # 3. Orientación relativa.
    pts = [(pi, pi, pi / 2), (amin, bmin, 0.0)]
    if not rapido:
        pts += [(pi / 2, pi / 2, pi / 2), (0.6 * pi, 0.85 * pi, 0.3 * pi)]
    else:
        pts = pts[1:]
    rv0 = np.zeros(3)
    vI, st, gap = v_dual(amin, bmin, 0.0, rv0)
    chk.check(st == "optimal" and abs(vI - vmin) < V_TOL,
              f"Q = I, abeliano ({amin / pi:.4f}, {bmin / pi:.4f}) pi: v_dual_cg = {vI:.10f}, "
              f"primal U(1) = {vmin:.10f}, |dif| = {abs(vI - vmin):.1e} [estado {st}, gap {gap:.1e}]")
    vx, st = v_dual(amin, bmin, 0.0, (pi, 0.0, 0.0))[:2]
    vu, _ = v_abel((amin, -amin), (bmin, -bmin))
    chk.check(st == "optimal" and abs(vx - vu) < V_TOL,
              f"Q = R_x(pi), abeliano: v_dual_cg = {vx:.10f}, primal U(1) con flujos (fA, fB), (-fA, -fB) "
              f"= {vu:.10f}, |dif| = {abs(vx - vu):.1e} [estado {st}]")

    rel = []
    for p in pts:
        nf = len(fallos)
        vq, sq, _ = v_dual(*p, rv0)
        vmin_q, rv = v_rel(*p, starts=1 if rapido else 5, maxfev=4 if rapido else MAXFEV)
        nev = sum(1 for k in _cache if k[:3] == _key(*p, rv0)[:3])
        capped = info_v_rel["arranques_en_tope"]
        if sq != "optimal":
            vq = np.nan  # no es un resultado (ya cuenta como FALLO en fallos)
        lab = f"({p[0] / pi:.4f}, {p[1] / pi:.4f}, {p[2] / pi:.4f}) pi"
        chk.info(f"{lab}: v(Q = I) = {vq:.10f} [{sq}], cota superior de min_Q v (Nelder-Mead con tope) = "
                 f"{vmin_q:.10f}, dif = {vmin_q - vq:+.2e}, Q = exp({np.array2string(rv, precision=4)}), "
                 f"{nev} evaluaciones, {capped} arranques en el tope")
        rel.append([*p, f"{vq:.12f}", sq, f"{vmin_q:.12f}", *rv, nev, len(fallos) - nf, capped])
    write_csv(f"diamond_chain2_rel{suf}.csv", ["theta_A", "theta_B", "gamma", "v_Q_I", "estado_Q_I",
                                         "v_min_Q_cota_sup", "rx", "ry", "rz",
                                         "evaluaciones", "fallos", "arranques_en_tope"], rel)
    for k, st, gap in fallos:
        chk.check(False, f"v_dual_cg en (theta_A, theta_B, gamma, rv) = {k}: estado {st}, gap {gap:.1e}; "
                  f"no se usa como valor")
    if not fallos:
        chk.check(True, f"todas las evaluaciones de v_dual_cg con estado 'optimal' ({len(_cache)} en la caché)")
    chk.finish()


if __name__ == "__main__":
    main()

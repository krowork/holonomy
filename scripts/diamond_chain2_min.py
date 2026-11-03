"""Cadena abierta de 2 diamantes (7 qubits): mínimo global de v y su testigo (exploración).

Datos: el par (H_A, H_B) de invariantes (theta_A, theta_B, gamma) en la
celda 1 y el mismo par conjugado por Q en la celda 2
(holonomy.diamond.relative_pairs, gauge de árbol de
holonomy.diamond.chain_rotations), Q = exp(rv) con rv su vector de rotación.
Cada v sale de GraphDual.v_dual_cg (el primal completo de 7 qubits no cabe);
solo cuenta el estado 'optimal' (la cota superior, a menos de tol de la
inferior); cualquier otro estado es un FALLO y nunca un valor (Nelder-Mead lo
ve como +inf).

Simetrías usadas para reducir la malla (demostradas aquí):
  S1. theta_A <-> theta_B. La reflexión de la cadena sigma: 0 <-> 6, 1 <-> 4,
      2 <-> 5, 3 fijo, es un automorfismo del grafo (lleva las aristas de una
      celda a las de la otra, algunas invertidas, y R_ba = R_ab^T). En el
      gauge de árbol, el lazo A de la celda nueva 1 es la imagen de
      j' -> u -> w -> j' de la celda vieja 0, de holonomía
      R_j'u R_uw R_wj' = H_A (H_B H_A)^T = H_B^T, y su lazo B la de
      u -> j -> w -> u, de holonomía H_A^T, ambas en el marco común (el
      árbol j -> u -> j' sigue siendo de identidades). Igual en la otra
      celda con Q. Así los datos nuevos son (Q P Q^T, P) con
      P = (H_B^T, H_A^T) = (R_{-n_B}(theta_B), R_{-z}(theta_A)), cuyos ejes
      forman el mismo ángulo gamma: P = G P' G^T con P' el par canónico de
      (theta_B, theta_A, gamma) y G la rotación z -> -n_B, n_B -> -z. Tras
      el gauge global (Q G)^T: v(theta_A, theta_B, gamma, Q) =
      v(theta_B, theta_A, gamma, G^T Q^T G), y Q -> G^T Q^T G recorre SO(3),
      así que min_Q v es simétrico y basta theta_A <= theta_B. Se comprueba
      en un punto genérico con Q explícita (paso 1 de las comprobaciones).
  S2. theta_A = pi o theta_B = pi: gamma <-> pi - gamma. R_n(pi) = R_{-n}(pi);
      con theta_A = pi el par de ángulo pi - gamma es el de gamma conjugado
      por R_x(pi) (fija R_z(pi), lleva R_y(gamma) z a R_y(pi - gamma) z); con
      theta_B = pi, por R_z(pi). Es un gauge global (Q pasa a G Q G^T): basta
      gamma <= pi/2.
  S3. theta_A = 0 o theta_B = 0: gamma no cuenta (un solo gamma, como en el
      plan); con theta_A = theta_B = 0 todo R es la identidad y v no depende
      de Q (una sola evaluación, v = 1/3).
Malla del plan con theta_A <= theta_B: 5 puntos con algún theta = 0,
4 pares con theta = pi x 4 gammas, 6 pares x 7 gammas: 63 puntos.

Etapa 1, cribado (CRIBADO: sus valores no son resultados): cada punto de la
malla con min_Q v por Nelder-Mead sobre rv desde Q = I (un arranque) con
tope MAXFEV_CRIBA y tol del CG TOL_CRIBA (v es una cota superior certificada
a menos de TOL_CRIBA de la inferior). El cribado puede perder un mínimo cuya
diferencia con los tres elegidos sea menor que la precisión del cribado,
que es TOL_CRIBA más lo que el tope deja sin bajar en Q (en (pi, pi, pi/2),
tarea 4: Q = I da 0,32965 y el min_Q de 5 arranques 0,32818).

Etapa 2, precisión completa (tol TOL_FINO = 1e-8): desde los 3 mejores puntos
del cribado, min_Q v con 5 arranques (el mejor Q del cribado y 4 rotaciones
al azar de semilla fija), tope MAXFEV por arranque, y después Nelder-Mead
conjunto en (theta_A, theta_B, gamma, rv) desde el punto y su mejor Q, con
tope MAXFEV_CONJ, los ángulos reflejados a [0, pi] (como
scripts/ladder_open3_min.py). Los topes son duros (el Nelder-Mead se corta en
la evaluación número tope): cada mínimo es una cota superior del mínimo local
de su arranque (un v certificado de datos concretos), no el mínimo
convergido. El mínimo final P es el menor v de la etapa 2.

Paso 2 en P:
  - v_dual_cg de nuevo (sin caché) para tener el testigo y sus subespacios
    finales; debe coincidir con el v de la caché a 1e-7;
  - perturbaciones P + d (sin reflejar, los datos son función suave de los
    6 parámetros) en los 6 ejes (eps = 0,1; 0,05; 0,025; ambos signos) y en
    12 direcciones al azar (eps = 0,1; 0,05), como scripts/ladder_perp_min.py,
    más las 15 diagonales (e_i + e_j)/sqrt(2) con eps = 0,05 (sin ellas los
    15 términos cruzados solo salen de 12 direcciones y el ajuste tiene rango
    30 de 33); 75 en total: dv > 0 en todas, y ajuste de dv a forma cuadrática + lineales + cúbicos
    diagonales: forma cuadrática definida positiva (P no es crítico por
    simetría, así que los términos lineales y el punto estacionario del
    ajuste se dan como información: miden cuánto dejó el tope sin converger);
    el ajuste solo cuenta si la matriz de diseño tiene rango completo (33),
    si no es un FALLO;
  - witness_spread con eps = 1e-4, 1e-5, 1e-6 sobre los subespacios finales
    (U = last_U): anchura máxima de los 90 coeficientes del testigo. Con U el
    conjunto comprimido contiene al verdadero, así que cada anchura es una
    cota superior de la verdadera: si las cotas bajan, las verdaderas también
    (prueba de unicidad); si no bajan, no se concluye nada (inconcluso, no
    "hay una cara"). Veredicto "testigo único" (OK) si los dos cocientes,
    1e-4 -> 1e-5 y 1e-5 -> 1e-6, pasan de 2,5 (~3,16 si va como sqrt(eps),
    10 si va como eps) y la anchura en 1e-6 es menor que ANCHO_REL = 0,1
    veces el mayor |y_m| del testigo: el testigo queda fijado al 10 % de su
    propia escala (los y_m del arranque de Heisenberg rotado son ~1/(3|E|) =
    1/30; una cota absoluta fija dependería de esa normalización N(y) = 1).
    Si no, "inconcluso (rango comprimido: solo cota superior)", como
    información. Los SDP optimal_inaccurate de witness_spread se cuentan en
    el resumen del Checker (status_note) y el veredicto lleva la marca
    "inexacto" con su número; no se descartan.
Comprobaciones también: v(0, 0, 0) = 1/3, S1 en un punto genérico, y cada
evaluación no 'optimal' es un FALLO.

Caché reanudable data/diamond_chain2_min_evals.csv (una fila por evaluación,
con su tol; una evaluación vale para cualquier petición con tol mayor o
igual, y no se repite al relanzar). Al arrancar se lee, solo para leer, la
caché de la tarea 4 (data/diamond_chain2_evals.csv, tol 1e-8): una clave que
esté allí se copia a la caché propia en lugar de recalcularse. La caché de la
tarea 4 nunca se escribe. El arranque en caliente (y0 de la última evaluación
'optimal', R0_WARM autovectores) no está en la clave: al reanudar, un valor
recalculado puede diferir ~1e-9 del de una corrida sin cortes.

Coste (medido 2026-11-03 con /usr/bin/time, máquina cargada, ~9 de 12
núcleos, en (pi, pi, pi/2)): una evaluación de v_dual_cg cuesta 101 s en frío
y 87 s en caliente con tol = 1e-5, 120 s con 1e-6 y 148 s con 1e-8 (en
caliente); pico 842 MB. witness_spread (180 SDP) ~20 min por eps. Estimación
de la corrida completa: cribado 62 x 8 llamadas, ~470 nuevas (las de
(pi, pi, pi/2), (0, pi, 0) y (pi/2, pi/2, pi/2) salen de la caché de la
tarea 4) x 87 s = ~11,5 h; etapa 2, 3 x (5 x 20 + 60) = 480 x 148 s =
~20 h; paso 2, 76 x 148 s = ~3,1 h más ~1 h de witness_spread: ~36 h en
total (~43 h si la máquina sigue cargada), pico ~1 GB.
--rapido medido: 1710 s (10 evaluaciones nuevas, 1 witness_spread) y 846 MB.

Escribe data/diamond_chain2_min.csv (caso: cribado | minQ | conjunto |
minimo; theta_A, theta_B, gamma, rx, ry, rz, v, tol, evaluaciones, en_tope),
data/diamond_chain2_min_pert.csv (d en los 6 parámetros, dv),
data/diamond_chain2_min_testigo.csv (eps, anchura máxima, SDP inexactos) y
la caché.

Uso: (ulimit -v 8000000; /usr/bin/time -f "%M KB" python scripts/diamond_chain2_min.py [--rapido])
--rapido (prueba de humo): malla (0, 0, 0) y (pi, pi, pi/2), tol 1e-4 y 1e-5,
topes 4 / 2 / 3, un arranque en Q, 4 perturbaciones, witness_spread con
eps = 1e-4; las comprobaciones de convergencia (dv > 0, rango del ajuste,
unicidad) pasan a información. Ficheros con sufijo _rapido (caché aparte).
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import diamond_chain2 as dc2
from holonomy import diamond
from holonomy.gdual import GraphDual
from holonomy.paulis import Ry
from holonomy.report import DATA, write_csv

N, K = 7, 2
EDGES = diamond.chain_edges(K)
pi = np.pi
SEED = 20261103
EVALS = DATA / "diamond_chain2_min_evals.csv"
EVALS_T4 = DATA / "diamond_chain2_evals.csv"  # caché de la tarea 4: solo lectura
EVAL_COLS = dc2.EVAL_COLS + ["tol", "origen"]
TOL_T4 = 1e-8  # tol de v_dual_cg en la tarea 4 (la de omisión)
R0_WARM = dc2.R0_WARM
V_TOL = 1e-7
ANCHO_REL = 0.1   # anchura del testigo en el menor eps, relativa a max |y_m|
COCIENTE = 2.5    # bajada mínima de la anchura por cada factor 10 en eps
GAP_NEG = GraphDual.GAP_NEG

# Configuración de la corrida completa (main la cambia con --rapido).
CFG = dict(TOL_CRIBA=1e-5, TOL_FINO=1e-8, MAXFEV_CRIBA=8, MAXFEV=20, MAXFEV_CONJ=60,
           MEJORES=3, ARRANQUES=5, EPS_PERT=(0.1, 0.05, 0.025), EPS_AZAR=(0.1, 0.05), N_AZAR=12,
           EJES=6, PARES=15, EPS_PAR=0.05, EPS_TESTIGO=(1e-4, 1e-5, 1e-6))
CFG_RAPIDO = dict(TOL_CRIBA=1e-4, TOL_FINO=1e-5, MAXFEV_CRIBA=4, MAXFEV=2, MAXFEV_CONJ=3,
                  MEJORES=1, ARRANQUES=1, EPS_PERT=(0.05,), EPS_AZAR=(0.05,), N_AZAR=1,
                  EJES=1, PARES=1, EPS_PAR=0.05, EPS_TESTIGO=(1e-4,))

_cache = {}      # clave -> [(v, estado, gap, tol, origen), ...]
_t4 = {}         # clave -> (v, estado, gap) de la caché de la tarea 4
_dual = None
_warm = None
fallos = {}      # clave -> (estado, gap): evaluaciones no 'optimal' usadas
cuenta = {"nuevas": 0, "cache": 0, "tarea4": 0}


def _parse(row):
    """(clave, (v, estado, gap, tol, origen)) de una fila de la caché propia,
    o None si no es válida."""
    if len(row) != len(EVAL_COLS) or row[-1] not in ("propia", "tarea4"):
        return None
    ev = dc2._parse_eval(row[:len(dc2.EVAL_COLS)])
    if ev is None:
        return None
    try:
        tol = float(row[-2])
    except ValueError:
        return None
    return ev[0], (*ev[1], tol, row[-1])


def _guardar(k, e):
    """Entrada e para la clave k: en memoria y añadida al fichero de caché
    (con cabecera si el fichero no existe o está vacío)."""
    _cache.setdefault(k, []).append(e)
    new = not EVALS.exists() or EVALS.stat().st_size == 0
    with open(EVALS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(EVAL_COLS)
        w.writerow([*k, *e])


def load():
    """Carga la caché propia (todas las entradas de cada clave) y la de la
    tarea 4 (solo lectura). Caché propia: una última línea corrupta o
    truncada se descarta con aviso y se quita del fichero; una corrupta en
    medio, o una cabecera distinta, lanza ValueError (como dc2.load). Caché de
    la tarea 4: otro proceso puede estar añadiendo filas, así que solo se
    toman las líneas completas y válidas y nunca se escribe."""
    _cache.clear()
    _t4.clear()
    if EVALS.exists():
        with open(EVALS, newline="") as fh:
            lines = fh.read().splitlines(keepends=True)
        if lines:
            if next(csv.reader(lines[:1])) != EVAL_COLS:
                raise ValueError(f"{EVALS}: cabecera inesperada")
            for i, line in enumerate(lines[1:], start=2):
                row = next(csv.reader([line]), [])
                if not row:
                    continue
                ev = _parse(row) if line.endswith("\n") else None
                if ev is None:
                    if i == len(lines):
                        warnings.warn(f"{EVALS}: última línea {i} corrupta o truncada, se descarta: {line!r}")
                        with open(EVALS, "w", newline="") as fh:
                            fh.writelines(lines[:-1])
                        break
                    raise ValueError(f"{EVALS}: línea {i} corrupta en medio del fichero: {line!r}")
                _cache.setdefault(ev[0], []).append(ev[1])
    if EVALS_T4.exists():
        with open(EVALS_T4, newline="") as fh:
            lines = fh.read().splitlines(keepends=True)
        if lines and next(csv.reader(lines[:1])) == dc2.EVAL_COLS:
            for line in lines[1:]:
                ev = dc2._parse_eval(next(csv.reader([line]), [])) if line.endswith("\n") else None
                if ev is not None:
                    _t4[ev[0]] = ev[1]


def rotations(x):
    """R_e de la cadena de 2 en x = (theta_A, theta_B, gamma, rx, ry, rz)."""
    Q = Rotation.from_rotvec(np.asarray(x[3:6], float)).as_matrix()
    return diamond.chain_rotations(K, diamond.relative_pairs(x[0], x[1], x[2], Q))


def _pick(k, tol):
    """La entrada de la caché para k que vale con tolerancia tol (tol propia
    <= tol): la 'optimal' de menor tol si hay alguna (un FALLO a tol más
    fina no tapa un valor válido), si no la no 'optimal' de menor tol;
    None si ninguna vale."""
    ok = [e for e in _cache.get(k, []) if e[3] <= tol]
    if not ok:
        return None
    return min(ok, key=lambda e: (e[1] != "optimal", e[3]))


def v_eval(x, tol):
    """v_dual_cg en x = (theta_A, theta_B, gamma, rx, ry, rz) con tolerancia
    tol, con caché (vale una evaluación de tol <= la pedida). Devuelve
    (v, estado, gap); solo 'optimal' es un resultado."""
    global _dual, _warm
    k = dc2._key(x[0], x[1], x[2], x[3:6])
    e = _pick(k, tol)
    t4 = _t4.get(k) if TOL_T4 <= tol else None
    if e is not None and (e[1] == "optimal" or t4 is None or t4[1] != "optimal"):
        cuenta["cache"] += 1
    elif t4 is not None:
        e = (*t4, TOL_T4, "tarea4")
        _guardar(k, e)
        cuenta["tarea4"] += 1
    else:
        if _dual is None:
            _dual = GraphDual(N, EDGES)
        warm = dict(y0=_warm, r0=R0_WARM) if _warm is not None else {}
        v, y, st, gap = _dual.v_dual_cg(rotations(k), tol=tol, **warm, **CLARABEL_OPTS)
        if st == "optimal":
            _warm = y
        e = (float(v), st, float(gap), float(tol), "propia")
        _guardar(k, e)
        cuenta["nuevas"] += 1
    if e[1] != "optimal":
        fallos[k] = (e[1], e[2])
    return e[0], e[1], e[2]


class _Tope(Exception):
    pass


def nelder_mead(f, x0, step, cap):
    """Nelder-Mead de f desde x0 con símplex inicial x0 + diag(step) y tope
    duro de cap llamadas a f (contando las que salen de la caché, para que la
    trayectoria sea la misma al reanudar). Devuelve (mejor f, mejor x,
    llamadas, si llegó al tope)."""
    best = [np.inf, np.asarray(x0, float).copy()]
    n = [0]

    def g(x):
        if n[0] >= cap:
            raise _Tope
        n[0] += 1
        val = f(x)
        if val < best[0]:
            best[0], best[1] = val, np.array(x, float)
        return val

    x0 = np.asarray(x0, float)
    simplex = x0 + np.vstack([np.zeros(len(x0)), np.diag(step)])
    try:
        minimize(g, x0, method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=1e-9, maxfev=10 ** 9, initial_simplex=simplex))
        capped = False
    except _Tope:
        capped = True
    return best[0], best[1], n[0], capped


def _val(x, tol):
    v, st, _ = v_eval(x, tol)
    return v if st == "optimal" else np.inf


def min_q(p, tol, x0s, cap):
    """min_Q v en p = (theta_A, theta_B, gamma) desde los vectores de rotación
    x0s, un Nelder-Mead con tope por arranque. Devuelve (v, rv, llamadas,
    arranques en el tope); (inf, nan) si nada fue 'optimal'."""
    best, rvb, calls, capped = np.inf, np.full(3, np.nan), 0, 0
    for x0 in x0s:
        v, rv, n, c = nelder_mead(lambda rv: _val((*p, *rv), tol), x0, [0.3] * 3, cap)
        calls, capped = calls + n, capped + c
        if v < best:
            best, rvb = v, rv
    return best, rvb, calls, capped


def fold(a):
    """Refleja cada ángulo a [0, pi]."""
    return np.abs(np.mod(np.asarray(a, float) + pi, 2 * pi) - pi)


def joint(x0, tol, cap):
    """Nelder-Mead conjunto en (theta_A, theta_B, gamma, rv) desde x0, con los
    ángulos reflejados a [0, pi]. Devuelve (v, x reflejado, llamadas, tope)."""
    v, x, n, c = nelder_mead(lambda x: _val((*fold(x[:3]), *x[3:]), tol), x0,
                             [-0.1 * pi] * 3 + [0.3] * 3, cap)
    return v, np.concatenate([fold(x[:3]), x[3:]]), n, c


def grid():
    """Malla del plan reducida por S1-S3 (ver el docstring)."""
    th = np.array([0, 0.25, 0.5, 0.75, 1.0]) * pi
    ga = np.arange(7) / 6 * pi
    pts = []
    for i, a in enumerate(th):
        for b in th[i:]:
            gs = ga[:1] if a == 0 or b == 0 else (ga[:4] if b == pi else ga)
            pts += [(a, b, g) for g in gs]
    return pts


def swap_q(g, rv):
    """Q' = G^T Q^T G de la simetría S1: v(tA, tB, g, Q) = v(tB, tA, g, Q')."""
    z = np.array([0.0, 0.0, 1.0])
    n = Ry(g) @ z
    F1 = np.column_stack([z, n, np.cross(z, n)])
    F2 = np.column_stack([-n, -z, np.cross(n, z)])
    G = F2 @ np.linalg.inv(F1)
    Q = Rotation.from_rotvec(rv).as_matrix()
    return Rotation.from_matrix(G.T @ Q.T @ G).as_rotvec()


def feats(d):
    """Cuadráticos (21), lineales (6) y cúbicos diagonales (6) en 6 variables."""
    iu = np.triu_indices(6)
    q = np.outer(d, d)[iu] * np.where(iu[0] == iu[1], 1.0, 2.0)
    return np.concatenate([q, d, d ** 3])


def quad_form(c):
    H = np.zeros((6, 6))
    H[np.triu_indices(6)] = c[:21]
    return H + np.triu(H, 1).T


def verdict(epss, widths, ymax):
    """(único, mensaje) de la unicidad del testigo con las anchuras máximas
    `widths` (cotas superiores, por el U comprimido) en los eps `epss`
    decrecientes: único si cada cociente entre eps consecutivos pasa de
    COCIENTE y la última anchura es < ANCHO_REL * ymax (ver el docstring);
    si no, inconcluso. Necesita al menos 3 eps."""
    if len(widths) < 3 or not np.all(np.isfinite(widths)):
        return False, (f"testigo: inconcluso (hacen falta 3 anchuras válidas, hay "
                       f"{int(np.sum(np.isfinite(widths)))})")
    rs = [widths[i] / widths[i + 1] for i in range(len(widths) - 1)]
    tope = ANCHO_REL * ymax
    txt = (", ".join(f"{epss[i]:.0e} -> {epss[i + 1]:.0e}: {r:.2f}" for i, r in enumerate(rs))
           + f"; anchura en {epss[-1]:.0e} = {widths[-1]:.2e} (cota {ANCHO_REL} max|y| = {tope:.2e})")
    if all(r > COCIENTE for r in rs) and widths[-1] < tope:
        return True, f"testigo único: la anchura baja (cocientes > {COCIENTE}) {txt}"
    return False, f"testigo: inconcluso (rango comprimido: solo cota superior), {txt}"


def lab(x):
    s = f"({x[0] / pi:.4f}, {x[1] / pi:.4f}, {x[2] / pi:.4f}) pi"
    if len(x) > 3:
        s += f", Q = exp({np.array2string(np.asarray(x[3:6]), precision=4)})"
    return s


def main():
    global EVALS, CFG
    rapido = "--rapido" in sys.argv
    if rapido:
        CFG = CFG_RAPIDO
    c = CFG
    chk = Checker("diamond_chain2_min" + (" --rapido" if rapido else ""))
    suf = "_rapido" if rapido else ""
    EVALS = DATA / f"diamond_chain2_min_evals{suf}.csv"
    load()
    chk.info(f"caché: {len(_cache)} claves propias, {len(_t4)} de la tarea 4 (solo lectura)")
    rows = []

    # Comprobaciones previas: v(0,0,0) = 1/3 y la simetría S1.
    v0, st, gap = v_eval((0, 0, 0, 0, 0, 0), c["TOL_CRIBA"])
    chk.check(st == "optimal" and abs(v0 - 1 / 3) < c["TOL_CRIBA"] + GAP_NEG,
              f"sin holonomía: v = {v0:.10f}, |v - 1/3| = {abs(v0 - 1 / 3):.1e} [estado {st}, gap {gap:.1e}]")
    ps, rvs = (0.6 * pi, 0.85 * pi, 0.3 * pi), np.array([0.5, 0.3, -0.2])
    rvs2 = swap_q(ps[2], rvs)
    va, sa, _ = v_eval((*ps, *rvs), c["TOL_FINO"])
    vb, sb, _ = v_eval((ps[1], ps[0], ps[2], *rvs2), c["TOL_FINO"])
    tol_s = max(2 * c["TOL_FINO"], V_TOL)
    chk.check(sa == sb == "optimal" and abs(va - vb) < tol_s,
              f"simetría S1 en {lab((*ps, *rvs))}: v = {va:.10f}, con theta_A <-> theta_B y "
              f"Q' = exp({np.array2string(rvs2, precision=4)}) v = {vb:.10f}, |dif| = {abs(va - vb):.1e} "
              f"(< {tol_s:.0e}) [estados {sa}, {sb}]")

    # Etapa 1: cribado.
    pts = grid() if not rapido else [(0.0, 0.0, 0.0), (pi, pi, pi / 2)]
    crib = []
    for p in pts:
        if p[0] == 0 and p[1] == 0:
            v, st, _ = v_eval((*p, 0, 0, 0), c["TOL_CRIBA"])
            v, rv, n, capped = (v if st == "optimal" else np.inf), np.zeros(3), 1, False
        else:
            v, rv, n, capped = nelder_mead(lambda rv, p=p: _val((*p, *rv), c["TOL_CRIBA"]),
                                           np.zeros(3), [0.3] * 3, c["MAXFEV_CRIBA"])
        crib.append((v, p, rv))
        rows.append(["cribado", *p, *rv, f"{v:.12f}", c["TOL_CRIBA"], n, int(capped)])
        chk.info(f"cribado (no es resultado) {lab((*p, *rv))}: v = {v:.8f}, {n} llamadas"
                 + (", en el tope" if capped else ""))
    crib = sorted((t for t in crib if np.isfinite(t[0])), key=lambda t: (round(t[0], 9), t[1]))
    for v, p, rv in crib[:5]:
        chk.info(f"mejores del cribado (no son resultados): {lab((*p, *rv))}, v = {v:.8f}")

    # Etapa 2: precisión completa desde los mejores.
    rng = np.random.default_rng(SEED)
    fin = []
    for _, p, rv in crib[:c["MEJORES"]]:
        x0s = [rv] + [Rotation.random(random_state=rng.integers(2 ** 31)).as_rotvec()
                      for _ in range(c["ARRANQUES"] - 1)]
        vq, rvq, n, capped = min_q(p, c["TOL_FINO"], x0s, c["MAXFEV"])
        rows.append(["minQ", *p, *rvq, f"{vq:.12f}", c["TOL_FINO"], n, capped])
        chk.info(f"min_Q (cota superior, tope {c['MAXFEV']}) en {lab((*p, *rvq))}: v = {vq:.10f}, "
                 f"{n} llamadas, {capped} de {len(x0s)} arranques en el tope")
        fin.append((vq, np.array([*p, *rvq])))
        if not np.isfinite(vq):
            continue
        vj, xj, n, capped = joint(np.array([*p, *rvq]), c["TOL_FINO"], c["MAXFEV_CONJ"])
        rows.append(["conjunto", *xj, f"{vj:.12f}", c["TOL_FINO"], n, int(capped)])
        chk.info(f"Nelder-Mead conjunto (cota superior, tope {c['MAXFEV_CONJ']}) -> {lab(xj)}: "
                 f"v = {vj:.10f}, {n} llamadas" + (", en el tope" if capped else ""))
        fin.append((vj, xj))
    vP, P = min(fin, key=lambda t: t[0])
    if not np.isfinite(vP):
        chk.check(False, "ninguna evaluación 'optimal' en la etapa 2: no hay mínimo")
        chk.finish()
    rows.append(["minimo", *P, f"{vP:.12f}", c["TOL_FINO"], "", ""])
    chk.info(f"MÍNIMO (cota superior de min v, tol {c['TOL_FINO']:.0e}): {lab(P)}, v = {vP:.10f}, "
             f"déficit 1/3 - v = {1 / 3 - vP:.6f}")
    write_csv(f"diamond_chain2_min{suf}.csv",
              ["caso", "theta_A", "theta_B", "gamma", "rx", "ry", "rz", "v", "tol", "evaluaciones", "en_tope"],
              rows)

    # Paso 2a: testigo en P (sin caché, para tener los subespacios finales).
    global _dual
    if _dual is None:
        _dual = GraphDual(N, EDGES)
    warm = dict(y0=_warm, r0=R0_WARM) if _warm is not None else {}
    vF, yF, stF, gapF = _dual.v_dual_cg(rotations(P), tol=c["TOL_FINO"], **warm, **CLARABEL_OPTS)
    UF = [u.copy() for u in _dual.last_U]
    tol_r = max(2 * c["TOL_FINO"], V_TOL)
    chk.check(stF == "optimal" and abs(vF - vP) < tol_r,
              f"v_dual_cg de nuevo en P: v = {vF:.10f}, caché {vP:.10f}, |dif| = {abs(vF - vP):.1e} "
              f"(< {tol_r:.0e}) [estado {stF}, gap {gapF:.1e}, rango {_dual.last_ranks}, "
              f"solver {_dual.last_solver_status}]")

    # Paso 2b: perturbaciones.
    dirs = [s * eps * e for eps in c["EPS_PERT"] for e in np.eye(6)[:c["EJES"]] for s in (1, -1)]
    rng = np.random.default_rng(0)
    for _ in range(c["N_AZAR"]):
        u = rng.normal(size=6)
        u /= np.linalg.norm(u)
        dirs += [eps * u for eps in c["EPS_AZAR"]]
    E = np.eye(6)
    pares = [(i, j) for i in range(6) for j in range(i + 1, 6)][:c["PARES"]]
    dirs += [c["EPS_PAR"] * (E[i] + E[j]) / np.sqrt(2) for i, j in pares]
    prow = []
    for d in dirs:
        v, st, _ = v_eval(P + d, c["TOL_FINO"])
        dv = v - vP if st == "optimal" else np.nan
        prow.append((*d, dv))
        eps = np.linalg.norm(d)
        chk.info(f"d = {np.array2string(d, precision=4, sign='+')}: dv = {dv:+.3e}, dv/eps^2 = {dv / eps ** 2:+.5f}")
    dvs = np.array([r[6] for r in prow])
    nbad = int(np.sum(~np.isfinite(dvs)))
    ok = nbad == 0 and dvs.min() > 0
    if nbad:
        msg = (f"{len(dvs)} perturbaciones: {nbad} sin v 'optimal' (FALLO de v_dual_cg, abajo), "
               f"no se puede decir dv > 0 en todas")
    else:
        msg = f"{len(dvs)} perturbaciones: dv > 0 en todas (mínimo {dvs.min():.2e})"
    if rapido:
        chk.info("--rapido, informativo: " + msg)
    else:
        chk.check(ok, msg)
    D = np.array([r[:6] for r in prow])
    F = np.array([feats(d) for d in D])
    rank = np.linalg.matrix_rank(F)
    if nbad:
        chk.info(f"sin ajuste cuadrático: {nbad} perturbaciones sin v 'optimal'")
    elif rank < F.shape[1]:
        msg = f"ajuste cuadrático: matriz de diseño de rango {rank} de {F.shape[1]}, H no determinada"
        if rapido:
            chk.info("--rapido, informativo: " + msg)
        else:
            chk.check(False, msg)
    else:
        coef, *_ = np.linalg.lstsq(F, dvs, rcond=None)
        H, gl = quad_form(coef), coef[21:27]
        ev = np.linalg.eigvalsh(H)
        chk.check(ev.min() > 0, f"forma cuadrática del ajuste: autovalores {np.array2string(ev, precision=5)}; "
                  f"residuo {np.abs(F @ coef - dvs).max():.1e}")
        if ev.min() > 0:
            xs = -0.5 * np.linalg.solve(H, gl)
            chk.info(f"términos lineales {np.array2string(gl, precision=2)}; punto estacionario del ajuste "
                     f"P + {np.array2string(xs, precision=4)} (|.| = {np.linalg.norm(xs):.3f}), "
                     f"dv predicho {-0.25 * gl @ np.linalg.solve(H, gl):+.2e}")
    write_csv(f"diamond_chain2_min_pert{suf}.csv",
              ["d_theta_A", "d_theta_B", "d_gamma", "d_rx", "d_ry", "d_rz", "dv"], prow)

    # Paso 2c: unicidad del testigo.
    trow, widths, inac_tot, nsdp = [], [], 0, 0
    if stF == "optimal":
        for eps in c["EPS_TESTIGO"]:
            try:
                sp = _dual.witness_spread(rotations(P), vF, eps, U=UF, **CLARABEL_OPTS)
            except RuntimeError as exc:
                chk.check(False, f"witness_spread con eps = {eps:.0e}: {exc}")
                widths.append(np.nan)
                continue
            w = max(hi - lo for lo, hi in sp)
            ninac = int(sum(a + b for a, b in _dual.last_spread_inaccurate))
            for _ in range(ninac):
                chk.status_note("optimal_inaccurate")
            inac_tot += ninac
            nsdp += 2 * len(sp)
            widths.append(w)
            trow.append((eps, w, ninac))
            chk.info(f"testigo, eps = {eps:.0e}: anchura máxima {w:.3e} ({ninac} de {2 * len(sp)} SDP "
                     f"optimal_inaccurate)")
        ymax = float(np.max(np.abs(yF)))
        unico, msg = verdict(c["EPS_TESTIGO"], widths, ymax)
        if inac_tot:
            msg += f" [inexacto: {inac_tot} de {nsdp} SDP optimal_inaccurate]"
        if unico:
            chk.check(True, msg)
        else:
            chk.info(msg)
    else:
        chk.check(False, "sin testigo 'optimal' en P: no hay witness_spread")
    write_csv(f"diamond_chain2_min_testigo{suf}.csv", ["eps", "anchura_max", "sdp_inexactos"], trow)

    for k, (st, gap) in fallos.items():
        chk.check(False, f"v_dual_cg en {k}: estado {st}, gap {gap:.1e}; no se usa como valor")
    if not fallos:
        chk.check(True, f"todas las evaluaciones usadas con estado 'optimal' ({len(_cache)} claves en la caché)")
    chk.info(f"evaluaciones: {cuenta['nuevas']} nuevas, {cuenta['cache']} de la caché propia, "
             f"{cuenta['tarea4']} copiadas de la tarea 4")
    chk.finish()


if __name__ == "__main__":
    main()

"""Cadena abierta de 2 diamantes (7 qubits): mínimo global de v y su testigo (exploración).

Artículo (~/fisica/holonomia_diamantes.tex): §6.1, El mínimo (Lema de reflexión, búsqueda, x*, órbita de
gauge, mínimo transverso y testigo).

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

Paso 2, en torno al punto exacto x* (diagnóstico de la corrida de 38 h, que
dio 5 FALLO en el paso 2 en el P de Nelder-Mead): el mínimo exacto es
x* = (theta_A, theta_B, gamma, Q) = (0, pi, 0, R_x(pi/2)), con
v_dual_cg(x*) = 0,3255461236 <= v(P) = 0,3255461398. Las A son la identidad y
las B son giros de pi de ejes n_B = R_y(gamma) z y Q n_B perpendiculares
(alpha = arccos(n_B . Q n_B) = pi/2). Con theta_A = 0 los datos solo dependen de
theta_B y alpha, así que {theta_A = 0, theta_B = pi, alpha = pi/2} es una
órbita de gauge de dimensión 3, parametrizada por (g, phi1, phi2):
gamma = g, Q = R_y(g) R_z(phi1) R_x(pi/2) R_z(phi2) R_y(g)^T. Por eso el
ajuste cuadrático en las 6 coordenadas tiene 3 autovalores ~0 y la estrictez
en 6 coordenadas es falsa por construcción: la estrictez solo cabe en las 3
direcciones transversas (theta_A, theta_B - pi, alpha - pi/2).
  (a) v(x*) <= v(P) + V_TOL (x* es un punto de la órbita exacta; P, la cota
      del Nelder-Mead); se da theta_A, theta_B - pi y alpha - pi/2 en P;
  (b) planitud de la órbita: |v - v(x*)| < V_TOL en 3 puntos de la órbita con
      desplazamiento finito de (g, phi1, phi2);
  (c) estrictez transversa: dv = v - v(x*) > 0 con theta_A = eps en 16 puntos
      de la órbita (4 valores de g x 4 pares (phi1, phi2)), a lo largo de los
      ejes theta_B y alpha, en las diagonales (dthB, dalpha) y en las mezclas de
      theta_A con cada eje; y ajuste de dv en z = (theta_A, dthB, dalpha) con
      los datos del punto (0, 0, 0) de la órbita (3 lineales + 6 cuadráticos):
      FALLO si la matriz de diseño no tiene rango 9, OK si la forma cuadrática
      3x3 es definida positiva (los lineales se dan como información);
  - las 75 perturbaciones de P en las 6 coordenadas (6 ejes x 3 eps x 2 signos,
    12 direcciones al azar x 2 eps y 15 diagonales con eps = 0,05, como
    scripts/ladder_perp_min.py) y su ajuste cuadrático + lineales + cúbicos se
    siguen calculando, pero solo como información (3 direcciones planas de
    gauge con theta_A = 0: ver la órbita);
  - testigo en x*, solo información: v_dual_cg de nuevo (sin caché, para tener
    los subespacios finales U = last_U) y witness_spread con eps = 1e-4, 1e-5,
    1e-6 y on_fail = "nan" (anchura máxima de los coeficientes finitos). Con U
    el conjunto comprimido contiene al verdadero, así que cada anchura es una
    cota superior de la verdadera. Clarabel se estanca en este punto (acaba en
    AlmostSolved con brecha relativa del 3 %, o en NumericalError según los
    subespacios U), lo que no es un fallo del código: si algún SDP no converge
    el veredicto es "inconcluso: N de M SDP sin converger". Si todos
    convergen, veredicto de unicidad como antes (cocientes 1e-4 -> 1e-5 y
    1e-5 -> 1e-6 > 2,5 y anchura en 1e-6 < ANCHO_REL = 0,1 veces el mayor |y_m|;
    si no, inconcluso). Los optimal_inaccurate se cuentan en el resumen del
    Checker (status_note) y llevan la marca "inexacto".
Comprobaciones también: v(0, 0, 0) = 1/3 a V_TOL con tol TOL_FINO, S1 en un
punto genérico, y cada evaluación no 'optimal' es un FALLO.

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
~20 h; paso 2 (revisado), ~52 evaluaciones nuevas x 135 s = ~2 h más ~1 h de
testigo (las 75 perturbaciones de P salen de la caché): ~34 h estimadas con
la caché vacía (la corrida de 2026-11-04/06 midió 136 645 s = 38 h para todo
salvo el paso 2 revisado, pico 1,99 GB); relanzar con esa caché solo calcula
el paso 2 revisado, ~3 h.
Medido (2026-11-06): relanzada con esa caché, 52 evaluaciones nuevas,
7 320 s (2,0 h), pico 0,85 GB; 10/10 OK.
Relanzada el 2026-11-06 tras la revisión final (control sin holonomía a tol
1e-8, 1 evaluación nueva): 914 s, pico 0,82 GB; 10/10 OK. El testigo cambia
entre ejecuciones (53 y 111 de 540 SDP sin converger), según los subespacios.
--rapido medido (con el paso 2 en x*): 2569 s (25 evaluaciones nuevas, 1 witness_spread) y 839 MB.

Escribe data/diamond_chain2_min.csv (caso: cribado | minQ | conjunto |
minimo; theta_A, theta_B, gamma, rx, ry, rz, v, tol, evaluaciones, en_tope),
data/diamond_chain2_min_pert.csv (d en los 6 parámetros, dv),
data/diamond_chain2_min_orbita.csv (g, phi1, phi2, thA, dthB, dalpha, v, dv),
data/diamond_chain2_min_testigo.csv (eps, anchura máxima, SDP inexactos, SDP
fallidos) y la caché. El caso "xstar" de diamond_chain2_min.csv es x*.

Uso: (ulimit -v 8000000; /usr/bin/time -f "%M KB" python scripts/diamond_chain2_min.py [--rapido])
--rapido (prueba de humo): malla (0, 0, 0) y (pi, pi, pi/2), tol 1e-4 y 1e-5,
topes 4 / 2 / 3, un arranque en Q, 4 perturbaciones, órbita reducida (un
punto, un eps), witness_spread con eps = 1e-4; las comprobaciones de convergencia (dv > 0, rango del ajuste,
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
from holonomy.paulis import Rx, Ry, Rz
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
           EJES=6, PARES=15, EPS_PAR=0.05, EPS_TESTIGO=(1e-4, 1e-5, 1e-6),
           XSTAR=(0.0, pi, 0.0, pi / 2, 0.0, 0.0),
           ORBITA_G=(0.0, pi / 4, pi / 2, 3 * pi / 4),
           ORBITA_PHI=((0.0, 0.0), (pi / 2, 0.0), (0.0, pi / 2), (pi / 3, 2 * pi / 3)),
           EPS_A=(0.1, 0.05), EPS_T=(0.1, 0.05), EPS_DIAG=0.05,
           PLANITUD=((0.7, 0.0, 0.0), (0.0, 0.9, 0.0), (0.0, 0.0, 0.9)))
CFG_RAPIDO = dict(TOL_CRIBA=1e-4, TOL_FINO=1e-5, MAXFEV_CRIBA=4, MAXFEV=2, MAXFEV_CONJ=3,
                  MEJORES=1, ARRANQUES=1, EPS_PERT=(0.05,), EPS_AZAR=(0.05,), N_AZAR=1,
                  EJES=1, PARES=1, EPS_PAR=0.05, EPS_TESTIGO=(1e-4,),
                  XSTAR=(0.0, pi, 0.0, pi / 2, 0.0, 0.0),
                  ORBITA_G=(0.0,), ORBITA_PHI=((0.0, 0.0),),
                  EPS_A=(0.1,), EPS_T=(0.1,), EPS_DIAG=0.1,
                  PLANITUD=((0.7, 0.0, 0.0),))

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


def reflexion_g(g):
    """G de la simetría S1: la rotación con G z = -n_B y G n_B = -z,
    n_B = R_y(g) z. Para gamma en (0, pi) sale de las bases (z, n, z x n) y
    (-n, -z, n x z). Con gamma = 0 o pi (n = +-z) esa base es singular: con
    n = z, G es el giro de pi alrededor de un eje perpendicular a z (el Lema
    de reflexión), aquí R_x(pi); con n = -z, G = I."""
    z = np.array([0.0, 0.0, 1.0])
    n = Ry(g) @ z
    c = float(n @ z)
    if abs(abs(c) - 1) < 1e-12:
        return Rx(pi) if c > 0 else np.eye(3)
    F1 = np.column_stack([z, n, np.cross(z, n)])
    F2 = np.column_stack([-n, -z, np.cross(n, z)])
    return F2 @ np.linalg.inv(F1)


def swap_q(g, rv):
    """Q' = G^T Q^T G de la simetría S1: v(tA, tB, g, Q) = v(tB, tA, g, Q'),
    con G = reflexion_g(g) (vale también con gamma = 0 y pi)."""
    G = reflexion_g(g)
    Q = Rotation.from_rotvec(rv).as_matrix()
    return Rotation.from_matrix(G.T @ Q.T @ G).as_rotvec()


def feats(d):
    """Cuadráticos (21), lineales (6) y cúbicos diagonales (6) en 6 variables."""
    iu = np.triu_indices(6)
    q = np.outer(d, d)[iu] * np.where(iu[0] == iu[1], 1.0, 2.0)
    return np.concatenate([q, d, d ** 3])


def quad_form(c, n=6):
    H = np.zeros((n, n))
    H[np.triu_indices(n)] = c[:n * (n + 1) // 2]
    return H + np.triu(H, 1).T


def feats_z(z):
    """Cuadráticos (6) y lineales (3) en z = (theta_A, dthB, dalpha)."""
    iu = np.triu_indices(3)
    q = np.outer(z, z)[iu] * np.where(iu[0] == iu[1], 1.0, 2.0)
    return np.concatenate([q, z])


def orbit_point(g, phi1, phi2, thA=0.0, dthB=0.0, dalpha=0.0):
    """x = (theta_A, theta_B, gamma, rx, ry, rz) en la órbita de gauge de x*
    (thA = dthB = dalpha = 0) o desplazado de ella: gamma = g,
    Q = R_y(g) R_z(phi1) R_x(pi/2 + dalpha) R_z(phi2) R_y(g)^T,
    theta_A = thA, theta_B = pi + dthB."""
    Ryg = Ry(g)
    Q = Ryg @ Rz(phi1) @ Rx(pi / 2 + dalpha) @ Rz(phi2) @ Ryg.T
    return np.array([thA, pi + dthB, g, *Rotation.from_matrix(Q).as_rotvec()])


def axis_angle(x):
    """alpha: ángulo entre n_B = R_y(gamma) z y Q n_B."""
    nB = Ry(x[2]) @ np.array([0.0, 0.0, 1.0])
    Q = Rotation.from_rotvec(np.asarray(x[3:6], float)).as_matrix()
    return float(np.arccos(np.clip(nB @ Q @ nB, -1.0, 1.0)))


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
    # A la tol fina (la caché solo tenía este punto a TOL_CRIBA: una evaluación
    # nueva al relanzar) y con V_TOL; en --rapido la tol fina es 1e-5.
    v0, st, gap = v_eval((0, 0, 0, 0, 0, 0), c["TOL_FINO"])
    tol0 = max(V_TOL, c["TOL_FINO"])
    chk.check(st == "optimal" and abs(v0 - 1 / 3) < tol0,
              f"sin holonomía: v = {v0:.10f}, |v - 1/3| = {abs(v0 - 1 / 3):.1e} (< {tol0:.0e}) "
              f"[estado {st}, gap {gap:.1e}, tol {c['TOL_FINO']:.0e}]")
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
    # Paso 2, punto exacto x* y su órbita de gauge (ver el docstring).
    def conv(ok, msg):
        """Comprobación de convergencia: info en --rapido (tol gruesa)."""
        if rapido:
            chk.info("--rapido, informativo: " + msg)
        else:
            chk.check(ok, msg)

    xs = np.array(c["XSTAR"], float)
    vX, stX, gapX = v_eval(xs, c["TOL_FINO"])
    vX = vX if stX == "optimal" else np.nan
    rows.append(["xstar", *xs, f"{vX:.12f}", c["TOL_FINO"], "", ""])
    write_csv(f"diamond_chain2_min{suf}.csv",
              ["caso", "theta_A", "theta_B", "gamma", "rx", "ry", "rz", "v", "tol", "evaluaciones", "en_tope"],
              rows)
    chk.info(f"x* = {lab(xs)}: v = {vX:.10f} [estado {stX}, gap {gapX:.1e}]")
    conv(bool(vX <= vP + V_TOL),
         f"v(x*) = {vX:.10f} <= v(P) + {V_TOL:.0e} = {vP + V_TOL:.10f} (v(P) - v(x*) = {vP - vX:+.2e})")
    chk.info(f"P: theta_A = {P[0]:.5f}, theta_B - pi = {P[1] - pi:+.5f}, alpha - pi/2 = {axis_angle(P) - pi / 2:+.5f}")

    orows = []

    def ev_orb(g, p1, p2, thA=0.0, dthB=0.0, dal=0.0):
        """(v, dv) en el punto de la órbita (g, p1, p2) desplazado, dv = v - v(x*)
        (nan si no es 'optimal'); la fila se añade al CSV de la órbita."""
        v, st, _ = v_eval(orbit_point(g, p1, p2, thA, dthB, dal), c["TOL_FINO"])
        v = v if st == "optimal" else np.nan
        dv = v - vX
        orows.append((g, p1, p2, thA, dthB, dal, v, dv))
        return v, dv

    # (b) planitud de la órbita.
    for g, p1, p2 in c["PLANITUD"]:
        v, dv = ev_orb(g, p1, p2)
        conv(bool(abs(dv) < V_TOL),
             f"planitud de la órbita en (g, phi1, phi2) = ({g}, {p1}, {p2}): v = {v:.10f}, "
             f"|v - v(x*)| = {abs(dv):.1e} (< {V_TOL:.0e})")

    # (c) estrictez transversa en torno a x*.
    dvA, mins = [], []
    for g in c["ORBITA_G"]:
        for p1, p2 in c["ORBITA_PHI"]:
            dd = [ev_orb(g, p1, p2, thA=e)[1] for e in c["EPS_A"]]
            dvA += dd
            mins.append((g, p1, p2, np.nanmin(dd) if np.isfinite(dd).any() else np.nan,
                         np.nanmin([d / e ** 2 for d, e in zip(dd, c["EPS_A"])]) if np.isfinite(dd).any() else np.nan))
            chk.info(f"theta_A = eps en la órbita (g, phi1, phi2) = ({g:.4f}, {p1:.4f}, {p2:.4f}): "
                     f"min dv = {mins[-1][3]:+.3e}, min dv/eps^2 = {mins[-1][4]:+.5f}")
    dvA = np.array(dvA)
    conv(bool(np.all(np.isfinite(dvA)) and dvA.min() > 0),
         f"theta_A = eps (eps = {c['EPS_A']}) en {len(c['ORBITA_G']) * len(c['ORBITA_PHI'])} puntos de la órbita: "
         f"dv > 0 en las {len(dvA)} (mínimo {np.nanmin(dvA) if np.isfinite(dvA).any() else np.nan:.2e}, "
         f"{int(np.sum(~np.isfinite(dvA)))} sin v 'optimal')")
    d2 = c["EPS_DIAG"] / np.sqrt(2)
    zs = [(e, 0.0, 0.0) for e in c["EPS_A"]]                              # (thA, dthB, dalpha)
    zs += [(0.0, s * e, 0.0) for e in c["EPS_T"] for s in (1, -1)]
    zs += [(0.0, 0.0, s * e) for e in c["EPS_T"] for s in (1, -1)]
    zs += [(0.0, s1 * d2, s2 * d2) for s1 in (1, -1) for s2 in (1, -1)]
    zs += [(d2, s * d2, 0.0) for s in (1, -1)] + [(d2, 0.0, s * d2) for s in (1, -1)]
    dvz = []
    for z in zs:
        v, dv = ev_orb(0.0, 0.0, 0.0, *z)
        dvz.append(dv)
        eps = np.linalg.norm(z)
        chk.info(f"z = (theta_A, dthB, dalpha) = {np.array2string(np.array(z), precision=4, sign='+')}: "
                 f"dv = {dv:+.3e}, dv/eps^2 = {dv / eps ** 2:+.5f}")
    dvz = np.array(dvz)
    conv(bool(np.all(np.isfinite(dvz)) and dvz.min() > 0),
         f"ejes theta_A / theta_B / alpha (eps_T = {c['EPS_T']}), diagonales y mezclas ({len(zs)} puntos en (0, 0, 0)): "
         f"dv > 0 en todas (mínimo {np.nanmin(dvz) if np.isfinite(dvz).any() else np.nan:.2e}, "
         f"{int(np.sum(~np.isfinite(dvz)))} sin v 'optimal')")
    if not np.all(np.isfinite(dvz)):
        conv(False, f"sin ajuste transverso: {int(np.sum(~np.isfinite(dvz)))} puntos sin v 'optimal'")
    else:
        Fz = np.array([feats_z(z) for z in zs])
        rz_ = np.linalg.matrix_rank(Fz)
        if rz_ < Fz.shape[1]:
            conv(False, f"ajuste transverso en z: matriz de diseño de rango {rz_} de {Fz.shape[1]}, H no determinada")
        else:
            coef, *_ = np.linalg.lstsq(Fz, dvz, rcond=None)
            Hz, gz = quad_form(coef, 3), coef[6:9]
            evz = np.linalg.eigvalsh(Hz)
            conv(bool(evz.min() > 0),
                 f"forma cuadrática transversa en (theta_A, dthB, dalpha): autovalores {np.array2string(evz, precision=5)}; "
                 f"residuo {np.abs(Fz @ coef - dvz).max():.1e}")
            chk.info(f"términos lineales transversos (theta_A, dthB, dalpha) = {np.array2string(gz, precision=3)}")
    orows = list(dict.fromkeys(orows))   # (0, 0, 0) con theta_A = eps está en (c) dos veces
    write_csv(f"diamond_chain2_min_orbita{suf}.csv",
              ["g", "phi1", "phi2", "thA", "dthB", "dalpha", "v", "dv"], orows)

    # Perturbaciones de P en las 6 coordenadas: solo información (3 direcciones
    # planas de gauge con theta_A = 0).
    nota = " (3 direcciones planas de gauge con theta_A = 0: ver la órbita)"
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
    if nbad:
        chk.info(f"{len(dvs)} perturbaciones de P: {nbad} sin v 'optimal'" + nota)
    else:
        chk.info(f"{len(dvs)} perturbaciones de P: mínimo dv {dvs.min():.2e}" + nota)
    D = np.array([r[:6] for r in prow])
    F = np.array([feats(d) for d in D])
    rank = np.linalg.matrix_rank(F)
    if nbad:
        chk.info(f"sin ajuste cuadrático en P: {nbad} perturbaciones sin v 'optimal'" + nota)
    elif rank < F.shape[1]:
        chk.info(f"ajuste cuadrático en P: matriz de diseño de rango {rank} de {F.shape[1]}, H no determinada" + nota)
    else:
        coef, *_ = np.linalg.lstsq(F, dvs, rcond=None)
        H, gl = quad_form(coef), coef[21:27]
        ev = np.linalg.eigvalsh(H)
        chk.info(f"forma cuadrática del ajuste en P: autovalores {np.array2string(ev, precision=5)}; "
                 f"residuo {np.abs(F @ coef - dvs).max():.1e}" + nota)
        if ev.min() > 0:
            xp = -0.5 * np.linalg.solve(H, gl)
            chk.info(f"términos lineales {np.array2string(gl, precision=2)}; punto estacionario del ajuste "
                     f"P + {np.array2string(xp, precision=4)} (|.| = {np.linalg.norm(xp):.3f}), "
                     f"dv predicho {-0.25 * gl @ np.linalg.solve(H, gl):+.2e}")
    write_csv(f"diamond_chain2_min_pert{suf}.csv",
              ["d_theta_A", "d_theta_B", "d_gamma", "d_rx", "d_ry", "d_rz", "dv"], prow)

    # Testigo en x* (sin caché, para tener los subespacios finales): solo información.
    global _dual
    if _dual is None:
        _dual = GraphDual(N, EDGES)
    warm = dict(y0=_warm, r0=R0_WARM) if _warm is not None else {}
    vF, yF, stF, gapF = _dual.v_dual_cg(rotations(xs), tol=c["TOL_FINO"], **warm, **CLARABEL_OPTS)
    UF = [u.copy() for u in _dual.last_U]
    chk.info(f"v_dual_cg de nuevo en x*: v = {vF:.10f}, caché {vX:.10f}, |dif| = {abs(vF - vX):.1e} "
             f"[estado {stF}, gap {gapF:.1e}, rango {_dual.last_ranks}, solver {_dual.last_solver_status}]")
    trow, widths, inac_tot, fail_tot, nsdp = [], [], 0, 0, 0
    if stF == "optimal":
        for eps in c["EPS_TESTIGO"]:
            try:
                sp = _dual.witness_spread(rotations(xs), vF, eps, U=UF, on_fail="nan", **CLARABEL_OPTS)
            except RuntimeError as exc:
                chk.info(f"witness_spread con eps = {eps:.0e}: {exc}")
                widths.append(np.nan)
                continue
            anch = [hi - lo for lo, hi in sp if np.isfinite(lo) and np.isfinite(hi)]
            w = max(anch) if anch else np.nan
            ninac = int(sum(a + b for a, b in _dual.last_spread_inaccurate))
            nfail = int(sum(a + b for a, b in _dual.last_spread_failed))
            for _ in range(ninac):
                chk.status_note("optimal_inaccurate")
            inac_tot += ninac
            fail_tot += nfail
            nsdp += 2 * len(sp)
            widths.append(w)
            trow.append((eps, w, ninac, nfail))
            chk.info(f"testigo en x*, eps = {eps:.0e}: anchura máxima {w:.3e} ({ninac} de {2 * len(sp)} SDP "
                     f"optimal_inaccurate, {nfail} sin converger)")
        if fail_tot:
            chk.info(f"testigo en x*: inconcluso: {fail_tot} de {nsdp} SDP sin converger "
                     f"(Clarabel se estanca en este punto)")
        else:
            ymax = float(np.max(np.abs(yF)))
            unico, msg = verdict(c["EPS_TESTIGO"], widths, ymax)
            if inac_tot:
                msg += f" [inexacto: {inac_tot} de {nsdp} SDP optimal_inaccurate]"
            chk.info(msg)
    else:
        chk.info(f"sin testigo 'optimal' en x* (estado {stF}): no hay witness_spread")
    write_csv(f"diamond_chain2_min_testigo{suf}.csv",
              ["eps", "anchura_max", "sdp_inexactos", "sdp_fallidos"], trow)

    for k, (st, gap) in fallos.items():
        chk.check(False, f"v_dual_cg en {k}: estado {st}, gap {gap:.1e}; no se usa como valor")
    if not fallos:
        chk.check(True, f"todas las evaluaciones usadas con estado 'optimal' ({len(_cache)} claves en la caché)")
    chk.info(f"evaluaciones: {cuenta['nuevas']} nuevas, {cuenta['cache']} de la caché propia, "
             f"{cuenta['tarea4']} copiadas de la tarea 4")
    chk.finish()


if __name__ == "__main__":
    main()

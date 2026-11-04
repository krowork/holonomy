"""Cadena cerrada de k diamantes: límite k -> infinito del déficit 1/3 - v
en un punto fijo (exploración).

El punto se da como (modo, f, psi), con psi un número o "min" (el mínimo en
psi, el mayor déficit). Modos y datos como en diamond_ring: (fA, fB) =
(f, f) en "igual" y (f, -f) en "opuesto", fases de
diamond.abelian_phases(k, fA, fB, periodic=True, psi). Los dos puntos
elegidos por Alex (tarea 7):
  A: igual, f = 0, psi = 0, solo k impar (3, 5, 7, ...). En f = 0 el déficit
     es un efecto de paridad: con k = 3, v = 0.3077932 < 1/3, y v(0, psi) fue
     plano en psi (el mismo v en toda la fila psi del mapa 13 x 13 de
     diamond_ring, confirmado por el primal reducido a 9e-9); con k = 2 y 4,
     v = 1/3. Por eso psi = 0 basta; con k = 3 se comprueba la planitud en dos
     psi más (información).
  B: igual, f dado (provisionalmente pi/3), mínimo en psi en cada k, para
     k = 2, 3, 4, ...: malla de NPSI puntos en [0, pi] y Brent (acotado,
     xatol = 1e-5) entre los vecinos del menor, como diamond_ring; el mínimo
     de la malla gana si Brent no lo mejora en más de 1e-8 (así un mínimo en
     el borde se da exactamente en 0 o pi y no a ~1e-4 por ruido de CG).
     Basta psi en [0, pi]: v(f, psi) = v(f, -psi) (comprobado en
     diamond_ring) y v tiene periodo 2 pi en psi, luego también
     v(f, pi + x) = v(f, pi - x) (reflexión en pi); v queda determinado por
     sus valores en [0, pi].

Cada evaluación: DiamondDual(k, dense_max=100, half=True).v_dual_cg con
r0 = 12 en frío (como diamond_ring) y tol = 1e-8, en un proceso hijo nuevo
(multiprocessing.Process, uno por evaluación, como el Pool con
maxtasksperchild=1 de ladder_limit: encadenados en un mismo proceso la
memoria crece). El hijo da su pico de memoria residente (ru_maxrss) y virtual
(VmPeak de /proc/self/status; el criterio de parada es ulimit -v, que limita
la virtual). Solo el estado 'optimal' es un resultado: cualquier otro es un
FALLO y su v no se escribe. Si el hijo muere sin dar resultado (p. ej. lo
mata el OOM-killer) o lanza una excepción (p. ej. MemoryError bajo
ulimit -v), ese k es un FALLO ("proceso hijo muerto" o "excepción en el
hijo") y no se sube a más k. Un 'optimal_inaccurate' de CVXPY en algún SDP
restringido (PeriodicDual.last_solver_status) no es un FALLO: el estado es el
de v_dual_cg (cotas cerradas); se da como información "inexacto" y en la
columna estado_solver.

Comprobaciones, en cada k:
  1. estado 'optimal' y |diferencia de cotas| < 1e-7 (no 1e-8: en la tarea 6
     el ruido de v_dual_cg en estos problemas llegó a 3e-8);
  2. con k <= 4, coincidencia a 1e-7 con los CSV de diamond_ring (psi fijo:
     data/diamond_ring_L<k>_<modo>.csv si (f, psi) está en su malla; "min":
     data/diamond_ring_min_L<k>_<modo>.csv si f está en ella); si el CSV no
     existe (k = 4 mientras corre su mapa), una línea de información;
  3. punto A con k = 3: v = 0.3077932355 (diamond_ring y primal reducido)
     a 1e-7;
  4. v <= 1/3 + 1e-7 (cota del diamante simple).

Reanudable: cada evaluación 'optimal' se guarda en
data/diamond_limit_<etiqueta>_evals.csv al terminarla y no se repite al
relanzar (Brent repite la misma secuencia y la lee de la caché); las fallidas
no se guardan y se repiten. La clave de la caché incluye los ajustes
(AJUSTES: r0, dense_max, tol de v_dual_cg y CLARABEL_OPTS): las filas con
otros ajustes, o sin estado 'optimal', no se leen. Al terminar cada k se
reescribe data/diamond_limit_<etiqueta>.csv con una fila por k: k, modo, f,
psi_modo, psi (el del mínimo en B), v, déficit, estado, dif_cotas,
estado_solver (el peor de PeriodicDual.last_solver_status en las
evaluaciones de ese k), picos residente y virtual (GB, los mayores de las
evaluaciones), tiempo (s, suma de las evaluaciones, incluidas las de caché),
número de evaluaciones y ajustes. Si el estado no es 'optimal', psi, v,
déficit y dif_cotas quedan vacíos, y al relanzar la fila se trata como
ausente. Una etiqueta ya usada con otro punto o con otros ajustes se
rechaza.

Uso: (ulimit -v 16000000; /usr/bin/time -f "%M KB %e s" python scripts/diamond_limit.py \\
          etiqueta modo f psi|min k [k ...])
     p. ej. ... diamond_limit.py A igual 0 0 3 5 7
            ... diamond_limit.py B igual pi/3 min 2 3 4 5
     f y psi admiten expresiones con pi (pi/3, 2*pi/3).
Medido (2026-11-04, OMP_NUM_THREADS = 2, con otros dos cálculos en la
máquina; pico = ru_maxrss del hijo, el de /usr/bin/time es solo el padre,
~0,12 GB; etiquetas de prueba pruebaA y pruebaB, ver el informe de la tarea 7):
  A (1 evaluación por k): k = 2: 0,2 s, 0,11 GB; k = 3: 1,5 s, 0,18 GB;
     k = 4: 4,7 s, 0,26 GB; k = 5: 10 s, 0,41 GB.
  B (28-30 evaluaciones por k): k = 2: 6 s, 0,11 GB; k = 3: 125 s, 0,22 GB;
     k = 4: 1811 s (62 s por evaluación), 1,41 GB.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import multiprocessing as mp
import resource
import sys
import time
import traceback

import numpy as np
from scipy.optimize import minimize_scalar

from holonomy import diamond
from holonomy.report import DATA, write_csv

THIRD = 1 / 3
DENSE_MAX = 100
R0 = 12  # subespacio inicial de v_dual_cg, en frío (ver diamond_ring)
TOL_CG = 1e-8  # tol de v_dual_cg (diferencia de cotas para 'optimal')
TOL_GAP = 1e-7
TOL_REF = 1e-7
TOL_BOUND = 1e-7
TOL_BRENT = 1e-8  # mejora mínima de Brent sobre el mínimo de la malla
V_A3 = 0.3077932355  # k = 3, f = 0: diamond_ring y primal reducido (tarea 6)
NPSI = 7  # malla en psi del punto "min": paso pi/6, contenida en la de 13 de diamond_ring
REF_MAX_K = 4
MODES = {"igual": 1.0, "opuesto": -1.0}  # fB = signo * fA
# Lo que determina la solución, además del punto: parte de la clave de la caché.
AJUSTES = ";".join([f"r0={R0}", f"dense_max={DENSE_MAX}", f"tol_cg={TOL_CG!r}"]
                   + [f"{a}={b!r}" for a, b in sorted(CLARABEL_OPTS.items())])

EVAL_HEADER = ["k", "modo", "f", "psi", "v_max", "estado", "dif_cotas", "estado_solver",
               "pico_GB", "vmpeak_GB", "tiempo_s", "ajustes"]
SUM_HEADER = ["k", "modo", "f", "psi_modo", "psi", "v_max", "deficit", "estado", "dif_cotas",
              "estado_solver", "pico_GB", "vmpeak_GB", "tiempo_s", "evaluaciones", "ajustes"]


class ChildFailure(Exception):
    """El proceso hijo murió sin resultado o lanzó una excepción."""


def angle(s):
    """Número o expresión con pi (pi/3, 2*pi/3)."""
    return float(eval(s, {"__builtins__": {}}, {"pi": np.pi}))


def _key(k, mode, f, psi):
    return (int(k), mode, round(float(f), 12), round(float(psi), 12))


def _vmpeak_gb():
    """Pico de memoria virtual del proceso (VmPeak, en GB)."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def point(args):
    """Una evaluación en frío:
    (v, estado, dif_cotas, estado_solver, pico_GB, vmpeak_GB, tiempo_s)."""
    k, mode, f, psi = args
    t0 = time.time()
    D = diamond.DiamondDual(k, dense_max=DENSE_MAX, half=True)
    ph = diamond.abelian_phases(k, f, MODES[mode] * f, periodic=True, psi=psi)
    v, _, st, gap = D.v_dual_cg(ph, tol=TOL_CG, r0=R0, **CLARABEL_OPTS)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return (float(v), st, float(gap), D.last_solver_status or "", mem, _vmpeak_gb(),
            time.time() - t0)


def _child(conn, args):
    try:
        conn.send(("ok", point(args)))
    except BaseException:
        conn.send(("error", traceback.format_exc(limit=3)))
    finally:
        conn.close()


def run_child(args):
    """point(args) en un proceso hijo nuevo. ChildFailure si muere sin
    resultado (OOM-killer, señal) o si lanza una excepción."""
    recv, send = mp.Pipe(duplex=False)
    p = mp.get_context("fork").Process(target=_child, args=(send, args))
    p.start()
    send.close()
    try:
        kind, payload = recv.recv()  # EOFError si el hijo muere sin enviar
    except EOFError:
        kind, payload = "dead", None
    recv.close()
    p.join()
    if kind == "dead":
        raise ChildFailure(f"proceso hijo muerto (exitcode {p.exitcode})")
    if kind == "error":
        raise ChildFailure("excepción en el hijo: " + payload.strip().splitlines()[-1])
    return payload


class Evaluator:
    """Evaluaciones con caché en disco (solo las 'optimal' con AJUSTES)."""

    def __init__(self, tag):
        self.path = DATA / f"diamond_limit_{tag}_evals.csv"
        self.cache = {}
        self.skipped = 0
        self.log = []  # evaluaciones del k en curso (de caché o nuevas)
        if self.path.exists():
            for r in csv.DictReader(open(self.path)):
                if r["estado"] != "optimal" or r.get("ajustes") != AJUSTES:
                    self.skipped += 1
                    continue
                self.cache[_key(r["k"], r["modo"], r["f"], r["psi"])] = (
                    float(r["v_max"]), r["estado"], float(r["dif_cotas"]), r["estado_solver"],
                    float(r["pico_GB"]), float(r["vmpeak_GB"]), float(r["tiempo_s"]))

    def __call__(self, k, mode, f, psi):
        key = _key(k, mode, f, psi)
        if key in self.cache:
            res = self.cache[key]
        else:
            res = run_child((k, mode, f, psi))
            if res[1] == "optimal":
                self.cache[key] = res
                DATA.mkdir(exist_ok=True)
                new = not self.path.exists()
                with open(self.path, "a", newline="") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(EVAL_HEADER)
                    w.writerow([k, mode, repr(float(f)), repr(float(psi))] + [
                        repr(x) if isinstance(x, float) else x for x in res] + [AJUSTES])
        self.log.append((psi,) + res)
        return res


def ref_value(k, mode, f, psi):
    """v de diamond_ring en el mismo punto: (v, archivo), (None, archivo) si
    el archivo no existe o el punto no está en él."""
    if psi is None:
        name = f"diamond_ring_min_L{k}_{mode}.csv"
        if not (DATA / name).exists():
            return None, name
        for r in csv.DictReader(open(DATA / name)):
            if abs(float(r["f"]) - f) < 1e-9 and r["estado"] == "optimal":
                return float(r["v_min"]), name
        return None, name
    name = f"diamond_ring_L{k}_{mode}.csv"
    if not (DATA / name).exists():
        return None, name
    for r in csv.DictReader(open(DATA / name)):
        if abs(float(r["f"]) - f) < 1e-9 and abs(float(r["psi"]) - psi) < 1e-9 \
                and r["estado"] == "optimal":
            return float(r["v_max"]), name
    return None, name


def minimum(ev, k, mode, f):
    """Mínimo en psi en [0, pi]: (psi, v, estado); estado 'optimal' si todas
    las evaluaciones lo fueron."""
    psis = np.linspace(0, np.pi, NPSI)
    vals = []
    fails = []
    for p in psis:
        v, st = ev(k, mode, f, p)[:2]
        if st != "optimal":
            fails.append(st)
        vals.append(v if st == "optimal" else np.nan)
    vals = np.array(vals)
    if np.all(np.isnan(vals)):
        return np.nan, np.nan, "fallos:" + ",".join(fails)
    j = int(np.nanargmin(vals))
    lo, hi = psis[max(j - 1, 0)], psis[min(j + 1, NPSI - 1)]

    def g(p):
        v, st = ev(k, mode, f, p)[:2]
        if st != "optimal":
            fails.append(st)
            return np.inf
        return v

    res = minimize_scalar(g, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-5))
    # Brent acotado no evalúa los extremos lo, hi: un mínimo en el borde 0 o pi
    # lo da el punto de la malla, que gana salvo que Brent lo mejore en más
    # de TOL_BRENT (el ruido de CG es ~1e-9).
    if res.fun < vals[j] - TOL_BRENT:
        pm, vm = float(res.x), float(res.fun)
    else:
        pm, vm = float(psis[j]), float(vals[j])
    return pm, vm, "optimal" if not fails else "fallos:" + ",".join(fails)


def _fmt(x):
    return repr(float(x))


def main():
    if len(sys.argv) < 6 or sys.argv[2] not in MODES:
        sys.exit("uso: diamond_limit.py etiqueta igual|opuesto f psi|min k [k ...]")
    tag, mode, f = sys.argv[1], sys.argv[2], angle(sys.argv[3])
    psi_fix = None if sys.argv[4] == "min" else angle(sys.argv[4])
    psi_modo = "min" if psi_fix is None else "fijo"
    ks = [int(a) for a in sys.argv[5:]]
    if min(ks) < 2:
        sys.exit("la cadena cerrada exige k >= 2")
    spec = f"modo {mode}, f/pi = {f / np.pi:.6f}, psi " + (
        "mínimo" if psi_fix is None else f"/pi = {psi_fix / np.pi:.6f}")

    sum_name = f"diamond_limit_{tag}.csv"
    rows = {}
    if (DATA / sum_name).exists():
        for r in csv.DictReader(open(DATA / sum_name)):
            # psi fijo: se compara el f y el modo de psi; el psi de una fila
            # fallida está vacío, así que el psi fijo se mira solo en las 'optimal'.
            same = (r["modo"] == mode and abs(float(r["f"]) - f) < 1e-12
                    and r["psi_modo"] == psi_modo and r.get("ajustes") == AJUSTES
                    and (psi_fix is None or r["estado"] != "optimal"
                         or abs(float(r["psi"]) - psi_fix) < 1e-12))
            if not same:
                sys.exit(f"{sum_name} es de otro punto o de otros ajustes (fila k = {r['k']}); "
                         "use otra etiqueta")
            if r["estado"] == "optimal":  # las filas fallidas cuentan como ausentes
                rows[int(r["k"])] = [r[h] for h in SUM_HEADER]

    chk = Checker(f"diamond_limit {tag}: {spec}")
    ev = Evaluator(tag)
    chk.info(f"{len(ev.cache)} evaluaciones en caché ({ev.path.name}); {ev.skipped} filas "
             f"descartadas (no 'optimal' u otros ajustes); ajustes {AJUSTES}")
    for k in ks:
        print(f"== k = {k} ({3 * k} sitios)")
        ev.log = []
        dead = None
        try:
            if psi_fix is None:
                psi, v, st = minimum(ev, k, mode, f)
                ok_gap = all(e[2] == "optimal" and abs(e[3]) < TOL_GAP for e in ev.log)
                gap = max(abs(e[3]) for e in ev.log)
                if st == "optimal" and not ok_gap:
                    st = "dif_cotas"
            else:
                psi = psi_fix
                v, st, gap = ev(k, mode, f, psi)[:3]
                if st == "optimal" and not abs(gap) < TOL_GAP:
                    st = "dif_cotas"
        except ChildFailure as e:
            dead = str(e)
            st = dead
        good = st == "optimal"
        if good:
            msg = (f"k = {k}: v = {v:.10f}, déficit = {THIRD - v:.4e}, "
                   + (f"mínimo en psi/pi = {psi / np.pi:.5f}, " if psi_fix is None else "")
                   + f"máx |dif cotas| = {abs(gap):.1e} [{len(ev.log)} evaluaciones]")
        else:
            msg = f"k = {k}: sin resultado, estado {st} [{len(ev.log)} evaluaciones]"
        chk.check(good, msg)
        sst = "optimal" if ev.log else ""
        for e in ev.log:
            sst = diamond.DiamondDual._worse(sst, e[4] or "optimal")
        if sst != "optimal":
            chk.info(f"   inexacto: algún SDP restringido acabó '{sst}'")
        n_ev = len(ev.log)
        mem = max((e[5] for e in ev.log), default=float("nan"))
        vmp = max((e[6] for e in ev.log), default=float("nan"))
        dt = sum(e[7] for e in ev.log)
        chk.info(f"   pico residente {mem:.2f} GB, virtual {vmp:.2f} GB, {dt:.0f} s "
                 f"({dt / max(n_ev, 1):.1f} s por evaluación)")
        if good:
            rows[k] = [k, mode, _fmt(f), psi_modo, _fmt(psi), _fmt(v), _fmt(THIRD - v), st,
                       _fmt(gap), sst, f"{mem:.3f}", f"{vmp:.3f}", f"{dt:.1f}", n_ev, AJUSTES]
        else:
            rows[k] = [k, mode, _fmt(f), psi_modo, "", "", "", st, "", sst, f"{mem:.3f}",
                       f"{vmp:.3f}", f"{dt:.1f}", n_ev, AJUSTES]
        if good:
            chk.check(v <= THIRD + TOL_BOUND, f"   v <= 1/3 + {TOL_BOUND:.0e}: v - 1/3 = {v - THIRD:.1e}")
            if k <= REF_MAX_K:
                ref, name = ref_value(k, mode, f, psi_fix)
                if ref is None:
                    chk.info(f"   sin contraste con diamond_ring: {name} no existe o no tiene el punto")
                else:
                    chk.check(abs(v - ref) < TOL_REF, f"   = {name}: {ref:.10f}, |dif| = {abs(v - ref):.1e}")
            if k == 3 and mode == "igual" and f == 0.0 and psi_fix is not None:
                chk.check(abs(v - V_A3) < TOL_REF,
                          f"   = {V_A3} (diamond_ring y primal reducido), |dif| = {abs(v - V_A3):.1e}")
                for p in (np.pi / 3, np.pi):
                    try:
                        vp, sp, gp = ev(k, mode, f, p)[:3]
                    except ChildFailure as e:
                        chk.info(f"   planitud en psi/pi = {p / np.pi:.3f}: sin resultado ({e})")
                        continue
                    if sp == "optimal":
                        chk.info(f"   planitud en psi: psi/pi = {p / np.pi:.3f}: v = {vp:.10f}, "
                                 f"|v - v(psi/pi = {psi / np.pi:.3f})| = {abs(vp - v):.1e} "
                                 f"[cotas a {gp:.1e}]")
                    else:
                        chk.info(f"   planitud en psi/pi = {p / np.pi:.3f}: sin resultado ({sp})")
        write_csv(sum_name, SUM_HEADER, [rows[q] for q in sorted(rows)])
        if dead is not None:
            chk.info(f"   no se sube a más k: {dead}")
            break
    chk.finish()


if __name__ == "__main__":
    main()

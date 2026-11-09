"""Escalera periódica 2 x L con f = 0: testigo fijo a L grande y familia exacta del testigo óptimo.

Artículo (~/fisica/holonomia.tex): sección «Rigidez del testigo» de las escaleras
(párrafo «Testigo fijo»); problema abierto de las escaleras (ley de caída con f = 0).

Testigos (holonomy.ladder_ed): W = -4 H con H la escalera XXZ periódica con giro Psi en
las dos patas, patas con acoplamiento 1 y anisotropía Delta, peldaños con XY x y ZZ c_r.
Con los datos de f = 0 y holonomía psi = Psi de las patas, N(W) = L[2(2+Delta) + 2x + c_r]
y v(L, 0, psi) <= -4 E_0 / N. La cota se certifica con el residuo r de Lanczos:
E_0 >= E_Ritz - r (si el valor de Ritz es el del fundamental; arranque aleatorio), así que
v <= -4 (E_Ritz - r)/N.

Familias exactas. En psi = 0 todos los datos son triviales: por el promedio sobre SU(2) global
y el intercambio de las patas hay un testigo óptimo de Heisenberg (Delta = Delta_r = 1) con un
solo parámetro x, de modo que v(L, 0, 0) = min_x (-4 E_0(x)/(3L(2+x))). En psi = pi, el promedio
sobre U(1), traslación e intercambio de patas deja (alpha, beta, c) en las patas y (alpha_r, c_r)
en los peldaños; la antiunitaria K compuesta con el gauge de 2pi conserva los datos de psi = pi
(holonomía -pi = pi) y lleva la fase chi de las patas a 2pi/L - chi, así que el promedio tiene
chi = pi/L: la familia (x, Delta, c_r) con giro pi contiene un óptimo y
v(L, 0, pi) = min (-4 E_0/N). El gradiente sale de Hellmann-Feynman (TwistedLadder.parts).

Etapas:
  fijo      x en {1, 0,8}, Delta = 1, c_r = x, Psi en {0, pi}, L de la lista: E_0, residuo,
            cota; la respuesta del testigo fijo al giro es dE = E_0(pi) - E_0(0) y, en
            unidades de v, R = -4 dE / N.
  familia   min en x (psi = 0, Brent) y en (x, Delta, c_r) (psi = pi, L-BFGS-B con
            gradiente), con arranque en caliente durante la búsqueda y el punto final
            reevaluado en frío con arranque aleatorio y su residuo.
  analisis  (también tras calcular) contraste con data/ladder_limit_F0_L<L>.csv (L <= 10),
            exponentes locales y ajustes de dE (fijo) y de v(pi) - v(0) (familia).

Cada evaluación de E_0 se guarda en data/escalera_testigo_fijo_evals.csv (reanudable: un
punto ya calculado no se repite). Escribe data/escalera_testigo_fijo.csv (etapa fijo) y
data/escalera_familia.csv (etapa familia).

Uso: (ulimit -v 14000000; OMP_NUM_THREADS=6 python scripts/escalera_testigo_fijo.py fijo 4 6 8 10 12 14)
     python scripts/escalera_testigo_fijo.py familia 4 6 8 10 12
     python scripts/escalera_testigo_fijo.py analisis
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv
import resource
import sys
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from holonomy.ladder_ed import TwistedLadder
from holonomy.report import DATA

EVALS = DATA / "escalera_testigo_fijo_evals.csv"
EV_HEAD = ["L", "Psi", "x", "Delta", "c_r", "E0", "res", "p_lxy", "p_lzz", "p_rxy", "p_rzz", "frio", "tiempo_s"]
FIJO = DATA / "escalera_testigo_fijo.csv"
FAMILIA = DATA / "escalera_familia.csv"
XS_FIJOS = (1.0, 0.8)


def clave(L, Psi, x, D, cr, frio):
    return (int(L), round(float(Psi), 12), round(float(x), 13), round(float(D), 13), round(float(cr), 13), int(frio))


def leer_evals():
    if not EVALS.exists():
        return {}
    with open(EVALS) as fh:
        return {clave(r["L"], r["Psi"], r["x"], r["Delta"], r["c_r"], r["frio"]): r for r in csv.DictReader(fh)}


class Evaluador:
    """E_0 de la escalera con caché en disco; el arranque en caliente usa el último vector."""

    def __init__(self, L, Psi):
        self.L, self.Psi = L, Psi
        self.T = None
        self.u = None
        self.cache = leer_evals()

    def __call__(self, x, D, cr, frio=False):
        k = clave(self.L, self.Psi, x, D, cr, frio)
        if k in self.cache:
            r = self.cache[k]
            return float(r["E0"]), float(r["res"]), np.array([float(r[c]) for c in EV_HEAD[7:11]])
        if self.T is None:
            self.T = TwistedLadder(self.L, Psi=self.Psi)
        self.T.set_coefs(x, D, cr)
        t0 = time.time()
        E, u, res = self.T.ground(seed=12345, v0=None if (frio or self.u is None) else self.u)
        if not frio:
            self.u = u
        P = self.T.parts(u)
        fila = [self.L, repr(float(self.Psi)), repr(float(x)), repr(float(D)), repr(float(cr)), repr(E), repr(res),
                *[repr(float(p)) for p in P], int(frio), f"{time.time() - t0:.1f}"]
        nuevo = not EVALS.exists()
        with open(EVALS, "a", newline="") as fh:
            w = csv.writer(fh)
            if nuevo:
                w.writerow(EV_HEAD)
            w.writerow(fila)
        self.cache[k] = dict(zip(EV_HEAD, map(str, fila)))
        print(f"      L = {self.L}, Psi = {self.Psi:.4f}, (x, D, c_r) = ({x:.10f}, {D:.10f}, {cr:.10f}): "
              f"E0 = {E:.12f}, res {res:.1e}, {time.time() - t0:.1f} s, "
              f"pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
        return E, res, P


def norma(L, x, D, cr):
    return L * (2 * (2 + D) + 2 * x + cr)


def cota(L, E, res, x, D, cr):
    """Cota superior certificada de v y el valor sin el residuo."""
    N = norma(L, x, D, cr)
    return -4 * (E - res) / N, -4 * E / N


def etapa_fijo(chk, Ls):
    filas = []
    for L in Ls:
        for x in XS_FIJOS:
            for Psi in (0.0, np.pi):
                E, res, _ = Evaluador(L, Psi)(x, 1.0, x, frio=True)
                u_cert, u = cota(L, E, res, x, 1.0, x)
                chk.check(res < 1e-8, f"fijo L = {L}, x = {x}, Psi = {Psi:.4f}: E0 = {E:.12f}, residuo {res:.1e}")
                filas.append([L, x, Psi, E, res, u, u_cert])
    _escribir(FIJO, ["L", "x", "Psi", "E0", "res", "cota", "cota_cert"], filas)


def etapa_familia(chk, Ls):
    filas = []
    for L in Ls:
        # psi = 0: Heisenberg, un parámetro.
        ev = Evaluador(L, 0.0)

        def f0(x):
            E, _, _ = ev(x, 1.0, x)
            return -4 * E / norma(L, x, 1.0, x)
        r0 = minimize_scalar(f0, bounds=(0.3, 1.5), method="bounded", options=dict(xatol=1e-7))
        x0 = float(r0.x)
        E, res, _ = ev(x0, 1.0, x0, frio=True)
        u0c, u0 = cota(L, E, res, x0, 1.0, x0)
        chk.check(res < 1e-8 and abs(u0 - r0.fun) < 1e-9,
                  f"familia L = {L}, psi = 0: v = {u0:.12f} en x = {x0:.6f} (residuo {res:.1e}, frío frente a caliente "
                  f"{u0 - r0.fun:.1e})")
        filas.append([L, 0.0, x0, 1.0, x0, u0, u0c, res, r0.nfev])

        # psi = pi: (x, Delta, c_r), lineal en el testigo; gradiente de Hellmann-Feynman.
        ev = Evaluador(L, np.pi)

        def fpi(y):
            x, D, cr = y
            E, _, P = ev(x, D, cr)
            N = norma(L, x, D, cr)
            dE = np.array([P[2], P[1], P[3]])  # d/dx, d/dDelta, d/dc_r
            dN = L * np.array([2.0, 2.0, 1.0])
            return -4 * E / N, -4 * (dE * N - E * dN) / N**2
        rp = minimize(fpi, [x0, 1.0, x0], jac=True, method="L-BFGS-B",
                      bounds=[(-1.0, 2.0), (-1.0, 3.0), (-1.0, 3.0)], options=dict(ftol=1e-14, gtol=1e-9, maxiter=200))
        xp, Dp, crp = map(float, rp.x)
        E, res, _ = ev(xp, Dp, crp, frio=True)
        upc, up = cota(L, E, res, xp, Dp, crp)
        borde = any(np.isclose(y, b, atol=1e-6) for y, bb in zip(rp.x, [(-1.0, 2.0), (-1.0, 3.0), (-1.0, 3.0)]) for b in bb)
        chk.check(res < 1e-8 and abs(up - rp.fun) < 1e-9 and not borde,
                  f"familia L = {L}, psi = pi: v = {up:.12f} en (x, Delta, c_r) = ({xp:.6f}, {Dp:.6f}, {crp:.6f}) "
                  f"(residuo {res:.1e}, {rp.nfev} evaluaciones, interior, {rp.message})")
        filas.append([L, np.pi, xp, Dp, crp, up, upc, res, rp.nfev])
    previas = _leer(FAMILIA)
    hechas = {(int(f[0]), round(f[1], 6)) for f in filas}
    filas = [f for f in previas if (int(f[0]), round(float(f[1]), 6)) not in hechas] + filas
    _escribir(FAMILIA, ["L", "psi", "x", "Delta", "c_r", "v", "v_cert", "res", "evaluaciones"],
              sorted(filas, key=lambda f: (int(f[0]), float(f[1]))))


def _leer(ruta):
    if not ruta.exists():
        return []
    with open(ruta) as fh:
        r = csv.reader(fh)
        next(r)
        return [[float(c) for c in fila] for fila in r]


def _escribir(ruta, cab, filas):
    if ruta == FIJO:
        previas = _leer(FIJO)
        hechas = {(int(f[0]), f[1], round(f[2], 6)) for f in filas}
        filas = [f for f in previas if (int(f[0]), f[1], round(f[2], 6)) not in hechas] + filas
        filas.sort(key=lambda f: (f[1], int(f[0]), f[2]))
    with open(ruta, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cab)
        for f in filas:
            w.writerow([int(f[0])] + [repr(float(c)) for c in f[1:]])
    print(f"escrito {ruta.relative_to(DATA.parent)}", flush=True)


def locales(Ls, ys):
    return [-np.log(ys[i + 1] / ys[i]) / np.log(Ls[i + 1] / Ls[i]) for i in range(len(Ls) - 1)]


def ajustes(Ls, ys):
    """Mínimos cuadrados en ln y: (nombre, parámetro, residuo cuadrático medio en ln y)."""
    Ls, ly = np.asarray(Ls, float), np.log(ys)
    out = []
    for nombre, pre in (("e^{-L/xi}", 0), ("L^-1 e^{-L/xi}", 1), ("L^-2 e^{-L/xi}", 2)):
        A = np.column_stack([np.ones_like(Ls), -Ls])
        c, *_ = np.linalg.lstsq(A, ly + pre * np.log(Ls), rcond=None)
        rms = np.sqrt(np.mean((A @ c - pre * np.log(Ls) - ly) ** 2))
        out.append((nombre, 1 / c[1], rms))
    A = np.column_stack([np.ones_like(Ls), -np.log(Ls)])
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    out.append(("L^-p", c[1], np.sqrt(np.mean((A @ c - ly) ** 2))))
    return out


def etapa_analisis(chk):
    fijo = _leer(FIJO)
    if fijo:
        for x in XS_FIJOS:
            filas = sorted((f for f in fijo if f[1] == x), key=lambda f: (int(f[0]), f[2]))
            Ls = sorted({int(f[0]) for f in filas if f[0] % 2 == 0})
            E = {(int(f[0]), round(f[2], 6)): f[3] for f in filas}
            Ls = [L for L in Ls if (L, 0.0) in E and (L, round(np.pi, 6)) in E]
            if len(Ls) < 3:
                continue
            dE = np.array([E[(L, round(np.pi, 6))] - E[(L, 0.0)] for L in Ls])
            R = np.array([-4 * d / norma(L, x, 1, x) for L, d in zip(Ls, dE)])
            chk.check(np.all(dE > 0), f"testigo fijo x = {x}: el giro pi sube E_0 en todos los L pares")
            chk.info(f"x = {x}: L = {Ls}")
            chk.info("  dE = E0(pi) - E0(0): " + ", ".join(f"{d:.4e}" for d in dE))
            chk.info("  R = -4 dE/N:        " + ", ".join(f"{r:.4e}" for r in R))
            chk.info("  cocientes dE_L/dE_{L+2}: " + ", ".join(f"{dE[i] / dE[i + 1]:.4f}" for i in range(len(Ls) - 1)))
            chk.info("  xi local, 2/ln(cociente): " + ", ".join(f"{2 / np.log(dE[i] / dE[i + 1]):.4f}" for i in range(len(Ls) - 1)))
            chk.info("  exponentes locales:  " + ", ".join(f"{p:.3f}" for p in locales(Ls, dE)))
            for nombre, par, rms in ajustes(Ls, dE):
                chk.info(f"  {nombre:16s} parámetro {par:8.4f}  residuo en ln {rms:.4f}")
            for nombre, par, rms in ajustes(Ls[-4:], dE[-4:]):
                chk.info(f"  últimos 4: {nombre:16s} parámetro {par:8.4f}  residuo en ln {rms:.4f}")
    fam = _leer(FAMILIA)
    if fam:
        v = {(int(f[0]), round(f[1], 6)): f for f in fam}
        Ls = sorted({int(f[0]) for f in fam})
        for L in Ls:
            ruta = DATA / f"ladder_limit_F0_L{L}.csv"
            if not ruta.exists():
                continue
            with open(ruta) as fh:
                cg = {round(float(r["psi"]), 6): float(r["v_max"]) for r in csv.DictReader(fh)}
            for psi in (0.0, round(np.pi, 6)):
                if (L, psi) in v and psi in cg:
                    d = v[(L, psi)][5] - cg[psi]
                    chk.check(abs(d) < 2e-8, f"familia frente a generación de columnas, L = {L}, psi = {psi:.4f}: "
                              f"{v[(L, psi)][5]:.10f} frente a {cg[psi]:.10f} (dif {d:.1e})")
        Lp = [L for L in Ls if (L, 0.0) in v and (L, round(np.pi, 6)) in v and L % 2 == 0]
        if len(Lp) >= 3:
            dv = np.array([v[(L, 0.0)][5] - v[(L, round(np.pi, 6))][5] for L in Lp])
            chk.info(f"familia, L = {Lp}: v(0) = " + ", ".join(f"{v[(L, 0.0)][5]:.8f}" for L in Lp))
            chk.info("  v(0) - v(pi) = " + ", ".join(f"{d:.4e}" for d in dv))
            chk.info("  exponentes locales: " + ", ".join(f"{p:.3f}" for p in locales(Lp, dv)))
            chk.info("  xi local: " + ", ".join(f"{2 / np.log(dv[i] / dv[i + 1]):.4f}" for i in range(len(Lp) - 1)))
            for nombre, par, rms in ajustes(Lp, dv):
                chk.info(f"  {nombre:16s} parámetro {par:8.4f}  residuo en ln {rms:.4f}")


def main():
    chk = Checker("escalera_testigo_fijo")
    etapa = sys.argv[1] if len(sys.argv) > 1 else "analisis"
    Ls = [int(a) for a in sys.argv[2:]]
    if etapa == "fijo":
        etapa_fijo(chk, Ls)
    elif etapa == "familia":
        etapa_familia(chk, Ls)
    elif etapa != "analisis":
        raise SystemExit(f"etapa desconocida: {etapa}")
    etapa_analisis(chk)
    chk.finish()


if __name__ == "__main__":
    main()

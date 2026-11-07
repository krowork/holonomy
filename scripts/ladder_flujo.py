"""Escalera periódica con flujo total fijo F (f = F/L) leída como una franja de
una hoja esférica (exploración; solo lee data/, no resuelve ningún SDP).

En una esfera unidad, el paralelo de colatitud theta tiene holonomía de
transporte paralelo 2pi(1 - cos theta). Si los ejes de Werner son los del
marco tangente, las dos patas son dos paralelos y la escalera es la franja
entre ellos: F = 2pi(cos theta_t - cos theta_b) es su área, y psi, la media de
las holonomías de las patas, es 2pi(1 - c) mód 2pi con c el coseno medio de la
colatitud (c = seno de la latitud del centro de la franja). El ecuador da
psi = 0. Con v(f, psi) = v(f, -psi) (Lema simfpsi de escaleras) norte y sur
coinciden, y psi en [0, pi] corresponde a c = psi/2pi (latitud 0 a 30 grados)
o a c = 1 - psi/2pi (de 30 grados al polo), si la franja cabe: F <= 4pi(1 - c).

Lee data/ladder_limit_F<F/pi>_L<L>.csv (ladder_limit.py --F) para
F = 0, pi/2, pi, 2pi, L = 4..10 y psi = 0, pi/4, pi/2, 3pi/4, pi. Comprobaciones:
  1. los datos están completos y certificados (cotas a < 1e-8);
  2. cada psi se recupera de su latitud con 2pi(1 - c);
  3. la oscilación de paridad, v(L) - (v(L-1) + v(L+1))/2, se amortigua: su
     máximo en L = 8, 9 es menor que en L = 5, 6, en cada (F, psi);
  4. a psi fijo, v baja al subir F en L = 9 y 10 (valores certificados);
  5. extrapolación (Shanks con L = 6, 8, 10 y 5, 7, 9; orientativa): en F = 0
     los límites de L pares e impares coinciden a < 2e-3 en cada latitud;
  6. extrapolación: el límite medio de pares e impares baja al subir F en
     cada latitud.
Da además el exponente p de |paridad| ~ L^-p (ajuste en L = 6..9; p <= 0 si
no decae de forma regular). El umbral 2e-3 de la comprobación 5 se fijó tras
ver los datos (diferencia máxima 1,8e-3): es una descripción, no una predicción.

Escribe data/ladder_flujo_latitud.csv.

Uso: python scripts/ladder_flujo.py   (~1 s)
"""
import _common  # noqa: F401
from _common import Checker

import csv
from pathlib import Path

import numpy as np

from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
FS = ("0", "0.5", "1", "2")
LS = range(4, 11)
PSIS = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * np.pi
TOL = 1e-8
EXTRAP_TOL = 2e-3


def shanks(a, b, c):
    """Transformación de Shanks de tres términos consecutivos."""
    return c - (c - b) ** 2 / ((c - b) - (b - a))


def latitudes(psi, F):
    """Latitudes (grados) del centro de una franja de área F con media psi."""
    c = psi / (2 * np.pi)
    out = [np.degrees(np.arcsin(c))]
    if 0 < c < 0.5 and F <= 4 * np.pi * c + 1e-12:
        out.append(np.degrees(np.arcsin(1 - c)))
    return out


def fold(x):
    """Ángulo a [0, pi] con x ~ -x y periodo 2pi."""
    x = np.mod(x, 2 * np.pi)
    return min(x, 2 * np.pi - x)


def main():
    chk = Checker("ladder_flujo")
    V, ok = {}, True
    for F in FS:
        for L in LS:
            path = DATA / f"ladder_limit_F{F}_L{L}.csv"
            rows = list(csv.DictReader(open(path))) if path.exists() else []
            psis = np.array([float(r["psi"]) for r in rows])
            ok &= (len(rows) == len(PSIS) and np.allclose(psis, PSIS, atol=1e-12)
                   and all(abs(float(r["dif_cotas"])) < TOL for r in rows))
            V[F, L] = [float(r["v_max"]) for r in rows]
    chk.check(ok, f"datos completos y certificados: F/pi = {', '.join(FS)}, L = 4..10, 5 psi, "
              f"cotas a < {TOL:g}")
    if not ok:
        chk.finish()
        return

    err = max(abs(fold(2 * np.pi * (1 - np.sin(np.radians(lat)))) - psi)
              for psi in PSIS for F in FS for lat in latitudes(psi, float(F) * np.pi))
    chk.check(err < 1e-12, f"psi = 2pi(1 - c) mód 2pi en todas las latitudes, |dif| = {err:.1e}")

    rows, damp, lims = [], True, {}
    for F in FS:
        for j, psi in enumerate(PSIS):
            s = {L: V[F, L][j] for L in LS}
            par = {L: s[L] - (s[L - 1] + s[L + 1]) / 2 for L in range(5, 10)}
            early, late = max(abs(par[5]), abs(par[6])), max(abs(par[8]), abs(par[9]))
            damp &= late < early
            Lf = np.arange(6, 10)
            p = -np.polyfit(np.log(Lf), np.log([abs(par[L]) for L in Lf]), 1)[0]
            ev, od = shanks(s[6], s[8], s[10]), shanks(s[5], s[7], s[9])
            lims[F, j] = (ev, od)
            lat = " o ".join(f"{x:.1f}" for x in latitudes(psi, float(F) * np.pi))
            chk.info(f"F/pi={F:3s} psi/pi={psi / np.pi:.2f} (latitud {lat}): paridad "
                     f"{' '.join(f'{par[L]:+.4f}' for L in range(5, 10))} (p = {p:.1f}); "
                     f"límite pares {ev:.4f}, impares {od:.4f}")
            rows.append((float(F) * np.pi, psi, lat, *(par[L] for L in range(5, 10)), p, ev, od))
    chk.check(damp, "la paridad se amortigua: máx. en L = 8, 9 < máx. en L = 5, 6 en cada (F, psi)")

    for L in (9, 10):
        dec = all(V[a, L][j] > V[b, L][j] for j in range(len(PSIS)) for a, b in zip(FS, FS[1:]))
        chk.check(dec, f"L = {L}: v baja al subir F en cada psi")

    d0 = max(abs(lims["0", j][0] - lims["0", j][1]) for j in range(len(PSIS)))
    chk.check(d0 < EXTRAP_TOL, f"extrapolación, F = 0: límites de pares e impares a {d0:.1e} "
              f"(< {EXTRAP_TOL:g}) en cada latitud")
    dF = max(abs(lims[F, j][0] - lims[F, j][1]) for F in FS[1:] for j in range(len(PSIS)))
    chk.info(f"extrapolación, F > 0: pares e impares difieren hasta {dF:.1e}")
    mid = {k: (a + b) / 2 for k, (a, b) in lims.items()}
    dec = all(mid[a, j] > mid[b, j] for j in range(len(PSIS)) for a, b in zip(FS, FS[1:]))
    chk.check(dec, "extrapolación: el límite medio baja al subir F en cada latitud ("
              + ", ".join(f"F/pi={F}: {min(mid[F, j] for j in range(5)):.4f}–"
                          f"{max(mid[F, j] for j in range(5)):.4f}" for F in FS) + ")")

    write_csv("ladder_flujo_latitud.csv",
              ["F", "psi", "latitud", "paridad_L5", "paridad_L6", "paridad_L7", "paridad_L8",
               "paridad_L9", "exponente", "limite_pares", "limite_impares"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

"""Números del artículo unificado (solo lectura).

Artículo (~/fisica/holonomia.tex, «Holonomía y pegado de perspectivas de
Werner: la rigidez del testigo óptimo frente a un flujo», versión única y
autocontenida): todas las secciones. No resuelve ningún SDP ni diagonaliza
nada.

El artículo reúne, literalmente, partes de cinco artículos anteriores
(anillos, escaleras, diamantes, esfera y rigidez). Tres comprobaciones:
  1. Cifras contra los CSV: las de Kohn, anillos, escaleras y diamantes que
     comprueba rigidez_numeros.py (sus funciones anillos, escaleras y
     diamantes, sobre el texto de este artículo).
  2. Procedencia: todo número del texto (decimales y m\\times10^{e}, extraídos
     como en rigidez_numeros.py) que no se haya comprobado en 1 tiene que
     aparecer en alguno de los artículos de origen. Esas cifras ya las
     comprueban los guiones de cada artículo (table62.py, edge.py,
     baedge2.py, diamond_numeros.py, diamond_punto_a.py...) contra data/, así
     que lo que se vigila aquí es la copia: un número mal transcrito no está en
     ninguna fuente. Los pocos números escritos de nuevo están en NUEVOS, cada
     uno con su motivo.
  3. Autocontenido: el texto no cita ni menciona los artículos de origen.
Si faltan los artículos de origen (no están en git), la comprobación 2 se
salta y se avisa; si falta el propio artículo, solo se hace la 1 contra los
CSV.

Uso: python scripts/holonomia_numeros.py [ruta del .tex]
     (por defecto ~/fisica/holonomia.tex; menos de 1 s, sin SDP)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import re
import sys
from pathlib import Path

from math import comb

import numpy as np

from escalera_testigo_fijo import ajustes, locales
from rigidez_numeros import Cifras, anillos, cuerpo, diamantes, escaleras, f, leer, numeros

FISICA = Path.home() / "fisica"
TEX = FISICA / "holonomia.tex"
FUENTES = ["holonomia_werner.tex", "holonomia_escaleras.tex", "holonomia_diamantes.tex",
           "holonomia_esfera.tex", "holonomia_rigidez.tex"]

# Números escritos de nuevo en el artículo unificado, con su motivo.
NUEVOS = {
    # Teorema thm:igualdad (cadena cerrada de diamantes, f = 0); todos de la salida de
    # scripts/diamond_igualdad.py, que los comprueba.
    "0,526": "constante de Lipschitz 2 sqrt(2) a* del exceso del anillo girado, k = 3",
    "0,495": "mínimo de la malla de 61 puntos menos la constante por media separación, k = 3",
    "0,284": "exceso del anillo en psi = 0, k = 2",
    "0,698": "exceso del anillo en psi = 0, k = 4",
    "2e-31": "peso del sector sin singletes en el promedio, k = 2, 3, 4",
    "2,0e-10": "error de marginales del estado girado, k = 3 (columna err_girado del CSV)",
    "2,4e-15": "error de marginales del estado girado, k = 2 y 4 (columna err_girado del CSV)",
    "2,2e-2": "error de marginales del estado sin girar, mínimo (columna err_sin_girar del CSV)",
    # Escaleras, testigo fijo: dato de la bibliografía.
    "3,19": "longitud de correlación de espín de la escalera isótropa, White, Noack y Scalapino",
}

# Citas y menciones de los artículos de origen, que el texto no debe tener.
ORIGEN = re.compile(r"\\cite(\[[^]]*\])?\{[^}]*\b(anillos|escaleras|diamantes|esfera|rigidez)\b"
                    r"|trabajos anteriores|esta serie")


def testigo_fijo(ck, num):
    """Párrafos «Familias exactas» y «Testigo fijo» de las escaleras (escalera_testigo_fijo.py)."""
    fam = {(int(r["L"]), round(f(r, "psi"), 6)): r for r in leer("escalera_familia.csv")}
    fij = {(int(r["L"]), f(r, "x"), round(f(r, "Psi"), 6)): r for r in leer("escalera_testigo_fijo.csv")}
    pi6 = round(np.pi, 6)
    num("4e7", comb(28, 14), "escaleras: dimensión del sector S^z = 0 en L = 14")
    num("1,1e-11", max(f(r, "res") for r in list(fam.values()) + list(fij.values())),
        "escaleras: residuo máximo de Lanczos", "<=")
    dif = 0.0
    for L in (4, 6, 8, 10):
        cg = {round(f(r, "psi"), 6): f(r, "v_max") for r in leer(f"ladder_limit_F0_L{L}.csv")}
        for p in (0.0, pi6):
            dif = max(dif, abs(f(fam[(L, p)], "v") - cg[p]))
    num("2,5e-9", dif, "escaleras: familia frente a generación de columnas, L <= 10", "<=")
    for L, tv, tx in ((12, "0,5106739", "0,785"), (14, "0,5103064", "0,778")):
        num(tv, f(fam[(L, 0.0)], "v"), f"escaleras: v(L, 0, 0), L = {L}")
        num(tx, f(fam[(L, 0.0)], "x"), f"escaleras: x óptimo, L = {L}")
    Ls = [4, 6, 8, 10, 12, 14]
    dv = [f(fam[(L, 0.0)], "v") - f(fam[(L, pi6)], "v") for L in Ls]
    num("1,11e-3", dv[4], "escaleras: v(0) - v(pi), L = 12")
    num("5,47e-4", dv[5], "escaleras: v(0) - v(pi), L = 14")
    for t, p in zip(("3,54", "3,64", "3,90", "4,23", "4,62"), locales(Ls, dv)):
        num(t, p, "escaleras: exponente local de v(0) - v(pi)")
    for x, txts in ((1.0, ("0,280", "0,107", "4,73e-2", "2,21e-2", "1,06e-2", "5,19e-3")), (0.8, None)):
        dE = [f(fij[(L, x, pi6)], "E0") - f(fij[(L, x, 0.0)], "E0") for L in Ls]
        loc = locales(Ls, dE)
        aj = {n: (par, rms) for n, par, rms in ajustes(Ls, dE)}
        if txts:
            for t, d in zip(txts, dE):
                num(t, d, "escaleras: dE del testigo fijo x = 1")
            for t, p in zip(("2,38", "2,84", "3,41", "4,02", "4,64"), loc):
                num(t, p, "escaleras: exponente local de dE, x = 1")
            num("0,17", aj["L^-p"][1], "escaleras: residuo de L^-p, x = 1")
            num("0,074", aj["e^{-L/xi}"][1], "escaleras: residuo de e^{-L/xi}, x = 1")
            num("0,0083", aj["L^-1 e^{-L/xi}"][1], "escaleras: residuo de L^-1 e^{-L/xi}, x = 1")
            num("3,67", aj["L^-1 e^{-L/xi}"][0], "escaleras: xi, x = 1")
        else:
            num("2,08", loc[0], "escaleras: primer exponente local de dE, x = 0,8")
            num("3,65", loc[-1], "escaleras: último exponente local de dE, x = 0,8")
            num("0,011", aj["L^-1 e^{-L/xi}"][1], "escaleras: residuo de L^-1 e^{-L/xi}, x = 0,8")
            num("4,97", aj["L^-1 e^{-L/xi}"][0], "escaleras: xi, x = 0,8")
            num("0,117", aj["L^-p"][1], "escaleras: residuo de L^-p, x = 0,8")


def main():
    ck = Checker("holonomia_numeros")
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else TEX
    s = cuerpo(ruta.read_text()) if ruta.exists() else None
    if s is None:
        ck.info(f"no está {ruta}: solo se comprueban las cifras contra los CSV")

    num = Cifras(ck, s)
    anillos(ck, num)
    escaleras(ck, num)
    diamantes(ck, num)
    testigo_fijo(ck, num)
    if s is None:
        ck.finish()

    todos = numeros(s)
    resto = sorted(set(todos) - num.vistas, key=todos.index)
    fuentes = [FISICA / n for n in FUENTES if (FISICA / n).exists()]
    if len(fuentes) < len(FUENTES):
        ck.info(f"faltan {len(FUENTES) - len(fuentes)} artículos de origen en {FISICA}: "
                "no se comprueba la procedencia")
    else:
        en_fuentes = set()
        for p in fuentes:
            en_fuentes |= set(numeros(cuerpo(p.read_text())))
        nuevos = [t for t in resto if t in NUEVOS]
        huerfanos = [t for t in resto if t not in en_fuentes and t not in NUEVOS]
        ck.check(not huerfanos,
                 f"procedencia: {len(set(todos))} números distintos, {len(num.vistas & set(todos))} "
                 f"contra los CSV, {len(resto) - len(nuevos) - len(huerfanos)} copiados de los "
                 f"artículos de origen, {len(nuevos)} nuevos con motivo"
                 + (f"; sin procedencia: {', '.join(huerfanos)}" if huerfanos else ""))
        sobran = sorted(set(NUEVOS) - set(todos))
        ck.check(not sobran, "NUEVOS solo lista números que están en el texto"
                 + (f"; sobran: {', '.join(sobran)}" if sobran else ""))

    menciones = sorted({m.group(0) for m in ORIGEN.finditer(s)})
    ck.check(not menciones, "autocontenido: no cita ni menciona los artículos de origen"
             + (f"; aparece: {', '.join(menciones)}" if menciones else ""))
    ck.finish()


if __name__ == "__main__":
    main()

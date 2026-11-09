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

from rigidez_numeros import Cifras, anillos, cuerpo, diamantes, escaleras, numeros

FISICA = Path.home() / "fisica"
TEX = FISICA / "holonomia.tex"
FUENTES = ["holonomia_werner.tex", "holonomia_escaleras.tex", "holonomia_diamantes.tex",
           "holonomia_esfera.tex", "holonomia_rigidez.tex"]

# Números escritos de nuevo en el artículo unificado, con su motivo.
NUEVOS = {
}

# Citas y menciones de los artículos de origen, que el texto no debe tener.
ORIGEN = re.compile(r"\\cite(\[[^]]*\])?\{[^}]*\b(anillos|escaleras|diamantes|esfera|rigidez)\b"
                    r"|trabajos anteriores|esta serie")


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

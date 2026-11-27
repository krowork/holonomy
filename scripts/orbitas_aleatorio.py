"""Teorema D″ con perturbación cinética aleatoria: ¿qué órbita de defecto gana? (celda_k_notas.md, Sec. 7.14).

h = -P_asc + eps K (K de celda_k_aleatorio.termino con delta = 0: solo elementos que conservan la carga, la clase
(C) intacta). Con 3 ∤ n, las tres órbitas de un defecto tienen cargas Q = -1, 0, +1. Para cada muestra y cada n:
  - E_Q = min_Theta E_0(Q, Theta) para las tres cargas (malla de NTHETA valores);
  - Q* (la ganadora), el hueco entre órbitas g_orb = (segunda) - (primera);
  - R_n del fundamental global, min_Q E_0(Q, Theta), y n^2 R_n.
Con QSOLO=q solo se calcula ese sector (para R_n de una órbita con malla fina).
Uso: [QSOLO=q] [NTHETA=m] python scripts/orbitas_aleatorio.py eps semilla "muestras" n [n ...]   (muestras: "0-11" o "3,5")
"""
import os
import sys

import numpy as np

from celda_k_aleatorio import Anillo, termino

NTHETA = int(os.environ.get("NTHETA", 12))
QSOLO = os.environ.get("QSOLO")  # si se da, solo ese sector (R_n de la órbita, malla fina)


def main():
    eps, semilla = float(sys.argv[1]), int(sys.argv[2])
    sel = sys.argv[3]
    muestras = (list(range(int(sel.split("-")[0]), int(sel.split("-")[1]) + 1)) if "-" in sel
                else [int(x) for x in sel.split(",")])
    ns = [int(x) for x in sys.argv[4:]]
    rng = np.random.default_rng(semilla)
    hs = [termino(rng, eps, 0.0) for _ in range(max(muestras) + 1)]
    thetas = np.linspace(0, 2 * np.pi, NTHETA + 1)[:-1]
    anillos = {}
    for s in muestras:
        partes = []
        for n in ns:
            if n not in anillos:
                anillos[n] = Anillo(n)
            A = anillos[n]
            if QSOLO is not None:
                Eq = np.array([A.e0(hs[s], int(QSOLO), t) for t in thetas])
                R = Eq.max() - Eq.min()
                partes.append(f"n={n}: Q={int(QSOLO):+d} R={R:.3e} n2R={n * n * R:.4f}")
                continue
            E = np.array([[A.e0(hs[s], Q, t) for t in thetas] for Q in (-1, 0, 1)])
            EQ = E.min(1)
            orden = np.argsort(EQ)
            Qs = orden[0] - 1
            gorb = EQ[orden[1]] - EQ[orden[0]]
            Eg = E.min(0)
            R = Eg.max() - Eg.min()
            partes.append(f"n={n}: Q*={Qs:+d} g_orb={gorb:.3f} R={R:.2e} n2R={n * n * R:.4f}")
        print(f"[{s}] " + " | ".join(partes), flush=True)


if __name__ == "__main__":
    main()

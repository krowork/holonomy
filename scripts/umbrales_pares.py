"""Umbrales de los pares complejos del XXZ retorcido con Delta < 0 (robustez_mps.tex, párrafo
«Delta_inf < 0» y problema abierto 3).

Derivación (sin rigor). Si j raíces llegan a la vez a lambda = +infinito (o -infinito), se suman sus
ecuaciones de Bethe: los desfases mutuos se cancelan (theta es impar), y en el infinito
p -> +-(pi - gamma) y theta(+-infinito - lambda_k) -> +-(pi - 2 gamma) frente a las M - j raíces
finitas. Con los números cuánticos de holonomy.bethe (las j superiores con n par; las j inferiores con
n impar y la rama -1) sale

    n par:    Phi = 2 j (pi - gamma),          n impar:  pi - Phi = (2 j + 1)(pi - gamma),

sin n: el umbral no depende del tamaño. Con j impar una raíz real pasa por el infinito a la recta
Im lambda = pi/2 (un paso suave en momentos; j = 1 con n par y Phi = pi es Delta = 0); con j par esa
raíz y la siguiente real forman un par complejo, el par k = j/2: pi - gamma = Phi/(4k) o
(pi - Phi)/(4k + 1).

Comprobaciones con holonomy.bethe_p:
  1. alternancia: a cada lado de los umbrales j = 1..5 (Delta_j +- 2e-3), el número de pares y de
     raíces reales en la recta Im lambda = pi/2 cambia como se predice (n = 10, 11, 40, 41; tres Phi
     por paridad);
  2. bisección del cambio en el número de pares para j = 2 y 4 (n = 10, 11, 40, 41): coincide con
     la fórmula a 1e-8 en Delta con j = 2 y a 3e-7 con j = 4. Cerca del umbral j el grupo de raíces
     se acerca al infinito como |Delta - Delta_j|^(1/j) (en p), así que con j = 4 la ecuación es muy
     plana y el punto exacto se resuelve peor.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from holonomy import bethe_p  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def umbral(n, Phi, j):
    """Delta_j de la fórmula derivada."""
    return -np.cos(Phi / (2 * j)) if n % 2 == 0 else -np.cos((np.pi - Phi) / (2 * j + 1))


def estado(S, D, Phi, b):
    """(pares, reales en la recta Im lambda = pi/2)."""
    p = S.solve(D, Phi, b).astype(complex)
    c = np.pi - np.arccos(D)
    re = p[p.imag == 0].real
    return len(S._pares(p)), int(np.sum(np.abs(re) > c))


def esperado(j):
    """(pares, reales en Im = pi/2) justo por debajo del umbral j."""
    return j // 2, j % 2


if __name__ == "__main__":
    casos = [(n, Phi) for n in (10, 40) for Phi in (np.pi / 2, 2 * np.pi / 3, np.pi)] + \
            [(n, Phi) for n in (11, 41) for Phi in (0.0, np.pi / 6, np.pi / 3)]
    print("1. alternancia (pares, reales en Im = pi/2) encima -> debajo de cada umbral")
    ok_alt = True
    for n, Phi in casos:
        b = 0 if n % 2 == 0 else -1
        S = bethe_p.MomentumSolver(n)
        fila = []
        for j in range(1, 6):
            Dj = umbral(n, Phi, j)
            if Dj < -0.995:
                break
            arriba, abajo = estado(S, Dj + 2e-3, Phi, b), estado(S, Dj - 2e-3, Phi, b)
            ok_alt &= arriba == esperado(j - 1) and abajo == esperado(j)
            fila.append(f"j={j}: {arriba}->{abajo}")
        print(f"   n={n:3d} Phi/pi={Phi / np.pi:.3f}  " + "  ".join(fila))
    check(ok_alt, "alternancia: j impar, una raíz pasa a Im lambda = pi/2; j par, se forma un par (j = 1..5)")

    print("2. bisección del umbral de los pares 1 y 2 (j = 2, 4)")
    err = {2: 0.0, 4: 0.0}
    for n, Phi in casos:
        b = 0 if n % 2 == 0 else -1
        for j in (2, 4):
            Dj = umbral(n, Phi, j)
            if Dj < -0.995:
                continue
            S = bethe_p.MomentumSolver(n)
            # encima: j/2 - 1 pares; debajo: j/2. Intervalo asimétrico: justo en el umbral el grupo
            # de raíces está en el infinito y la ecuación es singular.
            hi, lo = Dj + 2e-3, Dj - 1.3e-3
            for D in np.arange(-0.6, hi, -0.02):
                S.solve(D, Phi, b)
            while hi - lo > 1e-8:
                m = (hi + lo) / 2
                if estado(S, m, Phi, b)[0] >= j // 2:
                    lo = m
                else:
                    hi = m
            err[j] = max(err[j], abs((hi + lo) / 2 - Dj))
            print(f"   n={n:3d} Phi/pi={Phi / np.pi:.3f} j={j}: bisección {(hi + lo) / 2:.9f}, fórmula {Dj:.9f}")
    check(err[2] < 1e-8, f"umbral del primer par = fórmula (error máximo {err[2]:.1e} en Delta)")
    check(err[4] < 3e-7, f"umbral del segundo par = fórmula (error máximo {err[4]:.1e} en Delta)")
    sys.exit(0 if OK else 1)

"""Anatomía del testigo óptimo del dual para los datos de llenado 1/3 (exploración).

Para cada n y Theta, resuelve el dual (dual_celda, método suavizado) y descompone el testigo óptimo w (normalizado
con tr(w tau) = 1 y sin la parte proporcional a la identidad) frente a w0 = -h0 normalizado igual:
  - alpha = <off(w), off(w0)> / |off(w0)|^2, el peso de la parte cuántica de w0 en w (off = fuera de la diagonal);
  - la fracción de la parte fuera de la diagonal de w que no es proporcional a la de w0;
  - la diagonal de w (energías clásicas de los 9 pares) frente a la de w0;
  - el extremo superior del espectro del anillo con w, para ver la degeneración.

Uso: OMP_NUM_THREADS=1 python scripts/testigo_optimo.py eps lam theta_sobre_pi n [n ...]
"""
import sys

import numpy as np

from dual_celda import _tope, anillo, dual
from testigo_datos import Q, datos, termino

PARES = [f"{a}{b}" for a in range(3) for b in range(3)]


def sin_identidad(w):
    return w - np.trace(w) / 9 * np.eye(9)


def main():
    eps, lam, th = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) * np.pi
    ns = [int(x) for x in sys.argv[4:]]
    h = termino(eps, lam)
    sigma, _ = datos(h, 12, lam)
    tau = sigma - np.eye(9) / 9
    w0 = -h / np.real(np.trace(-h @ tau))
    off0 = w0 - np.diag(np.diag(w0))
    d0 = np.real(np.diag(sin_identidad(w0)))
    print(f"eps={eps} lam={lam} Theta/pi={th / np.pi}")
    print("  diagonal de w0 (sin identidad):", " ".join(f"{p}:{x:+.3f}" for p, x in zip(PARES, d0)))
    for n in ns:
        sup, w, inf = dual(n, 3, Q, tau, th, w0=-h, metodo="suave")
        w = sin_identidad(w)
        off = w - np.diag(np.diag(w))
        alpha = np.real(np.vdot(off0, off)) / np.real(np.vdot(off0, off0))
        resto = np.linalg.norm(off - alpha * off0) / max(np.linalg.norm(off), 1e-300)
        d = np.real(np.diag(w))
        lam_w, _ = _tope(anillo(n, 3, Q, w, th), 2 * n)
        print(f"n={n} (n%3={n % 3}): p en [{inf:.10f}, {sup:.10f}]  alpha={alpha:.4f}  "
              f"|off(w)|/|off(w0)|={np.linalg.norm(off) / np.linalg.norm(off0):.4f}  resto no paralelo={resto:.3f}")
        print("   diagonal de w:", " ".join(f"{p}:{x:+.3f}" for p, x in zip(PARES, d)))
        print("   espectro superior del anillo (lambda_1 - lambda_k):",
              " ".join(f"{lam_w[0] - x:.2e}" for x in lam_w[1:2 * n]), flush=True)


if __name__ == "__main__":
    main()

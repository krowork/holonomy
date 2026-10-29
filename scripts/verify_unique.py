"""Verificación simbólica (SymPy) de los elementos de matriz de la unicidad.

Paso 3(f) del teorema del anillo de 4 qubits: con kappa > 0 como símbolo y los
estados |K=0>, |K=pi> del paso 3(b), comprueba de forma exacta que

* <K=0|Z_0|K=pi>     = 1/(1+kappa^2),
* <K=0|Z_0 Z_1|K=pi> = -i kappa^2/(1+kappa^2),
* los elementos diagonales de Z_0 se anulan en ambos estados K, y
* los elementos diagonales de Z_0 Z_1 valen -f = -1/(1+kappa^2).

Convención: el qubit 0 es el más significativo (orden q0 ⊗ q1 ⊗ q2 ⊗ q3),
|0> es el autoestado de Z con valor +1 y un «giro» es un qubit en |1>.

Uso (desde la raíz del repositorio):  python scripts/verify_unique.py
Sale con estado 1 si alguna comprobación falla.
"""
import sys
from functools import reduce

import sympy as sp

N_SITES = 4
DIM = 2 ** N_SITES

kappa = sp.symbols("kappa", positive=True)

I2 = sp.eye(2)
Z = sp.Matrix([[1, 0], [0, -1]])

f = 1 / (1 + kappa**2)
N_norm = 1 / sp.sqrt(2 * (1 + kappa**2))

NEEL = ["1010", "0101"]
ADJ = ["1100", "0110", "0011", "1001"]
S = [1, -1, 1, -1]

_results = []


def check(ok, description):
    """Imprime «OK» o «FALLO» y guarda el resultado."""
    ok = bool(ok)
    print(("OK  " if ok else "FALLO ") + description)
    _results.append(ok)


def is_zero(expr):
    return sp.simplify(expr) == 0


def op(factors):
    """Producto tensorial con factors[k] en el qubit k e identidad en el resto."""
    return reduce(sp.kronecker_product, [factors.get(k, I2) for k in range(N_SITES)])


def ket(coeffs):
    """Vector de 16 componentes a partir de {cadena de bits: amplitud}."""
    v = sp.zeros(DIM, 1)
    for bits, amp in coeffs.items():
        v[int(bits, 2)] += amp
    return v


def matrix_element(bra, operator, k):
    return sp.simplify((bra.H * operator * k)[0, 0])


def main():
    # Estados K del paso 3(b).
    K0 = ket({NEEL[0]: N_norm, NEEL[1]: N_norm,
              **{ADJ[j]: -kappa * N_norm / sp.sqrt(2) for j in range(4)}})
    Kpi = ket({NEEL[0]: -N_norm, NEEL[1]: N_norm,
               **{ADJ[j]: sp.I * kappa * N_norm / sp.sqrt(2) * S[j] for j in range(4)}})
    check(is_zero((K0.H * K0)[0, 0] - 1) and is_zero((Kpi.H * Kpi)[0, 0] - 1)
          and is_zero((K0.H * Kpi)[0, 0]), "|K=0>, |K=pi> ortonormales")

    Z0 = op({0: Z})
    Z0Z1 = op({0: Z, 1: Z})

    # Coherencias (elementos fuera de la diagonal).
    check(is_zero(matrix_element(K0, Z0, Kpi) - 1 / (1 + kappa**2)),
          "<K=0|Z_0|K=pi> = 1/(1+kappa^2)")
    check(is_zero(matrix_element(K0, Z0Z1, Kpi) + sp.I * kappa**2 / (1 + kappa**2)),
          "<K=0|Z_0 Z_1|K=pi> = -i kappa^2/(1+kappa^2)")

    # Elementos diagonales.
    check(is_zero(matrix_element(K0, Z0, K0)) and is_zero(matrix_element(Kpi, Z0, Kpi)),
          "<K|Z_0|K> = 0 para K = 0, pi")
    check(is_zero(matrix_element(K0, Z0Z1, K0) + f) and is_zero(matrix_element(Kpi, Z0Z1, Kpi) + f),
          "<K|Z_0 Z_1|K> = -f = -1/(1+kappa^2) para K = 0, pi")

    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} comprobaciones correctas")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

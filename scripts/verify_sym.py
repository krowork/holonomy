"""Verificación simbólica (SymPy) del testigo del anillo de 4 qubits.

Comprueba de forma exacta, con kappa > 0 como símbolo, lo que la sección
«Teorema del anillo de 4 qubits» (pasos 1 y 2) y la sección «Verificación y
reproducibilidad» afirman sobre el testigo W:

* las identidades de normalización 2*J*kappa + c = 1/4, c^2 + 4*J^2 = 1/16,
  2*c + 1/2 = 1/(1+kappa^2), 2*sqrt(2)*J = kappa/(sqrt(2)*(1+kappa^2)) y
  1/(1+kappa^2) = 2/(3+sin(theta/2)) con kappa = cos(pi/4 - theta/4);
* que W conserva el número de giros y su espectro sector a sector;
* el bloque T = <A_j|W|N_k>, su forma explícita y T^dagger T = 16 J^2 I_2;
* que lambda_max(W) = f = 1/(1+kappa^2) con multiplicidad exactamente 2 en
  todo kappa en [1/sqrt(2), 1].

Convención: el qubit 0 es el más significativo (orden q0 ⊗ q1 ⊗ q2 ⊗ q3),
|0> es el autoestado de Z con valor +1 y un «giro» es un qubit en |1>.

Uso (desde la raíz del repositorio):  python scripts/verify_sym.py
Sale con estado 1 si alguna comprobación falla.
"""
import sys
from functools import reduce

import sympy as sp

N_SITES = 4
DIM = 2 ** N_SITES

kappa = sp.symbols("kappa", positive=True)
theta, phi, lam = sp.symbols("theta phi lambda", real=True)

I2 = sp.eye(2)
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])

# Coeficientes del testigo (paso 1).
J = kappa / (4 * (1 + kappa**2))
c = (1 - kappa**2) / (4 * (1 + kappa**2))
a = J / sp.sqrt(2)
f = 1 / (1 + kappa**2)

_results = []


def check(ok, description):
    """Imprime «OK» o «FALLO» y guarda el resultado."""
    ok = bool(ok)
    print(("OK  " if ok else "FALLO ") + description)
    _results.append(ok)


def is_zero(expr):
    return sp.simplify(expr) == 0


def is_zero_matrix(m):
    return all(is_zero(e) for e in m)


def op(factors):
    """Producto tensorial con factors[k] en el qubit k e identidad en el resto."""
    return reduce(sp.kronecker_product, [factors.get(k, I2) for k in range(N_SITES)])


def edge_ops(i):
    """A_i, B_i, C_i sobre la arista (i, i+1) con sitios módulo 4."""
    j = (i + 1) % N_SITES
    A = -(op({i: X, j: X}) + op({i: Y, j: Y}))
    B = op({i: X, j: Y}) - op({i: Y, j: X})
    C = -op({i: Z, j: Z})
    return A, B, C


def index(bits):
    """Índice en la base computacional de una cadena de bits (qubit 0 primero)."""
    return int(bits, 2)


def n_spins(idx):
    return bin(idx).count("1")


def build_witness():
    W = sp.zeros(DIM, DIM)
    for i in range(N_SITES):
        A, B, C = edge_ops(i)
        W += a * (A + B) + c * C
    return W


def main():
    # --- Identidades de normalización (pasos 1 y 2) ---
    check(is_zero(2 * J * kappa + c - sp.Rational(1, 4)), "2*J*kappa + c = 1/4")
    check(is_zero(c**2 + 4 * J**2 - sp.Rational(1, 16)), "c^2 + 4*J^2 = 1/16")
    check(is_zero(2 * c + sp.Rational(1, 2) - f), "2*c + 1/2 = 1/(1+kappa^2)")
    check(is_zero(2 * c - sp.Rational(1, 2) + kappa**2 * f),
          "2*c - 1/2 = -kappa^2/(1+kappa^2)")
    check(is_zero(2 * sp.sqrt(2) * J - kappa / (sp.sqrt(2) * (1 + kappa**2))),
          "2*sqrt(2)*J = kappa/(sqrt(2)*(1+kappa^2))")
    # Con theta = 4*phi y kappa = cos(pi/4 - phi), expandiendo las funciones trigonométricas.
    k_theta = sp.cos(sp.pi / 4 - theta / 4)
    expr = (1 / (1 + k_theta**2) - 2 / (3 + sp.sin(theta / 2))).subs(theta, 4 * phi)
    check(is_zero(sp.expand_trig(expr)),
          "1/(1+kappa^2) = 2/(3+sin(theta/2)) con kappa = cos(pi/4 - theta/4)")

    # --- Construcción de W y conservación del número de giros ---
    W = build_witness()
    check(is_zero_matrix(W - W.H), "W es hermítico")
    n_op = sp.zeros(DIM, DIM)
    for k in range(N_SITES):
        n_op += (sp.eye(DIM) - op({k: Z})) / 2
    check(is_zero_matrix(W * n_op - n_op * W), "[W, N_giros] = 0 (W conserva el número de giros)")

    # --- Espectro por sectores (polinomio característico exacto de cada bloque) ---
    s2 = kappa / (sp.sqrt(2) * (1 + kappa**2))
    expected = {
        0: [(kappa**2 - 1) / (kappa**2 + 1)],
        4: [(kappa**2 - 1) / (kappa**2 + 1)],
        1: [s2, s2, -s2, -s2],
        3: [s2, s2, -s2, -s2],
        2: [f, f, -kappa**2 * f, -kappa**2 * f, 0, 0],
    }
    labels = {
        0: "0 giros: autovalor (kappa^2-1)/(kappa^2+1) = -4c",
        4: "4 giros: autovalor (kappa^2-1)/(kappa^2+1) = -4c",
        1: "1 giro: autovalores ±kappa/(sqrt(2)(1+kappa^2)), cada uno doble",
        3: "3 giros: autovalores ±kappa/(sqrt(2)(1+kappa^2)), cada uno doble",
        2: "2 giros: autovalores 1/(1+kappa^2), -kappa^2/(1+kappa^2) y 0, cada uno doble",
    }
    check(is_zero(-4 * c - (kappa**2 - 1) / (kappa**2 + 1)), "-4c = (kappa^2-1)/(kappa^2+1)")
    for k in sorted(expected):
        idx = [s for s in range(DIM) if n_spins(s) == k]
        block = W.extract(idx, idx)
        # PurePoly puede devolver un generador distinto de lam; se usa el suyo.
        poly = block.charpoly(lam)
        x = poly.gen
        charpoly = poly.as_expr()
        target = sp.Mul(*[x - e for e in expected[k]])
        check(is_zero(sp.expand(charpoly - target)), labels[k])

    # --- Estructura del sector de 2 giros y bloque T ---
    neel = [index("1010"), index("0101")]
    adj = [index("1100"), index("0110"), index("0011"), index("1001")]
    check(all(is_zero(W[r, r] - 4 * c) for r in neel), "diagonal de los Néel = 4c")
    check(all(is_zero(W[r, r]) for r in adj), "diagonal de los adyacentes = 0")
    check(is_zero_matrix(W.extract(neel, neel) - 4 * c * sp.eye(2)),
          "W no conecta Néel con Néel")
    check(is_zero_matrix(W.extract(adj, adj)), "W no conecta adyacente con adyacente")

    T = W.extract(adj, neel)
    u = [1, 1, 1, 1]
    s = [1, -1, 1, -1]
    T_paper = sp.Matrix(4, 2, lambda j, k: -sp.sqrt(2) * J * (u[j] + sp.I * s[j] * (1 if k == 0 else -1)))
    check(is_zero_matrix(T - T_paper), "T = -sqrt(2) J [u⊗(1,1) + i s⊗(1,-1)]")
    TT = (T.H * T).applyfunc(sp.simplify)
    check(TT[0, 1] == 0 and TT[1, 0] == 0, "(T^dagger T) no diagonal exactamente nulo")
    check(is_zero_matrix(TT - 16 * J**2 * sp.eye(2)), "T^dagger T = 16 J^2 I_2")

    # --- lambda_max(W) = f con multiplicidad exactamente 2 ---
    # W es diagonal por bloques en el número de giros (comprobado arriba), así que su
    # espectro es la unión de los espectros por sector. Se comprueba la suma de
    # multiplicidades y que f supera estrictamente a cada otro autovalor en todo el
    # intervalo kappa en [1/sqrt(2), 1] (resolución exacta de la desigualdad).
    all_eigs = [e for k in expected for e in expected[k]]
    check(len(all_eigs) == DIM, "las multiplicidades por sector suman 16")
    mult_f = sum(1 for e in all_eigs if is_zero(e - f))
    check(mult_f == 2, "f = 1/(1+kappa^2) aparece exactamente 2 veces en el espectro")
    domain = sp.Interval(1 / sp.sqrt(2), 1)
    others = {sp.simplify(e) for e in all_eigs if not is_zero(e - f)}
    strict = all(
        sp.solveset(sp.factor(sp.simplify(f - e)) <= 0, kappa, domain) == sp.S.EmptySet
        for e in others
    )
    check(strict, "f - lambda > 0 para todo otro autovalor y todo kappa en [1/sqrt(2), 1]: "
                  "lambda_max(W) = f con multiplicidad 2")

    # Control numérico exacto adicional: autovalores de W completo en kappa racionales.
    ok_spot = True
    for kv in [sp.Rational(3, 4), sp.Rational(4, 5), sp.Rational(9, 10), sp.Rational(1)]:
        Wk = W.subs(kappa, kv)
        eig = {sp.nsimplify(sp.simplify(e)): m for e, m in Wk.eigenvals().items()}
        fk = 1 / (1 + kv**2)
        top = max(eig, key=lambda e: float(e))
        ok_spot &= is_zero(top - fk) and eig[top] == 2
    check(ok_spot, "control: autovalores exactos de W para kappa = 3/4, 4/5, 9/10, 1 "
                   "dan lambda_max = f con multiplicidad 2")

    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} comprobaciones correctas")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

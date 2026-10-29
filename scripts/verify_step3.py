"""Verificación simbólica (SymPy) del paso 3 del teorema del anillo de 4 qubits.

Comprueba de forma exacta, con kappa > 0 como símbolo:

* (a) que los vectores psi(n) con x = T n / f son autovectores de W con
  autovalor f = 1/(1+kappa^2) y forman una base ortonormal del espacio propio;
* (b) que la traslación P (sitio i -> i+1) conmuta con W, preserva el espacio
  propio y actúa en la base psi(n), n = (1,0), (0,1), como [[0,1],[1,0]]; que
  Psi = [|K=0>, |K=pi>] cumple (W - f) Psi = 0 y Psi^dagger Psi = I;
* (c) los valores por arista tr A_i, tr B_i, tr C_i en |K=0> y |K=pi>, incluido
  <A_tot> = 4m en |K=0> con m = 2 sqrt(2) kappa/(1+kappa^2);
* (e) que las 15 correlaciones (9 de dos sitios y 6 de un sitio) de las 4
  aristas de rho_star coinciden con las de rho_e(f), con phi simbólico mediante
  la parametrización racional t = tan(phi/2), y además en varios t racionales.

Convención: el qubit 0 es el más significativo (orden q0 ⊗ q1 ⊗ q2 ⊗ q3),
|0> es el autoestado de Z con valor +1 y un «giro» es un qubit en |1>.

Uso (desde la raíz del repositorio):  python scripts/verify_step3.py
Sale con estado 1 si alguna comprobación falla.
"""
import sys
from functools import reduce

import sympy as sp

N_SITES = 4
DIM = 2 ** N_SITES

kappa = sp.symbols("kappa", positive=True)
t = sp.symbols("t", nonnegative=True)

I2 = sp.eye(2)
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])
PAULI = {"X": X, "Y": Y, "Z": Z}

# Coeficientes del testigo (paso 1).
J = kappa / (4 * (1 + kappa**2))
c = (1 - kappa**2) / (4 * (1 + kappa**2))
a = J / sp.sqrt(2)
f = 1 / (1 + kappa**2)
m = 2 * sp.sqrt(2) * kappa / (1 + kappa**2)
N_norm = 1 / sp.sqrt(2 * (1 + kappa**2))

_results = []


def check(ok, description):
    """Imprime «OK» o «FALLO» y guarda el resultado."""
    ok = bool(ok)
    print(("OK  " if ok else "FALLO ") + description)
    _results.append(ok)


def is_zero(expr):
    return sp.simplify(expr) == 0


def is_zero_matrix(mat):
    return all(is_zero(e) for e in mat)


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


def ket(coeffs):
    """Vector de 16 componentes a partir de {cadena de bits: amplitud}."""
    v = sp.zeros(DIM, 1)
    for bits, amp in coeffs.items():
        v[index(bits)] += amp
    return v


def expval(bra, operator, k):
    return sp.simplify((bra.H * operator * k)[0, 0])


def build_witness():
    W = sp.zeros(DIM, DIM)
    for i in range(N_SITES):
        A, B, C = edge_ops(i)
        W += a * (A + B) + c * C
    return W


def build_translation():
    """P|b_0 b_1 b_2 b_3> = |b'> con b'_{i+1} = b_i (sitio i -> i+1)."""
    P = sp.zeros(DIM, DIM)
    for s in range(DIM):
        bits = format(s, f"0{N_SITES}b")
        new = "".join(bits[(k - 1) % N_SITES] for k in range(N_SITES))
        P[index(new), s] = 1
    return P


NEEL = ["1010", "0101"]
ADJ = ["1100", "0110", "0011", "1001"]
U = [1, 1, 1, 1]
S = [1, -1, 1, -1]


def main():
    W = build_witness()
    P = build_translation()
    Id = sp.eye(DIM)

    # --- (b) Traslación ---
    check(is_zero_matrix(P * W - W * P), "[P, W] = 0")
    check(P * ket({NEEL[0]: 1}) == ket({NEEL[1]: 1}) and P * ket({NEEL[1]: 1}) == ket({NEEL[0]: 1}),
          "P intercambia N0 <-> N1")
    check(all(P * ket({ADJ[j]: 1}) == ket({ADJ[(j + 1) % 4]: 1}) for j in range(4)),
          "P permuta A1 -> A2 -> A3 -> A4 -> A1")

    # --- (a) Espacio propio en la base psi(n) ---
    T = W.extract([index(b) for b in ADJ], [index(b) for b in NEEL])

    x_ok = []

    def psi(n0, n1):
        x = (T * sp.Matrix([n0, n1]) / f).applyfunc(sp.simplify)
        x_paper = [-sp.sqrt(2) * kappa / 4 * ((n0 + n1) * U[j] + sp.I * (n0 - n1) * S[j])
                   for j in range(4)]
        x_ok.append(all(is_zero(x[j] - x_paper[j]) for j in range(4)))
        amps = {NEEL[0]: n0, NEEL[1]: n1}
        amps.update({ADJ[j]: x[j] for j in range(4)})
        return ket(amps) / sp.sqrt(1 + kappa**2)

    Phi = sp.Matrix.hstack(psi(1, 0), psi(0, 1))
    check(all(x_ok), "x = T n / f = -(sqrt(2) kappa/4)[(n0+n1) u + i (n0-n1) s] para n = (1,0), (0,1)")
    check(is_zero_matrix((W - f * Id) * Phi), "(W - f) psi(n) = 0 para n = (1,0), (0,1)")
    check(is_zero_matrix(Phi.H * Phi - sp.eye(2)), "psi(n) ortonormales (|x|^2 = kappa^2)")
    P_E = (Phi.H * P * Phi).applyfunc(sp.simplify)
    check(is_zero_matrix(P * Phi - Phi * P_E), "P preserva el espacio propio E")
    check(P_E == sp.Matrix([[0, 1], [1, 0]]),
          "P restringida a E en la base psi(1,0), psi(0,1) = [[0,1],[1,0]]")

    # --- (b) Estados |K=0>, |K=pi> ---
    K0 = ket({NEEL[0]: N_norm, NEEL[1]: N_norm,
              **{ADJ[j]: -kappa * N_norm / sp.sqrt(2) for j in range(4)}})
    Kpi = ket({NEEL[0]: -N_norm, NEEL[1]: N_norm,
               **{ADJ[j]: sp.I * kappa * N_norm / sp.sqrt(2) * S[j] for j in range(4)}})
    Psi = sp.Matrix.hstack(K0, Kpi)
    check(is_zero_matrix((W - f * Id) * Psi), "(W - f) Psi = 0 con Psi = [|K=0>, |K=pi>]")
    check(is_zero_matrix(Psi.H * Psi - sp.eye(2)), "Psi^dagger Psi = I")
    check(is_zero_matrix(P * K0 - K0) and is_zero_matrix(P * Kpi + Kpi),
          "P|K=0> = |K=0>, P|K=pi> = -|K=pi>")
    check(is_zero_matrix(K0 - (Phi[:, 0] + Phi[:, 1]) / sp.sqrt(2)),
          "|K=0> = (psi(1,0) + psi(0,1))/sqrt(2)")

    # --- (c) Valores por arista ---
    edges = [edge_ops(i) for i in range(N_SITES)]
    A_tot = sum((e[0] for e in edges), sp.zeros(DIM, DIM))
    B_tot = sum((e[1] for e in edges), sp.zeros(DIM, DIM))
    C_tot = sum((e[2] for e in edges), sp.zeros(DIM, DIM))
    check(is_zero(m - 2 * f * sp.sqrt(2) * kappa), "m = 2 f (cos phi + sin phi) con cos phi + sin phi = sqrt(2) kappa")
    check(is_zero(expval(K0, A_tot, K0) - 4 * m), "<A_tot> = 4m = 8 sqrt(2) kappa/(1+kappa^2) en |K=0>")
    check(is_zero(expval(K0, B_tot, K0)), "<B_tot> = 0 en |K=0>")
    check(is_zero(expval(Kpi, A_tot, Kpi)), "<A_tot> = 0 en |K=pi>")
    check(is_zero(expval(Kpi, B_tot, Kpi) - 4 * m), "<B_tot> = 4m en |K=pi>")
    neel_proj = sp.zeros(DIM, DIM)
    for b in NEEL:
        neel_proj[index(b), index(b)] = 1
    check(is_zero(expval(K0, neel_proj, K0) - f) and is_zero(expval(Kpi, neel_proj, Kpi) - f),
          "peso sobre los Néel 2N^2 = f en ambos estados")
    check(is_zero(expval(K0, C_tot, K0) - 4 * f) and is_zero(expval(Kpi, C_tot, Kpi) - 4 * f),
          "<C_tot> = 4f en ambos estados")
    expected_edge = {"K=0": (K0, m, 0, f), "K=pi": (Kpi, 0, m, f)}
    for name, (k, eA, eB, eC) in expected_edge.items():
        ok = all(is_zero(expval(k, A, k) - eA) and is_zero(expval(k, B, k) - eB)
                 and is_zero(expval(k, C, k) - eC) for A, B, C in edges)
        desc = "tr A_i = m, tr B_i = 0" if name == "K=0" else "tr A_i = 0, tr B_i = m"
        check(ok, f"|{name}>: {desc}, tr C_i = f en las 4 aristas")

    # --- (e) Las 15 correlaciones de rho_star frente a rho_e(f) ---
    # Parametrización racional: t = tan(phi/2), phi en [0, pi/4] <=> t en [0, tan(pi/8)].
    cos_phi = (1 - t**2) / (1 + t**2)
    sin_phi = 2 * t / (1 + t**2)
    kappa_t = (cos_phi + sin_phi) / sp.sqrt(2)
    p0 = cos_phi / (cos_phi + sin_phi)
    ppi = sin_phi / (cos_phi + sin_phi)
    f_t = 1 / (1 + kappa_t**2)

    def target(label, v):
        """Correlación de rho_e(v) en la arista: <P_i Q_{i+1}> o marginal de un sitio."""
        table = {"XX": -v * cos_phi, "YY": -v * cos_phi, "XY": v * sin_phi,
                 "YX": -v * sin_phi, "ZZ": -v}
        return sp.S(table.get(label, 0))

    # Valores esperados en los estados K como funciones de kappa (rápido); luego se
    # combinan con p0, p_pi y se sustituye kappa = kappa(t).
    correlations = []
    for i in range(N_SITES):
        j = (i + 1) % N_SITES
        for p in "XYZ":
            for q in "XYZ":
                correlations.append((i, p + q, op({i: PAULI[p], j: PAULI[q]})))
        for p in "XYZ":
            correlations.append((i, p + "1", op({i: PAULI[p]})))
            correlations.append((i, "1" + p, op({j: PAULI[p]})))
    check(len(correlations) == 4 * 15, "60 correlaciones (15 por arista x 4 aristas)")

    residues = []
    for i, label, O in correlations:
        e0 = expval(K0, O, K0)
        epi = expval(Kpi, O, Kpi)
        value = p0 * e0.subs(kappa, kappa_t) + ppi * epi.subs(kappa, kappa_t)
        residues.append((i, label, sp.factor(sp.cancel(sp.together(value - target(label, f_t))))))
    bad = [(i, lab) for i, lab, r in residues if r != 0]
    check(not bad, "las 15 correlaciones de las 4 aristas de rho_star = rho_e(f), phi simbólico "
                   "(t = tan(phi/2))" + (f"; discrepancias: {bad}" if bad else ""))

    # Control adicional en t racionales de [0, tan(pi/8)] (tan(pi/8) ~ 0.4142).
    spot_t = [sp.Integer(0), sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(1, 3), sp.Rational(2, 5)]
    ok_spot = all(is_zero(r.subs(t, tv)) for tv in spot_t for _, _, r in residues)
    # Comprobación independiente: rho_star como matriz 16x16 evaluada en cada t racional.
    for tv in spot_t:
        kv = kappa_t.subs(t, tv)
        rho = (p0.subs(t, tv) * K0 * K0.H + ppi.subs(t, tv) * Kpi * Kpi.H).subs(kappa, kv)
        ok_spot &= is_zero(rho.trace() - 1)
        for i, label, O in correlations:
            ok_spot &= is_zero((rho * O).trace() - target(label, f_t).subs(t, tv))
    check(ok_spot, "control: rho_star = rho_e(f) en t = 0, 1/10, 1/5, 1/3, 2/5 (y tr rho_star = 1)")

    n_fail = _results.count(False)
    print(f"\n{len(_results) - n_fail}/{len(_results)} comprobaciones correctas")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

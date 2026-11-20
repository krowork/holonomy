"""AKLT en el anillo de 4 sitios con Theta = 0: p(4, 0) = 40/41 exacto
(robustez_mps.tex, Proposición prop:aklt4). Todo en aritmética racional (SymPy).

Primal: psi = 5 X - 4 D, con s el singlete de dos espines 1 (|+-> - |00> + |-+>),
D = s_12 s_34 + s_23 s_41 y X = s_13 s_24. Es un singlete invariante por traslación, y
sus pesos de arista en J = 0, 1, 2 son (121/369, 27/41, 5/369), los datos con p = 40/41.

Dual: h = 81 P0 + 110 P1. lambda_max(sum_j h_{j,j+1}) = 396, simple (en psi); el
siguiente autovalor es 355. Con tr h = 411 y tr(h tau) = 164/3, la cota dual es
(396/4 - 411/9) / (164/3) = 40/41.
"""
import itertools
import sys

import sympy as sp

OK = True
N = 4


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def kron(*Ms):
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.kronecker_product(out, M)
    return out


# espín 1, base (+1, 0, -1)
Sz = sp.diag(1, 0, -1)
Sp = sp.Matrix([[0, sp.sqrt(2), 0], [0, 0, sp.sqrt(2)], [0, 0, 0]])
SS = kron(Sz, Sz) + (kron(Sp, Sp.T) + kron(Sp.T, Sp)) / 2
I9 = sp.eye(9)
P = [(SS + I9) * (SS - I9) / 3, -(SS + 2 * I9) * (SS - I9) / 2, (SS + 2 * I9) * (SS + I9) / 6]

IDX = list(itertools.product(range(3), repeat=N))
POS = {s: i for i, s in enumerate(IDX)}


def ring(op):
    H = sp.zeros(3 ** N, 3 ** N)
    for j in range(N):
        k = (j + 1) % N
        for a, s in enumerate(IDX):
            for x in range(3):
                for y in range(3):
                    v = op[3 * x + y, 3 * s[j] + s[k]]
                    if v != 0:
                        t = list(s)
                        t[j], t[k] = x, y
                        H[POS[tuple(t)], a] += v
    return H


def dimers(p, q):
    """s_pq s_rt con {r, t} el par restante."""
    sing = {(0, 2): 1, (1, 1): -1, (2, 0): 1}
    r, t = [i for i in range(N) if i not in (p, q)]
    v = sp.zeros(3 ** N, 1)
    for a, s in enumerate(IDX):
        v[a] = sing.get((s[p], s[q]), 0) * sing.get((s[r], s[t]), 0)
    return v


if __name__ == "__main__":
    E = [ring(PJ) for PJ in P]
    check(all(sp.simplify(PJ * PJ - PJ).is_zero_matrix for PJ in P)
          and [PJ.trace() for PJ in P] == [1, 3, 5], "proyectores P_J de rango 1, 3, 5")

    # primal
    D = dimers(0, 1) + dimers(1, 2)
    X = dimers(0, 2)
    psi = 5 * X - 4 * D
    T = sp.zeros(3 ** N, 3 ** N)
    for a, s in enumerate(IDX):
        T[POS[s[1:] + s[:1]], a] = 1
    site = lambda op, j: kron(*[op if i == j else sp.eye(3) for i in range(N)])
    Stz = sum((site(Sz, j) for j in range(N)), sp.zeros(81, 81))
    Stp = sum((site(Sp, j) for j in range(N)), sp.zeros(81, 81))
    S2 = Stz * Stz + (Stp * Stp.T + Stp.T * Stp) / 2
    check((T * psi - psi).is_zero_matrix and (S2 * psi).is_zero_matrix,
          "psi = 5X - 4D es invariante por traslación y singlete")
    B = sp.Matrix.hstack(D, X)
    check(B.T * B == sp.Matrix([[24, 6], [6, 9]]), "Gram de (D, X) = [[24, 6], [6, 9]]")
    forms = [B.T * EJ * B for EJ in E]
    check(forms == [sp.Matrix([[64, 16], [16, 4]]), sp.Matrix([[12, -12], [-12, 12]]),
                    sp.Matrix([[20, 20], [20, 20]])],
          "formas de arista: W0 = 4(4a+b)^2, W1 = 12(a-b)^2, W2 = 20(a+b)^2")
    norm = (psi.T * psi)[0]
    w = [(psi.T * EJ * psi)[0] / (N * norm) for EJ in E]
    p = sp.Rational(40, 41)
    data = [(1 + 2 * p) / 9, (1 + p) / 3, 5 * (1 - p) / 9]
    check(norm == 369 and w == data, f"<psi|psi> = 369 y pesos {w} = datos con p = 40/41")

    # dual
    H = 81 * E[0] + 110 * E[1]
    check((H * psi - 396 * psi).is_zero_matrix, "H psi = 396 psi, con H = sum_j (81 P0 + 110 P1)_{j,j+1}")
    x = sp.symbols("x")
    mag = [sum(1 - t for t in s) for s in IDX]
    roots = []
    for M in range(-N, N + 1):
        I = [i for i in range(3 ** N) if mag[i] == M]
        for g, e in sp.factor_list(H.extract(I, I).charpoly(x).as_expr())[1]:
            check(sp.degree(g, x) <= 2, f"S^z = {M:+d}: factor {sp.expand(g)} (mult. {e})") if M == 0 else None
            roots += [(r, M) for r in sp.solve(g, x) for _ in range(e)]
    roots.sort(key=lambda t: -float(t[0]))
    check(roots[0] == (396, 0) and float(roots[1][0]) == 355,
          f"lambda_max = 396, simple; siguiente {roots[1][0]} (S^z = {roots[1][1]:+d})")
    h = 81 * P[0] + 110 * P[1]
    tau = sp.Rational(2, 9) * P[0] + sp.Rational(1, 9) * P[1] - sp.Rational(1, 9) * P[2]
    trh, trht = h.trace(), (h * tau).trace()
    bound = (sp.Rational(396, N) - trh / 9) / trht
    check(trh == 411 and trht == sp.Rational(164, 3) and bound == p,
          f"tr h = {trh}, tr(h tau) = {trht}, cota dual = {bound}")
    sys.exit(0 if OK else 1)

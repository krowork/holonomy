"""Coeficientes racionales exactos de K1(eps) y positividad graduada (celda_k_notas.md, 9.20).

K1_ab = (1/n) <psi| O_a (H - E) O_b |psi> / <psi|psi>,  O_a = sum_j o_a(j, j+1),  H = -P_asc + eps T (T: saltos
S+S- con amplitud 2). psi es el patrón 0,1,2,0,1,2,... vestido por Rayleigh-Schrödinger (normalización intermedia),
exacto en fracciones hasta el orden NORD. Hasta orden < n los demás patrones no se alcanzan, así que la serie es la
del anillo (y la de la cadena infinita): el coeficiente de orden m no depende de n si m < n - c (9.19).
Después: reducción graduada de Schur de K1(eps) en un complemento de las 5 direcciones nulas.
Uso: python scripts/k1_racional.py [n] [NORD]
"""
import sys
from fractions import Fraction

import sympy as sp

N_SITIOS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NORD = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # orden de psi; K1 se obtiene hasta este orden
n = N_SITIOS
KMAX = NORD
TRES = False  # 9.23: RS en el sector de momento 0 desde la suma de los tres patrones (n pequeño)


# ---------- series truncadas: listas de Fraction de longitud KMAX+1 ----------
def cero():
    return [Fraction(0)] * (KMAX + 1)


def sumar(a, b, c=1):
    return [x + c * y for x, y in zip(a, b)]


def mult(a, b):
    r = cero()
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j in range(KMAX + 1 - i):
            if b[j] != 0:
                r[i + j] += x * b[j]
    return r


def inv(a):
    assert a[0] != 0
    r = cero()
    r[0] = 1 / a[0]
    for k in range(1, KMAX + 1):
        s = sum(a[i] * r[k - i] for i in range(1, k + 1))
        r[k] = -s / a[0]
    return r


# ---------- modelo ----------
def energia0(s):
    return -sum(1 for j in range(n) if (s[(j + 1) % n] - s[j]) % 3 == 1)


def saltos(s):
    """T|s>: S+_j S-_{j+1} y su inverso, amplitud 2."""
    for j in range(n):
        k = (j + 1) % n
        a, b = s[j], s[k]
        if a < 2 and b > 0:
            t = list(s); t[j] += 1; t[k] -= 1
            yield tuple(t), 2
        if a > 0 and b < 2:
            t = list(s); t[j] -= 1; t[k] += 1
            yield tuple(t), 2


def H_serie(vec):
    """(H0 + eps T) sobre un vector de series {conf: serie}."""
    out = {}
    for s, c in vec.items():
        e = energia0(s)
        out[s] = sumar(out.get(s, cero()), c, e)
        sh = [Fraction(0)] + c[:-1]  # eps * c
        for t, amp in saltos(s):
            out[t] = sumar(out.get(t, cero()), sh, amp)
    return out


def rs():
    """psi = sum_m eps^m psi_m con <p|psi> = 1; E = sum_m eps^m E_m."""
    p = tuple(j % 3 for j in range(n))
    E0 = energia0(p)
    pats = [tuple((j + c) % 3 for j in range(n)) for c in range(3)] if TRES else [p]
    psi = [{q: Fraction(1) for q in pats}]
    E = [Fraction(E0)]
    for m in range(1, KMAX + 2):
        # (E0 - H0) psi_m = T psi_{m-1} - sum_{k=1}^{m} E_k psi_{m-k}
        rhs = {}
        for s, c in psi[m - 1].items():
            for t, amp in saltos(s):
                rhs[t] = rhs.get(t, 0) + amp * c
        Em = rhs.get(p, Fraction(0))  # <p| T psi_{m-1}>
        assert all(rhs.get(q, Fraction(0)) == Em for q in pats), "patrones no simétricos"
        E.append(Em)
        if m == KMAX + 1:
            break
        for k in range(1, m + 1):
            for s, c in psi[m - k].items():
                rhs[s] = rhs.get(s, 0) - E[k] * c
        new = {}
        for s, c in rhs.items():
            if s in pats or c == 0:
                continue
            new[s] = c / (E0 - energia0(s))
        psi.append(new)
    vec = {}
    for m, comp in enumerate(psi):
        for s, c in comp.items():
            vec.setdefault(s, cero())[m] += c
    return vec, E[:KMAX + 1]


# ---------- base de operadores de arista (racional, real) ----------
CARGA = [a + b for a in range(3) for b in range(3)]


def base():
    """('d', x) = |x><x|;  ('re', x, y) = |x><y| + |y><x|;  ('im', x, y) = i(|x><y| - |y><x|), x < y misma carga."""
    B = [("d", x) for x in range(9)]
    for x in range(9):
        for y in range(x + 1, 9):
            if CARGA[x] == CARGA[y]:
                B.append(("re", x, y))
    for x in range(9):
        for y in range(x + 1, 9):
            if CARGA[x] == CARGA[y]:
                B.append(("im", x, y))
    return B


def aplicar_o(o, vec):
    """O = sum_j o(j, j+1) sobre un vector de series. Los coeficientes de 'im' son imaginarios: se devuelven
    multiplicados por -i (es decir, i(|x><y| - |y><x|) -> se guarda (|x><y| - |y><x|) y el factor i aparte)."""
    out = {}
    for s, c in vec.items():
        for j in range(n):
            k = (j + 1) % n
            par = s[j] * 3 + s[k]
            dest = []
            if o[0] == "d":
                if par == o[1]:
                    dest.append((par, 1))
            else:
                x, y = o[1], o[2]
                sg = 1 if o[0] == "re" else -1
                if par == y:
                    dest.append((x, 1))   # |x><y|
                if par == x:
                    dest.append((y, sg))  # +-|y><x|
            for q, f in dest:
                t = list(s); t[j], t[k] = q // 3, q % 3
                t = tuple(t)
                out[t] = sumar(out.get(t, cero()), c, f)
    return out


def interior(u, v):
    r = cero()
    for s, c in u.items():
        if s in v:
            r = sumar(r, mult(c, v[s]))
    return r


def K1_series():
    psi, E = rs()
    norma = interior(psi, psi)
    B = base()
    Ops = [aplicar_o(o, psi) for o in B]
    # (H - E) O_b psi
    AOps = []
    for v in Ops:
        hv = H_serie(v)
        for s, c in v.items():
            hv[s] = sumar(hv.get(s, cero()), mult(E, c), -1)
        AOps.append(hv)
    inorm = inv(norma)
    K = {}
    for a, oa in enumerate(B):
        for b, ob in enumerate(B):
            if (oa[0] == "im") != (ob[0] == "im"):
                continue  # bloques Re / Im desacoplados (psi real)
            r = mult(interior(Ops[a], AOps[b]), inorm)
            # factor i de 'im': <i X psi| A |i Y psi> = <X psi|A|Y psi>
            K[(a, b)] = [x / n for x in r]
    return B, K, E


# ---------- reducción graduada ----------
eps = sp.Symbol("eps")


def a_matriz(B, K, idx):
    M = sp.zeros(len(idx), len(idx))
    for i, a in enumerate(idx):
        for j, b in enumerate(idx):
            M[i, j] = sum(sp.Rational(c.numerator, c.denominator) * eps**m for m, c in enumerate(K[(a, b)]))
    return M


def trunc(expr, orden):
    return sp.series(sp.expand(expr), eps, 0, orden + 1).removeO()


def graduada(M, orden, nivel=0, log=None):
    """Reduce M(eps) (PSD, polinómica) por órdenes: devuelve lista de (orden, matriz principal) cuyos espectros
    dan los autovalores ~ eps^orden * lambda."""
    log = [] if log is None else log
    if M.shape[0] == 0:
        return log
    # orden mínimo presente
    k = None
    for d in range(orden + 1):
        Md = M.applyfunc(lambda x: sp.expand(x).coeff(eps, d))
        if any(v != 0 for v in Md):
            k = d
            break
    if k is None:
        log.append(("> %d" % orden, M.shape[0]))
        return log
    Mk = M.applyfunc(lambda x: sp.expand(x).coeff(eps, k))
    if Mk.is_zero_matrix:
        return log
    R = Mk.columnspace()
    Nn = Mk.nullspace()
    Rm = sp.Matrix.hstack(*R) if R else sp.zeros(M.shape[0], 0)
    log.append((k, Rm.T * Mk * Rm))
    if not Nn:
        return log
    Nm = sp.Matrix.hstack(*Nn)
    A = (Rm.T * M * Rm) / eps**k
    Bm = (Rm.T * M * Nm) / eps**k
    C = (Nm.T * M * Nm)
    A = A.applyfunc(lambda x: sp.cancel(x))
    # A^{-1} en serie: A = A0 (1 + A0^{-1}(A - A0))
    A0 = A.applyfunc(lambda x: sp.expand(x).subs(eps, 0))
    A0i = A0.inv()
    X = A0i * (A - A0)
    Ai = sp.eye(A.shape[0])
    term = sp.eye(A.shape[0])
    for _ in range(orden):
        term = (-X * term).applyfunc(lambda x: trunc(x, orden))
        Ai = Ai + term
    Ai = (Ai * A0i).applyfunc(lambda x: trunc(x, orden))
    S = (C - eps**k * Bm.T * Ai * Bm).applyfunc(lambda x: trunc(sp.cancel(x), orden))
    return graduada(S, orden, nivel + 1, log)


if __name__ == "__main__":
    B, K, E = K1_series()
    print(f"n = {n}, orden {KMAX}.  E/n = " + " + ".join(f"({e / n})eps^{m}" for m, e in enumerate(E)))
    re = [i for i, o in enumerate(B) if o[0] != "im"]
    im = [i for i, o in enumerate(B) if o[0] == "im"]
    # complemento de las nulas en el bloque Re: quitar 5 diagonales tales que {I, Q(x)1, tel1, tel2, P_asc} + resto
    # sea base de las diagonales; se quitan |00>,|01>,|02>,|10>,|20> (comprobado abajo)
    Qd = [x // 3 - 1 for x in range(9)]
    Qb = [x % 3 - 1 for x in range(9)]
    nulas = [[1] * 9, Qd, [Qd[x] - Qb[x] for x in range(9)], [Qd[x]**2 - Qb[x]**2 for x in range(9)],
             [1 if (x % 3 - x // 3) % 3 == 1 else 0 for x in range(9)]]
    quitar = [1, 2, 3, 5, 6]  # |01>,|02>,|10>,|12>,|20>
    keep_d = [x for x in range(9) if x not in quitar]
    Mchk = sp.Matrix([row for row in nulas] + [[1 if y == x else 0 for y in range(9)] for x in keep_d])
    print("rango(nulas + diagonales que se conservan) =", Mchk.rank(), "(debe ser 9)")
    W = keep_d + [i for i in re if B[i][0] == "re"]
    for nombre, idx in (("Re", W), ("Im", im)):
        M = a_matriz(B, K, idx)
        print(f"\nBloque {nombre}: {[B[i] for i in idx]}")
        for k, Mk in graduada(M, KMAX):
            if isinstance(Mk, int):
                print(f"  orden {k}: {Mk} direcciones sin determinar")
                continue
            ev = [sp.nsimplify(v) for v in Mk.eigenvals(multiple=True)] if Mk.shape[0] <= 4 else None
            num = sorted(float(v) for v in sp.Matrix(Mk).evalf().eigenvals(multiple=True)) \
                if Mk.shape[0] > 4 else sorted(float(sp.re(v.evalf())) for v in ev)
            print(f"  orden eps^{k}: {Mk.shape[0]} direcciones; matriz principal = {Mk.tolist()}")
            print(f"    autovalores ≈ {num}" + (f"   exactos: {ev}" if ev else ""))


def sylvester(M):
    """Menores principales dominantes exactos (criterio de Sylvester)."""
    return [M[:k, :k].det() for k in range(1, M.shape[0] + 1)]


def chi_series():
    """Serie exacta de chi_ab = (1/n) <phi_a|(H - E)^{-1}|phi_b>, phi = (1 - P) O psi (9.21, Lema Z26).
    xi_b = (H - E)^{-1} phi_b orden a orden: (H0 - E0) xi_m = phi_m - T xi_{m-1} + sum_k E_k xi_{m-k},
    con <psi|xi> = 0. Requiere K + 2 < 2n/3 (si no, alcanza otra configuración degenerada)."""
    psi, E = rs()
    p = tuple(j % 3 for j in range(n))
    E0 = energia0(p)
    pats = [tuple((j + c) % 3 for j in range(n)) for c in range(3)] if TRES else [p]
    norma = interior(psi, psi)
    inorm = inv(norma)
    B = base()

    def orden(vec, m):
        return {s: c[m] for s, c in vec.items() if c[m] != 0}

    Psi = [orden(psi, m) for m in range(KMAX + 1)]
    phis = []
    for o in B:
        U = aplicar_o(o, psi)
        expo = mult(interior(psi, U), inorm)
        ph = {s: list(c) for s, c in U.items()}
        for s, c in psi.items():
            ph[s] = sumar(ph.get(s, cero()), mult(expo, c), -1)
        phis.append(ph)
    xis = []
    for b in range(len(B)):
        ph = phis[b]
        xi = []
        for m in range(KMAX + 1):
            r = {s: c[m] for s, c in ph.items() if c[m] != 0}
            if m >= 1:
                for s, c in xi[m - 1].items():
                    for t, amp in saltos(s):
                        r[t] = r.get(t, 0) - amp * c
            for k in range(1, m + 1):
                if E[k] != 0:
                    for s, c in xi[m - k].items():
                        r[s] = r.get(s, 0) + E[k] * c
            assert all(r.get(q, 0) == 0 for q in pats), ("compatibilidad", b, m)
            x = {}
            for s, c in r.items():
                if s in pats or c == 0:
                    continue
                d = energia0(s) - E0
                assert d != 0, "configuración degenerada alcanzada: n demasiado pequeño"
                x[s] = c / d
            # <psi|xi>_m = 0  ->  x[p] = - sum_{i>=1} <psi_i|xi_{m-i}>
            val = -sum(sum(c * xi[m - i].get(s, 0) for s, c in Psi[i].items()) for i in range(1, m + 1))
            for q in pats:
                x[q] = Fraction(val) / len(pats)
            xi.append(x)
        xis.append(xi)
    K = {}
    for a in range(len(B)):
        for b in range(len(B)):
            if (B[a][0] == "im") != (B[b][0] == "im"):
                continue
            ser = cero()
            for i in range(KMAX + 1):
                pa = {s: c[i] for s, c in phis[a].items() if c[i] != 0}
                for j in range(KMAX + 1 - i):
                    ser[i + j] += sum(c * xis[b][j].get(s, 0) for s, c in pa.items())
            K[(a, b)] = [x / n for x in mult(ser, inorm)]
    return B, K, E

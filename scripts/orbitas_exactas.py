"""w2 y w3 con aritmética racional exacta (celda_k_notas.md, 9.14). Modelo S+S- (k = 3, espín 1: amplitud 2).

En el anillo de n sitios, para cada configuración de un defecto p se suman los caminos de 2 y 3 saltos que vuelven
al subespacio de un defecto P, con denominadores enteros:
  H2 = -P V R V P ,  H3 = P V R V R V P ,  R = Q0/(E0 - H0) ,
y se restan los del vacío (anillo con 3 | n). Por la traslación, H2 y H3 son circulantes en cada órbita: se dan las
amplitudes de salto h(r) = <T^r p|H|p>, de donde w(kappa) = sum_r h(r) e^{i kappa r}.
Uso: python scripts/orbitas_exactas.py
"""
from collections import defaultdict
from fractions import Fraction
import itertools

K = 3
AMP = 2


def energia(s):
    n = len(s)
    return -sum(1 for j in range(n) if (s[(j + 1) % n] - s[j]) % K == 1)


def saltos(s):
    n = len(s)
    for j in range(n):
        a, b = s[j], s[(j + 1) % n]
        for da in (1, -1):
            if 0 <= a + da <= 2 and 0 <= b - da <= 2:
                t = list(s)
                t[j] += da
                t[(j + 1) % n] -= da
                yield tuple(t)


def efectivos(n, base):
    """H2 y H3 (fila de cada p de base) como dicts {p': valor}, con E0 = energía común de base."""
    E0 = energia(base[0])
    P = set(base)
    H2, H3 = {}, {}
    for p in base:
        h2, h3 = defaultdict(Fraction), defaultdict(Fraction)
        for m1 in saltos(p):
            if m1 in P:
                raise ValueError("primer orden no nulo")
            r1 = Fraction(1, E0 - energia(m1))
            for m2 in saltos(m1):
                if m2 in P:
                    h2[m2] += AMP * AMP * r1
                    continue
                r2 = Fraction(1, E0 - energia(m2))
                for m3 in saltos(m2):
                    if m3 in P:
                        h3[m3] += AMP ** 3 * r1 * r2
        H2[p], H3[p] = h2, h3
    return H2, H3


def traslada(s, r):
    n = len(s)
    return tuple(s[(j - r) % n] for j in range(n))


if __name__ == "__main__":
    # vacío: anillo de 9 sitios, patrón ascendente; e2 por enlace (los caminos de 3 saltos que vuelven son 0)
    pat = tuple(j % 3 for j in range(9))
    H2v, H3v = efectivos(9, [pat])
    e2 = H2v[pat][pat] / 9
    e3 = H3v[pat].get(pat, Fraction(0)) / 9
    print(f"vacío: e2 = {e2} por enlace, e3 = {e3}")
    for n in (10, 11, 13, 14):
        uno = [s for s in itertools.product(range(3), repeat=n)
               if sum(1 for j in range(n) if (s[(j + 1) % n] - s[j]) % K != 1) == 1]
        q = lambda s: sum(x - 1 for x in s)  # noqa: E731
        for Q in (-1, 0, 1):
            base = [s for s in uno if q(s) == Q]
            H2, H3 = efectivos(n, base)
            p = base[0]
            h2 = {r: H2[p].get(traslada(p, r), Fraction(0)) for r in range(n)}
            h3 = {r: H3[p].get(traslada(p, r), Fraction(0)) for r in range(n)}
            h2[0] -= n * e2
            nz2 = {(r if r <= n // 2 else r - n): v for r, v in h2.items() if v}
            nz3 = {(r if r <= n // 2 else r - n): v for r, v in h3.items() if v}
            # comprobación de circulancia: todas las filas iguales tras trasladar
            circ = all(H2[s].get(traslada(s, r), 0) == H2[p].get(traslada(p, r), 0)
                       for s in base for r in range(n) if traslada(p, r) in set(base))
            print(f"n={n} (n%3={n % 3}) Q={Q:+d}: h2 = {nz2}   h3 = {nz3}   |base| = {len(base)}")

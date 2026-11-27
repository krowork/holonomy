"""Conjuntos 4-conexos de N celdas de Z^2 que contienen una celda dada: c_N = N a_N, con a_N los poliominós fijos
(algoritmo de Redelmeier), y la cota de árboles r_N >= c_N (subárboles del recubrimiento universal: la raíz tiene 4
hijos y los demás nodos 3), que crece como (27/4)^N. Escribe scripts/animales.txt con N, c_N, r_N, y
scripts/perimetros.txt con N, p, N a_{N,p} (p = celdas vecinas fuera del poliominó), y scripts/filas.txt con
N, p, r_f, r_l, una_fila, cuenta (y = tiempo: r_f y r_l son las celdas de la primera y la última fila), y
scripts/perfiles.txt con el perfil completo de filas (primera, última e intermedias ordenadas).
Uso: python scripts/animales.py Nmax
"""
import sys
from math import comb

Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 13
a = [0] * (Nmax + 1)
from collections import defaultdict
per = defaultdict(int)
fil = defaultdict(int)
perf = defaultdict(int)
filas = defaultdict(int)
cnt = defaultdict(int)
estado = {"p": 0}


def redelmeier():
    # celdas (x, y) con y > 0, o y == 0 y x >= 0
    def vecinos(c):
        x, y = c
        return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

    def valida(c):
        return c[1] > 0 or (c[1] == 0 and c[0] >= 0)

    def paso(untried, n, poly, vistos):
        while untried:
            c = untried.pop()
            a[n + 1] += 1
            # perímetro de poly + c
            dp = sum(1 for v in vecinos(c) if cnt[v] == 0 and v not in poly) - (1 if cnt[c] > 0 else 0)
            per[(n + 1, estado["p"] + dp)] += 1
            filas[c[1]] += 1
            ymax = max(y for y, m in filas.items() if m > 0)
            fil[(n + 1, estado["p"] + dp, filas[0], filas[ymax], int(ymax == 0))] += 1
            tam = [filas[y] for y in range(ymax + 1)]
            if ymax == 0:
                perf[(n + 1, estado["p"] + dp, tam[0], 0, 1, ())] += 1
            else:
                perf[(n + 1, estado["p"] + dp, tam[0], tam[-1], ymax + 1, tuple(sorted(tam[1:-1])))] += 1
            if n + 1 < Nmax:
                nuevos = [v for v in vecinos(c) if valida(v) and v not in vistos and not any(
                    w in poly for w in vecinos(v) if w != c)]
                for v in nuevos:
                    vistos.add(v)
                poly.add(c)
                for v in vecinos(c):
                    cnt[v] += 1
                estado["p"] += dp
                paso(list(untried) + nuevos, n + 1, poly, vistos)
                estado["p"] -= dp
                for v in vecinos(c):
                    cnt[v] -= 1
                poly.remove(c)
                for v in nuevos:
                    vistos.discard(v)
            filas[c[1]] -= 1

    paso([(0, 0)], 0, set(), {(0, 0)})


sys.setrecursionlimit(10000)
redelmeier()


def arbol(N):
    # [x^N] x (1+B)^4, con B = x (1+B)^3 ; Lagrange: [x^m] (1+B)^k = k/m C(3m+k-1, m-1) para m >= 1
    if N == 1:
        return 1
    m = N - 1
    return 4 * comb(3 * m + 3, m - 1) // m


with open("scripts/animales.txt", "w") as f:
    for N in range(1, Nmax + 1):
        c = N * a[N]
        r = arbol(N)
        print(N, a[N], c, r, flush=True)
        f.write(f"{N} {c} {r}\n")

with open("scripts/perimetros.txt", "w") as f:
    for (N, pp), m in sorted(per.items()):
        f.write(f"{N} {pp} {N * m}\n")

with open("scripts/perfiles.txt", "w") as f:
    # N p r_primera r_ultima R c | filas intermedias (ordenadas); R = 1: una fila (r_ultima = 0)
    for (N, pp, rf, rl, R, mid), m in sorted(perf.items()):
        f.write(f"{N} {pp} {rf} {rl} {R} {N * m} | {' '.join(map(str, mid))}\n")

with open("scripts/filas.txt", "w") as f:
    for (N, pp, rf, rl, una), m in sorted(fil.items()):
        f.write(f"{N} {pp} {rf} {rl} {una} {N * m}\n")

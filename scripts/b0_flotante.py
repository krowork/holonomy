"""B(c_0 -> c_d) y 4|b_0| en coma flotante (sensibilidad_celda3.tex, Corolario de b_0), para n grande.

Mismo recuento que orden_theta.B (caminos de n saltos directos sin pasar por patrones), pero en coma flotante:
todos los sumandos tienen el signo (-1)^{n-1}, así que no hay cancelaciones. n = 15 reproduce 4476,36; n = 18
da 4|b_0| = 20789,16 (unos 3,5e5 estados en el frente, minutos).
Uso: python scripts/b0_flotante.py n
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from orden_theta import directos, energia
n = int(sys.argv[1])
pats = [tuple((c + j) % 3 for j in range(n)) for c in range(3)]
P = set(pats); E0 = energia(pats[0])
fr = {pats[0]: 1.0}
for paso in range(n):
    nu = defaultdict(float)
    for s, w in fr.items():
        for t in directos(s):
            nu[t] += 2*w
    if paso < n-1:
        fr = {t: w/(E0-energia(t)) for t, w in nu.items() if t not in P}
    else:
        res = {pats.index(t): w for t, w in nu.items() if t in P}
    print(paso, len(fr), flush=True, file=sys.stderr)
b0 = sum(res.values())
print(n, res, "4|b0| =", 4*abs(b0))

"""Orden en que el fundamental nota Theta con carga entera (celda_k_notas.md, 9.15). Modelo S+S- (k = 3).

Gauge Theta/n por enlace: el salto S+_j S-_{j+1} lleva e^{i Theta/n} (como celda_k_flujo / orbitas_entero), su
inverso e^{-i Theta/n}. Un camino lleva e^{i Theta J/n}, con J = (saltos directos) - (inversos).
Si el camino une dos estados del subespacio degenerado P (patrones si 3 | n; defectos de la órbita neutra si no),
J = Delta D mod n con D el dipolo, y Delta D es múltiplo de n (Q_c = 0, o Q = 0): con menos de n saltos, J = 0.
El primer término que depende de Theta es de orden n: caminos de n saltos todos en el mismo sentido, sin volver a P
(un subcamino que volviera tendría |J| < n y J != 0). Su coeficiente se calcula aquí con fracciones:

  B(p -> p') = sum_{caminos directos de n saltos} prod (amplitud 2) prod_{intermedios} 1/(E0 - E(m)) ,

y el término de Theta en lambda_p es eps^n (e^{i Theta} b_p + c.c.), b_p = suma de B con la fase de momento.
Uso: python scripts/orden_theta.py
"""
import cmath
import itertools
from collections import defaultdict
from fractions import Fraction

K = 3


def energia(s):
    n = len(s)
    return -sum(1 for j in range(n) if (s[(j + 1) % n] - s[j]) % K == 1)


def directos(s):
    """S+_j S-_{j+1}: s_j + 1, s_{j+1} - 1 (fase e^{+i Theta/n})."""
    n = len(s)
    for j in range(n):
        a, b = s[j], s[(j + 1) % n]
        if a < 2 and b > 0:
            t = list(s)
            t[j] += 1
            t[(j + 1) % n] -= 1
            yield tuple(t)


def B(n, p0, enP):
    """Suma de los caminos de n saltos directos de p0 a estados de P, sin pasar por P en medio."""
    E0 = energia(p0)
    frente = {p0: Fraction(1)}
    for paso in range(n):
        nuevo = defaultdict(Fraction)
        for s, w in frente.items():
            for t in directos(s):
                nuevo[t] += 2 * w
        if paso < n - 1:
            frente = {t: w / (E0 - energia(t)) for t, w in nuevo.items() if t not in enP}
        else:
            return {t: w for t, w in nuevo.items() if t in enP}


def traslada(s, r):
    n = len(s)
    return tuple(s[(j - r) % n] for j in range(n))


if __name__ == "__main__":
    print("3 | n: patrones c (s_j = c + j); lambda_p con |p> = sum_c w^{pc}|c>")
    for n in (6, 9):
        pats = [tuple((c + j) % 3 for j in range(n)) for c in range(3)]
        res = B(n, pats[0], set(pats))
        coef = {pats.index(t): v for t, v in res.items()}
        print(f"  n={n}: B(0 -> c) = {coef}")
        for p in range(3):
            bp = sum(complex(v) * cmath.exp(2j * cmath.pi * p * c / 3) for c, v in coef.items())
            print(f"    p={p}: b_p = {bp:.6g}, |b_p| = {abs(bp):.6g}  ->  R ~ 4|b_p| eps^n")
    print("3 ∤ n, órbita neutra: lambda_{0,kappa} = sum_r H[T^r p0, p0] e^{i kappa r}")
    for n in (7, 8, 10):
        uno = [s for s in itertools.product(range(3), repeat=n)
               if sum(1 for j in range(n) if (s[(j + 1) % n] - s[j]) % K != 1) == 1
               and sum(x - 1 for x in s) == 0]
        p0 = uno[0]
        res = B(n, p0, set(uno))
        porr = {}
        for t, v in res.items():
            r = next(r for r in range(n) if traslada(p0, r) == t)
            porr[r] = v
        print(f"  n={n}: B por desplazamiento r = {porr}")

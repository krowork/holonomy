"""Desarrollo en lazos de la entropía de máxima entropía: el término líder de cada ciclo.

Con marginales rho_e(v) = (I - v sum_ab (M_e)_ab s_a s_b)/4 y M_e en SO(3), sea S_G la entropía de máxima
entropía. Conjetura (término líder en v de cada ciclo simple C de longitud l):

    S_G(M) - S_G(1) = sum_C 2 (-v)^l [tr R_C - 3]/2 + ... = sum_C 2 (-1)^(l+1) v^l (1 - cos theta_C) + ...,

con R_C = M_{01} M_{12} ... M_{l-1,0} la holonomía del ciclo y theta_C su ángulo. Los puntos suspensivos son de
orden v^(l+1) para los ciclos de longitud l. Con l = 3 es la acción de Wilson de parches_cubico.py. Los ciclos
impares suben la entropía y los pares la bajan.

Argumento: S_G = min_x [ln Z(x) - x.t], con ln Z = N ln 2 + ln(1 + sum_k m_k/k!), m_k = 2^-N tr K^k,
K = sum_e sum_ab x_e,ab s_a s_b. Un término de m_k es no nulo solo si cada sitio lleva al menos dos Pauli,
así que el primero que recorre C entero está en k = l: las l aristas de C una vez, en l! órdenes que dan
lo mismo (en cada sitio hay dos Pauli y tr(s_a s_b) = 2 delta_ab), y aportan tr(X_01 ... X_{l-1,0}) a ln Z.
Con x* = -v M + O(v^2) y la estacionariedad de x*, ese término entra en S_G como (-v)^l tr R_C + O(v^(l+1)).

Comprueba, con el resolutor general de parches_boosts.MaxEnt:
1. anillos de 4 a 7 sitios con flujo pi/2 y pi: Delta S / [2 (-1)^(l+1) v^l (1 - cos f)] -> 1 cuando v -> 0;
2. el anillo de 4 con datos SO(3) al azar (no abelianos);
3. el cubo (escalera 2x4) con fases al azar: -2 v^4 sum_caras (1 - cos f_p);
4. la casa (un cuadrado y un triángulo con una arista común) con flujo solo en el cuadrado;
5. que el cociente de 1 es (r(v)/v)^l con r(v) común a los anillos, y que r(v) = e^(-1/xi(v)), con xi la longitud
   de decaimiento de la entropía de holonomía de los anillos (Holonomía II, tabla tab:xi).

Escribe data/lazos.csv.

Uso: OMP_NUM_THREADS=8 python scripts/lazos.py
"""
import _common  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation

from holonomy.report import write_csv
from parches_boosts import MaxEnt, rz
from parches_cubico import matriz

FALLOS = []
FILAS = []
VS = (0.02, 0.03, 0.05)


def ok(cond, msg):
    print(("OK   " if cond else "FALLO ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


def holonomia(edges, Ms, ciclo):
    R = np.eye(3)
    for a, b in zip(ciclo, ciclo[1:] + ciclo[:1]):
        R = R @ matriz(edges, Ms, a, b)
    return R


def prediccion(edges, Ms, ciclos, v):
    """Término líder: sum_C (-v)^l (tr R_C - 3)."""
    return sum((-v) ** len(c) * (np.trace(holonomia(edges, Ms, c)) - 3) for c in ciclos)


def serie(nombre, n, edges, Ms, ciclos, vs, tol):
    me = MaxEnt(n, edges)
    S0 = {v: me.resuelve([np.eye(3)] * len(edges), v)[0] for v in vs}
    cocientes = []
    for v in vs:
        S, res, _ = me.resuelve(Ms, v)
        dS = S - S0[v]
        pred = prediccion(edges, Ms, ciclos, v)
        cocientes.append(dS / pred)
        FILAS.append([nombre, v, dS, pred, dS / pred, res])
        print(f"  {nombre:22s} v = {v:.3f}: Delta S = {dS: .6e}, líder {pred: .6e}, cociente {dS / pred:.5f}"
              f"  (residuo {res:.1e})", flush=True)
    # cociente = 1 + a v + b v^2: extrapolación por los tres puntos
    c0 = np.polyfit(vs, cocientes, len(vs) - 1)[-1]
    ok(abs(c0 - 1) < tol, f"{nombre}: cociente extrapolado a v = 0: {c0:.5f}")
    return cocientes


def anillo(n):
    return tuple((i, (i + 1) % n) for i in range(n))


def main():
    print("\n1. Anillos con flujo en una arista")
    raices = {}
    for n in (4, 5, 6, 7):
        edges = anillo(n)
        for f in (np.pi / 2, np.pi):
            Ms = [np.eye(3)] * (n - 1) + [rz(f)]
            # con l = 7 y v = 0,02, Delta S ~ 1e-11: el redondeo de S_G (1e-15) limita la extrapolación
            cs = serie(f"anillo {n}, f = {f / np.pi:.2f} pi", n, edges, Ms, [list(range(n))], VS, 2e-3 if n < 7 else 5e-3)
            raices[n, f] = np.array(cs) ** (1 / n)
    # el cociente es (r(v)/v)^l con r(v) común a todos los anillos y flujos
    disp = np.ptp(np.array(list(raices.values())), axis=0)
    ok(disp.max() < 1e-3, f"anillos: (cociente)^(1/l) común a l = 4..7 y a los dos flujos, dispersión {disp.max():.1e}")
    print("  r(v) = v (cociente)^(1/l): " + ", ".join(f"{v}: {v * raices[4, np.pi][k]:.6f}" for k, v in enumerate(VS)))

    print("\n5. Anillos a v mayor: r(v) frente a la xi de la tabla tab:xi de fisher")
    XI = {0.2: (0.5937, 0.5946), 0.3: (0.7821, 0.7820), 0.4: (1.0401, 1.0401)}
    for v, (xp, xi) in XI.items():
        dS = {}
        for n in (6, 7, 8):
            me = MaxEnt(n, anillo(n))
            dS[n] = me.resuelve([np.eye(3)] * (n - 1) + [rz(np.pi)], v)[0] - me.resuelve([np.eye(3)] * n, v)[0]
        r67, r78 = -dS[7] / dS[6], -dS[8] / dS[7]
        A = dS[8] / (2 * (-1) ** 9 * r78**8 * 2)
        FILAS.append([f"anillo r(v)", v, r67, r78, A, np.nan])
        print(f"  v = {v}: r(6->7) = {r67:.5f}, r(7->8) = {r78:.5f}, e^(-1/xi) = {np.exp(-1 / xp):.5f} (pares), "
              f"{np.exp(-1 / xi):.5f} (impares); amplitud A/2 = {A:.4f}", flush=True)
        ok(abs(r78 - np.exp(-1 / xp)) < 0.02 * r78, f"v = {v}: r(v) = e^(-1/xi) a un 2 %")

    print("\n2. Anillo de 4 con datos SO(3) al azar")
    rng = np.random.default_rng(3)
    for k in range(3):
        Ms = list(Rotation.random(4, random_state=rng).as_matrix())
        serie(f"anillo 4, SO(3) #{k}", 4, anillo(4), Ms, [[0, 1, 2, 3]], VS, 2e-3)

    print("\n3. Cubo (escalera 2x4) con fases al azar")
    # sitios 0..3 abajo, 4..7 arriba; caras: dos tapas y cuatro laterales
    edges = tuple([(i, (i + 1) % 4) for i in range(4)] + [(4 + i, 4 + (i + 1) % 4) for i in range(4)]
                  + [(i, i + 4) for i in range(4)])
    caras = [[0, 1, 2, 3], [4, 5, 6, 7]] + [[i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i] for i in range(4)]
    Ms = [rz(f) for f in rng.uniform(-np.pi, np.pi, len(edges))]
    serie("cubo, fases al azar", 8, edges, Ms, caras, VS, 5e-3)

    print("\n4. Casa: cuadrado 0123 y triángulo 2 3 4, flujo solo en el cuadrado")
    edges = ((0, 1), (1, 2), (2, 3), (0, 3), (2, 4), (3, 4))
    Ms = [np.eye(3)] * 3 + [rz(-2.0)] + [np.eye(3)] * 2
    # el pentágono exterior 0 1 2 4 3 tiene la misma holonomía, pero entra a orden v^5
    serie("casa, f_cuadrado = 2", 5, edges, Ms, [[0, 1, 2, 3]], VS, 5e-3)

    write_csv("lazos.csv", ("caso", "v", "Delta_S", "lider", "cociente", "residuo"), FILAS)
    print(f"\n{'TODO OK' if not FALLOS else f'{len(FALLOS)} FALLOS'}")
    raise SystemExit(bool(FALLOS))


if __name__ == "__main__":
    main()

"""Artículo III (parches): desarrollo de la entropía de máxima entropía S_G hasta orden v^3.

Con marginales rho_e(v) = (I - v sum_ab (M_e)_ab s_a s_b)/4 en cada arista (M_ji = M_ij^T),

    S_G = N ln 2 - (v^2/2) sum_e ||M_e||_F^2 + v^3 [ sum_e det M_e - sum_{ijk} tr(M_ij M_jk M_ki) ] + O(v^4),

con la segunda suma sobre los triángulos del grafo. Con M_e = R_z(phi_e) B(b_e) y el triángulo i < j < k,

    tr(M_ij M_jk M_ki) = 1 + 2 cos f_p + b_ij . R(phi_jk) b_ik + b_jk . b_ik,   f_p = phi_ij + phi_jk - phi_ik,

así que la curvatura entra como 2 v^3 sum_p (1 - cos f_p) y la parte impar en los flujos es
O = v^3 sum_p sin(phi_jk) (b_ij x b_ik) + O(v^4).

Comprueba:
1. la traza cerrada del triángulo contra el producto de matrices;
2. que (S_G - S_3)/v^4 se estabiliza para v -> 0 en el triángulo, K_4 y el octaedro con curvatura, con y sin boosts
   (resolutor general de parches_boosts.MaxEnt);
3. el término de curvatura contra las entropías de data/parches_nervios_entropia.csv (v = 0,1 y 0,2);
4. la parte impar contra data/parches_boosts_paridad.csv (octaedro, v = 0,15) y su orden v^3 a v pequeño.

Escribe data/parches_cubico_{convergencia,curvatura,paridad}.csv.

Uso: OMP_NUM_THREADS=8 python scripts/parches_cubico.py
"""
import _common  # noqa: F401

import csv
import itertools

import numpy as np

from holonomy.report import write_csv, DATA
from parches_boosts import MaxEnt, datos, grafo, rz, boost

FALLOS = []


def ok(cond, msg):
    print(("OK   " if cond else "FALLO ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


def triangulos(n, edges):
    E = set(map(tuple, map(sorted, edges)))
    return [t for t in itertools.combinations(range(n), 3)
            if (t[0], t[1]) in E and (t[1], t[2]) in E and (t[0], t[2]) in E]


def matriz(edges, Ms, i, j):
    for (a, b), M in zip(edges, Ms):
        if (a, b) == (i, j):
            return M
        if (a, b) == (j, i):
            return M.T
    raise KeyError((i, j))


def traza_triangulo(edges, Ms, t):
    i, j, k = t
    return np.trace(matriz(edges, Ms, i, j) @ matriz(edges, Ms, j, k) @ matriz(edges, Ms, k, i))


def S3(n, edges, Ms, v):
    """Desarrollo hasta v^3."""
    s = n * np.log(2) - 0.5 * v**2 * sum((M**2).sum() for M in Ms)
    s += v**3 * sum(np.linalg.det(M) for M in Ms)
    s -= v**3 * sum(traza_triangulo(edges, Ms, t) for t in triangulos(n, edges))
    return s


def impar3(n, edges, phis, bs):
    """Coeficiente de v^3 de O = [S(phi,b) - S(-phi,b)]/2, en forma cerrada."""
    E = {tuple(e): (f, np.asarray(b)) for e, f, b in zip(map(tuple, edges), phis, bs)}
    cruz = lambda a, b: a[0] * b[1] - a[1] * b[0]
    return sum(np.sin(E[(j, k)][0]) * cruz(E[(i, j)][1], E[(i, k)][1]) for i, j, k in triangulos(n, edges))


def parte_traza(rng):
    print("\n1. Traza cerrada del triángulo")
    err = 0.0
    for _ in range(200):
        ph = rng.uniform(-np.pi, np.pi, 3)
        b = rng.normal(size=(3, 2))
        edges = ((0, 1), (1, 2), (0, 2))
        Ms = datos(ph, b)
        R2 = rz(ph[1])[:2, :2]
        cerrada = 1 + 2 * np.cos(ph[0] + ph[1] - ph[2]) + b[0] @ R2 @ b[2] + b[1] @ b[2]
        err = max(err, abs(traza_triangulo(edges, Ms, (0, 1, 2)) - cerrada))
    ok(err < 1e-12, f"tr(M_ij M_jk M_ki) = 1 + 2cos f + b_ij.R(phi_jk) b_ik + b_jk.b_ik, error máximo {err:.1e}")


def parte_convergencia(rng):
    print("\n2. Convergencia del resto (S_G - S_3)/v^4")
    filas = []
    for nombre in ("triangulo", "tetraedro", "octaedro"):
        n, edges, ph, _ = grafo(nombre)
        edges = tuple(tuple(e) for e in edges)
        me = MaxEnt(n, edges)
        for caso in ("curvatura", "curvatura+boosts"):
            bs = rng.normal(size=(len(edges), 2)) * 0.5 if "boosts" in caso else np.zeros((len(edges), 2))
            Ms = datos(ph, bs)
            restos = []
            for v in (0.01, 0.02, 0.04, 0.08):
                S, res, _ = me.resuelve(Ms, v)
                r = (S - S3(n, edges, Ms, v)) / v**4
                restos.append(r)
                filas.append([nombre, caso, v, S, S3(n, edges, Ms, v), r, res])
                print(f"  {nombre:9s} {caso:16s} v = {v:.2f}: S_G = {S:.12f}, resto/v^4 = {r: .4f}  (residuo {res:.1e})",
                      flush=True)
            # resto/v^4 = c4 + O(v): estable entre v = 0,01 y 0,02 (si el resto fuera O(v^3), se doblaría)
            ok(abs(restos[0] - restos[1]) < 0.03 * abs(restos[0]),
               f"{nombre} {caso}: resto O(v^4), resto/v^4 = {restos[0]:.2f} y {restos[1]:.2f} en v = 0,01 y 0,02")
    write_csv("parches_cubico_convergencia.csv", ("grafo", "datos", "v", "S_G", "S_3", "resto_v4", "residuo"), filas)


def parte_curvatura():
    print("\n3. Curvatura: Delta S_G = 2 v^3 sum_p (1 - cos f_p) + O(v^4)")
    with open(DATA / "parches_nervios_entropia.csv") as fh:
        S = {(r["grafo"], round(float(r["v"]), 6)): float(r["S"]) for r in csv.DictReader(fh)}
    filas = []
    for nombre in ("tetraedro", "octaedro", "icosaedro"):
        n, edges, ph, _ = grafo(nombre)
        edges = tuple(tuple(e) for e in edges)
        cero = np.zeros((len(edges), 2))
        for v in (0.1, 0.2):
            med = S[(f"{nombre} curvado", v)] - S[(f"{nombre} trivial", v)]
            pred = S3(n, edges, datos(ph, cero), v) - S3(n, edges, datos(0 * ph, cero), v)
            w = 2 * v**3 * sum(1 - np.cos(f) for f in
                               [np.arccos(np.clip((traza_triangulo(edges, datos(ph, cero), t) - 1) / 2, -1, 1))
                                for t in triangulos(n, edges)])
            filas.append([nombre, v, med, pred, med / pred])
            print(f"  {nombre:9s} v = {v}: medido {med:.6f}, orden v^3 {pred:.6f} (Wilson {w:.6f}), cociente {med / pred:.3f}")
            ok(abs(pred - w) < 1e-12, f"{nombre} v={v}: el término cúbico es la acción de Wilson")
            if v == 0.1:  # a v = 0,2 el icosaedro está en v/v_max = 0,78, fuera del régimen de v pequeño
                ok(med > 0 and 0.85 < med / pred < 1.0, f"{nombre} v=0,1: el orden v^3 da {med / pred:.0%} del aumento")
    write_csv("parches_cubico_curvatura.csv", ("grafo", "v", "Delta_S_medido", "Delta_S_v3", "cociente"), filas)


def parte_paridad():
    print("\n4. Parte impar O = v^3 sum_p sin(phi_jk) (b_ij x b_ik) + O(v^4), octaedro con curvatura")
    n, edges, ph, _ = grafo("octaedro")
    edges = tuple(tuple(e) for e in edges)
    E = len(edges)
    rng = np.random.default_rng(5)  # mismas direcciones que parches_boosts.parte_paridad
    u = rng.normal(size=(E, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    filas = []
    # a) frente a la tabla de parches_boosts (v = 0,15)
    with open(DATA / "parches_boosts_paridad.csv") as fh:
        tabla = [r for r in csv.DictReader(fh) if r["prueba"] == "impar"]
    v = 0.15
    for r in tabla:
        uu = np.tile([1.0, 0.0], (E, 1)) if r["direccion"] == "eje" else u
        nb, lam = float(r["b"]), float(r["escala_flujo"])
        pred = v**3 * impar3(n, edges, lam * ph, nb * uu)
        O = float(r["O"])
        filas.append(["tabla", r["direccion"], v, nb, lam, O, pred])
        print(f"  {r['direccion']:4s} |b| = {nb:3.1f}, flujo x {lam:4.2f}: O = {O: .3e}, v^3 = {pred: .3e}")
        if r["direccion"] == "eje":
            ok(pred == 0.0 and O == 0.0, f"boosts paralelos, |b|={nb}, x{lam}: O = 0 en los dos")
    # b) orden v^3 a v pequeño
    me = MaxEnt(n, edges)
    bs, phs = 0.4 * u, 0.5 * ph
    c3 = impar3(n, edges, phs, bs)
    cs = []
    for v in (0.01, 0.02, 0.04):
        Sp, rp, _ = me.resuelve(datos(phs, bs), v)
        Sm, rm, _ = me.resuelve(datos(-phs, bs), v)
        cs.append((Sp - Sm) / 2 / v**3)
        filas.append(["v_pequeno", "azar", v, 0.4, 0.5, (Sp - Sm) / 2, v**3 * c3])
        print(f"  v = {v:.2f}: O/v^3 = {cs[-1]: .6f}, predicho {c3: .6f}  (residuos {max(rp, rm):.1e})", flush=True)
    # O/v^3 = c3 + a v + b v^2: ajuste exacto por los tres puntos
    extrapolado = np.polyfit((0.01, 0.02, 0.04), cs, 2)[-1]
    ok(abs(extrapolado - c3) < 3e-4 * abs(c3),
       f"O/v^3 extrapolado a v = 0: {extrapolado:.6f} frente a {c3:.6f}")
    write_csv("parches_cubico_paridad.csv", ("serie", "direccion", "v", "b", "escala_flujo", "O", "O_v3"), filas)


def main():
    rng = np.random.default_rng(11)
    parte_traza(rng)
    parte_convergencia(rng)
    parte_curvatura()
    parte_paridad()
    print(f"\n{'TODO OK' if not FALLOS else f'{len(FALLOS)} FALLOS'}")
    raise SystemExit(bool(FALLOS))


if __name__ == "__main__":
    main()

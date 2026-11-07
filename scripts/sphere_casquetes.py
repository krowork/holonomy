"""Defecto de pegado de dos casquetes de un poliedro esférico:
delta = min(v(G1), v(G2)) - v(G1 unión G2), con datos triviales y de curvatura
(restringidos del poliedro completo).

Hay tres cortes. Cubo, dos caras: G1 y G2 inducidos por {0,..,5} y {2,..,7} de la
numeración del piloto, cuyas aristas son (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),
(7,4),(0,4),(1,5),(2,6),(3,7), que aquí se traduce a la numeración de holonomy.sphere
por coordenadas; cada uno tiene dos caras, el solape son dos aristas sueltas y dos caras
no están en ninguno. Cubo, hemisferios: H1 y H2, el cubo sin (-1,-1,-1) y sin (1,1,1);
tres caras cada uno, flujo 2 pi, solape un hexágono. Dodecaedro, hemisferios: eje de 5
pliegues por dos centros de cara opuestos, cuatro anillos de 5 vértices r1..r4 de arriba
abajo, G1 inducido por r1, r2, r3 y G2 por r2, r3, r4; seis caras cada uno, flujo 2 pi,
solape un decágono. Solo los dos cortes en hemisferios son comparables entre sí.

Cada casquete se resuelve con un testigo invariante bajo el subgrupo de isometrías
que conserva sus vértices, con (a, b, c) por órbita de aristas y planos de corte
(sphere.OrbitDual): cota superior lambda_max(y), inferior el LP, objetivo < 1e-8. Para
la unión se leen los valores de data/sphere_series.csv.

Comprobaciones:
  1. estructura: vértices y aristas de cada casquete, solape, anillos;
  2. cotas a < 1e-8, con la caja inactiva;
  3. el valor no cambia al repetir el óptimo con más vectores y arranque aleatorio;
  4. casquetes del cubo frente a primal_u1_graph a 1e-6 y, los de dos caras, al valor
     del piloto a 1e-5;
  5. 20 perturbaciones no simétricas del testigo con N = 1 en el bloque máximo (cota
     inferior de lambda_max): no bajan de v, salvo la diferencia de cotas;
  6. v(G) >= v(unión): delta >= 0, también con las cotas;
  7. en los hemisferios, holonomía del ciclo de solape = flujo del casquete, mód 2 pi;
  8. si existe data/sphere_casquetes_sin_simetria.csv: en el casquete G1 del dodecaedro,
     la cota inferior sin simetría con curvatura supera v sin curvatura, que es cota
     superior con cualquier testigo y no necesita el argumento de simetría.

Escribe data/sphere_casquetes.csv (un proceso hijo por caso; relanzar no repite lo
hecho) y data/sphere_casquetes_defecto.csv. Con --sin-simetria repite G1 del dodecaedro
curvado con un testigo sin simetría, sin exigir convergencia, y escribe
data/sphere_casquetes_sin_simetria.csv: el LP sigue siendo cota inferior.

Uso: (ulimit -v 25000000; python scripts/sphere_casquetes.py [--sin-simetria])
"""
import _common  # noqa: F401
from _common import Checker

import argparse
import csv
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from holonomy import sdp, sphere
from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
SALIDA = "sphere_casquetes.csv"
DEFECTO = "sphere_casquetes_defecto.csv"
SIN_SIMETRIA = "sphere_casquetes_sin_simetria.csv"
COLUMNS = ("poliedro", "subgrafo", "caso", "n", "E", "orden_grupo", "orbitas", "b_libres", "v",
           "cota_lp", "cota_multiplicadores", "dif", "iteraciones", "cortes", "k_max", "v_repetido",
           "v_primal", "pert_min", "y", "parametros", "segundos", "pico_GB")
PILOTO_XYZ = {0: (1, 1, 1), 1: (1, -1, 1), 2: (-1, -1, 1), 3: (-1, 1, 1),
              4: (1, 1, -1), 5: (1, -1, -1), 6: (-1, -1, -1), 7: (-1, 1, -1)}
PILOTO_ARISTAS = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7))
PILOTO_V = dict(trivial=(0.555556, 0.535565), curvado=(0.460328, 0.424853))
TOL_COTAS = 1e-8
N_PERT = 20
CORTES = (("cubo", "dos caras", "G1", "G2"), ("cubo", "hemisferios", "H1", "H2"),
          ("dodecaedro", "hemisferios", "G1", "G2"))
CASOS = [(pol, g, caso) for pol, _, *gs in CORTES for caso in ("trivial", "curvado") for g in gs]


def read_rows(name):
    path = DATA / name
    return list(csv.DictReader(open(path))) if path.exists() else []


def cube_pilot_map(p):
    """Vértice del piloto -> vértice de p, por coordenadas."""
    idx = {tuple(v): q for q, v in enumerate(p.verts)}
    return {k: idx[tuple(float(c) for c in xyz)] for k, xyz in PILOTO_XYZ.items()}


def rings(p):
    """Cuatro anillos de 5 vértices alrededor del eje por el centro de la cara 0."""
    axis = p.verts[list(p.faces[0])].mean(axis=0)
    h = sphere.unit_vertices(p) @ (axis / np.linalg.norm(axis))
    order = np.argsort(-h)
    return [sorted(int(v) for v in order[5 * q:5 * q + 5]) for q in range(4)]


def subsets(pol, p):
    """Vértices de G1 y G2 en la numeración de p."""
    if pol == "cubo":
        m = cube_pilot_map(p)
        idx = {tuple(v): q for q, v in enumerate(p.verts)}
        return dict(G1=[m[v] for v in range(6)], G2=[m[v] for v in range(2, 8)],
                    H1=[q for q in range(8) if q != idx[-1.0, -1.0, -1.0]],
                    H2=[q for q in range(8) if q != idx[1.0, 1.0, 1.0]])
    r = rings(p)
    return dict(G1=r[0] + r[1] + r[2], G2=r[1] + r[2] + r[3])


def data_phases(p, caso):
    return np.zeros(p.E) if caso == "trivial" else sphere.phases(p, sphere.face_fluxes(p))


def compute_case(pol, g, caso):
    """Resuelve un casquete y añade su fila."""
    t0 = time.time()
    p = sphere.polyhedron(pol)
    vs = subsets(pol, p)[g]
    sub, ph, _ = sphere.induced_subgraph(p, data_phases(p, caso), vs)
    grupo = sphere.subset_group(p, sphere.symmetry_group(p), vs)
    orbs = sphere.edge_orbits(sub, grupo)
    dual = sphere.OrbitDual(sub, ph, orbs)
    res = dual.minimize(log=100)
    print(f"      {pol} {g} {caso}: v = {res.v:.10f}, LP = {res.lower:.10f}, dif = {res.gap:.1e}, "
          f"{res.n_iter} iter, {res.n_cuts} cortes, {time.time() - t0:.0f} s", flush=True)
    k_max = int(np.argmax(res.block_values))
    # Repetición en el óptimo, con arranque aleatorio y más vectores por bloque.
    rep = sphere.OrbitDual(sub, ph, orbs, k_eigs=8, ncv=60, seed=23)
    v_rep = rep.value(res.y, all_blocks=True, add_cuts=False)[0]
    # Perturbaciones no simétricas con N = 1, solo en el bloque máximo.
    a, b, c = dual.edge_coefficients(res.y)
    rng = np.random.default_rng(101)
    E = len(sub.edges)
    pert = []
    for q in range(N_PERT):
        eps = (1e-3 if q % 2 == 0 else 1e-2) / E
        da, db, dc = rng.normal(size=(3, E)) * eps
        dc -= (2 * da + dc).sum() / E
        lam = sphere.witness_block_top(sub, ph, a + da, b + db, c + dc, k_max, seed=q)
        pert.append(lam - res.v)
    v_primal = ""
    if sub.n <= 7:
        v_primal = sdp.primal_u1_graph(sub.n, list(sub.edges), ph)
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    fila = dict(
        poliedro=pol, subgrafo=g, caso=caso, n=sub.n, E=E, orden_grupo=len(grupo),
        orbitas=";".join(f"{len(o.edges)}/{o.stabilizer}/{'b0' if o.b_zero else 'blibre'}"
                         for o in orbs),
        b_libres=sum(not o.b_zero for o in orbs), v=res.v, cota_lp=res.lower,
        cota_multiplicadores=res.lower_dual, dif=res.gap, iteraciones=res.n_iter,
        cortes=res.n_cuts, k_max="|".join(str(k) for k in
                                          np.flatnonzero(res.block_values > res.v - 1e-7)),
        v_repetido=v_rep, v_primal=v_primal, pert_min=min(pert),
        y=";".join(repr(float(x)) for x in res.y),
        parametros=";".join(f"{t}{q}" for t, q in res.params),
        segundos=time.time() - t0, pico_GB=pico)
    filas = read_rows(SALIDA)
    write_csv(SALIDA, COLUMNS, [[r[h] for h in COLUMNS] for r in filas + [fila]])


def sin_simetria():
    """G1 del dodecaedro curvado sin simetría: el LP es cota inferior aunque no converja."""
    t0 = time.time()
    p = sphere.polyhedron("dodecaedro")
    sub, ph, _ = sphere.induced_subgraph(p, data_phases(p, "curvado"), subsets("dodecaedro", p)["G1"])
    orbs = sphere.edge_orbits(sub, sphere.identity_group(sub))
    res = sphere.OrbitDual(sub, ph, orbs).minimize(max_iter=1200, log=100)
    print(f"      curvado: LP {res.lower:.7f}, lambda_max {res.v:.7f}, {res.n_iter} iter", flush=True)
    write_csv(SIN_SIMETRIA, ("poliedro", "subgrafo", "caso", "inf", "iteraciones", "segundos"),
              [["dodecaedro", "G1", "curvado", res.lower, res.n_iter, time.time() - t0]])


def overlap_holonomy(p, phis, vs1, vs2):
    """Holonomía del ciclo inducido por el solape; flujo de las caras dentro de G1."""
    solape = sorted(set(vs1) & set(vs2))
    sub, ph, _ = sphere.induced_subgraph(p, phis, solape)
    vecinos = {v: [] for v in range(sub.n)}
    fase = {}
    for (i, j), f in zip(sub.edges, ph, strict=True):
        vecinos[i].append(j)
        vecinos[j].append(i)
        fase[i, j], fase[j, i] = f, -f
    if any(len(w) != 2 for w in vecinos.values()):
        raise ValueError("el solape no es un ciclo")
    prev, cur, total, largo = 0, vecinos[0][0], fase[0, vecinos[0][0]], 1
    while cur != 0:
        sig = next(w for w in vecinos[cur] if w != prev)
        total += fase[cur, sig]
        prev, cur = cur, sig
        largo += 1
    dentro = set(vs1)
    flujo = sum(f for c, f in zip(p.faces, sphere.face_fluxes(p), strict=True)
                if set(c) <= dentro)
    return total, largo, flujo, len(sub.edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", nargs=3, metavar=("POLIEDRO", "G", "CASO"))
    ap.add_argument("--sin-simetria", action="store_true")
    args = ap.parse_args()
    if args.solo:
        compute_case(*args.solo)
        return
    if args.sin_simetria:
        sin_simetria()
        return

    chk = Checker("sphere_casquetes")
    cubo = sphere.polyhedron("cubo")
    m = cube_pilot_map(cubo)
    traducidas = {tuple(sorted((m[i], m[j]))) for i, j in PILOTO_ARISTAS}
    chk.check(traducidas == set(cubo.edges), "cubo: la numeración del piloto da las 12 aristas")
    dodeca = sphere.polyhedron("dodecaedro")
    r = rings(dodeca)
    alturas = sphere.unit_vertices(dodeca) @ (dodeca.verts[list(dodeca.faces[0])].mean(axis=0))
    salto = max(np.ptp(alturas[w]) for w in r)
    tam = {g: sphere.induced_subgraph(dodeca, np.zeros(30), v)[0]
           for g, v in subsets("dodecaedro", dodeca).items()}
    chk.check(salto < 1e-9 and all(t.n == 15 and len(t.edges) == 20 for t in tam.values()),
              f"dodecaedro: anillos de 5 a altura constante ({salto:.1e}), G1 y G2 con 15 qubits "
              "y 20 aristas")

    hechos = {(r["poliedro"], r["subgrafo"], r["caso"]) for r in read_rows(SALIDA)}
    for caso in CASOS:
        if caso in hechos:
            chk.info(f"{' '.join(caso)}: ya en data/{SALIDA}")
            continue
        chk.info(f"{' '.join(caso)}: calculando...")
        cod = subprocess.run([sys.executable, __file__, "--solo", *caso]).returncode
        chk.check(cod == 0, f"{' '.join(caso)}: el proceso hijo termina con código {cod}")
        if cod:
            chk.finish()

    filas = read_rows(SALIDA)
    for r in filas:
        et = f"{r['poliedro']} {r['subgrafo']} {r['caso']}"
        v, dif = float(r["v"]), float(r["dif"])
        chk.check(dif < TOL_COTAS, f"{et}: v = {v:.10f}, dif. de cotas {dif:.1e}, grupo de orden "
                  f"{r['orden_grupo']}, órbitas (tamaño/estabilizador/b) {r['orbitas']}")
        chk.check(abs(float(r["v_repetido"]) - v) < 1e-9,
                  f"{et}: repetido con k_eigs = 8 y arranque aleatorio, "
                  f"|dif| = {abs(float(r['v_repetido']) - v):.1e}")
        chk.check(float(r["pert_min"]) >= -dif - 1e-9,
                  f"{et}: {N_PERT} perturbaciones no simétricas, min(lambda - v) = "
                  f"{float(r['pert_min']):+.2e}")
        if r["v_primal"]:
            vp = float(r["v_primal"])
            if r["subgrafo"] in ("G1", "G2"):
                piloto = PILOTO_V[r["caso"]][0]
                chk.check(abs(vp - v) < 1e-6 and abs(v - piloto) < 1e-5,
                          f"{et}: primal {vp:.8f}, reducido {v:.8f}, piloto {piloto:.6f}")
            else:
                chk.check(abs(vp - v) < 1e-6, f"{et}: primal {vp:.8f}, reducido {v:.8f}")

    serie = {r["nombre"]: r for r in read_rows("sphere_series.csv")}
    if not chk.check({"cubo", "dodecaedro"} <= set(serie), "data/sphere_series.csv tiene cubo y dodecaedro "
                     "(si no, lanzar antes scripts/sphere_series.py)"):
        chk.finish()
    defecto = []
    for pol, corte, g1, g2 in CORTES:
        p = sphere.polyhedron(pol)
        for caso in ("trivial", "curvado"):
            s = [r for r in filas if r["poliedro"] == pol and r["caso"] == caso]
            v1, v2 = (float(next(r["v"] for r in s if r["subgrafo"] == g)) for g in (g1, g2))
            d1, d2 = (float(next(r["dif"] for r in s if r["subgrafo"] == g)) for g in (g1, g2))
            vu = float(serie[pol][f"v_{caso}"])
            du = float(serie[pol][f"dif_{caso}"])
            delta = min(v1, v2) - vu
            # v(G) >= v(G) - dif y v(unión) <= v(unión): cota inferior de delta.
            delta_inf = min(v1 - d1, v2 - d2) - vu
            # v(G) <= v(G) y v(unión) >= v(unión) - dif: cota superior de delta.
            delta_sup = min(v1, v2) - (vu - du)
            hol, flujo, largo = "", "", ""
            if corte == "hemisferios":
                vs = subsets(pol, p)
                total, largo, flujo, n_ar = overlap_holonomy(p, data_phases(p, caso), vs[g1], vs[g2])
                hol = float(np.mod(total + np.pi, 2 * np.pi) - np.pi)
                if caso == "curvado":
                    res = abs(np.mod(abs(total) - flujo + np.pi, 2 * np.pi) - np.pi)
                    chk.check(n_ar == largo and res < 1e-9,
                              f"{pol}: solape = ciclo de {largo} aristas, holonomía {total:.10f}, "
                              f"mód 2pi {hol:+.2e}; flujo del casquete {flujo / np.pi:.10f} pi")
            chk.check(delta_inf > -1e-8, f"{pol}, {corte}, {caso}: delta = {delta:.10f} (entre "
                      f"{delta_inf:.10f} y {delta_sup:.10f}); v = {v1:.10f}, {v2:.10f}, unión {vu:.10f}")
            if corte == "dos caras":
                piloto = PILOTO_V[caso][1]
                chk.check(abs(vu - piloto) < 1e-5, f"cubo {caso}: unión {vu:.8f}, piloto {piloto:.6f}")
            defecto.append([pol, corte, caso, v1, v2, d1, d2, vu, du, delta, delta_inf, delta_sup,
                            hol, flujo, largo])
    write_csv(DEFECTO, ("poliedro", "corte", "caso", "v_G1", "v_G2", "dif_G1", "dif_G2", "v_union",
                        "dif_union", "delta", "delta_inf", "delta_sup", "holonomia_solape",
                        "flujo_casquete", "largo_solape"), defecto)
    sim = read_rows(SIN_SIMETRIA)
    if sim:
        ci = float(sim[0]["inf"])
        ts = next(float(r["v"]) for r in filas if (r["poliedro"], r["subgrafo"], r["caso"])
                  == ("dodecaedro", "G1", "trivial"))
        chk.check(ci > ts, f"dodecaedro G1: con curvatura v >= {ci:.7f} sin simetría, sin curvatura "
                  f"v <= {ts:.7f}; la curvatura sube v en este casquete")
    else:
        chk.info(f"sin data/{SIN_SIMETRIA}: comprobación sin simetría no hecha (--sin-simetria)")
    chk.finish()


if __name__ == "__main__":
    main()

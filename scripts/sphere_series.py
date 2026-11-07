"""Serie de poliedros esféricos: v_max con datos triviales y con curvatura
uniforme por ángulo sólido (flujo de cada cara = su ángulo sólido; suma 4 pi).

Para cada poliedro, de menor a mayor tamaño, resuelve el dual reducido por simetría
de holonomy.sphere y escribe una fila de data/sphere_series.csv, cada una en un
proceso hijo (pico de memoria propio; relanzar no repite las filas ya escritas).
Cada fila lleva la geometría, el índice cromático, v trivial y v curvado, el déficit,
t óptimo y bloque máximo, la diferencia de cotas, |phi_e| en el gauge L1 óptimo y la
cota inferior de la Proposición 5 de esfera, v0/(1 + (chi' + 1) media|phi_e|).

Comprobaciones:
  1. Euler, flujos con suma 4 pi y 4 pi/F en los platónicos; residuo de fases;
  2. una órbita de aristas en cada poliedro, b nulo salvo en el cuboctaedro;
  3. índice cromático de los esperados (rombododecaedro: 4, por König);
  4. tetraedro, octaedro y cubo: dual reducido frente a primal_u1_graph a 1e-6
     (data/sphere_primal.csv);
  5. v trivial = valor en el punto de Heisenberg, t = 1/(3E) (SU(2));
  6. cotas a < 1e-8 (con b libre, planos de corte); v curvado por encima de la cota
     de la Proposición 5 de esfera;
  7. icosaedro y dodecaedro triviales frente a una diagonalización independiente con
     los operadores de sdp.u1_sector_operators; se compara también con los valores
     de memoria de E0, solo orientativos;
  8. dodecaedro: el mayor autovalor en el óptimo no cambia con el número de autovalores
     pedidos ni con ncv (data/sphere_convergencia.csv);
  9. gauge SO(3) ambiente, sin recalcular los v (se releen del CSV): longitudes
     geodésicas l_e, chi'_f y las cotas v0/(1 + 3 chi'_f l_max) <= v curvado <=
     v0/(1 - (chi'_f + 1) v0 l_max^2/2), la superior solo si el denominador es positivo
     (data/sphere_gauge.csv);
 10. dodecaedro curvado: 10 perturbaciones no simétricas del testigo óptimo con N = 1,
     en el bloque k = 10 (cota inferior de lambda_max), no bajan de v salvo la
     diferencia de cotas (data/sphere_perturbaciones.csv).

Escribe además data/sphere_orbitas.csv y data/sphere_bloques.csv.

Uso: (ulimit -v 25000000; python scripts/sphere_series.py)   (varios minutos)
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
from scipy.sparse.linalg import eigsh

from holonomy import sdp, sphere
from holonomy.report import write_csv

DATA = Path(__file__).resolve().parent.parent / "data"
SERIE = "sphere_series.csv"
ORBITAS = "sphere_orbitas.csv"
BLOQUES = "sphere_bloques.csv"
PRIMAL = "sphere_primal.csv"
CONVERGENCIA = "sphere_convergencia.csv"
GAUGE = "sphere_gauge.csv"
PERTURBACIONES = "sphere_perturbaciones.csv"
COLUMNS = ("nombre", "N", "E", "F", "Delta", "chi", "flujo_min", "flujo_max",
           "v_trivial", "v_curvado", "deficit", "N_deficit", "sqrtN_deficit",
           "t_trivial", "t_trivial_min", "t_trivial_max", "Dxxz_trivial", "b_trivial", "k_trivial",
           "dif_trivial",
           "t_curvado", "t_curvado_min", "t_curvado_max", "Dxxz_curvado", "b_curvado", "v_curvado_b0",
           "k_curvado", "dif_curvado", "cota_curvado", "phi_media", "phi_max", "cota_media",
           "segundos", "pico_GB")
CHI_ESPERADO = dict(tetraedro=3, octaedro=4, cubo=3, icosaedro=5, dodecaedro=3, cuboctaedro=4,
                    rombododecaedro=4)
PRIMAL_ESPERADO = dict(tetraedro=(1 / 3, 1 / 3), octaedro=(1 / 3, 0.296296),
                       cubo=(0.535565, 0.424853))
E0_MEMORIA = dict(icosaedro=-6.188, dodecaedro=-9.722)
TOL_COTAS = 1e-8
TOL_PRIMAL = 1e-6
TOL_E0 = 1e-8


def read_rows(name):
    path = DATA / name
    return list(csv.DictReader(open(path))) if path.exists() else []


def append_rows(name, header, new):
    write_csv(name, header, [[r[h] for h in header] for r in read_rows(name) + new])


def compute_row(nombre):
    """Calcula y escribe la fila del poliedro en el proceso actual."""
    t0 = time.time()
    p = sphere.polyhedron(nombre)
    flujos = sphere.face_fluxes(p)
    phis = sphere.phases(p, flujos)
    l1 = sphere.min_l1_phases(p, flujos)
    chi = sphere.chromatic_index(p)
    grupo = sphere.symmetry_group(p)
    orbs = sphere.edge_orbits(p, grupo)
    res0 = sphere.reduced_dual(p, np.zeros(p.E), orbs)
    print(f"      {nombre}: trivial v = {res0.v:.10f}, t = {res0.t:.8f}, k = {res0.kmax}, "
          f"dif = {res0.gap:.1e} ({res0.n_eval} eval, {time.time() - t0:.0f} s)", flush=True)
    res1 = sphere.reduced_dual(p, phis, orbs)
    print(f"      {nombre}: curvado v = {res1.v:.10f}, t = {res1.t:.8f}, b = {res1.b:.8f}, "
          f"k = {res1.kmax}, dif = {res1.gap:.1e} ({res1.n_eval} eval, {time.time() - t0:.0f} s)",
          flush=True)
    intervalos = []
    for caso, ph, res in (("trivial", np.zeros(p.E), res0), ("curvado", phis, res1)):
        signos = None
        if not orbs[0].b_zero:
            signos = np.array([orbs[0].signs[m] for m in range(p.E)], dtype=float)
        d = sphere.ReducedDual(p, ph, b_signs=signos)
        intervalos.append(sphere.optimal_interval(lambda t: d.value(t, res.b), res.t, res.v,
                                                  0.001 / (3 * p.E)))
    # Con b libre, el valor con b = 0 mide cuánto aporta b.
    v_b0 = res1.v
    if res1.free_b:
        dual_b0 = sphere.ReducedDual(p, phis)
        v_b0 = sphere.minimize_convex(dual_b0.value, res1.t, 0.1 / p.E)[1]
    d = res0.v - res1.v
    cota_media = res0.v / (1 + (chi + 1) * l1["mean"])
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    fila = dict(
        nombre=nombre, N=p.n, E=p.E, F=p.F, Delta=p.max_degree, chi=chi,
        flujo_min=flujos.min(), flujo_max=flujos.max(),
        v_trivial=res0.v, v_curvado=res1.v, deficit=d, N_deficit=p.n * d,
        sqrtN_deficit=np.sqrt(p.n) * d,
        t_trivial=res0.t, t_trivial_min=intervalos[0][0], t_trivial_max=intervalos[0][1],
        Dxxz_trivial=res0.delta, b_trivial=res0.b, k_trivial="|".join(map(str, res0.kmax)), dif_trivial=res0.gap,
        t_curvado=res1.t, t_curvado_min=intervalos[1][0], t_curvado_max=intervalos[1][1],
        Dxxz_curvado=res1.delta, b_curvado=res1.b, v_curvado_b0=v_b0, k_curvado="|".join(map(str, res1.kmax)),
        dif_curvado=res1.gap, cota_curvado=res1.bound_kind,
        phi_media=l1["mean"], phi_max=l1["max"], cota_media=cota_media,
        segundos=time.time() - t0, pico_GB=pico)
    append_rows(SERIE, COLUMNS, [fila])
    append_rows(ORBITAS, ("nombre", "orden_grupo", "orbita", "n_aristas", "estabilizador", "b_nulo"),
                [dict(nombre=nombre, orden_grupo=len(grupo), orbita=q, n_aristas=len(o.edges),
                      estabilizador=o.stabilizer, b_nulo=o.b_zero) for q, o in enumerate(orbs)])
    append_rows(BLOQUES, ("nombre", "caso", "k", "lambda_max"),
                [dict(nombre=nombre, caso=caso, k=k, lambda_max=lam)
                 for caso, res in (("trivial", res0), ("curvado", res1))
                 for k, lam in enumerate(res.block_values)])


def primal_checks(chk):
    """Tetraedro, octaedro y cubo: primal frente al dual reducido de la serie."""
    hechos = {(r["nombre"], r["caso"]) for r in read_rows(PRIMAL)}
    filas = {r["nombre"]: r for r in read_rows(SERIE)}
    for nombre, (v0, v1) in PRIMAL_ESPERADO.items():
        p = sphere.polyhedron(nombre)
        phis = sphere.phases(p, sphere.face_fluxes(p))
        for caso, ph, ve, col in (("trivial", np.zeros(p.E), v0, "v_trivial"),
                                  ("curvado", phis, v1, "v_curvado")):
            if (nombre, caso) not in hechos:
                t0 = time.time()
                v = sdp.primal_u1_graph(p.n, list(p.edges), ph)
                append_rows(PRIMAL, ("nombre", "caso", "v_primal", "segundos"),
                            [dict(nombre=nombre, caso=caso, v_primal=v, segundos=time.time() - t0)])
            r = next(x for x in read_rows(PRIMAL) if x["nombre"] == nombre and x["caso"] == caso)
            vp, vd = float(r["v_primal"]), float(filas[nombre][col])
            chk.check(abs(vp - vd) < TOL_PRIMAL and abs(vd - ve) < 1e-5,
                      f"{nombre} {caso}: primal {vp:.8f}, dual reducido {vd:.8f}, "
                      f"esperado {ve:.6f}")


def independent_heisenberg(nombre, k):
    """E0 de sum_e S_i.S_j en el sector de k giros, con los operadores de
    sdp.u1_sector_operators (bucle de Python, sin el código de sphere)."""
    p = sphere.polyhedron(nombre)
    _, ops = sdp.u1_sector_operators(p.n, list(p.edges), k)
    # S.S = (XX + YY + ZZ)/4 = -(A + C)/4.
    H = sum(-(ops["A", e] + ops["C", e]) / 4 for e in range(p.E)).tocsr().real
    if H.shape[0] <= 2000:
        return float(np.linalg.eigvalsh(H.toarray())[0])
    return float(eigsh(H, k=3, which="SA", tol=1e-12, ncv=60, return_eigenvectors=False).min())


def independent_checks(chk):
    filas = {r["nombre"]: r for r in read_rows(SERIE)}
    for nombre, k in (("icosaedro", 6), ("dodecaedro", 10)):
        if nombre not in filas:
            continue
        t0 = time.time()
        E0 = independent_heisenberg(nombre, k)
        E = int(filas[nombre]["E"])
        v = 4 * (-E0) / (3 * E)
        vd = float(filas[nombre]["v_trivial"])
        chk.check(abs(v - vd) < TOL_E0,
                  f"{nombre}: diagonalización independiente (k = {k}): E0 = {E0:.8f}, "
                  f"v = {v:.10f}; dual reducido {vd:.10f} ({time.time() - t0:.0f} s)")
        Em = E0_MEMORIA[nombre]
        chk.info(f"{nombre}: E0 de memoria {Em} (v = {-4 * Em / (3 * E):.4f}); "
                 f"calculado {E0:.5f}; diferencia {E0 - Em:+.5f}")


def convergence_checks(chk):
    """Dodecaedro: repite con arranque aleatorio y varios k_eigs y ncv los bloques
    próximos al máximo en el óptimo."""
    filas = {r["nombre"]: r for r in read_rows(SERIE)}
    if "dodecaedro" not in filas:
        return
    fila = filas["dodecaedro"]
    p = sphere.polyhedron("dodecaedro")
    phis = sphere.phases(p, sphere.face_fluxes(p))
    bloques = [r for r in read_rows(BLOQUES) if r["nombre"] == "dodecaedro"]
    hechos = {(r["caso"], r["k"], r["k_eigs"], r["ncv"]) for r in read_rows(CONVERGENCIA)}
    cab = ("caso", "k", "k_eigs", "ncv", "lambda", "referencia", "segundos")
    peor = 0.0
    for caso, ph, t in (("trivial", np.zeros(p.E), float(fila["t_trivial"])),
                        ("curvado", phis, float(fila["t_curvado"]))):
        dual = sphere.ReducedDual(p, ph)
        vals = {int(r["k"]): float(r["lambda_max"]) for r in bloques if r["caso"] == caso}
        top = max(vals.values())
        for k in [q for q, lam in vals.items() if lam > top - 0.02]:
            for k_eigs, ncv in ((1, 30), (3, 40), (6, 60), (10, 80)):
                if (caso, str(k), str(k_eigs), str(ncv)) not in hechos:
                    t0 = time.time()
                    lam, _ = dual.block_lmax(k, t, k_eigs=k_eigs, ncv=ncv, tol=1e-12, warm=False)
                    append_rows(CONVERGENCIA, cab,
                                [dict(caso=caso, k=k, k_eigs=k_eigs, ncv=ncv, referencia=vals[k],
                                      segundos=time.time() - t0, **{"lambda": lam})])
                    print(f"      dodecaedro {caso} k={k} k_eigs={k_eigs} ncv={ncv}: {lam:.12f}, "
                          f"referencia {vals[k]:.12f} ({time.time() - t0:.0f} s)", flush=True)
    for r in read_rows(CONVERGENCIA):
        peor = max(peor, abs(float(r["lambda"]) - float(r["referencia"])))
    chk.check(peor < 1e-9, f"dodecaedro: eigsh estable con k_eigs = 1, 3, 6, 10 y ncv = 30..80 "
              f"en los bloques próximos al máximo, |dif| máx. = {peor:.1e}")


def gauge_checks(chk):
    """Cotas con la geometría del gauge ambiente; los v se releen de la serie."""
    filas = []
    for r in read_rows(SERIE):
        p = sphere.polyhedron(r["nombre"])
        _, ls = sphere.ambient_rotations(p)
        chi_f = sphere.fractional_chromatic_index(p)
        v0, v1 = float(r["v_trivial"]), float(r["v_curvado"])
        lmax = float(ls.max())
        inf = v0 / (1 + 3 * chi_f * lmax)
        den = 1 - (chi_f + 1) * v0 * lmax**2 / 2
        sup = v0 / den if den > 0 else np.inf
        filas.append([r["nombre"], lmax, float(ls.mean()), chi_f, inf, den, sup, v0, v1])
        chk.check(inf <= v1 and v1 <= sup + float(r["dif_curvado"]),
                  f"{r['nombre']}: l_max = {lmax:.8f}, chi'_f = {chi_f:.6f}, cota inf. {inf:.6f} <= "
                  f"v curvado {v1:.6f} <= cota sup. {sup:.6f} (denominador {den:+.4f})")
    write_csv(GAUGE, ("nombre", "l_max", "l_media", "chi_f", "cota_inf", "denominador", "cota_sup",
                      "v_trivial", "v_curvado"), filas)


def perturbation_checks(chk):
    """Dodecaedro curvado: testigos no simétricos con N = 1 en el bloque k = 10."""
    filas = {r["nombre"]: r for r in read_rows(SERIE)}
    if "dodecaedro" not in filas:
        return
    r = filas["dodecaedro"]
    p = sphere.polyhedron("dodecaedro")
    phis = sphere.phases(p, sphere.face_fluxes(p))
    t, v, dif = float(r["t_curvado"]), float(r["v_curvado"]), float(r["dif_curvado"])
    a0, c0 = np.full(p.E, t), np.full(p.E, 1 / p.E - 2 * t)
    cab = ("indice", "eps", "N", "lambda_k10", "lambda_menos_v", "segundos")
    hechas = read_rows(PERTURBACIONES)
    rng = np.random.default_rng(20261107)
    for q in range(10):
        # Se sortea siempre, para que relanzar dé las mismas perturbaciones.
        eps = (1e-3 if q % 2 == 0 else 1e-2) / p.E
        da, db, dc = rng.normal(size=(3, p.E)) * eps
        dc -= (2 * da + dc).sum() / p.E
        if q < len(hechas):
            continue
        t0 = time.time()
        a, b, c = a0 + da, db, c0 + dc
        N = float((2 * a + c).sum())
        lam = sphere.witness_block_top(p, phis, a, b, c, 10, tol=1e-10, k_eigs=3, ncv=40, seed=q)
        append_rows(PERTURBACIONES, cab, [dict(indice=q, eps=eps, N=N, lambda_k10=lam, lambda_menos_v=lam - v,
                                               segundos=time.time() - t0)])
    filas = read_rows(PERTURBACIONES)
    dv = [float(x["lambda_menos_v"]) for x in filas]
    Ns = [abs(float(x["N"]) - 1) for x in filas]
    chk.check(min(dv) >= -dif - 1e-9 and max(Ns) < 1e-12,
              f"dodecaedro curvado: {len(dv)} perturbaciones no simétricas con N = 1 en k = 10, "
              f"min(lambda - v) = {min(dv):+.3e}, máx. {max(dv):+.3e}, dif. de cotas {dif:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", help="calcula solo este poliedro en este proceso")
    args = ap.parse_args()
    if args.solo:
        compute_row(args.solo)
        return

    chk = Checker("sphere_series")
    for nombre in sphere.NAMES:
        p = sphere.polyhedron(nombre)
        f = sphere.face_fluxes(p)
        ok = p.n - p.E + p.F == 2 and abs(f.sum() - 4 * np.pi) < 1e-10
        if nombre not in ("cuboctaedro", "rombododecaedro"):
            ok &= np.max(np.abs(f - 4 * np.pi / p.F)) < 1e-10
        h = sphere.face_holonomies(p, sphere.phases(p, f))
        res = np.max(np.abs(np.mod(h - f + np.pi, 2 * np.pi) - np.pi))
        chk.check(ok and res < 1e-9, f"{nombre}: V-E+F = {p.n}-{p.E}+{p.F} = 2, suma de flujos "
                  f"{f.sum() / np.pi:.12f} pi, residuo de holonomías {res:.1e}")

    hechos = {r["nombre"] for r in read_rows(SERIE)}
    for nombre in sphere.NAMES:
        if nombre in hechos:
            chk.info(f"{nombre}: ya en data/{SERIE}")
            continue
        chk.info(f"{nombre}: calculando...")
        cod = subprocess.run([sys.executable, __file__, "--solo", nombre]).returncode
        chk.check(cod == 0, f"{nombre}: el proceso hijo termina con código {cod}")
        if cod:
            chk.finish()

    filas = read_rows(SERIE)
    for r in filas:
        nombre = r["nombre"]
        E, N, chi = int(r["E"]), int(r["N"]), int(r["chi"])
        v0, v1 = float(r["v_trivial"]), float(r["v_curvado"])
        chk.check(chi == CHI_ESPERADO[nombre], f"{nombre}: chi' = {chi}")
        orbs = [o for o in read_rows(ORBITAS) if o["nombre"] == nombre]
        b_nulos = [o["b_nulo"] for o in orbs]
        chk.check(b_nulos == [str(nombre != "cuboctaedro")],
                  f"{nombre}: {len(orbs)} órbita(s), grupo de orden {orbs[0]['orden_grupo']}, "
                  f"estabilizador {orbs[0]['estabilizador']}, b nulo = {b_nulos}")
        dif = (float(r["dif_trivial"]), float(r["dif_curvado"]))
        chk.check(max(dif) < TOL_COTAS, f"{nombre}: diferencia de cotas {dif[0]:.1e} (trivial), "
                  f"{dif[1]:.1e} (curvado), {r['cota_curvado']}")
        p = sphere.polyhedron(nombre)
        orbitas = sphere.edge_orbits(p, sphere.symmetry_group(p))
        signos = None
        if not orbitas[0].b_zero:
            signos = np.array([orbitas[0].signs[m] for m in range(p.E)], dtype=float)
        if p.n <= 14:
            d = sphere.ReducedDual(p, np.zeros(p.E), b_signs=signos)
            vh = d.value(1 / (3 * E))
            chk.check(abs(vh - v0) < 1e-8, f"{nombre}: lambda_max en t = 1/(3E) = {vh:.10f} = "
                      f"v trivial {v0:.10f}")
        else:
            chk.check(abs(float(r["t_trivial"]) - 1 / (3 * E)) < 1e-4,
                      f"{nombre}: t trivial = {float(r['t_trivial']):.8f}, 1/(3E) = {1 / (3 * E):.8f}")
        chk.check(abs(float(r["b_trivial"])) < 1e-4,
                  f"{nombre}: b trivial = {float(r['b_trivial']):.1e} (SU(2) da b = 0)")
        chk.check(v1 >= float(r["cota_media"]) and v1 <= v0 + 1e-8,
                  f"{nombre}: Proposición 5 {float(r['cota_media']):.6f} <= v curvado {v1:.6f} "
                  f"<= v trivial {v0:.6f} (N = {N})")

    primal_checks(chk)
    independent_checks(chk)
    convergence_checks(chk)
    gauge_checks(chk)
    perturbation_checks(chk)
    chk.finish()


if __name__ == "__main__":
    main()

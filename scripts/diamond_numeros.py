"""Números del artículo de diamantes que salen de los CSV de data/ (solo lectura).

Artículo (~/fisica/holonomia_diamantes.tex): los números derivados de §6 (cadena de 2), §6.1 (mínimo),
§7 (cadena cerrada) y §8 (límite) que ningún otro guion imprime. No resuelve
ningún SDP: lee los CSV y comprueba cada cifra tal como está escrita en el
artículo, con su redondeo (OK si el valor del CSV, redondeado a las cifras
del texto, da el texto; FALLO si no). Si se regenera un CSV y una cifra
cambia, este guion lo dice.

"v = 1/3" quiere decir |1/3 - v| <= 1e-7 (V_TOL, la tolerancia de todos los
guiones de diamantes); "déficit" es 1/3 - v.

§6, cadena de 2, datos abelianos (data/diamond_chain2_abel.csv):
  - v = 1/3 en los 12 puntos de la recta fA = -fB; v <= 1/3 + 2,7e-8 en los
    181 puntos;
  - en la malla de 13x13, v = 1/3 exactamente en (0, 0) y en los 43 puntos
    con fA, fB >= pi/2 y fA + fB >= 5 pi/4; déficit > 1e-7 en el resto.
§6.1, mínimo (data/diamond_chain2_min_evals.csv, diamond_chain2_min.csv,
diamond_chain2_min_orbita.csv):
  - la caché tiene 1103 evaluaciones (filas) en 1100 puntos distintos, 473 de
    ellas del cribado a tol 1e-5, todas 'optimal';
  - la menor está en la órbita de x* (theta_A = 0, theta_B = pi,
    alpha = pi/2), con tol 1e-8, 9e-11 por debajo de v(x*) = 0,3255461236;
    v(P) - v(x*) = 1,6e-8; déficit de x*, 7,79e-3;
  - respecto del mínimo abeliano (0, pi) de la malla (0,3259024068), x* baja
    v en 3,6e-4;
  - |v - v(x*)| <= 1,7e-9 en los 3 puntos de la órbita (cota: el máximo,
    1,65e-9, redondeado hacia arriba);
  - curvatura dv/eps^2 con theta_A = eps = 0,1 en los 16 puntos de la órbita:
    de 7,2e-3 a 8,9e-3.
§7, cadena cerrada (data/diamond_ring_L<k>_<modo>.csv y
diamond_ring_min_L<k>_<modo>.csv, k = 2, 3, 4):
  - f = 0: v = 1/3 en los 13 valores de psi con k = 2 y 4; con k = 3,
    v = 0,3077932 con variación 7,3e-10 entre los 13;
  - modo opuesto: v = 1/3 en los 169 puntos con k = 2 y 4; con k = 3, déficit
    > 1e-7 en 164 de 169, y los otros 5 en f >= 11 pi/12, psi <= pi/6;
  - modo igual: v = 1/3 en 44 puntos de k = 2 (filas f = 0 y pi y un rincón
    con f >= 3 pi/4 junto a psi = 0, hasta psi = 2 pi/3 en f = 11 pi/12), en 6
    de k = 3 (f >= 11 pi/12, psi >= 5 pi/6) y en 39 de k = 4 (filas f = 0,
    11 pi/12 y pi);
  - tabla: mayor déficit del modo igual (mínimo en psi de cada fila):
    1,084e-2 en (pi/3, 0,189 pi) con k = 2; 2,554e-2 en f = 0 con k = 3;
    8,849e-3 en (5 pi/12, pi) con k = 4;
  - psi* (mínimo en psi de cada fila, modo igual): k = 2, de 0,10 pi a
    0,29 pi para pi/12 <= f <= pi/2 y ~pi para 7 pi/12 <= f <= 11 pi/12;
    k = 3, ~0 para f >= pi/6; k = 4, ~pi para pi/12 <= f <= pi/2 y entre
    0,50 pi y 0,56 pi para 7 pi/12 <= f <= 5 pi/6 ("~" = a 0,005 pi, lo que
    se lee 1,00 pi o 0,00 pi con dos decimales).
§8, límite (data/diamond_limit_A.csv y diamond_limit_B.csv):
  - A: k (1/3 - v) = 0,07662029 en k = 3, 5 y 7, con dispersión 5e-10;
  - B: el déficit no es monótono: sube de k = 3 a k = 4 en 7,8e-4; baja dentro
    de cada paridad; los k impares quedan por debajo de los pares vecinos.
En todos, solo cuentan las filas con estado 'optimal' (las demás son un
FALLO).

Uso: python scripts/diamond_numeros.py   (menos de 1 s, sin SDP)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv

import numpy as np
from scipy.spatial.transform import Rotation

from holonomy.report import DATA

pi = np.pi
THIRD = 1 / 3
V_TOL = 1e-7
PASO = pi / 12          # paso de las mallas de 13 puntos en [0, pi]
CERO_PI = 0.005         # psi/pi "~0" o "~1": a menos de 0,005


def leer(name):
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def cifra(x, txt):
    """True si x, redondeado a las cifras de txt, se escribe txt. txt con coma
    decimal: '0,3255461236' (decimales fijos) o '7,79e-3' (mantisa y
    exponente)."""
    t = txt.replace(",", ".")
    if "e" in t:
        m, e = t.split("e")
        d = len(m.split(".")[1]) if "." in m else 0
        return f"{x:.{d}e}" == f"{float(m):.{d}f}e{int(e):+03d}"
    d = len(t.split(".")[1]) if "." in t else 0
    return f"{x:.{d}f}" == t


def ij(r, a="f", b="psi"):
    """Índices de malla (paso pi/12) de las columnas a y b."""
    return round(float(r[a]) / PASO), round(float(r[b]) / PASO)


def chk_cifra(chk, x, txt, msg):
    chk.check(cifra(x, txt), f"{msg}: {x:.6e} -> «{txt}»")


def cadena2(chk):
    rows = leer("diamond_chain2_abel.csv")
    recta = [r for r in rows if r["caso"] == "fA=-fB"]
    malla = [r for r in rows if r["caso"] == "malla"]
    chk.check(len(recta) == 12 and len(malla) == 169,
              f"§6: {len(recta)} puntos en la recta fA = -fB y {len(malla)} en la malla (12 y 169)")
    dr = max(abs(float(r["v"]) - THIRD) for r in recta)
    chk.check(dr <= V_TOL, f"§6: v = 1/3 en toda la recta fA = -fB: máx |v - 1/3| = {dr:.1e}")
    vmax = max(float(r["v"]) for r in rows)
    chk_cifra(chk, vmax - THIRD, "2,7e-8", f"§6: v <= 1/3 + 2,7e-8 en los {len(rows)} puntos (máx v - 1/3)")
    uno = {ij(r, "fA", "fB") for r in malla if abs(float(r["v"]) - THIRD) <= V_TOL}
    region = {(i, j) for i in range(13) for j in range(13) if i >= 6 and j >= 6 and i + j >= 15}
    chk.check(len(region) == 43 and uno == region | {(0, 0)},
              f"§6: v = 1/3 en la malla exactamente en (0, 0) y en los {len(region)} puntos con "
              f"fA, fB >= pi/2 y fA + fB >= 5 pi/4 ({len(uno)} puntos con v = 1/3)")
    fuera = min(THIRD - float(r["v"]) for r in malla if ij(r, "fA", "fB") not in uno)
    chk.check(fuera > V_TOL, f"§6: déficit positivo en el resto de la malla (mínimo {fuera:.1e} > 1e-7)")
    return {ij(r, "fA", "fB"): float(r["v"]) for r in malla}


def minimo(chk, abel):
    ev = leer("diamond_chain2_min_evals.csv")
    cols = ("theta_A", "theta_B", "gamma", "rx", "ry", "rz")
    claves = {tuple(r[c] for c in cols) for r in ev}
    criba = sum(1 for r in ev if float(r["tol"]) == 1e-5)
    noopt = sum(1 for r in ev if r["estado"] != "optimal")
    chk.check(len(ev) == 1103 and len(claves) == 1100 and criba == 473 and noopt == 0,
              f"§6.1: la caché tiene {len(ev)} evaluaciones en {len(claves)} puntos, {criba} del cribado "
              f"(tol 1e-5), {noopt} no 'optimal' (1103, 1100, 473, 0)")
    best = min(ev, key=lambda r: float(r["v"]))
    x = np.array([float(best[c]) for c in cols])
    Q = Rotation.from_rotvec(x[3:]).as_matrix()
    nB = np.array([np.sin(x[2]), 0.0, np.cos(x[2])])
    alpha = float(np.arccos(np.clip(nB @ Q @ nB, -1, 1)))
    dist = max(abs(x[0]), abs(x[1] - pi), abs(alpha - pi / 2))
    chk.check(dist < 1e-9 and float(best["tol"]) <= 1e-8,
              f"§6.1: la menor evaluación está en la órbita de x* (theta_A, theta_B - pi, alpha - pi/2 "
              f"a {dist:.1e}), con tol {float(best['tol']):.0e}")
    mn = {r["caso"]: r for r in leer("diamond_chain2_min.csv") if r["caso"] in ("xstar", "minimo")}
    vX, vP = float(mn["xstar"]["v"]), float(mn["minimo"]["v"])
    chk_cifra(chk, vX, "0,3255461236", "§6.1: v(x*)")
    chk_cifra(chk, THIRD - vX, "7,79e-3", "§6.1: déficit de x*")
    chk_cifra(chk, vX - float(best["v"]), "9e-11", "§6.1: v(x*) - menor evaluación de la caché")
    chk_cifra(chk, vP - vX, "1,6e-8", "§6.1: v(P) - v(x*)")
    v0pi = abel[(0, 12)]
    chk_cifra(chk, v0pi, "0,3259024068", "§6: mínimo abeliano v(0, pi) de la malla")
    chk_cifra(chk, v0pi - vX, "3,6e-4", "§6.1: v(0, pi) - v(x*), lo que baja la orientación relativa")
    orb = leer("diamond_chain2_min_orbita.csv")
    num = ("thA", "dthB", "dalpha")
    plano = [abs(float(r["dv"])) for r in orb if all(float(r[c]) == 0 for c in num)]
    # Cota escrita redondeando hacia arriba a dos cifras: 1,6e-9 < máx <= 1,7e-9.
    chk.check(len(plano) == 3 and 1.6e-9 < max(plano) <= 1.7e-9,
              f"§6.1: |v - v(x*)| <= 1,7e-9 en los {len(plano)} puntos de la órbita "
              f"(máx {max(plano):.3e}, redondeado hacia arriba)")
    curv = [float(r["dv"]) / 0.1 ** 2 for r in orb
            if float(r["thA"]) == 0.1 and float(r["dthB"]) == 0 and float(r["dalpha"]) == 0]
    chk.check(len(curv) == 16, f"§6.1: {len(curv)} puntos de la órbita con theta_A = 0,1 (16)")
    chk_cifra(chk, min(curv), "7,2e-3", "§6.1: curvatura dv/eps^2 en theta_A, la menor de los 16")
    chk_cifra(chk, max(curv), "8,9e-3", "§6.1: curvatura dv/eps^2 en theta_A, la mayor de los 16")


def cerrada(chk):
    mapa, mins = {}, {}
    for k in (2, 3, 4):
        for modo in ("igual", "opuesto"):
            rows = leer(f"diamond_ring_L{k}_{modo}.csv")
            mr = leer(f"diamond_ring_min_L{k}_{modo}.csv")
            bad = sum(1 for r in rows + mr if r["estado"] != "optimal")
            chk.check(len(rows) == 169 and bad == 0,
                      f"§7: k = {k}, {modo}: {len(rows)} puntos, {bad} filas no 'optimal' (169, 0)")
            mapa[k, modo] = {ij(r): float(r["v_max"]) for r in rows}
            mins[k, modo] = {round(float(r["f"]) / PASO): (float(r["psi_min"]) / pi, float(r["deficit"]))
                             for r in mr}
    uno = {km: {p for p, v in m.items() if abs(v - THIRD) <= V_TOL} for km, m in mapa.items()}

    # Sin flujo de triángulo.
    for k in (2, 4):
        f0 = [v for (i, _), v in mapa[k, "igual"].items() if i == 0]
        chk.check(len(f0) == 13 and max(abs(v - THIRD) for v in f0) <= V_TOL,
                  f"§7: f = 0, k = {k}: v = 1/3 en los {len(f0)} valores de psi")
    f0 = [v for (i, _), v in mapa[3, "igual"].items() if i == 0]
    chk_cifra(chk, float(np.mean(f0)), "0,3077932", "§7: f = 0, k = 3: v (media de los 13 psi)")
    chk_cifra(chk, max(f0) - min(f0), "7,3e-10", "§7: f = 0, k = 3: variación entre los 13 valores de psi")

    # Modo opuesto.
    for k in (2, 4):
        chk.check(len(uno[k, "opuesto"]) == 169, f"§7: opuesto, k = {k}: v = 1/3 en "
                  f"{len(uno[k, 'opuesto'])} de 169 puntos (169)")
    u3 = uno[3, "opuesto"]
    chk.check(169 - len(u3) == 164 and len(u3) == 5 and all(i >= 11 and j <= 2 for i, j in u3),
              f"§7: opuesto, k = 3: déficit > 1e-7 en {169 - len(u3)} puntos; los otros {len(u3)}, "
              f"{sorted(u3)}, en f >= 11 pi/12, psi <= pi/6")

    # Modo igual.
    u2 = uno[2, "igual"]
    filas = {(i, j) for i in (0, 12) for j in range(13)}
    rincon = u2 - filas
    jmax = {i: max(j for ii, j in rincon if ii == i) for i in {i for i, _ in rincon}}
    ok2 = (len(u2) == 44 and filas <= u2 and all(i >= 9 for i, _ in rincon)
           and all({(i, j) for j in range(jm + 1)} <= rincon for i, jm in jmax.items())
           and jmax.get(11) == 8 and max(jmax.values()) == 8)
    chk.check(ok2, f"§7: igual, k = 2: v = 1/3 en {len(u2)} puntos (44): filas f = 0 y pi y un rincón "
                   f"f >= 3 pi/4 desde psi = 0, hasta psi = 2 pi/3 en f = 11 pi/12 "
                   f"(psi máximo por fila {dict(sorted(jmax.items()))})")
    u3 = uno[3, "igual"]
    chk.check(len(u3) == 6 and all(i >= 11 and j >= 10 for i, j in u3),
              f"§7: igual, k = 3: v = 1/3 en {len(u3)} puntos (6), en f >= 11 pi/12, psi >= 5 pi/6")
    u4 = uno[4, "igual"]
    chk.check(len(u4) == 39 and u4 == {(i, j) for i in (0, 11, 12) for j in range(13)},
              f"§7: igual, k = 4: v = 1/3 en {len(u4)} puntos (39), las filas f = 0, 11 pi/12 y pi")

    # Tabla: mayor déficit del modo igual.
    for k, txt, fi, psi_txt in ((2, "1,084e-2", 4, "0,189"), (3, "2,554e-2", 0, None),
                                (4, "8,849e-3", 5, "1")):
        f_max = max(mins[k, "igual"], key=lambda i: mins[k, "igual"][i][1])
        ps, d = mins[k, "igual"][f_max]
        ok = f_max == fi and cifra(d, txt) and (psi_txt is None or cifra(ps, psi_txt if psi_txt != "1" else "1,00"))
        chk.check(ok, f"§7 (tabla): igual, k = {k}: mayor déficit {d:.4e} en f = {f_max} pi/12, "
                      f"psi* = {ps:.4f} pi -> «{txt}» en f = {fi} pi/12" + (f", psi* «{psi_txt} pi»" if psi_txt else ""))

    # psi* del modo igual.
    m2 = mins[2, "igual"]
    ps = [m2[i][0] for i in range(1, 7)]
    chk.check(cifra(min(ps), "0,10") and cifra(max(ps), "0,29"),
              f"§7: k = 2, psi*/pi de {min(ps):.4f} a {max(ps):.4f} para pi/12 <= f <= pi/2 (0,10 a 0,29)")
    chk.check(all(abs(m2[i][0] - 1) < CERO_PI for i in range(7, 12)),
              "§7: k = 2, psi* ~ pi para 7 pi/12 <= f <= 11 pi/12: "
              + " ".join(f"{m2[i][0]:.4f}" for i in range(7, 12)))
    m3 = mins[3, "igual"]
    chk.check(all(abs(m3[i][0]) < CERO_PI for i in range(2, 13)),
              "§7: k = 3, psi* ~ 0 para f >= pi/6: " + " ".join(f"{m3[i][0]:.4f}" for i in range(2, 13)))
    m4 = mins[4, "igual"]
    chk.check(all(abs(m4[i][0] - 1) < CERO_PI for i in range(1, 7)),
              "§7: k = 4, psi* ~ pi para pi/12 <= f <= pi/2: " + " ".join(f"{m4[i][0]:.4f}" for i in range(1, 7)))
    ps = [m4[i][0] for i in range(7, 11)]
    chk.check(cifra(min(ps), "0,50") and cifra(max(ps), "0,56"),
              f"§7: k = 4, psi*/pi de {min(ps):.4f} a {max(ps):.4f} para 7 pi/12 <= f <= 5 pi/6 (0,50 a 0,56)")


def limite(chk):
    A = {int(r["k"]): r for r in leer("diamond_limit_A.csv")}
    B = {int(r["k"]): r for r in leer("diamond_limit_B.csv")}
    bad = [f"{t}{k}" for t, D in (("A", A), ("B", B)) for k, r in D.items() if r["estado"] != "optimal"]
    chk.check(sorted(A) == [3, 5, 7] and sorted(B) == [2, 3, 4, 5] and not bad,
              f"§8: A en k = {sorted(A)}, B en k = {sorted(B)}, filas no 'optimal': {bad or 'ninguna'}")
    kd = [k * (THIRD - float(A[k]["v_max"])) for k in (3, 5, 7)]
    chk.check(all(cifra(x, "0,07662029") for x in kd),
              "§8: A: k (1/3 - v) = 0,07662029 en k = 3, 5, 7: " + ", ".join(f"{x:.10f}" for x in kd))
    chk_cifra(chk, max(kd) - min(kd), "5e-10", "§8: A: dispersión de k (1/3 - v)")
    d = {k: THIRD - float(B[k]["v_max"]) for k in B}
    chk_cifra(chk, d[4] - d[3], "7,8e-4", "§8: B: el déficit sube de k = 3 a k = 4 (no monotonía)")
    chk.check(d[2] > d[4] and d[3] > d[5] and d[3] < min(d[2], d[4]) and d[5] < d[4],
              f"§8: B: baja dentro de cada paridad y los impares quedan por debajo de los pares vecinos: "
              + ", ".join(f"k = {k}: {d[k]:.4e}" for k in sorted(d)))


def main():
    chk = Checker("diamond_numeros")
    abel = cadena2(chk)
    minimo(chk, abel)
    cerrada(chk)
    limite(chk)
    chk.finish()


if __name__ == "__main__":
    main()

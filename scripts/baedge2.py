"""Coeficiente de 1/n^2, rigidez finita D_n y bordes de la ventana para n grande
con las ecuaciones de Bethe (secciones 6.2 y 6.3, tabla 3, figura 2).

Cada n se guarda en data/bethe_cache/n_XXXX.json, de modo que el cálculo se
puede interrumpir, reanudar y repartir entre procesos:

    python scripts/baedge2.py                 # todos los n de la lista
    python scripts/baedge2.py 1024 2048       # solo esos n (p. ej. en paralelo)
    python scripts/baedge2.py --report        # solo tabla y comprobaciones

Al final escribe data/bethe_large.csv con todos los n disponibles en caché.
"""
import json
import sys
import time

import numpy as np

from holonomy import asymptotics, bethe
from holonomy.report import DATA, Checker, read_csv, write_csv

EVEN = (22, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384,
        512, 640, 768, 1024, 1280, 1536, 2048)
ODD = tuple(n + 1 for n in EVEN if n + 1 <= 1537)
CACHE = DATA / "bethe_cache"

# Tabla 3 publicada: n -> (coef, D_n, 1 + r_n, psi/pi)
PUBLISHED = {
    64: (+1.7848, 0.2687, 1.0096, 0.0906), 256: (+1.7535, 0.2650, 1.0056, 0.1029),
    1024: (+1.7356, 0.2628, 1.0037, 0.1104), 2048: (+1.7289, 0.2619, 1.0031, 0.1134),
    65: (-1.8050, 0.2738, 1.0021, 0.0728), 257: (-1.7709, 0.2687, 1.0016, 0.0904),
    1025: (-1.7478, 0.2654, 1.0011, 0.1018), 1537: (-1.7426, 0.2646, 1.0009, 0.1044),
}
# Tabla 1 (bloque de Bethe): n -> (psi/pi, Delta* en el borde). En n = 2048 la
# versión anterior del artículo daba 0,113387; este código da 0,113388.
PUBLISHED_EDGE = {
    256: (0.102904, 0.99965), 1024: (0.110442, 0.99997), 2048: (0.113388, 0.99999),
    257: (0.090357, 1.00013), 1025: (0.101781, 1.00001), 1537: (0.104378, 1.00000),
}


def compute(n):
    path = CACHE / f"n_{n:04d}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    row = asymptotics.row(bethe.eps, n, slope=bethe.cusp_slope)
    row["seconds"] = time.time() - t0
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1))
    print(f"n={n}: {row['seconds']:.0f} s", flush=True)
    return row


def cached_rows():
    rows = [json.loads(p.read_text()) for p in sorted(CACHE.glob("n_*.json"))]
    return sorted(rows, key=lambda r: r["n"])


def report():
    ck = Checker()
    rows = cached_rows()
    print(f"{'n':>5} {'coef':>9} {'D_n':>7} {'1+r_n':>7} {'psi/pi':>9} {'pred':>9} {'Delta_borde':>11}")
    for r in rows:
        print(f"{r['n']:>5} {r['coef']:+9.4f} {r['D_n']:7.4f} {r['one_plus_r']:7.4f} "
              f"{r['psi_over_pi']:9.6f} {r['psi_pred_over_pi']:9.6f} {r['delta_edge']:11.5f}")
    by_n = {r["n"]: r for r in rows}
    for n, (coef, Dn, opr, psi) in PUBLISHED.items():
        if n not in by_n:
            ck.check(False, f"n={n}: falta en la caché")
            continue
        r = by_n[n]
        got = (round(r["coef"], 4), round(r["D_n"], 4), round(r["one_plus_r"], 4), round(r["psi_over_pi"], 4))
        ck.check(got == (coef, Dn, opr, psi), f"n={n}: tabla 3 {got} vs publicado {(coef, Dn, opr, psi)}")
    for n, (psi, De) in PUBLISHED_EDGE.items():
        if n in by_n:
            r = by_n[n]
            got = (round(r["psi_over_pi"], 6), round(r["delta_edge"], 5))
            ck.check(got == (psi, De), f"n={n}: tabla 1 {got} vs publicado {(psi, De)}")
    # psi/pi = 1 - 3 D_n/(2 eps_n): el error cae con n (Delta* -> 1, d -> 0)
    for parity in (0, 1):
        errs = [(r["n"], abs(r["psi_over_pi"] - r["psi_pred_over_pi"]))
                for r in rows if r["n"] >= 128 and r["n"] % 2 == parity]
        print(f"  error de psi/pi = 1 - 3 D_n/(2 eps_n), {'par' if parity == 0 else 'impar'}: "
              + ", ".join(f"n={int(m)}: {e:.1e}" for m, e in errs))
        big = [e for m, e in errs if m >= 1024]
        ck.check(max(big) < 1e-5, f"{'par' if parity == 0 else 'impar'}: la relación asintótica se cumple a "
                                  f"< 1e-5 para n >= 1024 (máx {max(big):.1e})")
    for parity in (0, 1):
        seq = [r["one_plus_r"] for r in rows if r["n"] % 2 == parity and r["n"] >= 100]
        ck.check(all(a > b for a, b in zip(seq, seq[1:])),
                 f"{'par' if parity == 0 else 'impar'}: 1 + r_n decrece de forma monótona desde n ~ 100")
    if 2048 in by_n:
        print(f"1 - Delta* en la cúspide (Theta = pi), n = 2048: {1 - by_n[2048]['delta_star_cusp_extreme']:.2e}")
    try:
        for r in read_csv("edges_ed.csv"):
            if int(r["n"]) in (8, 16):
                print(f"1 - Delta* en la cúspide (Theta = pi), n = {int(r['n'])}: {1 - r['delta_star_cusp_extreme']:.3f}")
    except FileNotFoundError:
        pass
    write_csv("bethe_large.csv", asymptotics.FIELDS, [[r[k] for k in asymptotics.FIELDS] for r in rows])
    ck.finish()


def main():
    args = sys.argv[1:]
    if "--report" not in args:
        ns = [int(a) for a in args] or sorted(EVEN + ODD)
        for n in ns:
            compute(n)
    report()


if __name__ == "__main__":
    main()

"""¿La sensibilidad s ~ lambda^{2n} es propia de AKLT, de SU(2) o genérica?

Con la regla de carga de (M2) el MPS es A^+ = a sigma^+, A^0 = diag(b1, b2), A^- = c sigma^-.
Reducción a forma canónica:
  * fases físicas diagonales (conmutan con el giro) quitan las fases de a, c y b1;
  * el gauge diag(1, g) deja solo el producto |ac|;
  * la fase de b2/b1 es una espiral: A^s -> A^s D con D = diag(1, e^{i alpha}) multiplica
    la marginal por (1 ⊗ u) con u diagonal, y por el lema del giro
        p_z(n, Theta + n alpha) = p_canon(n, Theta)   (mod 2 pi).
Queda la familia real b1 = 1, b2 = -r, a = c = sqrt(t); AKLT es r = 1, t = 2, el único
punto con SU(2). Al ser real, p_canon es par en Theta y basta Theta en [0, pi]; con
fase compleja (los aleatorios) la malla [0, pi] no ve los extremos para n alpha genérico.

Modos:
  canon    los aleatorios válidos de mps_aleatorio.py, en forma canónica (y una
           comprobación de la espiral en n = 6);
  familia  puntos (r, t) alrededor de AKLT: t != 2 con r = 1 (conserva b1 = -b2) y r != 1.
Salida: data/mps_su2_<modo>.csv. Uso: python scripts/mps_su2.py [canon|familia [nmax] [caso]];
sin argumentos, los dos modos (canon hasta n = 12, familia hasta n = 11). Con caso: solo ese,
salida data/mps_su2_<modo>_<caso>.csv, para lanzar casos en paralelo.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mps_aleatorio import MPSDual, aleatorios, jordan_size, marginal, normalize, tensors  # noqa: E402
from aklt_lanczos import check, sector_ops  # noqa: E402
import aklt_lanczos  # noqa: E402


def canon(z):
    a, b1, b2, c = z
    alpha = np.pi - np.angle(b2 / b1)          # b2 e^{i alpha} / b1 < 0
    return (abs(a), abs(b1), -abs(b2), abs(c)), alpha


def prep(z):
    A, L, mods, E = normalize(tensors(*z))
    s = marginal(A, L)
    w, U = np.linalg.eigh(s)
    k = U[:, w < 1e-12]
    return s - np.eye(9) / 9, k @ k.conj().T, mods, E


def estudiar(name, z, ns, thetas, rows):
    tau, Pk, mods, E = prep(z)
    lam = mods[1]
    print(f"\n{name}: z = {np.round(z, 4)}, |mu| = {np.round(mods, 4)}, lambda = {lam:.5f}, "
          f"J = {jordan_size(E, lam)}", flush=True)
    print("    n    1-p(0)      1-p(pi)     s=max-min    (1-p(0))/lam^2n   s/lam^2n   argmax   cert", flush=True)
    for n in ns:
        ops = [sector_ops(n, M) for M in (0, 1, 2, -1, -2)]
        h_prev, T_prev, ps, cert = None, None, [], 0.0
        for T in thetas:
            d = MPSDual(n, T, ops, tau, Pk)
            c0 = d.ceiling() if h_prev is None else d.rotated(h_prev, T_prev, T)
            try:
                r = d.solve(c0)
            except Exception as e:
                print(f"      n={n}, Theta={T / np.pi:.2f} pi: {type(e).__name__}; se corta este caso")
                return
            h_prev, T_prev = d.h(r["c"]), T
            sep = max(d.top(r["c"], k=1, ops=ops[k])[0][0] / n for k in range(1, 5)) - r["p_sup"]
            if sep >= -1e-9:
                print(f"      n={n}, Theta={T / np.pi:.2f} pi: máximo fuera de M = 0 ({sep:+.1e}); se corta")
                return
            ps.append(r["p_sup"])
            cert = max(cert, r["p_sup"] - r["p_inf"])
            rows.append(dict(caso=name, lam=lam, n=n, Theta=T, p_sup=r["p_sup"], p_inf=r["p_inf"]))
        ps = np.array(ps)
        s = ps.max() - ps.min()
        print(f"  {n:3d}  {1 - ps[0]:.3e}   {1 - ps[-1]:.3e}   {s:.3e}    {(1 - ps[0]) / lam ** (2 * n):10.3e}"
              f"     {s / lam ** (2 * n):9.3e}   {thetas[np.argmax(ps)] / np.pi:.2f}pi  {cert:.1e}", flush=True)
        if cert > 1e-8:
            print(f"      (n={n} sin certificar a 1e-8: se excluye del análisis)")


def correr(modo, nmax, solo):
    ns = list(range(6, nmax + 1))
    thetas = tuple(np.linspace(0, np.pi, 5))
    rows = []
    if modo == "canon":
        al = aleatorios(12)
        # comprobación de la espiral: p_z(6, Theta + 6 alpha) = p_canon(6, Theta)
        z = al["aleatorio1"]
        zc, alpha = canon(z)
        n = 6
        ops = [sector_ops(n, M) for M in (0, 1, 2, -1, -2)]
        err = 0.0
        for T in (0.0, 0.7, 2.0):
            pv = []
            for zz, TT in ((zc, T), (z, T + n * alpha)):
                tau, Pk, _, _ = prep(zz)
                d = MPSDual(n, TT, ops, tau, Pk)
                pv.append(d.solve(d.ceiling())["p_sup"])
            err = max(err, abs(pv[0] - pv[1]))
        check(err < 1e-9, f"espiral: p_z(n, Theta + n alpha) = p_canon(n, Theta), n = 6 (error {err:.1e})")
        casos = {f"{k}_canon": canon(al[k])[0] for k in ("aleatorio1", "aleatorio6", "aleatorio7")}
    else:
        casos = {}
        for r_, t_ in ((1.0, 2.0), (1.0, 1.5), (1.0, 2.5), (1.0, 3.5), (0.8, 2.0), (1.25, 2.0), (0.7, 1.5)):
            casos[f"r{r_:g}_t{t_:g}"] = (np.sqrt(t_), 1.0, -r_, np.sqrt(t_))
    if solo:
        casos = {k: v for k, v in casos.items() if k.startswith(solo)}
    for name, z in casos.items():
        estudiar(name, z, ns, thetas, rows)
    out = ROOT / "data" / (f"mps_su2_{modo}_{solo}.csv" if solo else f"mps_su2_{modo}.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"escrito {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        correr(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        correr("canon", 12, None)
        correr("familia", 11, None)
    sys.exit(0 if aklt_lanczos.OK else 1)

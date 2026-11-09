"""Cota tipo Kohn en la escalera periódica 2 x L: testigo óptimo, su gap y su
respuesta al giro de la holonomía de las patas (exploración).

Artículo: quinto trabajo (holonomía y rigidez), caso escalera de la cota tipo
Kohn (demostración en el docstring de `holonomy.rigidez`); usa la sección 8
de escaleras (límite L -> infinito en f = pi; datos de ladder_limit.py) y se
compara con la Proposición 14 de escaleras (cota de holonomía pequeña, de
orden 1/L), que la cota de aquí no repite.

Para f en {pi, 0} y cada L, el testigo óptimo y de LadderDual(L, dense_max =
100, half = True).v_dual_cg en psi0 = 0, en psi0 = pi y, si es otro, en el psi
de la malla donde v es mínimo (f = pi con L impar: pi/2) (datos de
ladder.abelian_phases(L, f, periodic=True, psi=psi0)), cada uno en frío en un
proceso hijo nuevo (como ladder_limit y diamond_limit). Con él:
  - witness_gap: lambda_max menos el primer autovalor distinto (tolerancia de
    degeneración 1e-8; también con 1e-6, porque en el óptimo del dual
    lambda_max es degenerado entre bloques solo a la precisión de v_dual_cg)
    y el multiplete superior (bloques k <= L);
  - el testigo girado W[Psi] = ladder_twist_y(y, L, Psi) hasta cada psi de la
    malla de ladder_limit (0, pi/4, pi/2, 3pi/4, pi), Psi = psi - psi0:
    v(psi) <= lambda_max(W[Psi]) (cota de Kohn) y la respuesta
    R(Psi) = lambda_max(W[Psi]) - lambda_max(W).
Toda cota superior final (lambda_max de y y de cada testigo girado) se
reevalúa con Lanczos arrancado en un vector aleatorio, r = 4 autovalores y
ncv = 40 (rigidez.lam_max_aleatorio; Review Focus 1 del plan) y se toma el
mayor de los dos valores (los dos son valores de Ritz, que nunca pasan del
verdadero lambda_max); su diferencia se da en la columna dif_aleatorio.

Comprobaciones, en cada (f, L, psi0):
  1. estado 'optimal' de v_dual_cg (cotas a < 1e-8);
  2. v = v(psi0) de data/ladder_limit_L<L>.csv (f = pi) o
     data/ladder_limit_F0_L<L>.csv (f = 0) a < 1e-7;
  3. N(W[Psi]) en los datos de psi = N(W) = 1 a < 1e-9 (normalización del giro);
  4. cota de Kohn: lambda_max(W[Psi]) >= v(psi) del CSV - 1e-8 en los cinco psi;
  5. |dif_aleatorio| < 1e-9 (la reevaluación aleatoria no cambia la cota).

Análisis (también con --analisis, que solo lee data/rigidez_escaleras.csv):
por (f, L), la variación de v en psi de los CSV, var = máx - mín, frente a su
cota de Kohn Kohn_var = R_min(psi_máx - psi_mín) (testigo óptimo del mínimo de
la malla girado hasta el máximo) y su cociente var/Kohn_var <= 1; la
horquilla de las dos caras
    -R_pi(-pi) <= v(pi) - v(0) <= R_0(pi);
R_0(pi/4) y R_0(pi/2); y por paridad de L los exponentes locales
p_L = -ln(x_{L+2}/x_L)/ln((L+2)/L) y los ajustes a e^{-L/xi} y a L^{-p}
(mínimos cuadrados en ln x, residuo cuadrático medio en ln x) de var,
Kohn_var, |v(pi) - v(0)|, |R_0(pi)|, |R_0(pi/2)| y el gap del testigo de
psi0 = 0. Es descriptivo: con 2 a 4 tamaños por paridad cada ajuste de dos
parámetros deja como mucho 2 grados de libertad.

Escribe data/rigidez_escaleras.csv, una fila por (f, L, psi0): v, estado,
diferencia de cotas, estado del solver, gap (tolerancias 1e-8 y 1e-6),
multiplete, cota_<psi> (lambda_max del testigo girado a cada psi), la mayor
dif_aleatorio, y0..y7 (el testigo óptimo, orden de LadderDual.KEYS), picos de
memoria residente y virtual y tiempo. Reanudable: la fila se escribe al
terminar cada punto, y al relanzar se saltan las 'optimal' con los mismos
AJUSTES.

Uso: (ulimit -v 16000000; /usr/bin/time -f "%M KB %e s" python scripts/rigidez_escaleras.py [L ...])
     por defecto L = 4..8;   python scripts/rigidez_escaleras.py --analisis
Medido (2026-11-09, OMP_NUM_THREADS = 2; pico = ru_maxrss del hijo): L = 4, 5: 15 s;
L = 6..8: 209 s (L = 8: 16-43 s y 0,26-0,61 GB por punto); L = 9, 10: 1434 s
(L = 9: 45-199 s, 0,5-2,1 GB; L = 10: 151-417 s, 1,2-4,3 GB, el mayor en f = pi,
psi0 = pi). En total ~27 min: no entra en run_all.sh. Con el CSV escrito, el
análisis (--analisis) tarda 0,5 s.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import multiprocessing as mp
import resource
import sys
import time
import traceback

import numpy as np

from holonomy import ladder
from holonomy.report import DATA, write_csv
from holonomy.rigidez import lam_max_aleatorio, ladder_twist_y, witness_gap

DENSE_MAX = 100
TOL_CG = 1e-8
TOL_REF = 1e-7
TOL_KOHN = 1e-8
TOL_NORM = 1e-9
TOL_ALEAT = 1e-9
SEED = 12345
PSIS = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * np.pi
PSI_NAMES = ("0", "pi4", "pi2", "3pi4", "pi")
PSI0S = (0.0, np.pi)
FS = (("pi", np.pi), ("0", 0.0))
AJUSTES = ";".join([f"dense_max={DENSE_MAX}", "half=True", f"tol_cg={TOL_CG!r}",
                    f"seed={SEED}", "r=4", "ncv=40"]
                   + [f"{a}={b!r}" for a, b in sorted(CLARABEL_OPTS.items())])
NAME = "rigidez_escaleras.csv"
HEADER = (["f", "L", "psi0", "v_max", "estado", "dif_cotas", "estado_solver", "gap",
           "gap_1e-6", "multiplete", "multiplete_1e-6"]
          + [f"cota_{p}" for p in PSI_NAMES] + ["dif_aleatorio", "dif_norma"]
          + [f"y{m}" for m in range(8)] + ["pico_GB", "vmpeak_GB", "tiempo_s", "ajustes"])


def _vmpeak_gb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def _mult(m):
    return " ".join(f"{k}:{q}x{n}" for (k, q), n in m)


def point(args):
    """Un punto en frío: el testigo óptimo en (f, psi0), su gap y sus giros."""
    L, f, psi0 = args
    t0 = time.time()
    D = ladder.LadderDual(L, dense_max=DENSE_MAX, half=True)
    ph0 = ladder.abelian_phases(L, f, periodic=True, psi=psi0)
    v, y, st, dif = D.v_dual_cg(ph0, tol=TOL_CG, **CLARABEL_OPTS)
    out = dict(v=float(v), estado=st, dif=float(dif), solver=D.last_solver_status or "",
               y=[float(c) for c in y])
    if st == "optimal":
        g8, m8 = witness_gap(D, y, tol=1e-8, detalle=True)
        g6, m6 = witness_gap(D, y, tol=1e-6, detalle=True)
        cotas, dal, dno = [], 0.0, 0.0
        n0 = D.norm(y, ph0)
        for psi in PSIS:
            yt = ladder_twist_y(y, L, psi - psi0)
            ph = ladder.abelian_phases(L, f, periodic=True, psi=psi)
            dno = max(dno, abs(D.norm(yt, ph) - n0), abs(n0 - 1))
            lam = D.lam_max(yt)
            lal = lam_max_aleatorio(D, yt, seed=SEED)
            if abs(lal - lam) > abs(dal):
                dal = lal - lam
            cotas.append(max(lam, lal))
        out.update(gap=g8, gap6=g6, mult=_mult(m8), mult6=_mult(m6), cotas=cotas,
                   dif_aleatorio=float(dal), dif_norma=float(dno))
    out.update(pico=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               vmpeak=_vmpeak_gb(), tiempo=time.time() - t0)
    return out


def _child(conn, args):
    try:
        conn.send(("ok", point(args)))
    except BaseException:
        conn.send(("error", traceback.format_exc(limit=3)))
    finally:
        conn.close()


def run_child(args):
    """point(args) en un proceso hijo nuevo; (None, motivo) si muere o lanza."""
    recv, send = mp.Pipe(duplex=False)
    p = mp.get_context("fork").Process(target=_child, args=(send, args))
    p.start()
    send.close()
    try:
        kind, payload = recv.recv()
    except EOFError:
        kind, payload = "dead", None
    recv.close()
    p.join()
    if kind == "dead":
        return None, f"proceso hijo muerto (exitcode {p.exitcode})"
    if kind == "error":
        return None, "excepción en el hijo: " + payload.strip().splitlines()[-1]
    return payload, None


def ref_csv(fname, L):
    """{psi: v} de ladder_limit para (f, L), o None si no existe."""
    name = f"ladder_limit_L{L}.csv" if fname == "pi" else f"ladder_limit_F0_L{L}.csv"
    if not (DATA / name).exists():
        return None, name
    return {round(float(r["psi"]), 9): float(r["v_max"])
            for r in csv.DictReader(open(DATA / name))}, name


def psi0s(ref):
    """0, pi y, si es otro, el psi de la malla donde v es mínimo (f = pi con L
    impar: pi/2): el testigo del mínimo acota la variación máx - mín."""
    out = list(PSI0S)
    if ref is not None:
        pm = PSIS[int(np.argmin([ref[round(p, 9)] for p in PSIS]))]
        if all(abs(pm - p) > 1e-9 for p in out):
            out.append(float(pm))
    return out


def load_rows():
    rows = {}
    if (DATA / NAME).exists():
        for r in csv.DictReader(open(DATA / NAME)):
            if r["estado"] == "optimal" and r["ajustes"] == AJUSTES:
                rows[(r["f"], int(r["L"]), round(float(r["psi0"]), 9))] = r
    return rows


def save(rows):
    keys = sorted(rows, key=lambda k: (k[0] != "pi", k[1], k[2]))
    write_csv(NAME, HEADER, [[rows[k][h] for h in HEADER] for k in keys])


def compute(Ls, chk):
    rows = load_rows()
    chk.info(f"{len(rows)} filas 'optimal' en {NAME} con los ajustes {AJUSTES}")
    for fname, f in FS:
        for L in Ls:
            ref, refname = ref_csv(fname, L)
            for psi0 in psi0s(ref):
                key = (fname, L, round(psi0, 9))
                tag = f"f = {fname}, L = {L}, psi0/pi = {psi0 / np.pi:.2f}"
                if key in rows:
                    chk.info(f"{tag}: en caché")
                    continue
                res, err = run_child((L, f, psi0))
                if res is None:
                    chk.check(False, f"{tag}: {err}")
                    return rows
                ok = res["estado"] == "optimal" and abs(res["dif"]) < TOL_CG
                chk.check(ok, f"{tag}: v = {res['v']:.10f}, cotas a {res['dif']:.1e} "
                          f"({res['estado']}, solver {res['solver']}; {res['tiempo']:.0f} s, "
                          f"pico {res['pico']:.2f} GB, virtual {res['vmpeak']:.2f} GB)")
                if not ok:
                    continue
                if ref is None:
                    chk.info(f"   {refname} no existe: sin contraste ni cota de Kohn")
                else:
                    vr = ref[round(psi0, 9)]
                    chk.check(abs(res["v"] - vr) < TOL_REF,
                              f"   = {refname}: {vr:.10f}, |dif| = {abs(res['v'] - vr):.1e}")
                    viol = [c - ref[round(p, 9)] for c, p in zip(res["cotas"], PSIS)]
                    chk.check(min(viol) >= -TOL_KOHN,
                              f"   Kohn: min_psi [lambda_max(W[psi - psi0]) - v(psi)] = {min(viol):.2e}")
                chk.check(res["dif_norma"] < TOL_NORM, f"   normalización del giro: |N - 1| <= {res['dif_norma']:.1e}")
                chk.check(abs(res["dif_aleatorio"]) < TOL_ALEAT,
                          f"   Lanczos aleatorio: máx |dif| = {abs(res['dif_aleatorio']):.1e}")
                chk.info(f"   gap = {res['gap']:.6e} (tol 1e-6: {res['gap6']:.6e}), "
                         f"multiplete {res['mult']} (tol 1e-6: {res['mult6']})")
                rows[key] = dict(zip(HEADER, [
                    fname, L, repr(psi0), repr(res["v"]), res["estado"], repr(res["dif"]),
                    res["solver"], repr(res["gap"]), repr(res["gap6"]), res["mult"], res["mult6"]]
                    + [repr(float(c)) for c in res["cotas"]]
                    + [repr(res["dif_aleatorio"]), repr(res["dif_norma"])]
                    + [repr(c) for c in res["y"]]
                    + [f"{res['pico']:.3f}", f"{res['vmpeak']:.3f}", f"{res['tiempo']:.1f}", AJUSTES]))
                save(rows)
    return rows


def local_exponents(Ls, xs):
    """p entre L y L + 2 (misma paridad): -ln(x_{L+2}/x_L)/ln((L+2)/L)."""
    d = dict(zip(Ls, xs))
    return [(L, -np.log(d[L + 2] / d[L]) / np.log((L + 2) / L)) for L in Ls if L + 2 in d]


def fits(Ls, xs):
    """Ajustes en ln x: a e^{-L/xi} y a L^{-p}. Devuelve
    ((xi, rms_exp), (p, rms_pot)), rms en ln x; None si hay menos de 3 puntos."""
    Ls, lx = np.asarray(Ls, float), np.log(np.asarray(xs, float))
    if Ls.size < 3:
        return None
    out = []
    for X in (Ls, np.log(Ls)):
        A = np.vstack([np.ones_like(X), X]).T
        c, *_ = np.linalg.lstsq(A, lx, rcond=None)
        rms = float(np.sqrt(np.mean((A @ c - lx) ** 2)))
        out.append((c[1], rms))
    (b_exp, r_exp), (b_pot, r_pot) = out
    return (-1 / b_exp, r_exp), (-b_pot, r_pot)


def analysis(rows):
    """Tabla por (f, L) y exponentes locales y ajustes por paridad de L."""
    print("== análisis (v de ladder_limit; R_a(Psi) = lambda_max(W_a[Psi]) - v(a), W_a el "
          "testigo óptimo de psi0 = a)")
    for fname, _ in FS:
        Ls = sorted({L for (fn, L, _) in rows if fn == fname})
        series = {}
        print(f"-- f = {fname}")
        print("   L   var_psi    Kohn_var   cociente  v(pi)-v(0)  [-R_pi(-pi), R_0(pi)]     "
              "R_0(pi/4)   R_0(pi/2)   gap_0      mult_0")
        for L in Ls:
            ref, _ = ref_csv(fname, L)
            if ref is None:
                continue
            vs = np.array([ref[round(p, 9)] for p in PSIS])
            jmin, jmax = int(np.argmin(vs)), int(np.argmax(vs))
            r0 = rows.get((fname, L, 0.0))
            rp = rows.get((fname, L, round(np.pi, 9)))
            rm = rows.get((fname, L, round(float(PSIS[jmin]), 9)))
            if r0 is None or rp is None or rm is None:
                continue
            var = vs[jmax] - vs[jmin]
            # Testigo del mínimo girado hasta el máximo: var <= K.
            K = float(rm[f"cota_{PSI_NAMES[jmax]}"]) - float(rm["v_max"])
            dv = vs[-1] - vs[0]
            R0 = [float(r0[f"cota_{p}"]) - float(r0["v_max"]) for p in PSI_NAMES]
            Rp = float(rp["cota_0"]) - float(rp["v_max"])  # R_pi(-pi)
            g0 = float(r0["gap"])
            print(f"  {L:2d}  {var:.3e}  {K:.3e}  {var / K:7.4f}  {dv:+.3e}  "
                  f"[{-Rp:+.3e}, {R0[-1]:+.3e}]  {R0[1]:+.3e}  {R0[2]:+.3e}  {g0:.3e}  "
                  f"{r0['multiplete']}")
            for name, x in (("var_psi", var), ("Kohn_var", K), ("|v(pi)-v(0)|", abs(dv)),
                            ("|R_0(pi)|", abs(R0[-1])), ("|R_0(pi/2)|", abs(R0[2])),
                            ("gap_0", g0)):
                series.setdefault(name, []).append((L, x))
        for name, pts in series.items():
            for par, lab in ((0, "pares"), (1, "impares")):
                sel = [(L, x) for L, x in pts if L % 2 == par and x > 1e-12]
                if len(sel) < 2:
                    continue
                Lp, xp = zip(*sel)
                pl = " ".join(f"{L}->{L + 2}: {p:.2f}" for L, p in local_exponents(Lp, xp))
                ft = fits(Lp, xp)
                fs = ("" if ft is None else
                      f"; e^(-L/xi): xi = {ft[0][0]:.3f}, rms {ft[0][1]:.2e}; "
                      f"L^-p: p = {ft[1][0]:.3f}, rms {ft[1][1]:.2e}")
                print(f"   {name:13s} L {lab}: exponentes locales {pl}{fs}")


def main():
    args = sys.argv[1:]
    chk = Checker("rigidez_escaleras")
    if args[:1] == ["--analisis"]:
        analysis(load_rows())
        chk.finish()
    Ls = [int(a) for a in args] or list(range(4, 9))
    rows = compute(Ls, chk)
    analysis(rows)
    chk.finish()


if __name__ == "__main__":
    main()

"""Correcciones de tamaño finito del anillo XXZ con Delta grande (docs/xxz_anillo_retorcido.tex).

Con n par y Delta = 1/epsilon grande, el Teorema de la energía del artículo predice
    |E0(n, 0) + n e_inf(Delta)| <= C (c/Delta)^{n/2},
y el desdoblamiento R_n = E0(n, pi) - E0(n, 0) es del mismo orden. Comprobaciones, con
diagonalización exacta en los sectores de magnetización mínima (n = 4..18 par) y la e_inf exacta
de Yang-Yang:
  1. E0(n, 0) + n e_inf < 0 (el anillo sin flujo queda por debajo de la cadena infinita);
  2. el cociente cada dos sitios de |E0(n, 0) + n e_inf| es < 2/Delta y crece con n (la tasa
     exacta, e^{-1/xi} con xi = 2 xi_JKM, es algo menor que 2/Delta);
  3. con flujo pi la corrección es de orden superior: |E0(n, pi) + n e_inf| <= 0,05 R_n desde n = 10;
  4. la tasa exacta: el cociente del último par de n (16 -> 18) frente a e^{-2/xi} sqrt(16/18), con
     xi = 2 xi_JKM y el prefactor n^{-1/2} de R_n, está a menos de un 2 % de 1.
  5. el orden principal en epsilon = 1/Delta (Observación del Teorema de la energía): en las unidades
     de H~ = (4/Delta) H, [E0(n,0) + n e_inf] 4 epsilon / epsilon^{n/2} -> -n C_{n/2-1} 2^{2-n/2},
     con C_k los números de Catalan, a menos de un 0,1 % con epsilon = 1/160 (n <= 10) y 1/80 (n = 12).
  6. la cota del Teorema de la energía con sus constantes, |E0(n,0) + n e_inf| <= 1,14 n (45/Delta)^{n/2-1},
     con Delta = 50, 100 y 200 y n = 4..14 par, la constante de contracción K(45) <= 1/2 del Lema de
     contracción, K'(54) <= 1/2 del de contracción con flujo, y las constantes del Lema de polímeros:
     M_2 = 85 e^{4 theta} ~ 232 (theta = 64/255), B' ~ 6,3e3, C = 16 (B' + 3,5) ~ 1,0e5, las cotas
     de rho_inf del Lema del límite y kappa_0 = 4,3e-3 < kappa(Delta_inf = 232).
  7. marginales retorcidas (Teorema del estado de prueba): la distancia en norma traza
     delta_n(Theta) = 2|dzz| + 2 max(|dzz|, |dg|) entre la marginal del fundamental del sector S^z = 0
     de H_n(Delta, Theta) (gauge uniforme) y la girada de la cadena infinita (zz = -e_inf',
     g = -e_inf + Delta e_inf') cumple delta_n <= 2 (2/Delta)^{n/2} desde n = 10, con Theta = 0, pi/2
     y pi, y delta_n(pi/2)/delta_n(0) difiere de 1 en menos de un 3 % (uniformidad en Theta).
  8. periodización con flujo: con n = 8 y phi = 0,37, los coeficientes g^(m) del anillo son la suma de
     los de Z (anillo de 24 sitios) sobre los levantamientos hasta m = n/2 - 1 = 3, y no en m = 4.
Escribe data/xxz_ising_tamfinito.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from holonomy import xxz  # noqa: E402
from neel_bethe import eps_inf  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


def xi_jkm(D):
    g = np.arccosh(D)
    m = np.arange(1, 400)
    return 1 / (g + 2 * np.sum((-1.0) ** m / m * np.tanh(m * g)))


rows = []
NS = list(range(4, 19, 2))
for D in [3.0, 5.0, 8.0, 12.0]:
    einf = eps_inf(D)
    d0, dpi, R = [], [], []
    for n in NS:
        e0 = -n * xxz.eps(n, D, 0.0)
        epi = -n * xxz.eps(n, D, np.pi)
        d0.append(e0 + n * einf)
        dpi.append(epi + n * einf)
        R.append(epi - e0)
        rows.append(dict(Delta=D, n=n, E0_0_mas_n_einf=d0[-1], E0_pi_mas_n_einf=dpi[-1], R_n=R[-1]))
    d0, dpi, R = map(np.array, (d0, dpi, R))
    rat = np.abs(d0[1:] / d0[:-1])
    print(f"\nDelta = {D}: |E0(n,0)+n e_inf| = " + ", ".join(f"{x:.2e}" for x in np.abs(d0)))
    print("  cociente cada dos sitios / (2/Delta): " + ", ".join(f"{x:.4f}" for x in rat / (2 / D)))
    check(np.all(d0 < 0), f"Delta = {D}: E0(n,0) + n e_inf < 0 en n = 4..18")
    check(np.all(rat < 2 / D) and np.all(np.diff(rat) > 0),
          f"Delta = {D}: cociente cada dos sitios < 2/Delta y creciente")
    sel = np.array(NS) >= 10
    check(np.all(np.abs(dpi[sel]) <= 0.05 * R[sel]),
          f"Delta = {D}: |E0(n,pi) + n e_inf| <= 0,05 R_n desde n = 10 "
          f"(máximo {np.max(np.abs(dpi[sel]) / R[sel]):.1e})")
    tasa = np.exp(-2 / (2 * xi_jkm(D)))
    q = rat[-1] / (tasa * np.sqrt(NS[-2] / NS[-1]))
    check(abs(q - 1) < 0.02, f"Delta = {D}: cociente(n=16->18) / [e^(-2/xi) sqrt(16/18)] = {q:.4f} "
          f"(e^(-2/xi) = {tasa:.5f}, 2/Delta = {2 / D:.5f})")

# --- 5: orden principal (números de Catalan) ---------------------------------------
from math import comb  # noqa: E402

print()
for n, D in [(4, 160.0), (6, 160.0), (8, 160.0), (10, 160.0), (12, 80.0)]:
    k = n // 2
    eps = 1 / D
    d = (-n * xxz.eps(n, D, 0.0) + n * eps_inf(D)) * 4 / D
    pred = -n * (comb(2 * k - 2, k - 1) // k) / 2 ** (k - 2)
    check(abs(d / eps ** k / pred - 1) < 1e-3,
          f"n = {n}: coeficiente de epsilon^{k} = {d / eps ** k:.4f} (Catalan: {pred})")

# --- 6: cota del teorema con constantes explícitas ---------------------------------
def K(M):
    d = 8 / M
    return np.exp(d) - 1 + d * np.exp(d) + (3 + d) * np.exp(d) / M


def B(M):
    s4 = 4 / M
    return 2 * ((np.exp(s4) - 1 - s4) + (1 + np.exp(s4)) / M)


print()
check(K(45) <= 0.5 < K(44), f"K(45) = {K(45):.4f} <= 1/2 < K(44) = {K(44):.4f}")
check(B(45) * 45 / 4 <= 1.14, f"B(45) * 45/4 = {B(45) * 45 / 4:.4f} <= 1,14")


def Kflujo(M):
    d = 8 / M
    return (np.exp(d) - 1 + d * np.exp(d) + (3 + d) * np.exp(d) / M
            + np.pi * (1.5 + d / 2) * np.exp(d / 2) / M)


check(Kflujo(54) <= 0.5 < Kflujo(53),
      f"contracción con flujo: K'(54) = {Kflujo(54):.4f} <= 1/2 < K'(53) = {Kflujo(53):.4f}")


def M2(Mb):
    return Mb * np.exp(4 * 64 / (3 * Mb))


th = 64 / (3 * 85)
Bp = 128 * np.exp(10 * th) / th
check(Kflujo(85) <= 0.5 and abs(M2(85) - 232) < 0.5 and M2(85) <= min(M2(m) for m in range(54, 400)),
      f"polímeros: M_2(85) = {M2(85):.2f} (mínimo entero de M e^(256/(3M))), K'(85) = {Kflujo(85):.3f}")
C2 = 16 * (Bp + 3.5)
check(abs(Bp / 6.3e3 - 1) < 0.01 and abs(C2 / 1.0e5 - 1) < 0.01,
      f"B' = {Bp:.0f} ~ 6,3e3, C = 16 (B' + 3,5) = {C2:.3g} ~ 1,0e5")
# Lema del límite (i): |e^KT + 1| <= B(45)/2 en |eps| < 1/45; Cauchy en |eps| <= 1/232.
dKT = B(45) / 2 / (1 / 45 - 1 / 232)
zmax, tmax = (1 + B(45) / 2 + dKT / 232) / 4, dKT / 4
check(B(45) / 2 <= 0.0506 and dKT <= 2.83 and zmax <= 0.266 and tmax <= 0.71
      and 1 + 4 * zmax + 2 * tmax <= 3.5,
      f"rho_inf: |e_KT'| <= {dKT:.3f}, |z| <= {zmax:.4f}, |t| <= {tmax:.3f}, suma <= {1 + 4*zmax + 2*tmax:.3f}")


# kappa_0: Delta_inf(kappa) = 232 por la tangencia de Yang--Yang (forma estable de delta_inf_kappa.py).
def g_dg(D):
    qq = 1 / (D + np.sqrt(D * D - 1))
    MS = np.arange(1, int(40 / -np.log(qq)) + 2)
    t = qq ** (2 * MS)
    S = np.sum(t / (1 + t))
    dS = np.sum(2 * MS * qq ** (2 * MS - 1) / (1 + t) ** 2)
    g = -qq / 2 + (1 / qq - qq) * S
    return g, (-0.5 + (-1 / qq ** 2 - 1) * S + (1 / qq - qq) * dS) / ((1 - 1 / qq ** 2) / 2)


gg, dg = g_dg(232.0)
k232 = (gg - 232.0 * dg) / (0.5 + 2 * dg)
check(k232 > 4.3e-3, f"kappa(Delta_inf = 232) = {k232:.6e} > kappa_0 = 4,3e-3")
peor = 0.0
for D in [50.0, 100.0, 200.0]:
    for n in range(4, 15, 2):
        d = abs(-n * xxz.eps(n, D, 0.0) + n * eps_inf(D))
        peor = max(peor, d / (1.14 * n * (45 / D) ** (n / 2 - 1)))
check(peor <= 1, f"|E0 + n e_inf| / [1,14 n (45/Delta)^(n/2-1)] <= {peor:.2e} (Delta = 50..200, n = 4..14)")

# --- 7: marginales retorcidas ------------------------------------------------------
def de_inf(D, h=1e-6):
    return (eps_inf(D + h) - eps_inf(D - h)) / (2 * h)


print()
for D in [5.0, 8.0, 12.0]:
    zzi, gi = -de_inf(D), -eps_inf(D) + D * de_inf(D)
    for n in range(10, 19, 2):
        diag, Kop = xxz.sector_operators(n, n // 2)
        dl = {}
        for Th in [0.0, np.pi / 2, np.pi]:
            _, v = xxz.ground_state(n, D, Th)
            zz = np.real(np.vdot(v, diag * v)) / n
            g = np.exp(1j * Th / n) * np.vdot(v, Kop @ v) / n
            dzz, dg = abs(zz - zzi), abs(g - gi)
            dl[Th] = 2 * dzz + 2 * max(dzz, dg)
        cota = 2 * (2 / D) ** (n / 2)
        check(max(dl.values()) <= cota and abs(dl[np.pi / 2] / dl[0.0] - 1) < 0.03,
              f"Delta = {D}, n = {n}: delta_n = {dl[0.0]:.2e}, {dl[np.pi / 2]:.2e}, {dl[np.pi]:.2e} "
              f"<= 2 (2/Delta)^(n/2) = {cota:.2e}")

# --- 8: periodización con flujo (Lema de periodización con flujo) --------------------
def kt_coefs(n, MMAX, phi):
    """Coeficientes de Taylor g^(m) del punto fijo de F_{n,phi} en el anillo de n sitios."""
    bond = [frozenset({k, (k + 1) % n}) for k in range(n)]
    Lam = frozenset(range(n))
    sgn = [(-1) ** k for k in range(n)]
    c, sn = np.cos(phi), np.sin(phi)

    def mul(A, B):
        C = {}
        for x, a in A.items():
            for y, b in B.items():
                C[x ^ y] = C.get(x ^ y, 0) + a * b
        return C

    def add(A, B, f=1):
        C = dict(A)
        for k, v in B.items():
            C[k] = C.get(k, 0) + f * v
        return C

    g = [dict() for _ in range(MMAX + 1)]
    selB = lambda l, j: {X: v for X, v in g[l].items() if len(X & bond[j]) == 1}
    selS = lambda l, j: {X: v for X, v in g[l].items() if j in X}
    QB = [[{frozenset(): 1}] for _ in range(n)]
    QS = [[{frozenset(): 1}] for _ in range(n)]

    def nextQ(Q, sel, j, m, upto):
        R = {}
        for l in range(1, upto + 1):
            R = add(R, mul(sel(l, j), Q[j][m - l]), l / m)
        return R

    for m in range(1, MMAX + 1):
        N = {}
        for j in range(n):
            N = add(N, nextQ(QB, selB, j, m, m - 1), -1)
            N = add(N, mul({bond[j]: c}, add(QB[j][m - 1], {frozenset(): 1} if m == 1 else {})))
            N = add(N, mul({bond[j]: -1j * sn * sgn[j]}, add(QS[j][m - 1], QS[(j + 1) % n][m - 1])))
        g[m] = {X: v / sum(len(b & X) == 1 for b in bond)
                for X, v in N.items() if X and X != Lam and abs(v) > 1e-14}
        for j in range(n):
            QB[j].append(nextQ(QB, selB, j, m, m))
            QS[j].append(nextQ(QS, selS, j, m, m))
    return g


print()
nper, phiper = 8, 0.37
gn, gN = kt_coefs(nper, 4, phiper), kt_coefs(3 * nper, 4, phiper)
for m in range(1, 5):
    lift = {}
    for Y, v in gN[m].items():
        X = frozenset(y % nper for y in Y)
        if len(X) == len(Y):
            lift[X] = lift.get(X, 0) + v / 3
    err = max(abs(gn[m].get(X, 0) - lift.get(X, 0)) for X in set(lift) | set(gn[m]))
    esperado = err < 1e-12 if m <= nper // 2 - 1 else err > 0.1
    check(esperado, f"periodización con flujo, n = {nper}, phi = {phiper}, m = {m}: "
          f"max |g_n - suma de levantamientos| = {err:.2e} ({'cumple' if m < nper // 2 else 'falla, como debe'})")

out = ROOT / "data" / "xxz_ising_tamfinito.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nescrito {out.relative_to(ROOT)}")
sys.exit(0 if OK else 1)

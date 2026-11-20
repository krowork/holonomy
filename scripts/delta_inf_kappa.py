"""Testigo límite con kappa pequeño (robustez_mps.tex, Proposición prop:limite (vi)).

Con Delta = cosh(gamma) > 1 y q = e^{-gamma}, la fórmula de Yang-Yang da
    e_inf = sinh(gamma) [1/2 + 2 sum_m q^{2m}/(1 + q^{2m})] - Delta/4 = Delta/4 + q/2 - q^3 + O(q^5),
y la condición de tangencia e_inf'(Delta) (Delta + 2 kappa) = e_inf(Delta) del mínimo de
F_kappa = 4 e_inf/(Delta + 2 kappa) define kappa como función de Delta_inf:
    kappa(Delta) = (e_inf - Delta e_inf') / (2 e_inf') = 2q + 2q^3 + O(q^5).
Invirtiendo, Delta_inf kappa = 1 + kappa^2/2 - 3 kappa^4/4 + 17 kappa^6/16 + O(kappa^8).

Comprobaciones:
  1. el desarrollo en q de e_inf - Delta/4 y de Delta kappa(Delta) (SymPy);
  2. la serie en kappa, invirtiendo la de kappa(q) (SymPy);
  3. Delta_inf(kappa) por minimización directa de F_kappa (en log Delta; el mínimo es plano, tolerancia 1e-5) frente
     a la raíz de la tangencia, para kappa de 0,9 a 1e-3;
  4. Delta_inf kappa -> 1, con el resto de la serie de orden kappa^8 (hasta el redondeo, 1e-11);
  5. los valores de la tabla de robustez_mps.tex (kappa = 0,4; 0,6; 0,8).
Escribe data/delta_inf_kappa.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neel_bethe import eps_inf  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("OK    " if cond else "FALLO ") + msg, flush=True)
    OK &= bool(cond)


# --- 1 y 2: series --------------------------------------------------------------
q, k = sp.symbols("q k", positive=True)
Sig = sum(q ** (2 * m) / (1 + q ** (2 * m)) for m in range(1, 8))
Dq = (1 / q + q) / 2
eq = (1 / q - q) / 2 * (sp.Rational(1, 2) + 2 * Sig) - Dq / 4
deq = sp.diff(eq, q) / sp.diff(Dq, q)
kap_q = sp.series((eq - Dq * deq) / (2 * deq), q, 0, 10).removeO()
check(sp.expand(sp.series(eq - Dq / 4, q, 0, 6).removeO() - (q / 2 - q ** 3 + 2 * q ** 5)) == 0,
      "e_inf - Delta/4 = q/2 - q^3 + 2 q^5 + O(q^7)")
check(sp.expand(sp.series(kap_q, q, 0, 8).removeO() - (2 * q + 2 * q ** 3 - 18 * q ** 5 - 10 * q ** 7)) == 0,
      "kappa(q) = 2q + 2q^3 - 18q^5 - 10q^7 + O(q^9)")
a = sp.symbols("a1:5")
qs = a[0] * k + a[1] * k ** 3 + a[2] * k ** 5 + a[3] * k ** 7
coef = sp.Poly(sp.expand(sp.series(kap_q.subs(q, qs) - k, k, 0, 9).removeO()), k).coeffs()
qk = qs.subs(sp.solve(coef, a, dict=True)[0])
serie = sp.series((Dq * k).subs(q, qk), k, 0, 8).removeO()
check(sp.expand(serie - (1 + k ** 2 / 2 - sp.Rational(3, 4) * k ** 4 + sp.Rational(17, 16) * k ** 6)) == 0,
      "Delta_inf kappa = 1 + kappa^2/2 - 3 kappa^4/4 + 17 kappa^6/16 + O(kappa^8)")
serie_f = sp.lambdify(k, serie)


# --- 3: minimización directa frente a tangencia ------------------------------------
# Forma estable para Delta grande: g = e_inf - Delta/4 = -q/2 + (1/q - q) S(q), S = sum q^{2m}/(1+q^{2m}).


def g_dg(D):
    """g(Delta) y g'(Delta), sin la cancelación de e_inf - Delta/4."""
    qq = 1 / (D + np.sqrt(D * D - 1))  # e^{-arccosh D}, sin cancelación
    MS = np.arange(1, int(40 / -np.log(qq)) + 2)
    t = qq ** (2 * MS)
    S = np.sum(t / (1 + t))
    dS = np.sum(2 * MS * qq ** (2 * MS - 1) / (1 + t) ** 2)
    g = -qq / 2 + (1 / qq - qq) * S
    dg_dq = -0.5 + (-1 / qq ** 2 - 1) * S + (1 / qq - qq) * dS
    dD_dq = (1 - 1 / qq ** 2) / 2
    return g, dg_dq / dD_dq


def kappa_de(D):
    g, dg = g_dg(D)
    return (g - D * dg) / (0.5 + 2 * dg)


def delta_tang(kappa):
    return brentq(lambda D: kappa_de(D) - kappa, 1.01, 1e6, xtol=1e-14, rtol=1e-15)


def delta_min(kappa):
    # F_kappa - 1 = (4 g - 2 kappa)/(Delta + 2 kappa)
    def f(x):
        D = np.exp(x)
        return (4 * g_dg(D)[0] - 2 * kappa) / (D + 2 * kappa)
    r = minimize_scalar(f, bounds=(1e-6, np.log(1e5)), method="bounded", options={"xatol": 1e-13})
    return np.exp(r.x)


for D in [1.2, 2.0, 5.0, 20.0]:
    check(abs(g_dg(D)[0] + D / 4 - eps_inf(D)) < 1e-12, f"forma estable = Yang-Yang en Delta = {D}")


kappas = [0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
rows = []
print(f"\n{'kappa':>7} {'Delta_inf':>12} {'Delta_inf kappa':>16} {'serie':>12} {'resto/k^8':>10}")
for kp in kappas:
    Dt, Dm = delta_tang(kp), delta_min(kp)
    prod, s = Dt * kp, serie_f(kp)
    rows.append(dict(kappa=kp, delta_inf=Dt, delta_min=Dm, producto=prod, serie=s))
    print(f"{kp:7.3f} {Dt:12.6f} {prod:16.12f} {s:12.9f} {(prod - s) / kp ** 8:10.3f}")
    check(abs(Dm - Dt) / Dt < 1e-5, f"kappa = {kp}: minimización directa = tangencia ({Dm:.8f})")

# --- 4: límite y resto ---------------------------------------------------------------
peq = [r for r in rows if r["kappa"] <= 0.1]
check(all(abs(r["producto"] - r["serie"]) <= 3 * r["kappa"] ** 8 + 1e-11 for r in peq),
      "|Delta_inf kappa - serie| <= 3 kappa^8 + 1e-11 con kappa <= 0,1")
check(abs(rows[-1]["producto"] - 1 - 0.5e-6) < 1e-9, f"Delta_inf kappa = {rows[-1]['producto']:.10f} en kappa = 1e-3")

# --- 5: valores de la tabla ----------------------------------------------------------
for kp, ref in [(0.4, 2.660), (0.6, 1.850), (0.8, 1.400)]:
    D = next(r["delta_inf"] for r in rows if r["kappa"] == kp)
    check(abs(D - ref) < 6e-4, f"kappa = {kp}: Delta_inf = {D:.4f} (tabla {ref})")

out = ROOT / "data" / "delta_inf_kappa.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nescrito {out.relative_to(ROOT)}")
sys.exit(0 if OK else 1)

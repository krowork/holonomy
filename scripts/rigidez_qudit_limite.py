"""Qudits sin holonomía: v_max(n, 0) en forma cerrada y su límite n → ∞.

Artículo: sexto trabajo (rigidez), sección de qudits (Proposición «qudits sin
holonomía» y su corolario); generaliza \\cite{anillos}, sección 3 (Θ = 0:
v_max = (4q − 1)/3 con el fundamental de Heisenberg) y el límite del Teorema 10.
Marco y notación: holonomy/qudit.py.

Proposición (qudits sin holonomía). Si la holonomía del anillo de n ≥ 3
qudits es trivial (θ = 0 módulo el centro de SU(d)),

    v_max(n, 0) = d/(d+1) · (1/d − e_P(n)),   e_P(n) = E_0(Σ_i P_{i,i+1})/n,

con P el intercambio de dos qudits y la suma sobre las n aristas del anillo.

Demostración. Con holonomía trivial, el gauge lleva los datos a U_e = I en
todas las aristas (qudit.py, como el Lema 1 de \\cite{anillos}): cada arista
lleva ρ(v), y tr(ρ(v) P) = 1/d − v(1 + 1/d).

(≤) Sea ρ factible en v. Como Σ_i P_{i,i+1} es suma de operadores de arista,
    tr(ρ Σ_i P_{i,i+1}) = Σ_i tr(ρ(v) P) = n [1/d − v(1 + 1/d)],
y el primer miembro es ≥ E_0 = n e_P(n). Despejando,
v ≤ d/(d+1)(1/d − e_P(n)). (Es la dualidad débil con el testigo
W = −Σ P + (n/d) I, de norma N = n(d + 1)/d: `qudit.su_d_witness` con θ = 0.)

(≥) Sea |ψ⟩ un fundamental de Σ P, T la traslación y dU la medida de Haar de
SU(d). Se define
    ρ̄ = (1/n) Σ_{k=0}^{n−1} ∫ dU  U^{⊗n} T^k |ψ⟩⟨ψ| T^{−k} U^{†⊗n}.
Σ P conmuta con U^{⊗n} y con T, así que tr(ρ̄ Σ P) = E_0. ρ̄ es invariante
bajo T, que lleva la arista (i, i+1) a (i+1, i+2) conservando el orden de los
dos qudits (también la de cierre), así que todas las aristas tienen el mismo
marginal σ. ρ̄ es invariante bajo U^{⊗n}, así que σ conmuta con U ⊗ U para
todo U ∈ SU(d) y, por la dualidad de Schur–Weyl, σ = αI + βP: los estados
U ⊗ U-invariantes son exactamente la familia de Werner ρ(w),
w ∈ [−(d−1)/(d+1), 1]. De tr(σ P) = E_0/n = e_P(n) sale
w = d/(d+1)(1/d − e_P(n)). Como ρ(w) conmuta con P, la orientación de la
arista no importa, y ρ̄ es factible en v = w. ∎

Observaciones. (i) El testigo óptimo es −Σ P, el de Heisenberg en d = 2
(P = (1 + σ⃗·σ⃗)/2, y la fórmula es v = (4q − 1)/3 con q = (1 − e_P)/2, la de
\\cite{anillos}, sección 3). (ii) Con n = d, el estado totalmente antisimétrico
da P = −1 en cada arista: e_P = −1 y v_max(d, 0) = 1.

Corolario (límite). v_max(n, 0) → v_∞(d) = d/(d+1)(1/d − e_∞(d)) =
[2(ψ(1) − ψ(1/d)) − (d − 1)]/(d + 1), con e_∞(d) = 1 − (2/d)[ψ(1) − ψ(1/d)]
la energía por sitio de Sutherland (1975; Messio–Mila 2012, ec. 4), que se
cita. El límite existe para toda sucesión de n (no solo n múltiplo de d): la
cadena abierta cumple E_ab(m + k) ≤ E_ab(m) + E_ab(k) + 1 (‖P‖ = 1), Fekete da
el límite de E_ab(n)/n, y el anillo difiere de la cadena abierta en un enlace,
|E_0 − E_ab(n)| ≤ 1. En d = 2, v_∞ = (4 ln 2 − 1)/3.

Resultado (tamaño finito; teoría de campos, no demostrado). SU(d)_1 con
velocidad u = 2π/d y carga central c = d − 1 da, para n múltiplo de d,
    e_P(n) = e_∞(d) − π u c/(6n²) + o(1/n²) = e_∞(d) − π²(d−1)/(3d n²) + …,
    v_max(n, 0) = v_∞(d) + π²(d−1)/(3(d+1) n²) + …,
con correcciones logarítmicas (operador marginal) sin amplitud verificada. El
ajuste del guion es descriptivo.

Comprueba:
  1. d = 2: v(n, 0) con la columna Θ = 0 de tab:valores (n = 3..8) y
     v_∞(2) = (4 ln 2 − 1)/3; n = d da v = 1 (d = 2..6);
  2. la fórmula frente a la cota del testigo SU(d) por bloques de color y
     momento (`lam_max_sectors`; d = 3, n = 3..6; d = 4, n = 3..5);
  3. la fórmula frente al primal reducido por el toro (`primal_ring_toro`):
     d = 3, n = 3, 4, 5; d = 4, n = 3, 4 (los dos con v = 1). Con
     --primal-d4n5 añade d = 4, n = 5 (98 s y 13,9 GB de pico: no por defecto);
  4. e_P(n) por sectores de color (d = 2, n ≤ 16; d = 3, n ≤ 12; d = 4,
     n ≤ 9): mínimo sobre todos los sectores, igual al del sector más
     equilibrado; Lanczos (sectores de más de 2000 estados) reevaluado con
     otra semilla aleatoria y con 6 autovalores y 80 vectores, con residuo
     ‖Hx − λx‖; denso frente a Lanczos en los sectores de 2000 a 5000;
  5. v_∞(d) para d = 2..6 (forma con digamma = forma con e_∞);
  6. ajuste descriptivo, n ≡ 0 (mod d): n²(e_P − e_∞) frente a −π²(d−1)/(3d)
     (desviación decreciente y < 5 % en el mayor n), y ajuste de dos puntos
     e_∞ + A/n² con los dos mayores n frente a Sutherland (la extrapolación
     queda al menos 5 veces más cerca de e_∞ que e_P del mayor n).

Escribe data/rigidez_qudit_limite.csv (filas por d y n; filas con n = inf
para el límite). Los v(n, 0) son exactos por la Proposición (no son cotas de
una minimización del dual): con diagonalización exacta, no hay que
etiquetarlos «cota superior».

Uso: python scripts/rigidez_qudit_limite.py [--primal-d4n5]
(19 s y 0,83 GB de pico sin la opción, medidos con /usr/bin/time -f "%M KB %e s";
el pico es el primal de d = 3, n = 5)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, TABLE, TABLE_TOL, Checker

import sys

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.special import digamma

from holonomy import qudit
from holonomy.report import write_csv

TOL = 1e-10
LANCZOS_TOL = 1e-9
PRIMAL_TOL = 5e-6  # Clarabel 'optimal_inaccurate' en v = 1 (óptimo puro, SDP degenerado)
TAMANOS = {2: 16, 3: 12, 4: 9}
DENSE_MAX = 2000


def v_formula(d, eP):
    return d / (d + 1) * (1 / d - eP)


def a_cft(d):
    """Coeficiente de 1/n² de e_P(n): −π u c/6 = −π²(d−1)/(3d)."""
    u, c = 2 * np.pi / d, d - 1
    return -np.pi * u * c / 6


def equilibrado(d, n):
    q, r = divmod(n, d)
    return tuple([q + 1] * r + [q] * (d - r))


def lanczos_residuo(d, n, counts, seed=12345):
    """Menor autovalor del sector con Lanczos de arranque aleatorio, 6
    autovalores y 80 vectores; devuelve (λ, ‖Hx − λx‖)."""
    H = qudit.uls_sector_matrix(d, n, counts)
    dim = H.shape[0]
    if dim <= DENSE_MAX:
        w, V = np.linalg.eigh(H.toarray())
        lam, x = w[0], V[:, 0]
    else:
        rng = np.random.default_rng(seed)
        w, V = eigsh(H, k=6, which="SA", v0=rng.standard_normal(dim), ncv=80, tol=0)
        i = int(np.argmin(w))
        lam, x = w[i], V[:, i]
    return float(lam), float(np.linalg.norm(H @ x - lam * x)), dim


def main():
    ck = Checker("rigidez_qudit_limite")
    rows = []

    # --- 5 (primero, lo usan los demás). Límite para d = 2..6 -------------------
    ck.info("Límite (Corolario): v_∞(d) = d/(d+1)(1/d − e_∞(d))")
    vinf = {}
    for d in range(2, 7):
        einf = qudit.uls_e_inf(d)
        vinf[d] = v_formula(d, einf)
        alt = (2 * (digamma(1.0) - digamma(1.0 / d)) - (d - 1)) / (d + 1)
        ck.check(abs(vinf[d] - alt) < 1e-14,
                 f"d = {d}: e_∞ = {einf:.9f}, v_∞ = {vinf[d]:.9f} (forma digamma {alt:.9f})")
    ck.check(abs(vinf[2] - (4 * np.log(2) - 1) / 3) < 1e-14,
             f"v_∞(2) = {vinf[2]:.12f} = (4 ln 2 − 1)/3 = {(4 * np.log(2) - 1) / 3:.12f}")

    # --- 1. Regresión d = 2 y n = d ---------------------------------------------
    for n in range(3, 9):
        v = v_formula(2, qudit.uls_ground_energy(2, n) / n)
        ck.check(abs(v - TABLE[n][0]) <= TABLE_TOL,
                 f"d = 2, n = {n}: v(n, 0) = {v:.6f} (tab:valores {TABLE[n][0]:.4f})")
    for d in range(2, 7):
        v = v_formula(d, qudit.uls_ground_energy(d, d) / d)
        ck.check(abs(v - 1) < TOL, f"n = d = {d}: v = {v:.12f} (antisimétrico, v = 1)")

    # --- 2. Testigo SU(d) por bloques de color y momento ------------------------
    for d, ns in ((3, range(3, 7)), (4, range(3, 6))):
        for n in ns:
            x = qudit.su_d_witness(d, np.zeros(d))
            cota = qudit.lam_max_sectors(d, n, x)[0] / qudit.ring_norm(d, n, x, np.zeros(d))
            v = v_formula(d, qudit.uls_ground_energy(d, n) / n)
            ck.check(abs(cota - v) < TOL,
                     f"d = {d}, n = {n}: testigo SU(d) {cota:.10f} = fórmula {v:.10f}")

    # --- 3. Primal reducido por el toro ------------------------------------------
    primal = {}
    casos = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4)]
    if "--primal-d4n5" in sys.argv:
        casos.append((4, 5))
    for d, n in casos:
        vp, st, _, diag = qudit.primal_ring_toro(d, n, np.zeros(d), return_state=True,
                                                 **CLARABEL_OPTS)
        v = v_formula(d, qudit.uls_ground_energy(d, n) / n)
        primal[(d, n)] = vp
        ck.check(abs(vp - v) < PRIMAL_TOL and st.startswith("optimal"),
                 f"d = {d}, n = {n}: primal {vp:.9f}, fórmula {v:.9f}, "
                 f"dif {vp - v:+.1e}, λ_min(ρ) {diag['lam_min']:.1e}, "
                 f"residuo {diag['residuo']:.1e}{ck.status_note(st)}")

    # --- 4. e_P(n) por sectores de color, con reevaluación de Lanczos -----------
    ck.info("e_P(n) por sectores de color (mínimo sobre todos los sectores)")
    ck.info(f"{'d':>2} {'n':>3} {'sector':>14} {'dim':>6} {'e_P(n)':>14} {'v(n,0)':>12} "
            f"{'n²(e_P−e_∞)':>12}")
    ep = {}
    for d, nmax in TAMANOS.items():
        einf = qudit.uls_e_inf(d)
        for n in range(3, nmax + 1):
            Ea = qudit.uls_sector_energies(d, n, dense_max=DENSE_MAX, seed=0)
            Eb = qudit.uls_sector_energies(d, n, dense_max=DENSE_MAX, seed=7)
            dif_sem = max(abs(Ea[c] - Eb[c]) for c in Ea)
            E0 = min(Ea.values())
            beq = equilibrado(d, n)
            lam, res, dim = lanczos_residuo(d, n, beq)
            ok = (dif_sem < LANCZOS_TOL and abs(Ea[beq] - E0) < TOL
                  and abs(lam - E0) < LANCZOS_TOL and res < 1e-8)
            if DENSE_MAX < dim <= 5000:
                Hd = qudit.uls_sector_matrix(d, n, beq).toarray()
                lam_d = float(np.linalg.eigvalsh(Hd)[0])
                ok = ok and abs(lam_d - E0) < LANCZOS_TOL
                extra = f", denso {lam_d - E0:+.0e}"
            else:
                extra = ""
            e = E0 / n
            ep[(d, n)] = e
            v = v_formula(d, e)
            dev = n * n * (e - einf)
            ck.check(ok, f"d = {d}, n = {n:2d}, sector {str(beq):>14} ({dim}): "
                         f"e_P = {e:.10f}, v = {v:.10f}, n²(e_P−e_∞) = {dev:+.4f} "
                         f"[semillas {dif_sem:.0e}, Lanczos {lam - E0:+.0e}, "
                         f"residuo {res:.0e}{extra}]")
            vp = primal.get((d, n), np.nan)
            rows.append([d, n, E0, e, v, vp, dev, a_cft(d), einf, vinf[d],
                         int(abs(Ea[beq] - E0) < TOL)])

    # --- 6. Ajuste descriptivo (n ≡ 0 mod d) -------------------------------------
    ck.info("Ajuste (descriptivo), n ≡ 0 (mod d): n²(e_P − e_∞) → −π u c/6 = −π²(d−1)/(3d)")
    for d, nmax in TAMANOS.items():
        einf = qudit.uls_e_inf(d)
        ns = [n for n in range(3, nmax + 1) if n % d == 0]
        devs = [n * n * (ep[(d, n)] - einf) for n in ns]
        rel = [abs(x / a_cft(d) - 1) for x in devs]
        ck.check(all(np.diff(rel) < 0) and rel[-1] < 0.05,
                 f"d = {d}: n = {ns}, n²(e_P − e_∞) = {[round(x, 4) for x in devs]} "
                 f"frente a {a_cft(d):.4f}; desviación relativa "
                 f"{[f'{r:.1%}' for r in rel]}")
        n1, n2 = ns[-2], ns[-1]
        e1, e2 = ep[(d, n1)], ep[(d, n2)]
        efit = (n2**2 * e2 - n1**2 * e1) / (n2**2 - n1**2)
        A = (e1 - efit) * n1**2
        vfit = v_formula(d, efit)
        # Descriptivo: la extrapolación mejora al menos 5 veces el mayor n.
        ck.check(abs(efit - einf) < abs(e2 - einf) / 5,
                 f"d = {d}: ajuste e_∞ + A/n² con n = {n1}, {n2}: e_∞ = {efit:.6f} "
                 f"(Sutherland {einf:.6f}, dif {efit - einf:+.1e}; e_P({n2}) − e_∞ = "
                 f"{e2 - einf:+.1e}), A = {A:.4f} "
                 f"(CFT {a_cft(d):.4f}), v_∞ = {vfit:.6f} (Corolario {vinf[d]:.6f})")

    for d in range(2, 7):
        einf = qudit.uls_e_inf(d)
        rows.append([d, float("inf"), np.nan, einf, vinf[d], np.nan, np.nan, a_cft(d),
                     einf, vinf[d], 1])
    write_csv("rigidez_qudit_limite.csv",
              ["d", "n", "E0", "e_P", "v_n0", "v_primal", "n2_dev", "a_cft", "e_inf",
               "v_inf", "min_en_equilibrado"], rows)
    ck.finish()


if __name__ == "__main__":
    main()

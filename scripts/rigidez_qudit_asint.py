"""Asintótica de qudits con holonomía: coeficiente de 1/n² y ventana por clase n mód d.

Artículo: sexto trabajo (rigidez), sección de qudits, asintótica; generaliza
holonomia_werner.tex (\\cite{anillos}), secciones 6.2 (coeficiente ±π²/6 y
rigidez de espín) y 6.3 (anchura límite de la ventana). Derivación completa,
referencias y tablas: docs/superpowers/notes/2026-11-09-rigidez-asintotica.md.
Funciones: holonomy/qudit_asint.py. Datos de contraste: los de
scripts/rigidez_qudit_flujo.py (tarea 7, data/rigidez_qudit_flujo.csv).

Qué se deriva (Resultado de teoría de campos, no teorema)
--------------------------------------------------------
Familia de testigos −P^s (el −Σ P girado por un flujo de Cartan s) más un campo
de color h de un sitio. Con los modos cero de SU(d)_1 (carga Q en la clase
r = n mód d, corriente λ ∈ Λ_w; rigidez isótropa B = 1/d, la de Λ_w por la
simetría de Weyl) y A = d ε_d/(d² − 1), ε_d = 1/d − e_∞(d):
    lím n²[v(n, 0) − v(n, t)] = d/(d+1) [H_r(0) − H_r(t)]       (`qudit_asint.H_cero`).
  · d = 2: ±π²/6 en Θ = π (n par / impar) y ventana ψ_∞/π = κ_2 =
    1 − 3/(8(ln 2 − ¼)); el campo no actúa (las dos cargas ±½ comparten red).
  · d = 3, n ≡ 0: +2π²/9 en C, +π²/6 en M; ventana = celda de Voronoi de
    2πΛ_w escalada por κ_3 = 1 − 8/(9 ε_3) = 0,14245: θ/π = 2κ_3/3 = 0,0950
    (raíz) y κ_3/√3 = 0,0822 (peso).
  · d = 3, n ≢ 0: el campo de color iguala los tres sectores de carga mínima:
    −π²/9 en C y −π²/18 en M (sin campo sería −π²/6); n ≡ 1 y n ≡ 2 iguales a
    este orden. No hay ventana de un solo nivel para ningún n (Proposición
    exacta, docstring de la nota y comprobación 6); el flujo del testigo queda
    anclado en 0 hasta θ = (2/3)(1 − κ_3) L, se desliza hasta M en
    θ = (2/3 + κ_3/3) L (recta peso); en la raíz, anclado en 0 hasta
    (1 − κ_3)/3·L, meseta en s = L/3 para θ/L ∈ [1/3, 1/3 + (1 − κ_3)/3] y
    llegada a M en (2/3 + κ_3/3) L.

Comprueba
---------
  1. Teoría: κ_2 y v_∞(2) en forma cerrada (1e-13); d = 2, coeficientes ±π²/6
     con y sin campo (1e-7); bordes de la ventana d = 2 (n par y n impar) y
     d = 3, n ≡ 0 (raíz y peso), y del deslizamiento d = 3, n ≢ 0 en la recta
     peso: el estado (anclado o no) cambia entre 0,999 y 1,001 veces el borde
     predicho; d = 3, coeficientes en C y M por clase en forma cerrada (1e-7) y
     r = 1 ≡ r = 2; con n ≢ 0 el óptimo deja siempre ≥ 2 cargas activas.
  2. Anillo SU(3) retorcido (n = 3..12, todos los sectores de color, Lanczos de
     arranque aleatorio con k = 4 y 40 vectores, denso hasta 300 estados):
     E_0(0) = `qudit.uls_ground_energy` (1e-9); gauge: giro de 2π en un color y
     desplazamiento común no cambian E_0 (1e-9), giro nulo = Σ P sin girar;
     consistencia con `qudit.ring_witness`/`ring_norm` (testigo con campo, n = 4, 5,
     1e-10); simetría de cada recta (copia de sectores) frente al cálculo
     completo (n = 6, 1e-10); Lanczos frente a denso (n = 8) y reevaluación con
     otra semilla y 60 vectores en n = 12 (1e-9) (Review Focus 1); un caso con
     un sector no equilibrado activo (n = 4 en M; quitarlo cambia el valor)
     (Review Focus 2).
  3. Campo: n ≡ 0 ⇒ el campo no mejora en C ni en M (1e-9, sector (q, q, q));
     n ≢ 0 ⇒ no mejora en C y sí en M (> 1e-4); v_tarea7 ≤ v_campo + 1e-7 (n ≤ 9
     y la sonda n = 10 en M).
  4. Coeficientes n²[v(n, 0) − v_campo(n, t)] (cota inferior rigurosa del de
     v_max) en C y M: monótonos hacia el límite por clase, sin cruzarlo.
  5. Rigidez local en el punto liso: n ≡ 0, raíz = peso (isotropía de Weyl,
     1e-6 relativo); n ≢ 0 (en M), anisótropa, dada como información.
  6. Proposición (exacta): para n ≢ 0, el multiplete superior de todos los
     óptimos de la tarea 7 ocupa ≥ 2 sectores con 0 en la envolvente convexa de
     sus cargas.
  7. τ = 0 (n = 4..7, SDP con las fases de z fijadas al flujo de Cartan s = t):
     v_tarea7 − 1e-6 ≤ v_τ0 ≤ v_campo + 1e-7 en C y M.
  8. Barrido de la familia con campo en las dos rectas (n = 4, 5, 7, 8; s en
     una malla de 25 puntos): v_familia ≥ v_tarea7 − 1e-7 en los 13 θ.
Información: rigidez de amplitud y de pendiente en las cúspides, ventana de
la familia isótropa (y su calibración en d = 2 frente a tab:ventana de
\\cite{anillos}), fracción ψ(n)/ψ_∞ de la tarea 7 por recta, energías de modos
cero n(E_0 − n e_∞) por clase, y s*/L del barrido frente a θ_W/L.

Escribe data/rigidez_qudit_asint.csv (formato largo: seccion, d, n, r,
punto, recta, magnitud, valor, teoria, tarea7, nota).

Uso: (ulimit -v 16000000; /usr/bin/time -f "%M KB %e s" \\
          .venv/bin/python scripts/rigidez_qudit_asint.py [nmax])   (nmax = 12)
Medido (2026-11-09, OMP_NUM_THREADS = 2, `ulimit -v 16000000`): 110 s y 1,20 GB
de pico (n = 12: ~40 s; τ = 0 con n = 7: ~12 s; barrido: ~20 s). En `run_all.sh`
(FAST).

Resultados (2026-11-09; n²[v(0) − v_campo], cota inferior del coeficiente de v_max)
-----------------------------------------------------------------------------------
  · n ≡ 0, C: 2,853 (3), 2,500 (6), 2,427 (9), 2,396 (12) → 2π²/9 = 2,193;
    M: 2,250, 1,915, 1,846, 1,817 → π²/6. Tarea 7 (v óptimo): C 2,858 (6),
    2,588 (9); M 2,568, 2,136: el exceso sobre la familia (anisotropía de dos
    sitios) decae como n^{−2} (cociente 0,45 de n = 6 a 9; (6/9)² = 0,44).
  · n ≢ 0, C: −2,019 (4), −0,560 (5), −1,708 (7), −0,835 (8), −1,564 (10),
    −0,943 (11) → −π²/9; M: −1,657, +0,043, −1,098, −0,231, −0,959, −0,341
    → −π²/18 (sin campo serían −2,49 … −1,63, hacia −π²/6). n ≡ 1 por debajo y
    n ≡ 2 por encima; medias de pares (4,5), (7,8), (10,11): C −1,289, −1,272,
    −1,253; M −0,807, −0,665, −0,650.
  · Rigidez: local en 0 (n ≡ 0) isótropa, 0,500 (3), 0,4033 (6), 0,3834 (9),
    0,3750 (12) → 1/3; en las cúspides,
    pendiente 0,349 (C) y 0,360 (M) en n = 12 frente a 1/3; local en M con
    n ≢ 0 anisótropa (raíz/peso 0,293/0,530 en n = 4 … 0,420/0,334 en n = 11),
    con la anisotropía decreciendo como ≈ 3/n y cambiando de signo con r.
  · Ventana n ≡ 0 (tarea 7): fracción del límite 0,210/0,211 (n = 6) y
    0,365/0,348 (n = 9) en raíz/peso; cociente raíz/peso 1,146 y 1,209 frente
    a 2/√3 = 1,155. La familia isótropa la sobrestima (0,065 y 0,033 en n = 6),
    como Δ = 1 en d = 2 (0,061 frente a 0,0186 en n = 6).
  · τ = 0 iguala a la tarea 7 en M (todos los n) y en C con n = 6; en C con
    n ≢ 0 deja n²Δv = 0,021, 0,090, 0,014 (n = 4, 5, 7).
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv
import sys
import time
import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from holonomy import qudit
from holonomy import qudit_asint as qa
from holonomy.report import DATA, write_csv

warnings.filterwarnings("ignore", message="Solution may be inaccurate")

D = 3
PI = np.pi
C = 2 * PI / 3 * np.array([1.0, 0.0, -1.0])
M_RAIZ = np.array([PI, 0.0, -PI])
U_RAIZ = np.array([1.0, 0.0, -1.0])               # t = θ·U_RAIZ, L = π
U_PESO = np.array([1.0, 1.0, -2.0]) / np.sqrt(3)  # t = θ·U_PESO, L = π/√3
L_RAIZ, L_PESO = PI, PI / np.sqrt(3)
M_PESO = L_PESO * U_PESO                          # ≡ M_RAIZ (la misma holonomía)
PERM = {"raiz": (2, 1, 0), "peso": (1, 0, 2)}
RECTAS = {"raiz": (U_RAIZ, L_RAIZ), "peso": (U_PESO, L_PESO)}

FILAS = []


def fila(seccion, magnitud, valor, d=D, n="", r="", punto="", recta="", teoria="", tarea7="",
         nota=""):
    FILAS.append([seccion, d, n, r, punto, recta, magnitud, valor, teoria, tarea7, nota])


# ---------------------------------------------------------------------------
# Datos de la tarea 7
# ---------------------------------------------------------------------------

def leer_tarea7():
    path = DATA / "rigidez_qudit_flujo.csv"
    filas = list(csv.DictReader(open(path)))
    malla = {}
    for f in filas:
        if f["tipo"] in ("malla", "sonda"):
            malla[(f["recta"], int(f["n"]), int(f["i_theta"]))] = f
    return filas, malla


def ventana_tarea7(filas, recta, n):
    """[θ del último liso, θ de la primera cúspide] desde θ = 0 (θ/π)."""
    fs = sorted((float(f["theta"]), f["clase"]) for f in filas
                if f["recta"] == recta and int(f["n"]) == n and f["tipo"] in ("malla", "borde"))
    ult = 0.0
    for th, cl in fs:
        if cl != "liso":
            return ult / PI, th / PI
        ult = th
    return ult / PI, np.nan


def cargas_multiplete(texto, n):
    """Cargas Q de los sectores «n0n1n2:k» del multiplete (sin repetir)."""
    out = {}
    for parte in texto.split("|"):
        sec = tuple(int(ch) for ch in parte.split(":")[0])
        if sum(sec) != n:   # n ≥ 10 con algún número de dos cifras: no ocurre (n ≤ 10)
            raise ValueError(texto)
        out[sec] = qa.carga(sec)
    return out


def cero_en_envolvente(Qs):
    Qs = np.array(Qs)
    k = len(Qs)
    res = linprog(np.zeros(k), A_eq=np.vstack([Qs.T, np.ones((1, k))]),
                  b_eq=np.r_[np.zeros(Qs.shape[1]), 1.0], bounds=[(0, None)] * k, method="highs")
    return res.status == 0


# ---------------------------------------------------------------------------
# 1. Teoría de orden dominante
# ---------------------------------------------------------------------------

def anclado(d, r, t, c, tol=1e-5):
    s = qa.H_cero(d, r, t)[1]
    return bool(np.linalg.norm(s - qa.proj(c)) < tol)


def teoria(ck):
    print("=== 1. Teoría de orden dominante")
    for d in (2, 3, 4):
        c = qa.constantes(d)
        ck.info(f"d = {d}: e_∞ = {qudit.uls_e_inf(d):.10f}, ε = {c['eps']:.10f}, "
                f"A = {c['A']:.10f}, B = {c['B']:.10f}, κ = {c['kappa']:.10f}, "
                f"v_∞ = {c['v_inf']:.10f}")
        for k in ("eps", "A", "B", "kappa", "v_inf"):
            fila("teoria", k, c[k], d=d)
    c2 = qa.constantes(2)
    k2 = 1 - 3 / (8 * (np.log(2) - 0.25))
    ck.check(abs(c2["kappa"] - k2) < 1e-13,
             f"d = 2: κ = 1 − 3/(8(ln2 − ¼)) = {k2:.12f} (dif {c2['kappa'] - k2:.1e})")
    ck.check(abs(c2["v_inf"] - (4 * np.log(2) - 1) / 3) < 1e-13, "d = 2: v_∞ = (4 ln2 − 1)/3")

    # d = 2: coeficientes y ventana con las dos paridades.
    tpi = np.array([PI / 2, -PI / 2])
    for r, signo in ((0, +1), (1, -1)):
        for campo in (True, False):
            cf = qa.coef_cero(2, r, tpi, campo)
            ck.check(abs(cf - signo * PI ** 2 / 6) < 1e-7,
                     f"d = 2, n ≡ {r}: lím n²[v(0) − v(π)] = {cf:+.10f} "
                     f"({'+' if signo > 0 else '−'}π²/6; {'con' if campo else 'sin'} campo)")
        fila("teoria", "coef", qa.coef_cero(2, r, tpi), d=2, r=r, punto="Theta=pi",
             teoria=signo * PI ** 2 / 6)
    u2 = np.array([0.5, -0.5])   # t = Θ·u2
    k2v = c2["kappa"]
    # n par: liso en 0, cúspide en Θ = π; n impar: liso en π, cúspide en 0.
    ok_p = (not anclado(2, 0, 0.999 * k2v * PI * u2, tpi)) and anclado(2, 0, 1.001 * k2v * PI * u2, tpi)
    ok_i = (not anclado(2, 1, (PI - 0.999 * k2v * PI) * u2, np.zeros(2))) and \
        anclado(2, 1, (PI - 1.001 * k2v * PI) * u2, np.zeros(2))
    ck.check(ok_p and ok_i, f"d = 2: ventana ψ_∞/π = κ_2 = {k2v:.6f} en las dos paridades "
                            "(anclado/no anclado a 0,999 y 1,001 veces el borde)")
    fila("teoria", "ventana_theta_sobre_pi", k2v, d=2, r=0, recta="Theta", teoria=k2)

    # d = 3: coeficientes por clase.
    c3 = qa.constantes(3)
    k3 = c3["kappa"]
    cerradas = {(0, "C", True): 2 * PI ** 2 / 9, (0, "M", True): PI ** 2 / 6,
                (1, "C", True): -PI ** 2 / 9, (1, "M", True): -PI ** 2 / 18,
                (1, "M", False): -PI ** 2 / 6, (1, "C", False): -PI ** 2 / 9,
                (0, "C", False): 2 * PI ** 2 / 9, (0, "M", False): PI ** 2 / 6}
    nombres = {2 * PI ** 2 / 9: "2π²/9", PI ** 2 / 6: "π²/6", -PI ** 2 / 9: "−π²/9",
               -PI ** 2 / 18: "−π²/18", -PI ** 2 / 6: "−π²/6"}
    for (r, p, campo), val in cerradas.items():
        t = C if p == "C" else M_RAIZ
        cf = qa.coef_cero(3, r, t, campo)
        ck.check(abs(cf - val) < 1e-7, f"d = 3, n ≡ {r}, {p}, {'con' if campo else 'sin'} campo: "
                                       f"coef = {cf:+.10f} ({nombres[val]})")
        fila("teoria", "coef" if campo else "coef_sin_campo", cf, r=r, punto=p, teoria=val)
    dif = max(abs(qa.coef_cero(3, 1, t) - qa.coef_cero(3, 2, t)) for t in (C, M_RAIZ, 0.4 * L_PESO * U_PESO))
    ck.check(dif < 1e-7, f"d = 3: n ≡ 1 y n ≡ 2 dan el mismo H a orden dominante (dif {dif:.1e})")

    # Ventana n ≡ 0: celda de Voronoi de 2πΛ_w escalada por κ_3.
    vr, vp = 2 * k3 / 3, k3 / np.sqrt(3)
    ok_r = (not anclado(3, 0, 0.999 * vr * PI * U_RAIZ, C)) and anclado(3, 0, 1.001 * vr * PI * U_RAIZ, C)
    ok_p = (not anclado(3, 0, 0.999 * vp * PI * U_PESO, M_PESO)) and \
        anclado(3, 0, 1.001 * vp * PI * U_PESO, M_PESO)
    ck.check(ok_r and ok_p, f"d = 3, n ≡ 0: ventana θ/π = 2κ/3 = {vr:.6f} (raíz) y κ/√3 = {vp:.6f} "
                            "(peso)")
    fila("teoria", "ventana_theta_sobre_pi", vr, r=0, recta="raiz", teoria=vr)
    fila("teoria", "ventana_theta_sobre_pi", vp, r=0, recta="peso", teoria=vp)

    # n ≢ 0: anclaje en 0, deslizamiento hasta M (recta peso).
    a, b = 2 * (1 - k3) / 3, 2 / 3 + k3 / 3
    ok = (anclado(3, 1, 0.999 * a * M_PESO, np.zeros(3))
          and not anclado(3, 1, 1.001 * a * M_PESO, np.zeros(3))
          and not anclado(3, 1, 0.999 * b * M_PESO, M_PESO) and anclado(3, 1, 1.001 * b * M_PESO, M_PESO))
    ck.check(ok, f"d = 3, n ≢ 0, recta peso: anclado en 0 hasta θ/L = 2(1−κ)/3 = {a:.6f}, "
                 f"deslizamiento hasta M en θ/L = 2/3 + κ/3 = {b:.6f}")
    fila("teoria", "anclado_en_0_hasta_theta_sobre_L", a, r=1, recta="peso", teoria=a)
    fila("teoria", "anclado_en_M_desde_theta_sobre_L", b, r=1, recta="peso", teoria=b)
    # Recta raíz: anclado en 0 hasta (1 − κ)/3, meseta en s = L/3 en [1/3, 1/3 + (1 − κ)/3].
    a2, b2 = (1 - k3) / 3, 1 / 3 + (1 - k3) / 3
    P3 = L_RAIZ / 3 * U_RAIZ
    ok = (anclado(3, 1, 0.999 * a2 * PI * U_RAIZ, np.zeros(3))
          and not anclado(3, 1, 1.001 * a2 * PI * U_RAIZ, np.zeros(3))
          and anclado(3, 1, 1.001 / 3 * PI * U_RAIZ, P3) and anclado(3, 1, 0.999 * b2 * PI * U_RAIZ, P3)
          and not anclado(3, 1, 1.001 * b2 * PI * U_RAIZ, P3)
          and anclado(3, 1, 1.001 * b * PI * U_RAIZ, M_RAIZ))
    ck.check(ok, f"d = 3, n ≢ 0, recta raíz: anclado en 0 hasta θ/L = (1−κ)/3 = {a2:.6f}, meseta en "
                 f"s = L/3 para θ/L ∈ [1/3, {b2:.6f}], en M desde {b:.6f}")
    fila("teoria", "anclado_en_0_hasta_theta_sobre_L", a2, r=1, recta="raiz", teoria=a2)
    fila("teoria", "meseta_L/3_hasta_theta_sobre_L", b2, r=1, recta="raiz", teoria=b2)
    # Recta raíz y número de cargas activas (n ≢ 0) en un barrido.
    min_act = 9
    perfil = {}
    for recta, (u, L) in RECTAS.items():
        for x in np.linspace(0, 1, 25):
            H, s, h, st, act = qa.H_cero(3, 1, x * L * u)
            min_act = min(min_act, len(act))
            perfil[(recta, round(x, 4))] = float(s @ u / (u @ u) / L)
            fila("teoria", "s_estrella_sobre_L", perfil[(recta, round(x, 4))], r=1, recta=recta,
                 nota=f"theta_sobre_L={x:.4f}")
            H0, s0 = qa.H_cero(3, 0, x * L * u)[:2]
            fila("teoria", "s_estrella_sobre_L", float(s0 @ u / (u @ u) / L), r=0, recta=recta,
                 nota=f"theta_sobre_L={x:.4f}")
    ck.check(min_act >= 2, f"d = 3, n ≢ 0: el óptimo deja ≥ 2 cargas activas en los 50 puntos "
                           f"(mínimo {min_act})")
    ck.info("n ≢ 0, s*/L en la raíz (θ/L: s*/L): " + ", ".join(
        f"{x:.3f}: {perfil[('raiz', round(x, 4))]:.3f}" for x in np.linspace(0, 1, 25)[::2]))
    return k3


# ---------------------------------------------------------------------------
# 2. Anillo SU(3) retorcido
# ---------------------------------------------------------------------------

def comprobaciones_anillo(ck):
    print("=== 2. Anillo SU(3) retorcido: comprobaciones")
    rng = np.random.default_rng(7)
    # Giro nulo = Σ P sin girar.
    for n in (5, 6):
        for c in [(2, 2, 1), (2, 2, 2), (3, 2, 1)]:
            if sum(c) != n:
                continue
            A = qa.uls_torcido(3, n, c, np.zeros(3)).toarray()
            B = qudit.uls_sector_matrix(3, n, c).toarray()
            ck.check(np.max(np.abs(A - B)) == 0, f"n = {n}, sector {c}: giro nulo = Σ P sin girar")
    # Gauge (Review Focus 3).
    for n in (5, 6):
        s = qa.proj(rng.normal(size=3))
        e = qa.energia_min(3, n, s)
        e1 = qa.energia_min(3, n, s + np.array([2 * PI, 0, 0]))
        e2 = qa.energia_min(3, n, s + 0.37 * np.ones(3))
        e3 = qa.energia_min(3, n, s + 2 * PI * np.array([1.0, -1.0, 0.0]))
        dmax = max(abs(e1 - e), abs(e2 - e), abs(e3 - e))
        ck.check(dmax < 1e-9, f"n = {n}: giro de 2π en un color, de 2π(1, −1, 0) y desplazamiento común: "
                              f"E_0 igual (dif {dmax:.1e})")
    # Consistencia con qudit.ring_witness / ring_norm (testigo con campo).
    for n in (4, 5):
        s, t, h = (qa.proj(rng.normal(size=3) * 2) for _ in range(3))
        h = 0.3 * h
        x = qudit.su_d_witness(3, s, n)
        for a_ in range(3):
            for b_ in range(a_, 3):
                x[qudit.param_index(3, "D", a_, b_)] -= h[a_] + h[b_]
        lam = qudit.lam_max_sectors(3, n, x)[0]
        E = qa.energias_torcidas(3, n, s)
        lam2 = n / 3 - min(E[c] + 2 * h @ qa.carga(c) for c in E)
        dn = qudit.ring_norm(3, n, x, t) - qa.norma_iso(3, n, t, s)
        ck.check(abs(lam - lam2) < 1e-10 and abs(dn) < 1e-12,
                 f"n = {n}: λ_max(W) = n/d − mín_N(E_N + 2h·Q_N) ({lam - lam2:.1e}) y N ({dn:.1e})")
    # Copia por simetría de cada recta.
    for recta, (u, L) in RECTAS.items():
        s = 0.37 * L * u
        E1 = qa.energias_torcidas(3, 6, s)
        E2 = qa.energias_torcidas(3, 6, s, perm=PERM[recta])
        dmax = max(abs(E1[c] - E2[c]) for c in E1)
        ck.check(dmax < 1e-10, f"n = 6, recta {recta}: E_N = E_(N∘σ) (copia por simetría, dif {dmax:.1e})")
    # Lanczos frente a denso.
    for c in [(3, 3, 2), (2, 3, 3)]:
        e_d = qa.energia_sector(3, 8, c, C, dense_max=10 ** 6)
        e_l = qa.energia_sector(3, 8, c, C, dense_max=0)
        ck.check(abs(e_d - e_l) < 1e-10, f"n = 8, {c}, en C: Lanczos = denso ({e_l - e_d:.1e})")
    # Review Focus 2: n = 4 en M, sector no equilibrado activo.
    E = qa.energias_torcidas(3, 4, M_RAIZ)
    G, h, act = qa.campo_lp(E)
    Eb = {c: e for c, e in E.items() if max(c) - min(c) <= 1}
    Gb = qa.campo_lp(Eb)[0]
    noeq = [c for c in act if max(c) - min(c) > 1]
    ck.check(bool(noeq) and Gb - G > 1e-3,
             f"n = 4, M: sector no equilibrado activo {noeq}; con solo los equilibrados G sube "
             f"{Gb - G:.4f} (v bajaría por debajo del óptimo)")


def energias_punto(n, t, recta):
    perm = PERM.get(recta)
    return qa.energias_torcidas(3, n, t, perm=perm)


def anillo(ck, ns, t7, filas7, k3):
    print("=== 3. Anillo SU(3) retorcido: rigidez finita, coeficientes y ventana")
    einf = qudit.uls_e_inf(3)
    lim = {(0, "C"): 2 * PI ** 2 / 9, (0, "M"): PI ** 2 / 6, (1, "C"): -PI ** 2 / 9,
           (1, "M"): -PI ** 2 / 18, (2, "C"): -PI ** 2 / 9, (2, "M"): -PI ** 2 / 18}
    coefs = {}
    res = {}
    for n in ns:
        t0 = time.time()
        r = n % 3
        E0 = qa.energias_torcidas(3, n, np.zeros(3))
        e0 = min(E0.values())
        ck.check(abs(e0 - qudit.uls_ground_energy(3, n)) < 1e-9, f"n = {n}: E_0(0) = uls_ground_energy")
        v0 = 0.75 * (1 / 3 - e0 / n)
        modo = (2 * PI ** 2 / 9) * (1 if r else -1)
        fila("energia", "n(E0-n*einf)", n * (e0 - n * einf), n=n, r=r, punto="0", teoria=modo,
             nota="modos cero: -pi^2(d-1)/(3d) + (pi^2/d)·2|omega_r|^2")
        res[n] = dict(v0=v0)
        for p, t, recta in (("C", C, "raiz"), ("M", M_RAIZ, "raiz")):
            E = energias_punto(n, t, recta)
            Emin = min(E.values())
            G, h, act = qa.campo_lp(E)
            vi = 0.75 * (1 / 3 - Emin / n)
            vc = 0.75 * (1 / 3 - G / n)
            res[n][p] = dict(E=E, Emin=Emin, G=G, h=h, act=act, vi=vi, vc=vc)
            ci, cc = n * n * (v0 - vi), n * n * (v0 - vc)
            coefs[(n, p)] = cc
            f7 = t7.get(("raiz", n, 8 if p == "C" else 12))
            v7 = float(f7["v_sup"]) if f7 else np.nan
            c7 = n * n * (v0 - v7) if f7 else np.nan
            res[n][p]["v7"] = v7
            fila("coef", "iso", ci, n=n, r=r, punto=p, teoria=lim[(r, p)], tarea7=c7)
            fila("coef", "campo", cc, n=n, r=r, punto=p, teoria=lim[(r, p)], tarea7=c7,
                 nota="cota inferior rigurosa de n^2[v(0)-v_max]")
            fila("coef", "v_campo", vc, n=n, r=r, punto=p, tarea7=v7)
            fila("energia", "n(Emin-E0)", n * (Emin - e0), n=n, r=r, punto=p)
            fila("energia", "n(G-E0)", n * (G - e0), n=n, r=r, punto=p,
                 nota="activos=" + ";".join("".join(map(str, c)) for c in act))
            if r == 0:
                ck.check(abs(G - Emin) < 1e-9 and act == [(n // 3,) * 3],
                         f"n = {n} ≡ 0, {p}: el campo no mejora (G − E_min = {G - Emin:.1e}), "
                         f"activo {act}")
            elif p == "C":
                ck.check(abs(G - Emin) < 1e-9, f"n = {n} ≢ 0, C: el campo no mejora "
                                               f"({G - Emin:.1e}; tres sectores ya degenerados)")
            else:
                ck.check(G - Emin > 1e-4, f"n = {n} ≢ 0, M: el campo mejora n(G − E_min) = "
                                          f"{n * (G - Emin):.4f}, activos {act}")
            if f7:
                ck.check(v7 <= vc + 1e-7, f"n = {n}, {p}: v_tarea7 = {v7:.7f} ≤ v_campo = {vc:.7f}")
        # Sonda n = 10 de la tarea 7 (M, recta peso).
        if n == 10 and ("peso", 10, 12) in t7:
            v7 = float(t7[("peso", 10, 12)]["v_sup"])
            ck.check(v7 <= res[n]["M"]["vc"] + 1e-7, f"n = 10, M: v_tarea7 (sonda) = {v7:.7f} ≤ "
                                                     f"v_campo = {res[n]['M']['vc']:.7f}")
            res[n]["M"]["v7"] = v7
            fila("coef", "tarea7_sonda", n * n * (v0 - v7), n=n, r=r, punto="M")
        # Rigidez local en el punto liso (0 para n ≡ 0; M para n ≢ 0).
        p0 = np.zeros(3) if r == 0 else M_RAIZ
        e_p = e0 if r == 0 else res[n]["M"]["Emin"]
        Bl = {}
        for recta, (u, L) in RECTAS.items():
            uu = u / np.linalg.norm(u)
            if r and recta == "peso":
                p0r, uu = M_PESO, uu   # mismo punto M en el gauge de la recta peso
                e_pr = min(energias_punto(n, M_PESO, "peso").values())
            else:
                p0r, e_pr = p0, e_p
            dl = 0.02
            b = [n * (min(energias_punto(n, p0r + j * dl * uu, recta).values()) - e_pr) / (j * dl) ** 2
                 for j in (1, 2)]
            Bl[recta] = (4 * b[0] - b[1]) / 3
            fila("rigidez", "B_local", Bl[recta], n=n, r=r, punto="0" if r == 0 else "M",
                 recta=recta, teoria=1 / 3)
        if r == 0:
            ck.check(abs(Bl["raiz"] - Bl["peso"]) < 1e-6 * Bl["raiz"],
                     f"n = {n}: rigidez local isótropa en 0 (Weyl): {Bl['raiz']:.6f} / {Bl['peso']:.6f}")
        else:
            ck.info(f"n = {n}: rigidez local en M (anisótropa): raíz {Bl['raiz']:.4f}, "
                    f"peso {Bl['peso']:.4f} (1/3 a orden dominante)")
        res[n]["Bl"] = Bl
        # Amplitud y pendiente en las cúspides; ventana de la familia isótropa (n ≡ 0).
        if r == 0:
            for recta, cc_, nom in (("raiz", C, "C"), ("peso", M_PESO, "M")):
                Ec = min(energias_punto(n, cc_, recta).values())
                R = float(np.linalg.norm(cc_))
                d1 = 1e-3 * R
                uu = cc_ / R
                Es = [Ec] + [min(energias_punto(n, cc_ - j * d1 * uu, recta).values()) for j in (1, 2)]
                dE = (3 * Es[0] - 4 * Es[1] + Es[2]) / (2 * d1)
                Bamp = n * (Ec - e0) / R ** 2
                Bpen = n * dE / (2 * R)
                y = qa.borde_ventana_iso(3, n, uu, R, Ec, dE)
                psi = y / np.sqrt(2) / PI
                v7a, v7b = ventana_tarea7(filas7, recta, n)
                psi_inf = 2 * k3 / 3 if recta == "raiz" else k3 / np.sqrt(3)
                res[n][("vent", recta)] = (psi, (v7a + v7b) / 2, psi_inf)
                fila("rigidez", "B_amplitud", Bamp, n=n, r=r, punto=nom, recta=recta, teoria=1 / 3)
                fila("rigidez", "B_pendiente", Bpen, n=n, r=r, punto=nom, recta=recta, teoria=1 / 3)
                fila("ventana", "iso_theta_sobre_pi", psi, n=n, r=r, recta=recta, teoria=psi_inf,
                     tarea7=(v7a + v7b) / 2 if n <= 9 else "")
                ck.info(f"n = {n}, {recta}: B_amp = {Bamp:.4f}, B_pend = {Bpen:.4f} (1/3); ventana "
                        f"iso {psi:.5f}, tarea 7 {(v7a + v7b) / 2 if n <= 9 else float('nan'):.5f}, "
                        f"límite {psi_inf:.5f}")
        else:
            # Distancia de la cúspide al punto liso más cercano (los puntos lisos
            # πq_a + 2πΛ_w tienen todos la energía de M): 0 → πq_a, C → M.
            ba = {}
            for nom, R in (("0", PI * np.sqrt(2 / 3)), ("C", PI * np.sqrt(2) / 3)):
                Ec = e0 if nom == "0" else res[n]["C"]["Emin"]
                ba[nom] = n * (Ec - res[n]["M"]["Emin"]) / R ** 2
                fila("rigidez", "B_amplitud", ba[nom], n=n, r=r, punto=nom, recta="raiz",
                     teoria=1 / 3, nota="desde el punto liso mas cercano, sin campo")
            ck.info(f"n = {n}: B_amp (sin campo, desde el punto liso) en 0: {ba['0']:.4f}, "
                    f"en C: {ba['C']:.4f} (1/3)")
        ck.info(f"n = {n} (≡ {r}): coef C iso {n * n * (v0 - res[n]['C']['vi']):+.4f} campo "
                f"{coefs[(n, 'C')]:+.4f}; M iso {n * n * (v0 - res[n]['M']['vi']):+.4f} campo "
                f"{coefs[(n, 'M')]:+.4f}; tarea 7 C {n * n * (v0 - res[n]['C']['v7']):+.4f} "
                f"M {n * n * (v0 - res[n]['M']['v7']):+.4f}  ({time.time() - t0:.1f} s)")
    # Monotonía por clase.
    for r in (0, 1, 2):
        for p in ("C", "M"):
            seq = [(n, coefs[(n, p)]) for n in ns if n % 3 == r]
            if len(seq) < 2:
                continue
            L_ = lim[(r, p)]
            dists = [c - L_ for _, c in seq]
            mono = all(abs(dists[i + 1]) < abs(dists[i]) for i in range(len(dists) - 1))
            lado = all(np.sign(x) == np.sign(dists[0]) for x in dists)
            ck.check(mono and lado, f"n ≡ {r}, {p}: coef campo {[round(c, 4) for _, c in seq]} → "
                                    f"{L_:+.4f} de forma monótona y por un solo lado")
    # Medias de clases consecutivas n ≡ 1, 2 (quitan la alternancia).
    for n in ns:
        if n % 3 == 1 and n + 1 in ns:
            for p in ("C", "M"):
                m = (coefs[(n, p)] + coefs[(n + 1, p)]) / 2
                fila("coef", "media_r1_r2", m, n=f"{n}-{n + 1}", punto=p, teoria=lim[(1, p)])
                ck.info(f"media n = {n}, {n + 1}, {p}: {m:+.4f} (límite {lim[(1, p)]:+.4f})")
    # Review Focus 1: reevaluación en el n mayor.
    nm = max(ns)
    for p, t in (("C", C), ("M", M_RAIZ)):
        E2 = qa.energias_torcidas(3, nm, t, seed=1, k=6, ncv=60)
        dm = max(abs(E2[c] - res[nm][p]["E"][c]) for c in E2)
        ck.check(dm < 1e-9, f"n = {nm}, {p}: todos los sectores reevaluados con otra semilla, "
                            f"k = 6, ncv = 60 (dif máx {dm:.1e})")
    return res, coefs


# ---------------------------------------------------------------------------
# 4. Calibración d = 2 (familia isótropa = Δ = 1)
# ---------------------------------------------------------------------------

TAB_VENTANA = {6: 0.018587, 7: 0.017261, 8: 0.032868, 9: 0.026632, 10: 0.043237, 11: 0.033113,
               12: 0.050947, 13: 0.038055, 14: 0.056863, 15: 0.042025, 16: 0.061534}


def calibracion_d2(ck):
    print("=== 4. Calibración d = 2: familia isótropa (Δ = 1) frente a tab:ventana")
    tpi = np.array([PI / 2, -PI / 2])
    for n in range(4, 17):
        e0 = qa.energia_min(2, n, np.zeros(2))
        ck.check(abs(e0 - qudit.uls_ground_energy(2, n)) < 1e-9, f"d = 2, n = {n}: E_0(0)")
        p, c = (np.zeros(2), tpi) if n % 2 == 0 else (tpi, np.zeros(2))
        Ec, dE, R = qa.pendiente_lateral(2, n, p, c)
        y = qa.borde_ventana_iso(2, n, (c - p) / R, R, Ec, dE)
        psi = y * np.sqrt(2) / PI
        fila("d2", "ventana_iso_Theta_sobre_pi", psi, d=2, n=n, r=n % 2,
             teoria=qa.constantes(2)["kappa"], tarea7=TAB_VENTANA.get(n, ""),
             nota="tarea7 = tab:ventana de anillos (Delta optimizado)")
        fila("d2", "B_pendiente", n * dE / (2 * R), d=2, n=n, r=n % 2, teoria=0.5)
        ck.info(f"d = 2, n = {n}: ψ_iso/π = {psi:.6f}, tab:ventana {TAB_VENTANA.get(n, '—')}, "
                f"B_pend = {n * dE / (2 * R):.4f} (→ 1/2)")


# ---------------------------------------------------------------------------
# 5. τ = 0: fases de z fijadas al flujo de Cartan
# ---------------------------------------------------------------------------

def v_tau0(n, t, s):
    """Mínimo de λ_max/N con z_ab = −r_ab e^{i(s_b − s_a)/n}, r_ab ≥ 0 (τ = 0) y
    D_ab libres: SDP por bloques (`qudit.sector_blocks`). Devuelve (v, estado),
    v = λ_max(W(x))/N(x) del x obtenido (cota superior)."""
    labels, _, ops = qudit.sector_blocks(3, n)
    m = qudit.n_params(3)
    Nv = np.array([qudit.ring_norm(3, n, e, t) for e in np.eye(m)])
    c0v = n * qudit._trace_coeffs(3) / 9
    x = cp.Variable(m)
    tt = cp.Variable()
    cons = [Nv @ x == 1, x[qudit.param_index(3, "D", 0, 0)] == 0]
    phi = np.asarray(s, float) / n
    for (a, b) in qudit._offpairs(3):
        dd = phi[b] - phi[a]
        ir, ii = qudit.param_index(3, "Re", a, b), qudit.param_index(3, "Im", a, b)
        cons += [x[ir] * np.sin(dd) - x[ii] * np.cos(dd) == 0,
                 x[ir] * np.cos(dd) + x[ii] * np.sin(dd) <= 0]
    for O in ops:
        r = O.shape[1]
        Wc = cp.reshape(O.reshape(m, r * r).T @ x, (r, r), order="C")
        Mx = (tt + c0v @ x) * np.eye(r) - Wc
        cons.append((Mx + Mx.H) / 2 >> 0)
    prob = cp.Problem(cp.Minimize(tt), cons)
    prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    xv = np.asarray(x.value)
    return qudit.lam_max_sectors(3, n, xv)[0] / qudit.ring_norm(3, n, xv, t), prob.status


def tau0(ck, res, t7):
    print("=== 5. τ = 0 (fases de Cartan) frente a la tarea 7 y la familia con campo")
    for n in range(4, 8):
        for p, t in (("C", C), ("M", M_RAIZ)):
            v, st = v_tau0(n, t, t)
            v7 = float(t7[("raiz", n, 8 if p == "C" else 12)]["v_sup"])
            vc = res[n][p]["vc"]
            ck.check(v7 - 1e-6 <= v <= vc + 1e-7,
                     f"n = {n}, {p}: v_tarea7 = {v7:.7f} ≤ v_τ0 = {v:.7f} ≤ v_campo = {vc:.7f}"
                     + ("" if st == "optimal" else f"  [{st}]"))
            fila("tau0", "v_tau0", v, n=n, r=n % 3, punto=p, tarea7=v7,
                 nota=f"v_campo={vc:.9f}; n2(v_tau0-v7)={n * n * (v - v7):.3e}; "
                      f"n2(v_campo-v_tau0)={n * n * (vc - v):.3e}")


# ---------------------------------------------------------------------------
# 6. Barrido de la familia con campo en las rectas
# ---------------------------------------------------------------------------

def barrido(ck, t7):
    print("=== 6. Barrido de la familia con campo (s en la recta) frente a la tarea 7")
    peor = np.inf
    for n in (4, 5, 7, 8):
        for recta, (u, L) in RECTAS.items():
            xs = np.linspace(0, L, 25)
            G = [qa.campo_lp(qa.energias_torcidas(3, n, x * u, perm=PERM[recta]))[0] for x in xs]
            linea = []
            for i in range(13):
                th = i * L / 12
                t = th * u
                vals = [(n / 3 - g) / qa.norma_iso(3, n, t, x * u) for g, x in zip(G, xs)]
                j = int(np.argmin(vals))
                f7 = t7[(recta, n, i)]
                v7, thw = float(f7["v_sup"]), float(f7["theta_W"]) / L
                peor = min(peor, vals[j] - v7)
                linea.append(f"{xs[j] / L:.2f}/{thw:+.2f}")
                fila("barrido", "s_estrella_sobre_L", xs[j] / L, n=n, r=n % 3, recta=recta,
                     tarea7=thw, nota=f"i_theta={i}; v_familia={vals[j]:.9f}; v_tarea7={v7:.9f}")
            ck.info(f"n = {n}, {recta}: s*/L (familia) / θ_W/L (tarea 7): " + " ".join(linea))
    ck.check(peor >= -1e-7, f"barrido: v_familia ≥ v_tarea7 en todos los puntos (mín dif {peor:.1e})")


# ---------------------------------------------------------------------------
# 7. Proposición en los óptimos de la tarea 7
# ---------------------------------------------------------------------------

def proposicion(ck, filas7):
    print("=== 7. Proposición: n ≢ 0 ⇒ multiplete superior en ≥ 2 sectores, 0 ∈ conv(Q)")
    tot = ok = 0
    malos = []
    for f in filas7:
        n = int(f["n"])
        if n % 3 == 0:
            continue
        Q = cargas_multiplete(f["multiplete"], n)
        tot += 1
        if len(Q) >= 2 and cero_en_envolvente(list(Q.values())):
            ok += 1
        else:
            malos.append((f["recta"], n, f["theta"], f["multiplete"]))
    ck.check(ok == tot, f"{ok}/{tot} óptimos de la tarea 7 con n ≢ 0 cumplen la Proposición "
                        f"{malos[:3] if malos else ''}")
    fila("proposicion", "filas_que_cumplen", ok, tarea7=tot)


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ns = list(range(3, nmax + 1))
    ck = Checker("rigidez_qudit_asint")
    filas7, t7 = leer_tarea7()
    k3 = teoria(ck)
    comprobaciones_anillo(ck)
    res, coefs = anillo(ck, ns, t7, filas7, k3)
    # Ventanas: fracciones de la tarea 7 y cociente raíz/peso.
    for n in (6, 9):
        if n in res:
            fr = {rc: res[n][("vent", rc)][1] / res[n][("vent", rc)][2] for rc in RECTAS}
            cociente = res[n][("vent", "raiz")][1] / res[n][("vent", "peso")][1]
            fila("ventana", "fraccion_tarea7", fr["raiz"], n=n, r=0, recta="raiz")
            fila("ventana", "fraccion_tarea7", fr["peso"], n=n, r=0, recta="peso")
            fila("ventana", "cociente_raiz_peso_tarea7", cociente, n=n, r=0, teoria=2 / np.sqrt(3))
            ck.info(f"n = {n}: ψ_tarea7/ψ_∞ = {fr['raiz']:.3f} (raíz), {fr['peso']:.3f} (peso); "
                    f"cociente raíz/peso {cociente:.3f} (2/√3 = {2 / np.sqrt(3):.3f})")
    calibracion_d2(ck)
    if nmax >= 7:
        tau0(ck, res, t7)
    if nmax >= 8:
        barrido(ck, t7)
    proposicion(ck, filas7)
    write_csv("rigidez_qudit_asint.csv",
              ["seccion", "d", "n", "r", "punto", "recta", "magnitud", "valor", "teoria", "tarea7",
               "nota"], FILAS)
    ck.finish()


if __name__ == "__main__":
    main()

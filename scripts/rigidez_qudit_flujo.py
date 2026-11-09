"""Qudits con holonomía (d = 3): cúspide y ventana del flujo del testigo óptimo.

Artículo: sexto trabajo (rigidez), sección de qudits; generaliza
holonomia_werner.tex (\\cite{anillos}), secciones 5.3 a 5.5 (flujo óptimo,
cúspide y ventana de desanclaje). Marco, notación y familia de testigos:
holonomy/qudit.py.

Pregunta. En qubits, el flujo óptimo del testigo queda atrapado en una cúspide
de la energía (doblete de corrientes opuestas) en Φ = π para n par y en Φ = 0
para n impar, salvo en una ventana de desanclaje junto al punto liso: eso
explica el efecto de paridad. Conjetura de trabajo para SU(3): la cúspide
depende de n mod 3 (efecto Z_3) y hay ventana. El guion lo pone a prueba y
describe lo que sale, también si no es así.

Holonomías: dos rectas y por qué
--------------------------------
La holonomía t = (t_0, t_1, t_2) (autofases, en el gauge uniforme
U_e = diag(e^{i t_a/n})) solo importa módulo: (a) un desplazamiento común
(fase global; incluye el centro Z_3), (b) 2π en cada componente, (c) las
permutaciones de los colores (gauge constante) y (d) t ↦ −t (conjugación
compleja K). (a) y (b) dejan la parte sin traza de t módulo 2π·Λ_w (red de
pesos de su(3)); con (c), módulo el grupo afín de Weyl, el dominio es la
alcoba, un triángulo equilátero de vértices 0, 2πω_1, 2πω_2 (los tres son el
centro, es decir, holonomía trivial); el centro Z_3 lo gira 120° y K lo
refleja, así que el dominio fundamental es un sexto de la alcoba: el triángulo
de vértices
    0 (holonomía trivial),
    C = (2π/3)(1, 0, −1) (centro de la alcoba: el «reloj» {1, ω, ω²}),
    M = (π, 0, −π) ≡ diag(1, 1, −1) (punto medio de un lado de la alcoba).
Las dos rectas recorren su borde:
  · recta «raiz», t = θ·(1, 0, −1) (dirección de la raíz α_1 + α_2): período
    2π, t(−θ) es la permutación 0 ↔ 2 de t(θ) y t(2π − θ) ≡ t(−θ), así que
    basta θ ∈ [0, π]; pasa de 0 a C (θ = 2π/3) y de C a M (θ = π);
  · recta «peso», t = θ·(1, 1, −2)/√3 (dirección del peso ω_2): en
    θ = 2π/√3, t = (2π/3)(1, 1, −2) ≡ (2π/3)(1, 1, 1), el centro, así que el
    período es 2π/√3; K da v(−θ) = v(θ) y basta θ ∈ [0, π/√3]; es el lado
    0 → M (en θ = π/√3, t ≡ (0, 0, π) ≡ M, el mismo punto que la recta raíz
    en θ = π).
Las dos rectas cubren el borde del dominio fundamental (0–C–M y 0–M), con los
tres puntos especiales y los dos tipos de lado (los segmentos 0–C y C–M son
fronteras de Voronoi de la red de pesos o de la de semipesos; ver abajo), pero
no su interior: lo que pase dentro del triángulo no se muestrea. Malla de 13
puntos θ_i = i·L/12, i = 0..12 (L = π o π/√3); en la recta raíz incluye C
(i = 8). El rango se comprueba con evaluaciones extra (n = 4, 5): v(−θ),
v(2π − θ) en la raíz y v(−θ), v(2π/√3 − θ) en la recta peso, iguales a v(θ).

Simetría residual de cada recta y simetrización. En la recta raíz, g = K∘(0 2)
deja los datos invariantes; en la recta peso, g = (0 1). Sobre x (orden de
qudit.py): una permutación π lleva D_ab a D_{π(a)π(b)} y z_ab a z_{π(a)π(b)}
(conjugado si π invierte el orden del par); K conjuga z. Se comprueba que
λ_max(W(gx))/N(gx) = λ_max(W(x))/N(x) y se usa x̄ = (x + gx)/2, que por
convexidad de λ_max (y N invariante) no es peor; así las degeneraciones que
fuerza la simetría salen exactas. En x̄, z_01 = z_12 (raíz) y z_01 real,
z_02 = z_12 (peso).

Flujo del testigo (convenio)
----------------------------
El término de intercambio z_ab |ab⟩⟨ba| tiene la misma fase en todas las
aristas (gauge uniforme). Se define el flujo del par (a, b) como la fase del
intercambio sumada en la vuelta, con el signo del testigo SU(d) (Y = −P):
    Φ_ab = n·arg(−z_ab)   módulo 2π, en (−π, π].
Para el testigo SU(d) girado por unos datos t, Φ_ab = t_b − t_a; en la recta
raíz los datos dan (Φ_01, Φ_12, Φ_02) = (−θ, −θ, −2θ) y en la recta peso
(0, −√3θ, −√3θ). Se guardan los tres Φ, los |z_ab| y la fase de triángulo
τ = Φ_01 + Φ_12 − Φ_02 (mód. 2π), invariante bajo gauges diagonales sitio a
sitio: si τ = 0 el flujo es abeliano, t_W con (t_W)_b − (t_W)_a = Φ_ab; si no,
el testigo tiene una fase de tres colores que no es un flujo de Cartan. El flujo
del testigo sobre la recta es θ_W = −Φ_01 (raíz) o θ_W = −Φ_02/√3 (peso),
comparable con θ.

Cúspide (convenio). Multiplete superior: todos los autovalores de W, en todos
los bloques (números de color, momento 2πk/n) y con su multiplicidad, a menos
de TOL_DEG = 1e-7 de λ_max (diagonalización densa de cada bloque). El testigo
está en una cúspide si el multiplete superior contiene al menos dos momentos k
distintos (cruce de niveles entre sectores de momento: λ_max no es derivable
en el flujo); «liso» si solo hay un momento y el gap al siguiente nivel (λ_max
menos el mayor autovalor por debajo de λ_max − TOL_DEG) pasa de GAP_LISO =
1e-5; con un solo momento y gap menor, «indet» (se cuenta como cúspide: los
planos de corte fijan v a 1e-9, pero un desdoblamiento que apenas cambia v,
p. ej. con θ → 0 en n = 3, queda sin resolver por debajo de ~1e-6). Se
guardan los sectores, la degeneración (con multiplicidad), los momentos, la
clase (liso, indet, doblete: dos momentos, triple: tres o más) y el gap. Ojo:
en el óptimo del dual (convexo, no derivable) varios bloques empatan casi
siempre; el criterio distingue si entre ellos hay momentos distintos.

Momentos y gauge; criterio invariante. Entre estados de un mismo sector de
color, una diferencia de momento no depende del gauge; entre sectores de color
distintos, sí: un gauge sitio a sitio (giro grande, fase e^{2πi j m_a/n} del
color a en el sitio j) desplaza el momento en 2π Σ_a m_a n_a/n, distinto en
cada sector. El criterio invariante es el desdoblamiento de corrientes: los
autovalores de P·(dW/ds)·P, con P el proyector sobre el multiplete superior y
s el flujo del testigo girado en una dirección de Cartan (z_ab ↦
z_ab e^{i s (u_b − u_a)/n}); si no son todos iguales, λ_max tiene un salto de
derivada en s (cruce, cúspide). Se calcula en la dirección de la recta (u) y
en la transversal sin traza (w: (1, −2, 1)/√3 en la raíz, (1, −1, 0) en la
recta peso) y se guardan los dos espectros (columnas corrientes_u,
corrientes_w) y el mayor de los dos desdoblamientos (desdoblamiento). Los
pares forzados por g en la raíz, (n0, n1, n2, k) ↔ (n2, n1, n0, −k), tienen
corrientes iguales a lo largo de u y opuestas en w. En M (θ = π en la raíz,
θ = π/√3 en la recta peso: la misma holonomía) los momentos del multiplete son
distintos en los dos gauges, y el multiplete se desdobla linealmente a lo largo
de cada recta: la clase en M depende de la dirección en que se sale de M, no
del gauge. Comprobación: toda fila doblete o triple tiene desdoblamiento > 1e-3
(mínimo medido 0,031), y las lisas tienen degeneración 1.

Cúspide forzada (certificación). El óptimo que se clasifica es el
simetrizado x̄, uno de los posibles: el conjunto óptimo es convexo y podría
contener otros. Para excluir que alguno sea liso se usa que, si un óptimo x*
tuviera el multiplete superior en un solo momento k, λ_max coincidiría cerca
de x* con λ_max^(k) (máximo solo sobre los bloques de momento k, todos los
sectores de color), x* sería mínimo local y por convexidad global de
λ_max^(k), y v = mín λ_max^(k). Basta pues, para cada k = 0..n−1, un x
explícito con λ_max^(k)(x)/N(x) < v: para los k fuera del multiplete de x̄ sirve
x̄ (margen = gap); para los del multiplete, la cota superior de
`v_dual_ring_cg(..., momentos=[k])` arrancado en x̄ (columnas cert_k, con el
margen v − sup_k por k del multiplete, y cert_margen, el menor; planos de corte
con tol 1e-9). Comprobación: margen > 1e-7 en toda fila doblete o triple de la
malla y de la sonda, salvo dos excepciones que se dan como información:
  · θ = 0 con n ≢ 0: el margen es 0 (a 1e-15). El multiplete de x̄ (el
    testigo SU(3), óptimo exacto) es un multiplete de color en cada momento
    ±k_0, y el de un solo momento ya fija v (el mínimo restringido a k_0 es v,
    alcanzado en x̄): la prueba no decide si hay otro óptimo con un solo
    momento; la cúspide de θ = 0 para n ≢ 0 es la de x̄, no está certificada
    como forzada.
  · filas de bisección (borde de la ventana): el margen es de segundo orden en
    la distancia δ al borde (n = 9, recta peso: 1,5e-8, 1,0e-7, 5,2e-7 y
    2,3e-6 con δ ≈ 3e-4, 9e-4, 2e-3 y 4e-3 rad; ~1e-16 en el propio borde):
    4 de las 33 filas de bisección en cúspide quedan con margen < 1e-7
    (n = 6, peso: 1; n = 9: 2 en la recta peso y 1 en la raíz); las otras 29
    se certifican.
Excluye todo óptimo liso (dentro de la familia invariante, que no pierde
generalidad) donde se certifica; no excluye otro óptimo, también en cúspide,
con un multiplete distinto del de x̄, ni dice nada de las filas «indet» (un solo
momento con gap < GAP_LISO), que no se certifican.

Ventana. Donde dos puntos vecinos de la malla tienen distinta clase
(cúspide o liso), se biseca en θ (NBIS = 10 pasos, resolución L/12/2¹⁰) y se
da el borde; la anchura de la ventana se mide desde el punto liso. Las filas
de la bisección van al CSV con tipo «borde».

Energía del anillo SU(3) con flujo (contraste con la red de pesos)
------------------------------------------------------------------
Aparte del dual, para el testigo isótropo −Σ P girado por t (`su_d_witness`)
se calcula E_0(t) = E_0(Σ P_t) por bloques en 25 puntos de cada recta, con los
sectores del fundamental, y se contrasta con la fórmula de la nota de
literatura (sección 4; derivada allí, no verificada en fuente, Resultado de
teoría de campos a orden dominante):
    ΔE(t) = E_0(t) − E_0(0) ≈ min_{λ ∈ Λ_w} |t − 2πλ|²/(3n)   (n ≡ 0 mód 3),
con cúspides en las fronteras de Voronoi de 2πΛ_w (en las rectas: el lado
C–M de la raíz y el punto M de la recta peso; C es un punto triple). Para
n ≢ 0 la nota solo dice que el mínimo se desplaza. Como hipótesis propia
(descriptiva, no derivada) se contrasta
    E_0(t) − E_0(M) ≈ min_{s ∈ S} |t − 2πs|²/(3n),  S = ½Λ_w ∖ Λ_w,
la red de semipesos sin los pesos (mínimo en M y sus imágenes, cúspides en 0 y
en el lado 0–C de la raíz, punto triple en C). Escribe
data/rigidez_qudit_flujo_energia.csv.

Comprueba
---------
  1. Energía isótropa: E_0(0) = `uls_ground_energy`; para n = 6..9, la
     estructura de cúspides en las dos rectas (clase de los fundamentales en
     cada punto) es la de Λ_w (n ≡ 0) o la de S (n ≢ 0) descrita arriba; el
     cociente con la fórmula se da como información.
  2. Dual en cada punto (rectas raíz y peso, 13 θ, n = 3..9): planos de corte
     `v_dual_ring_cg` arrancados en el x del θ anterior; estado 'optimal' y
     horquilla sup − inf < 1e-7; simetría residual (|Δv| < 1e-9) y x̄ no peor;
     la cota superior final (λ_max(W(x̄))/N(x̄)) se reevalúa con
     diagonalización densa en todos los bloques (`lam_max_sectors`) y, de
     forma independiente, con Lanczos de arranque aleatorio sobre W en los 3^n
     estados (4 autovalores, 40 vectores): diferencia < 1e-8 (Review Focus 1).
  3. θ = 0: v = d/(d+1)(1/d − e_P(n)) (tarea 6, exacto) a 1e-8.
  4. Primal reducido por el toro (`primal_ring_toro`), n = 3, 4, 5: los 13 θ
     de las dos rectas (78 puntos de malla; no en los de bisección);
     |dual − primal| < 1e-6, con estado 'optimal' u 'optimal_inaccurate': se
     acepta el segundo (Clarabel con tolerancias de 1e-10 lo da en casi todos)
     porque la coincidencia con la cota superior del dual, que no depende del
     primal, se exige igual a 1e-6. Con n > 5, v es «cota superior» (con la
     horquilla de los planos de corte como certificado numérico).
  5. Las dos rectas coinciden en θ = 0 y en M (raíz θ = π, peso θ = π/√3), a
     1e-7; rango (n = 4, 5, θ = θ_5): v(−θ), v(2π − θ) (raíz) y v(−θ),
     v(2π/√3 − θ) (peso) iguales a v(θ) a 1e-7, con `v_dual_ring_cg` en
     estado 'optimal'.
  5b. Cúspides: desdoblamiento de corrientes > 1e-3 en toda fila doblete o
     triple, y certificación de cúspide forzada (margen > 1e-7) salvo las dos
     excepciones de «Cúspide forzada» (θ = 0 con n ≢ 0; bisección).
  6. Ventana: bisección donde cambia la clase (ver arriba); tablas de v,
     flujo y clase por recta y n (información).

Escribe data/rigidez_qudit_flujo.csv (una fila por punto de malla o de
bisección: d, n, recta, θ, v cota superior, v_inf, horquilla, v sin
simetrizar, v primal (n ≤ 5), flujos, τ, θ_W, |z|, multiplete superior
(sectores «n0n1n2:k»), degeneración, momentos, clase, cúspide, gap, x y
segundos) y data/rigidez_qudit_flujo_energia.csv. Reanudable: cada punto se
añade al CSV al terminarlo y no se repite al relanzar (clave: recta, n, θ); al
final se reescribe ordenado. Un cálculo a la vez (serie, un proceso).

Resultados (2026-11-09; descriptivos, n = 3..9, más la sonda n = 10)
--------------------------------------------------------------------
  · Energía isótropa: la estructura de cúspides depende de n mod 3 y es la de
    Λ_w (n ≡ 0) o la de S (n ≢ 0) en todos los n = 6..9 (y cualitativamente en
    3..5); n ≡ 1 y n ≡ 2 tienen las mismas cúspides. El exceso sobre la fórmula
    de Λ_w es de 14–21 % en n = 6 y 11–15 % en n = 9; con S el cociente va de
    0,85 a 1,6 y depende de la dirección (rigidez anisótropa en M).
  · Testigo óptimo, n ≡ 0: liso solo en una ventana junto a θ = 0, de anchura
    θ/π = 0 (n = 3, por debajo de la resolución), 0,0199 (n = 6) y 0,0346
    (n = 9) en la raíz, y 0, 0,0174 y 0,0286 en la recta peso; en la ventana el
    flujo pasa de forma continua de 0 al de anclaje y el gap se cierra en el
    borde. Fuera, anclado: en la raíz, en el cruce triple k = 0, ±n/3 del sector
    (n/3, n/3, n/3) (el de C), con θ_W/L entre 0,66 y 0,84 y τ ≠ 0 (n = 6, 9;
    n = 3, con multipletes de 9 a 11 estados, es aparte); en la recta peso, en
    M (θ_W = L), doblete o triple.
  · n ≢ 0: el testigo está en una cúspide en todos los puntos de las dos rectas
    (n = 4, 5, 7, 8, y n = 10 en M y en la raíz θ_11). En la recta peso, anclado
    exactamente en t = 0 (θ_W = 0, doblete ±k) hasta θ = 5L/12 (n = 5), 3L/4
    (n = 7), L/2 (n = 8) (n = 4: 7L/12, y luego deriva); en la raíz, sobre la
    cresta 0–C (doblete ±k) con θ_W ≤ 0,035 L hasta θ = L/3 (n = 7) o L/4
    (n = 8). Después, cruces con k = 0 y θ_W hacia M. No hay ventana junto a M,
    el punto liso de la energía, en la malla (resolución L/12; una ventana más
    estrecha entre dos puntos en cúspide no se vería).
  · v: el signo de v(C) − v(0) es − para n ≡ 0 y + para n ≢ 0 (n = 3..9).

Uso: (ulimit -v 16000000; /usr/bin/time -f "%M KB %e s" \\
          .venv/bin/python scripts/rigidez_qudit_flujo.py [n ...] [--sonda])
     (sin n: 3..9). --analisis: solo lee el CSV y escribe las tablas.
Medido (2026-11-09, OMP_NUM_THREADS = 2; cifras de la primera versión, sin
corrientes ni certificación ni el primal de la recta peso, que se añadieron
después reanudando: 884 s más con --sonda para 24 primales, corrientes y una
primera certificación, y 1101 s con --sonda para recertificar todo con tol 1e-9,
pico 5,8–6,0 GB por la sonda): en frío, 1774 s y 1,75 GB de pico
(n = 9: 23 s por punto de media; las bisecciones de n = 6, 9, ~10 puntos cada
una); con el CSV escrito, 31 s y 0,94 GB (energía, rango y recomprobación de
las filas, también corrientes y certificación); --sonda (n = 10, tres puntos,
~85 s cada uno) añade 250 s y sube el pico a 6,6 GB; --analisis, 1,7 s.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import io
import resource
import sys
import time
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.sparse.linalg import eigsh

from holonomy import qudit
from holonomy.report import DATA, write_csv

D = 3
NS_DEFECTO = list(range(3, 10))
NTH = 13
NBIS = 10
TOL_DEG = 1e-7
GAP_LISO = 1e-5
TOL_HORQUILLA = 1e-7
TOL_SIM = 1e-9
TOL_LANCZOS = 1e-8
TOL_PRIMAL = 1e-6
TOL_CONTROL = 1e-8
TOL_RECTAS = 1e-7
NPRIMAL = 5
TOL_CORRIENTE = 1e-3
TOL_CERT = 1e-7
NSONDA = 10
SONDA = (("raiz", 12), ("peso", 12), ("raiz", 11))

RECTAS = {
    "raiz": dict(u=np.array([1.0, 0.0, -1.0]), L=np.pi, perm=(2, 1, 0), conj=True),
    "peso": dict(u=np.array([1.0, 1.0, -2.0]) / np.sqrt(3), L=np.pi / np.sqrt(3),
                 perm=(1, 0, 2), conj=False),
}
# Dirección transversal sin traza (para las corrientes).
W_PERP = {"raiz": np.array([1.0, -2.0, 1.0]) / np.sqrt(3), "peso": np.array([1.0, -1.0, 0.0])}
CSV_NAME = "rigidez_qudit_flujo.csv"
CSV_ENERGIA = "rigidez_qudit_flujo_energia.csv"
NPAR = qudit.n_params(D)
HEADER = (["tipo", "recta", "d", "n", "i_theta", "theta", "theta_sobre_L",
           "v_sup", "v_inf", "horquilla", "estado", "v_sin_simetrizar", "dif_simetria",
           "dif_lanczos", "v_primal", "estado_primal", "dif_primal",
           "etiqueta_v", "Phi01", "Phi12", "Phi02", "tau", "theta_W", "abs_z01", "abs_z12", "abs_z02",
           "multiplete", "degeneracion", "momentos", "clase", "cuspide", "gap",
           "corrientes_u", "corrientes_w", "desdoblamiento", "cert_k", "cert_margen"]
          + [f"x{p}" for p in range(NPAR)] + ["segundos"])


def wrap(a):
    """Ángulo en (−π, π]."""
    return float(np.pi - (np.pi - a) % (2 * np.pi))


def holonomia(recta, theta):
    return theta * RECTAS[recta]["u"]


def grid(recta):
    return [i * RECTAS[recta]["L"] / (NTH - 1) for i in range(NTH)]


def clave(recta, n, theta):
    return (recta, int(n), round(float(theta), 10))


# ---------------------------------------------------------------------------
# Simetría, flujos y multiplete superior
# ---------------------------------------------------------------------------

def accion(x, perm, conj):
    """x del testigo transformado por la permutación de colores perm (y K si
    conj): D_ab ↦ D_{π(a)π(b)}, z_ab ↦ z_{π(a)π(b)} (conjugado si π invierte el
    par), y K conjuga z."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for a, b in qudit._pairs(D):
        y[qudit.param_index(D, "D", perm[a], perm[b])] = x[qudit.param_index(D, "D", a, b)]
    for a, b in qudit._offpairs(D):
        z = x[qudit.param_index(D, "Re", a, b)] + 1j * x[qudit.param_index(D, "Im", a, b)]
        pa, pb = perm[a], perm[b]
        if pa > pb:
            pa, pb, z = pb, pa, np.conj(z)
        if conj:
            z = np.conj(z)
        y[qudit.param_index(D, "Re", pa, pb)] = z.real
        y[qudit.param_index(D, "Im", pa, pb)] = z.imag
    return y


def zs(x):
    return {ab: x[qudit.param_index(D, "Re", *ab)] + 1j * x[qudit.param_index(D, "Im", *ab)]
            for ab in qudit._offpairs(D)}


def flujos(recta, n, x):
    """Φ_ab = n·arg(−z_ab) (mód. 2π), τ = Φ_01 + Φ_12 − Φ_02 y θ_W."""
    z = zs(x)
    Phi = {ab: wrap(n * np.angle(-z[ab])) for ab in z}
    tau = wrap(Phi[(0, 1)] + Phi[(1, 2)] - Phi[(0, 2)])
    if recta == "raiz":
        thw = wrap(-Phi[(0, 1)])
    else:
        thw = wrap(-Phi[(0, 2)]) / np.sqrt(3)
    L = RECTAS[recta]["L"]
    if abs(thw + L) < 1e-6:  # −L ≡ L (el punto M)
        thw = L
    return Phi, tau, thw, {ab: abs(z[ab]) for ab in z}


def espectro_superior(n, x):
    """(λ_max, multiplete [(colores, k), ...] con multiplicidad, gap)."""
    labels, _, ops = qudit.sector_blocks(D, n)
    c0 = n * (qudit._trace_coeffs(D) @ x) / D**2
    ev = []
    for lab, O in zip(labels, ops):
        w = np.linalg.eigvalsh(np.tensordot(x, O, axes=1)) - c0
        ev.extend((float(e), lab) for e in w)
    ev.sort(key=lambda q: -q[0])
    lam = ev[0][0]
    multi = [lab for e, lab in ev if e > lam - TOL_DEG]
    resto = [e for e, _ in ev if e <= lam - TOL_DEG]
    gap = lam - resto[0] if resto else np.nan
    return lam, multi, gap


def clase(multi, gap=np.inf):
    """Momentos del multiplete y clase: liso (un momento y gap > GAP_LISO),
    indet (un momento y gap ≤ GAP_LISO: no se distingue de un cruce),
    doblete (dos momentos) o triple (tres o más)."""
    ks = sorted({k for _, k in multi})
    if len(ks) == 1:
        c = "liso" if gap > GAP_LISO else "indet"
    else:
        c = "doblete" if len(ks) == 2 else "triple"
    return ks, c


def fmt_multi(multi):
    out = []
    for (cols, k) in multi:
        s = "".join(str(c) for c in cols) + f":{k}"
        out.append(s)
    return "|".join(out)


def dx_giro(n, x, u):
    """dx/ds al girar el flujo del testigo s·u (z_ab ↦ z_ab e^{i s (u_b − u_a)/n})."""
    dx = np.zeros_like(x)
    for (a, b) in qudit._offpairs(D):
        z = x[qudit.param_index(D, "Re", a, b)] + 1j * x[qudit.param_index(D, "Im", a, b)]
        dz = 1j * (u[b] - u[a]) / n * z
        dx[qudit.param_index(D, "Re", a, b)] = dz.real
        dx[qudit.param_index(D, "Im", a, b)] = dz.imag
    return dx


def corrientes(n, x, u):
    """Autovalores de P·(dW/ds)·P (P: multiplete superior a TOL_DEG), bloque a
    bloque (dW/ds conserva color y momento)."""
    labels, _, ops = qudit.sector_blocks(D, n)
    c0 = n * (qudit._trace_coeffs(D) @ x) / D**2
    dx = dx_giro(n, x, u)
    esp = [np.linalg.eigh(np.tensordot(x, O, axes=1)) for O in ops]
    lam = max(w[-1] for w, _ in esp) - c0
    out = []
    for (w, V), O in zip(esp, ops):
        sel = (w - c0) > lam - TOL_DEG
        if sel.any():
            Vt = V[:, sel]
            M = Vt.conj().T @ np.tensordot(dx, O, axes=1) @ Vt
            out.extend(np.linalg.eigvalsh((M + M.conj().T) / 2))
    return np.sort(np.array(out))


def fmt_vec(v):
    return "|".join(f"{c:+.5f}" for c in v)


def certificar(n, t, x, v, multi):
    """Cúspide forzada: para cada k, margen v − mín_x λ_max^(k)/N acotado por un x
    explícito (x̄ para todo k; para los k del multiplete, además, la cota
    superior de los planos de corte restringidos a ese momento, arrancados en
    x̄). Devuelve (texto «k:margen» de los k del multiplete, menor margen)."""
    labels, _, ops = qudit.sector_blocks(D, n)
    c0 = n * (qudit._trace_coeffs(D) @ x) / D**2
    mejor = {}
    for lab, O in zip(labels, ops):
        e = np.linalg.eigvalsh(np.tensordot(x, O, axes=1))[-1] - c0
        mejor[lab[1]] = max(mejor.get(lab[1], -np.inf), e)
    texto = []
    for k in sorted({k for _, k in multi}):
        sup, _, _, _ = qudit.v_dual_ring_cg(D, n, t, x0=x, momentos=[k])
        mejor[k] = min(mejor[k], sup)
        texto.append(f"{k}:{v - mejor[k]:.3e}")
    return "|".join(texto), float(min(v - b for b in mejor.values()))


@lru_cache(maxsize=None)
def ops_completos(n):
    return qudit._param_ops(D, n, np.arange(D**n))


def lam_lanczos(n, x, seed=2026):
    """λ_max de W en los 3^n estados con Lanczos de arranque aleatorio
    (independiente de los bloques de momento)."""
    ops = ops_completos(n)
    c0 = n * (qudit._trace_coeffs(D) @ x) / D**2
    W = ops[0] * x[0]
    for xp, O in zip(x[1:], ops[1:]):
        W = W + xp * O
    rng = np.random.default_rng(seed)
    dim = D**n
    v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    if dim <= 300:
        return float(np.linalg.eigvalsh(W.toarray())[-1] - c0)
    w = eigsh(W, k=4, which="LA", v0=v0, ncv=40, tol=0)[0]
    return float(np.max(w) - c0)


# ---------------------------------------------------------------------------
# Un punto
# ---------------------------------------------------------------------------

def evaluar(recta, n, theta, x0, primal, tipo="malla", i_theta=-1):
    t0 = time.time()
    R = RECTAS[recta]
    t = holonomia(recta, theta)
    sup, inf, x, st = qudit.v_dual_ring_cg(D, n, t, x0=x0)
    N = qudit.ring_norm(D, n, x, t)
    v_x = qudit.lam_max_sectors(D, n, x)[0] / N
    gx = accion(x, R["perm"], R["conj"])
    v_gx = qudit.lam_max_sectors(D, n, gx)[0] / qudit.ring_norm(D, n, gx, t)
    xs = (x + gx) / 2
    xs = xs / qudit.ring_norm(D, n, xs, t)
    lam, multi, gap = espectro_superior(n, xs)
    v_sup = lam  # N(xs) = 1
    lam_blq = qudit.lam_max_sectors(D, n, xs)[0]
    dif_lz = lam_lanczos(n, xs) - lam_blq
    ks, cl = clase(multi, gap)
    Phi, tau, thw, az = flujos(recta, n, xs)
    vp, stp = np.nan, ""
    if primal:
        vp, stp = qudit.primal_ring_toro(D, n, t, **CLARABEL_OPTS)
    row = dict(
        tipo=tipo, recta=recta, d=D, n=n, i_theta=i_theta, theta=theta,
        theta_sobre_L=theta / R["L"], v_sup=v_sup, v_inf=inf, horquilla=v_sup - inf,
        estado=st, v_sin_simetrizar=v_x, dif_simetria=v_gx - v_x, dif_lanczos=dif_lz,
        v_primal=vp, estado_primal=stp, dif_primal=(v_sup - vp) if primal else np.nan,
        Phi01=Phi[(0, 1)], Phi12=Phi[(1, 2)], Phi02=Phi[(0, 2)], tau=tau, theta_W=thw,
        abs_z01=az[(0, 1)], abs_z12=az[(1, 2)], abs_z02=az[(0, 2)],
        multiplete=fmt_multi(multi), degeneracion=len(multi),
        momentos="|".join(str(k) for k in ks), clase=cl, cuspide=int(cl != "liso"), gap=gap,
        **{f"x{p}": float(xs[p]) for p in range(NPAR)}, segundos=time.time() - t0)
    row["etiqueta_v"] = etiqueta(row)
    completar_extras(row, xs)
    return row


def completar_extras(row, xs=None):
    """Corrientes y certificación de una fila (si faltan). Devuelve True si cambia."""
    cambia = False
    n, recta = row["n"], row["recta"]
    x = xde(row) if xs is None else xs
    if not isinstance(row.get("corrientes_u"), str) or row["corrientes_u"] == "":
        cu = corrientes(n, x, RECTAS[recta]["u"])
        cw = corrientes(n, x, W_PERP[recta])
        row["corrientes_u"], row["corrientes_w"] = fmt_vec(cu), fmt_vec(cw)
        row["desdoblamiento"] = float(max(np.ptp(cu), np.ptp(cw)))
        cambia = True
    cert = row.get("cert_margen")
    if row["clase"] in ("doblete", "triple") and not (isinstance(cert, float) and np.isfinite(cert)):
        multi = [(None, int(m.split(":")[1])) for m in row["multiplete"].split("|")]
        row["cert_k"], row["cert_margen"] = certificar(
            n, holonomia(recta, row["theta"]), x, row["v_sup"], multi)
        cambia = True
    elif row["clase"] not in ("doblete", "triple") and row.get("cert_k") is None:
        row["cert_k"], row["cert_margen"] = "", np.nan
    return cambia


def completar(ck, row):
    """Rellena lo que falte en una fila guardada (primal en n ≤ NPRIMAL de la
    malla, corrientes, certificación) y la añade de nuevo al CSV (la última fila
    de cada clave es la que vale al leer)."""
    cambia = False
    if (row["tipo"] == "malla" and row["n"] <= NPRIMAL
            and not (isinstance(row["v_primal"], float) and np.isfinite(row["v_primal"]))):
        t0 = time.time()
        vp, stp = qudit.primal_ring_toro(D, row["n"], holonomia(row["recta"], row["theta"]),
                                         **CLARABEL_OPTS)
        row["v_primal"], row["estado_primal"] = vp, stp
        row["dif_primal"] = row["v_sup"] - vp
        row["etiqueta_v"] = etiqueta(row)
        row["segundos"] += time.time() - t0
        cambia = True
    t0 = time.time()
    if completar_extras(row):
        row["segundos"] += time.time() - t0
        cambia = True
    if cambia:
        anadir(row)
    return cambia


# ---------------------------------------------------------------------------
# CSV reanudable
# ---------------------------------------------------------------------------

_TEXTO = {"tipo", "recta", "estado", "estado_primal", "multiplete", "momentos", "clase",
          "etiqueta_v", "corrientes_u", "corrientes_w", "cert_k"}


def etiqueta(r):
    """«primal» si el primal reducido por el toro confirma v a TOL_PRIMAL;
    «exacto_theta0» en θ = 0 (Proposición de la tarea 6); si no, «cota_superior»
    (con la horquilla de los planos de corte como certificado numérico)."""
    dp = r["dif_primal"]
    if isinstance(dp, float) and np.isfinite(dp) and abs(dp) < TOL_PRIMAL:
        return "primal"
    if r["theta"] == 0.0:
        return "exacto_theta0"
    return "cota_superior"


def _num(k, s):
    if k in _TEXTO:
        return s
    return float(s) if s != "" else np.nan


def leer():
    """Filas del CSV (la última de cada clave). Descarta una última línea
    incompleta (corte a medio escribir) y, si la cabecera es de una versión
    anterior, reescribe el fichero con la actual (columnas nuevas vacías) para
    que las filas que se añadan queden alineadas."""
    path = DATA / CSV_NAME
    out = {}
    if path.exists():
        texto = path.read_text()
        if texto and not texto.endswith("\n"):
            texto = texto[:texto.rfind("\n") + 1]
            path.write_text(texto)
            print("aviso: descartada una última línea incompleta del CSV", flush=True)
        lector = csv.DictReader(io.StringIO(texto))
        for r in lector:
            if None in r or None in r.values():
                print("aviso: fila mal formada descartada", flush=True)
                continue
            r = {h: r.get(h, "") for h in HEADER}
            r = {k: _num(k, v) for k, v in r.items()}
            for k in ("n", "d", "i_theta", "degeneracion", "cuspide"):
                r[k] = int(r[k])
            r["etiqueta_v"] = etiqueta(r)
            if abs(r["theta_W"] + RECTAS[r["recta"]]["L"]) < 1e-6:
                r["theta_W"] = RECTAS[r["recta"]]["L"]
            out[clave(r["recta"], r["n"], r["theta"])] = r
        if lector.fieldnames != HEADER:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(HEADER)
                for r in out.values():
                    w.writerow([r[h] for h in HEADER])
            print("aviso: CSV reescrito con la cabecera actual", flush=True)
    return out


def anadir(row):
    path = DATA / CSV_NAME
    nuevo = not path.exists()
    DATA.mkdir(exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(HEADER)
        w.writerow([row[h] for h in HEADER])


def xde(row):
    return np.array([row[f"x{p}"] for p in range(NPAR)])


# ---------------------------------------------------------------------------
# Energía del anillo SU(3) con flujo
# ---------------------------------------------------------------------------

_LW = [np.array(m, dtype=float) - np.mean(m) for m in product(range(-3, 4), repeat=3)]
_S = []
for _h in (p / 2 for p in _LW):
    if not any(np.allclose(_h, p) for p in _LW) and not any(np.allclose(_h, s) for s in _S):
        _S.append(_h)


def formula_red(n, t, red):
    t = np.asarray(t, dtype=float) - np.mean(t)
    return min(float(np.sum((t - 2 * np.pi * p) ** 2)) for p in red) / (D * n)


def energia(n, t):
    """E_0(Σ P_t) y fundamentales [(colores, k)] (a 1e-9), por bloques."""
    x = qudit.su_d_witness(D, t, n)
    lam, _, _ = espectro_superior(n, x)
    labels, _, ops = qudit.sector_blocks(D, n)
    c0 = n * (qudit._trace_coeffs(D) @ x) / D**2
    fund = []
    for lab, O in zip(labels, ops):
        w = np.linalg.eigvalsh(np.tensordot(x, O, axes=1)) - c0
        fund.extend([lab] * int(np.sum(w > lam - 1e-9)))
    # W = −Σ P_t + (n/d) I: E_0(Σ P_t) = n/d − λ_max(W).
    return n / D - lam, fund


def esperado_energia(n, recta, i, nfino):
    """Clase esperada del fundamental en el punto i de nfino (n ≥ 6)."""
    th_c = 2 * (nfino - 1) // 3  # índice de C en la recta raíz (2π/3 de π)
    cero = n % 3 == 0
    if recta == "raiz":
        if i == 0:
            return "liso" if cero else "doblete"
        if i < th_c:
            return "liso" if cero else "doblete"
        if i == th_c:
            return "triple"
        return "doblete" if cero else "liso"
    if i == 0:
        return "liso" if cero else "doblete"
    if i == nfino - 1:
        return "doblete" if cero else "liso"
    return "liso"


def parte_energia(ck, ns):
    ck.info("Energía del anillo SU(3) con flujo t (testigo isótropo −Σ P girado)")
    nfino = 25
    M = holonomia("raiz", np.pi)
    rows = []
    for n in ns:
        e0, _ = energia(n, np.zeros(3))
        ck.check(abs(e0 - qudit.uls_ground_energy(D, n)) < 1e-9,
                 f"n = {n}: E_0(t = 0) = {e0:.10f} = uls_ground_energy")
        eM, _ = energia(n, M)
        ref, red = (e0, _LW) if n % 3 == 0 else (eM, _S)
        for recta in RECTAS:
            clases, cocientes = [], []
            for i in range(nfino):
                th = i * RECTAS[recta]["L"] / (nfino - 1)
                t = holonomia(recta, th)
                e, fund = energia(n, t)
                ks, cl = clase(fund)
                f = formula_red(n, t, red)
                q = (e - ref) / f if f > 1e-9 else np.nan
                clases.append(cl)
                if np.isfinite(q):
                    cocientes.append(q)
                rows.append([n, recta, th, th / RECTAS[recta]["L"], e, e - e0, e - eM,
                             "Lw" if n % 3 == 0 else "S", f, q, fmt_multi(fund), len(fund),
                             "|".join(map(str, ks)), cl])
            resumen = "".join({"liso": "·", "doblete": "2", "triple": "3"}[c] for c in clases)
            ck.info(f"n = {n} ({n % 3} mód 3), recta {recta}: clase en θ = 0..L "
                    f"[· liso, 2 doblete, 3 triple] {resumen}; "
                    f"(E − E_ref)/fórmula({'Λ_w' if n % 3 == 0 else 'S'}) en "
                    f"[{min(cocientes):.3f}, {max(cocientes):.3f}]")
            if n >= 6:
                esp = [esperado_energia(n, recta, i, nfino) for i in range(nfino)]
                ck.check(clases == esp,
                         f"n = {n}, recta {recta}: estructura de cúspides de "
                         f"{'Λ_w (n ≡ 0)' if n % 3 == 0 else 'S (n ≢ 0)'}")
    write_csv(CSV_ENERGIA, ["n", "recta", "theta", "theta_sobre_L", "E0", "E0_menos_E0_0",
                            "E0_menos_E0_M", "red", "formula", "cociente", "fundamental",
                            "degeneracion", "momentos", "clase"], rows)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def v_formula(n):
    return D / (D + 1) * (1 / D - qudit.uls_ground_energy(D, n) / n)


def comprobar_punto(ck, r):
    nota = (f"n = {r['n']}, {r['recta']} θ = {r['theta']:.6f} ({r['tipo']})")
    ck.check(r["estado"] == "optimal" and abs(r["horquilla"]) < TOL_HORQUILLA
             and abs(r["dif_simetria"]) < TOL_SIM
             and r["v_sup"] <= r["v_sin_simetrizar"] + 1e-12
             and abs(r["dif_lanczos"]) < TOL_LANCZOS,
             f"{nota}: v ≤ {r['v_sup']:.10f}, horquilla {r['horquilla']:+.1e} "
             f"[{r['estado']}], simetría {r['dif_simetria']:+.0e}, x̄ − x "
             f"{r['v_sup'] - r['v_sin_simetrizar']:+.0e}, Lanczos {r['dif_lanczos']:+.0e}; "
             f"{r['clase']} ({r['multiplete']}), gap {r['gap']:.2e}")
    if isinstance(r["v_primal"], float) and np.isfinite(r["v_primal"]):
        # 'optimal_inaccurate' se acepta: la coincidencia a TOL_PRIMAL con la cota
        # superior del dual se exige igual.
        ck.check(abs(r["dif_primal"]) < TOL_PRIMAL and str(r["estado_primal"]).startswith("optimal"),
                 f"{nota}: primal {r['v_primal']:.10f}, dual − primal {r['dif_primal']:+.1e} "
                 f"[{r['estado_primal']}]")
    if r["clase"] in ("doblete", "triple"):
        texto = (f"{nota}: cúspide, desdoblamiento de corrientes {r['desdoblamiento']:.3f}; "
                 f"margen de cúspide forzada {r['cert_margen']:.2e} (k:margen {r['cert_k']})")
        if r["tipo"] == "borde" or (r["theta"] == 0.0 and r["n"] % 3 != 0):
            # Excepciones documentadas (docstring, «Cúspide forzada»): junto al
            # borde de la ventana el margen es de segundo orden en la distancia al
            # borde; en θ = 0 con n ≢ 0 el multiplete de color de un solo momento ya
            # fija v (mín. restringido = v): no certificable.
            ck.check(r["desdoblamiento"] > TOL_CORRIENTE, texto)
            if not r["cert_margen"] > TOL_CERT:
                ck.info("  (sin certificar: " + ("borde de ventana" if r["tipo"] == "borde"
                                                  else "θ = 0, n ≢ 0") + ")")
        else:
            ck.check(r["desdoblamiento"] > TOL_CORRIENTE and r["cert_margen"] > TOL_CERT, texto)


def calcular(ck, ns, datos):
    for recta in RECTAS:
        for n in ns:
            x0 = None
            for i, th in enumerate(grid(recta)):
                key = clave(recta, n, th)
                if key not in datos:
                    datos[key] = evaluar(recta, n, th, x0, n <= NPRIMAL, "malla", i)
                    anadir(datos[key])
                else:
                    completar(ck, datos[key])
                comprobar_punto(ck, datos[key])
                x0 = xde(datos[key])
            # Bisección donde cambia la clase (cúspide frente a liso).
            g = grid(recta)
            for i in range(NTH - 1):
                a, b = datos[clave(recta, n, g[i])], datos[clave(recta, n, g[i + 1])]
                if a["cuspide"] == b["cuspide"]:
                    continue
                lo, hi = a, b
                for _ in range(NBIS):
                    th = (lo["theta"] + hi["theta"]) / 2
                    key = clave(recta, n, th)
                    if key not in datos:
                        datos[key] = evaluar(recta, n, th, xde(lo), False, "borde", -1)
                        anadir(datos[key])
                    else:
                        completar(ck, datos[key])
                    comprobar_punto(ck, datos[key])
                    m = datos[key]
                    if m["cuspide"] == lo["cuspide"]:
                        lo = m
                    else:
                        hi = m


def comprobar_rango(ck):
    """v en puntos equivalentes por las simetrías (n = 4, 5, θ = θ_5)."""
    for n in (4, 5):
        for recta, L in (("raiz", 2 * np.pi), ("peso", 2 * np.pi / np.sqrt(3))):
            th = grid(recta)[5]
            vs, sts = [], []
            for t in (th, -th, L - th):
                v, _, _, st = qudit.v_dual_ring_cg(D, n, holonomia(recta, t))
                vs.append(v)
                sts.append(st)
            ck.check(max(vs) - min(vs) < TOL_RECTAS and all(st == "optimal" for st in sts),
                     f"rango, n = {n}, recta {recta}: v(θ), v(−θ), v({L / np.pi:.4f}π − θ) = "
                     f"{', '.join(f'{v:.10f}' for v in vs)} [{', '.join(sts)}]")


def analisis(ck, ns, datos):
    filas = sorted(datos.values(), key=lambda r: (r["recta"], r["n"], r["theta"]))
    for n in ns:
        r0 = datos.get(clave("raiz", n, 0.0))
        if r0 is None:
            continue
        vf = v_formula(n)
        ck.check(abs(r0["v_sup"] - vf) < TOL_CONTROL,
                 f"n = {n}, θ = 0: v = {r0['v_sup']:.10f}, d/(d+1)(1/d − e_P) = {vf:.10f}")
        p0 = datos.get(clave("peso", n, 0.0))
        rM = datos.get(clave("raiz", n, np.pi))
        pM = datos.get(clave("peso", n, np.pi / np.sqrt(3)))
        if p0 and rM and pM:
            ck.check(abs(p0["v_sup"] - r0["v_sup"]) < TOL_RECTAS
                     and abs(rM["v_sup"] - pM["v_sup"]) < TOL_RECTAS,
                     f"n = {n}: rectas en 0 ({r0['v_sup'] - p0['v_sup']:+.0e}) y en M "
                     f"(raíz π {rM['v_sup']:.10f}, peso π/√3 {pM['v_sup']:.10f})")
    pr = [r for r in filas if isinstance(r["v_primal"], float) and np.isfinite(r["v_primal"])]
    if pr:
        dif = np.array([r["dif_primal"] for r in pr])
        ck.info(f"primal: {len(pr)} puntos, |dual − primal| ≤ {np.max(np.abs(dif)):.1e}; "
                f"primal por debajo del dual en {int(np.sum(dif > 0))} (máx. {np.max(dif):+.1e}); "
                f"estados {sorted({r['estado_primal'] for r in pr})}")
    cus = [r for r in filas if r["clase"] in ("doblete", "triple")
           and isinstance(r["cert_margen"], float) and np.isfinite(r["cert_margen"])]
    if cus:
        nc = [r for r in cus if not r["cert_margen"] > TOL_CERT]
        ck.info(f"cúspides: {len(cus)} filas doblete o triple; desdoblamiento mínimo "
                f"{min(r['desdoblamiento'] for r in cus):.3f}; certificadas como forzadas "
                f"{len(cus) - len(nc)}; sin certificar {len(nc)}: "
                + ", ".join(f"{r['recta']} n={r['n']} θ={r['theta']:.5f} ({r['tipo']}, "
                            f"{r['cert_margen']:.1e})" for r in nc))
    for recta in RECTAS:
        ck.info(f"Recta {recta}: v(n, θ_i) (cota superior; n ≤ 5: primal a 1e-6 en los 13 θ), "
                f"θ_i = i·L/12, L = {RECTAS[recta]['L'] / np.pi:.4f}π")
        ck.info("  n  " + " ".join(f"{i:>8d}" for i in range(NTH)))
        for n in ns:
            g = [datos.get(clave(recta, n, th)) for th in grid(recta)]
            if any(r is None for r in g):
                continue
            ck.info(f"{n:3d}  " + " ".join(f"{r['v_sup']:.6f}" for r in g))
        ck.info(f"Recta {recta}: θ_W/L (flujo del testigo) | clase "
                f"[· liso, ? indeterminado, 2 doblete, 3 triple o más] | τ")
        for n in ns:
            g = [datos.get(clave(recta, n, th)) for th in grid(recta)]
            if any(r is None for r in g):
                continue
            L = RECTAS[recta]["L"]
            cl = "".join({"liso": "·", "indet": "?", "doblete": "2", "triple": "3"}[r["clase"]] for r in g)
            ck.info(f"{n:3d}  " + " ".join(f"{r['theta_W'] / L:+.3f}" for r in g)
                    + f"  {cl}  τ∈[{min(r['tau'] for r in g):+.3f}, "
                    f"{max(r['tau'] for r in g):+.3f}]")
        ck.info(f"Recta {recta}: bordes (cambio de clase, bisección)")
        for n in ns:
            bs = [r for r in filas if r["recta"] == recta and r["n"] == n]
            g = grid(recta)
            for i in range(NTH - 1):
                a, b = datos.get(clave(recta, n, g[i])), datos.get(clave(recta, n, g[i + 1]))
                if a is None or b is None or a["cuspide"] == b["cuspide"]:
                    continue
                dentro = [r for r in bs if g[i] <= r["theta"] <= g[i + 1]]
                lisos = [r["theta"] for r in dentro if not r["cuspide"]]
                cusps = [r["theta"] for r in dentro if r["cuspide"]]
                if a["cuspide"]:
                    borde = (max(cusps), min(lisos))
                else:
                    borde = (max(lisos), min(cusps))
                L = RECTAS[recta]["L"]
                ck.info(f"  n = {n}: entre θ_{i} y θ_{i + 1}: borde en θ/L ∈ "
                        f"[{borde[0] / L:.6f}, {borde[1] / L:.6f}] "
                        f"(θ/π ∈ [{borde[0] / np.pi:.6f}, {borde[1] / np.pi:.6f}])")
    write_csv(CSV_NAME, HEADER, [[r[h] for h in HEADER] for r in filas])


def sonda(ck, datos):
    """n = 10 (≡ 1 mód 3), solo en M (las dos rectas) y en la raíz θ_11: ¿se
    abre junto al punto liso de la energía (M) una ventana para n ≢ 0?"""
    for recta, i in SONDA:
        th = grid(recta)[i]
        key = clave(recta, NSONDA, th)
        if key not in datos:
            datos[key] = evaluar(recta, NSONDA, th, None, False, "sonda", i)
            anadir(datos[key])
        else:
            completar(ck, datos[key])
        comprobar_punto(ck, datos[key])
        r = datos[key]
        ck.info(f"sonda n = {NSONDA}, {recta} θ_{i}: v ≤ {r['v_sup']:.10f}, θ_W/L = "
                f"{r['theta_W'] / RECTAS[recta]['L']:+.4f}, τ = {r['tau']:+.4f}, {r['clase']} "
                f"({r['multiplete']}), gap {r['gap']:.2e}, {r['segundos']:.0f} s")
    a = datos.get(clave("raiz", NSONDA, np.pi))
    b = datos.get(clave("peso", NSONDA, np.pi / np.sqrt(3)))
    if a and b:
        ck.check(abs(a["v_sup"] - b["v_sup"]) < TOL_RECTAS,
                 f"sonda n = {NSONDA}: rectas en M ({a['v_sup']:.10f}, {b['v_sup']:.10f})")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ns = [int(a) for a in args] or NS_DEFECTO
    ck = Checker("rigidez_qudit_flujo")
    datos = leer()
    if "--analisis" not in sys.argv:
        parte_energia(ck, ns)
        comprobar_rango(ck)
        calcular(ck, ns, datos)
        if "--sonda" in sys.argv:
            sonda(ck, datos)
    else:
        for r in datos.values():
            if r["tipo"] == "sonda":
                ck.info(f"sonda n = {r['n']}, {r['recta']} θ = {r['theta']:.6f}: "
                        f"v ≤ {r['v_sup']:.10f}, {r['clase']} ({r['multiplete']})")
    analisis(ck, ns, datos)
    ck.info(f"pico de memoria residente {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    ck.finish()


if __name__ == "__main__":
    main()

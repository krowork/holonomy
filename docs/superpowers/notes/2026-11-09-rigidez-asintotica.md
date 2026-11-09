# Asintótica de qudits con holonomía: coeficiente de 1/n² y ventana (tarea 8)

Fecha: 2026-11-09. Nota de trabajo para redactar el artículo (tarea 9); no es parte del artículo.
Código: [holonomy/qudit_asint.py](file:///home/alex/fisica/pyhton/holonomy/holonomy/qudit_asint.py),
guion [scripts/rigidez_qudit_asint.py](file:///home/alex/fisica/pyhton/holonomy/scripts/rigidez_qudit_asint.py),
datos [data/rigidez_qudit_asint.csv](file:///home/alex/fisica/pyhton/holonomy/data/rigidez_qudit_asint.csv)
(113/113 comprobaciones, 110 s, 1,20 GB). Contraste con la tarea 7:
[data/rigidez_qudit_flujo.csv](file:///home/alex/fisica/pyhton/holonomy/data/rigidez_qudit_flujo.csv).

**Etiquetas.** «Lema» y «Proposición» son exactos para todo n (con demostración aquí). «Resultado» es
de teoría de campos, al nivel de \cite{anillos} sección 6.2: no es un teorema.

## 0. Resumen

1. **El coeficiente y la ventana dependen de la clase r = n mód d**, como se sospechaba, pero el
   desplazamiento de la red no es por el peso fundamental ω_r (que está en Λ_w y no desplazaría nada),
   sino por **½ω** con ω en la órbita de Weyl de ω_r: los «semipesos» de la tarea 7 son exactamente
   ½W·ω_r + Λ_w. En d = 3, r = 1 y r = 2 dan el mismo conjunto.
2. **Mecanismo nuevo para n ≢ 0: el campo de color.** La familia de testigos invariante ya contiene
   términos de un sitio h·n_a (D_ab = h_a + h_b). No cambian la normalización N, pero desplazan cada
   sector de color en 2h·Q. Para n ≢ 0 todos los sectores tienen carga Q ≠ 0 y el dual los equilibra.
   - Eso cambia el coeficiente a orden dominante: en M pasa de −π²/6 a **−π²/18**.
   - Obliga a que el nivel superior de todo óptimo ocupe al menos dos sectores de color (Proposición 2,
     exacta). Que eso sea además una cúspide, y que por tanto no haya ventana lisa, es un **Resultado** de
     teoría de campos para d ≥ 3, no una consecuencia de la Proposición: en d = 2 con n impar hay dos
     sectores y el punto es liso (§2).
   - Explica el «exceso de amplitud» y la «rigidez anisótropa» que la tarea 7 atribuía a la anisotropía.
3. **Coeficientes límite, lím n²[v(n,0) − v(n,t)]** (Resultado):

   | d | clase | punto | límite | valor |
   |---|---|---|---|---|
   | 2 | n par | Θ = π | +π²/6 | +1,6449 |
   | 2 | n impar | Θ = π | −π²/6 | −1,6449 |
   | 3 | n ≡ 0 | C | +2π²/9 | +2,1932 |
   | 3 | n ≡ 0 | M | +π²/6 | +1,6449 |
   | 3 | n ≢ 0 | C | −π²/9 | −1,0966 |
   | 3 | n ≢ 0 | M | −π²/18 | −0,5483 (sin campo: −π²/6) |

4. **Ventana límite** (Resultado): con n ≡ 0 es la celda de Voronoi de 2πΛ_w escalada por
   κ_d = 1 − (d² − 1)/(d² ε_d) alrededor de t = 0.
   - d = 2: κ_2 = 1 − 3/(8(ln 2 − ¼)) = 0,15378.
   - d = 3: κ_3 = 0,14245, es decir, θ/π = 2κ_3/3 = **0,0950** en la recta raíz y κ_3/√3 = **0,0822** en la
     recta peso.
   - Con n ≢ 0, **Resultado** (teoría de campos, d ≥ 3): no hay ventana lisa; el testigo queda siempre en
     un cruce de sectores con corrientes distintas. En el límite, su flujo se desancla en un intervalo,
     pero en un cruce de tres sectores de color. Lo exacto es solo la Proposición 2 (≥ 2 sectores).
5. **Prueba d = 2 superada:** ±π²/6 con las dos paridades y ψ_∞/π = 1 − 3/(8(ln 2 − ¼)), con y sin
   campo (en d = 2 el campo no actúa).

## 1. Familia de testigos y Lema exacto

En el gauge uniforme, los datos t (autofases de la holonomía, sin traza) dan
U_t = diag(e^{i t_a/n}). Se toma el testigo
$$Y = -(1\otimes U_s)\,P\,(1\otimes U_s)^\dagger - (h\otimes1 + 1\otimes h),$$
con s un flujo de Cartan propio del testigo y h = diag(h_a) sin traza: es decir, z_ab = −e^{i(s_b−s_a)/n},
D_ab = −δ_ab − (h_a + h_b) en la parametrización de `qudit.py`.

**Lema 1 (exacto).** Sea V = U_s^† U_t y P^s el intercambio girado. Entonces
$$N = \frac{n\,(|\mathrm{tr}\,V|^2-1)}{d(d-1)},\qquad
\lambda_{\max}(W) = \frac nd-\min_{\mathbf N}\big[E_{\mathbf N}(s)+2h\cdot Q_{\mathbf N}\big],$$
con E_N(s) la energía fundamental de Σ_i P^s_{i,i+1} en el sector de números de color **N** y
Q_N = **N** − (n/d)(1,…,1) su carga (un peso). Por tanto
$$v_{\max}(n,t)\le\frac{n/d-G(s)}{N},\qquad G(s)=\max_h\min_{\mathbf N}\big[E_{\mathbf N}(s)+2h\cdot Q_{\mathbf N}\big]
=\min_{\mu\ge0,\ \sum\mu=1,\ \sum\mu Q=0}\ \sum_{\mathbf N}\mu_{\mathbf N}E_{\mathbf N}(s).$$

*Demostración.* Para N:
- P_A = (I − P)/2, y P(1⊗V)P = V⊗1, así que tr(P(1⊗V)P(1⊗V^†)) = |tr V|²;
- de ahí tr(Yρ_e(1)) = −(d − |tr V|²)/(d(d−1)), mientras que tr(Y)/d² = −1/d;
- el campo no contribuye, porque las marginales de ρ_e son I/d.

Para λ_max:
- W = ΣY_e − c_0 con c_0 = n tr(Y)/d² = −n/d (h no tiene traza);
- Σ_e(h⊗1 + 1⊗h) = 2h·N̂, que en el sector **N** vale 2h·Q_N;
- W conserva los números de color.

La última igualdad es la dualidad de la programación lineal. ∎

En s = t, N = n(d+1)/d. Con h = 0 queda la cota «isótropa». En d = 2 la familia es el testigo de
Heisenberg (Δ = 1) con flujo. Comprobado contra `qudit.ring_witness`/`ring_norm` a 1e-10.

## 2. Proposición exacta: el nivel superior ocupa varios sectores si n ≢ 0

**Proposición 2 (exacta, todo n y toda holonomía).** Sea x* un óptimo de la familia invariante de
`qudit.py` (invariante bajo el toro diagonal y la traslación; restringirse a ella no pierde generalidad,
porque el promedio de cualquier óptimo sobre ese grupo compacto es otro óptimo invariante, y solo para
ellos tienen sentido los sectores de color). Esta familia da v_max. Entonces 0 está en la envolvente convexa de las cargas Q_N de los sectores de
color que contienen el nivel superior de W(x*). En particular, si n ≢ 0 (mód d), el nivel superior ocupa
al menos dos sectores de color distintos.

*Demostración.*
1. Los términos de un sitio D_ab = h_a + h_b están en la familia y no cambian N (Lema 1). Añaden
   −2h·Q_N en el sector **N**.
2. Si 0 ∉ conv{Q_N : N superior}, por separación existe h con h·Q_N ≥ c > 0 en esos sectores.
3. Para δ > 0 pequeño, W(x*) − 2δ h·Q̂ baja λ_max:
   - los sectores superiores bajan 2δc;
   - los demás, separados por un gap γ > 0, suben como mucho 2δ|h| máx|Q| < γ.
4. Como N no cambia, v baja, lo que contradice la optimalidad de x*.
5. Si n ≢ 0 (mód d), ningún sector tiene Q = 0, porque los números de color no pueden ser iguales. ∎

**Comprobación:** los 107 óptimos de la tarea 7 con n ≢ 0 la cumplen (sección 7 del guion).

**Lo que la Proposición no dice.** «Al menos dos sectores» **no** implica que el testigo esté en una
cúspide ni que no haya ventana. En la tarea 7 y en \cite{anillos}, «liso» significa un solo momento, o
que las corrientes P·dW/ds·P no se desdoblan; dos sectores degenerados con la misma corriente son lisos.
El contraejemplo es d = 2 con n impar (párrafo siguiente). Que la tarea 7 no encontrara filas «liso» con
n ≢ 0 es **coherente** con la Proposición, no una consecuencia de ella.

**Comparación con los qubits.** En d = 2 y n impar, la condición se cumple sola: la simetría
R_x(π)∘reflexión hace degenerados los sectores S^z = ±½, y por eso el punto liso de los qubits sigue
siendo liso.

**Por qué en d = 3 no ocurre lo mismo.** Con holonomía t ≠ 0, el estabilizador de t en el grupo de Weyl
no actúa transitivamente sobre la órbita de cargas {q_a}. La degeneración tiene que forzarla el campo.

**Resultado (teoría de campos, no demostrado):** para d ≥ 3 y n ≢ 0 ese cruce es una cúspide en el flujo,
porque los sectores degenerados tienen corrientes distintas: sus redes de puntos lisos πq_a + 2πΛ_w
difieren con la carga (en d = 2 las dos cargas ±q comparten red y las corrientes coinciden). Por tanto
no hay ventana lisa con n ≢ 0.

**Apoyo numérico** (no prueba):
- las cúspides certificadas como forzadas en la tarea 7 (200 de 212 filas; sin certificar θ = 0 con
  n ≢ 0 y 4 filas de bisección);
- el desdoblamiento de corrientes > 1e-3 en todas sus filas doblete o triple;
- los 107 óptimos con n ≢ 0 que cumplen la Proposición.

## 3. Entrada de teoría de campos

**Normalización.** En Σ P, el anillo SU(d) es crítico y su límite es SU(d)_1:
- velocidad v = 2π/d y c = d − 1 (Messio–Mila 2012; Affleck 1988);
- e_∞(d) = 1 − (2/d)[ψ(1) − ψ(1/d)] (Sutherland 1975).

**Modos cero** (bosones en el retículo de pesos; marco de SU(d)_1 de Affleck 1986, 1988 y
Di Francesco–Mathieu–Sénéchal 1997, cap. 15). **La ecuación (1) no está contrastada con el original de
Affleck**: es mi formulación del espectro estándar de SU(d)_1, no una cita textual. Un estado de baja
energía tiene:
- momentos de Cartan (p_L, p_R), con p_L ∈ Λ_w;
- carga Q = p_L + p_R en la clase de n fundamentales, es decir, Q ≡ ω_r (mód Λ_r) con r = n mód d.

Un flujo de Cartan s en el enlace de cierre desplaza p_L → p_L + s/2π y p_R → p_R − s/2π. Con
λ = −p_L ∈ Λ_w:
$$E_{Q,\lambda}(s)=n\,e_\infty-\frac{\pi^2(d-1)}{3dn}+\frac1{dn}\Big[\pi^2|Q|^2+|s-\pi Q-2\pi\lambda|^2\Big]+o(1/n).\tag{1}$$
(El corchete es 2π²(|p_L|² + |p_R|²) con p_L = −λ y p_R = Q + λ, es decir, (2πv/n)(h + h̄) con
h = |p_L|²/2 y h̄ = |p_R|²/2.)

**Contrastes de (1) que sí se pueden hacer** (de coherencia, no sustituyen la lectura del original):
- con Q = ω_r y λ = 0 da el primario (0, ω_r), de peso h̄ = |ω_r|²/2 = r(d − r)/(2d): para r = 1,
  (d − 1)/(2d), el peso conforme de la fundamental de SU(d)_1 (¼ en d = 2, ⅓ en d = 3);
- en d = 2 con n impar reproduce las torres (½, 0) ⊕ (0, ½), con E − n e_∞ = −π²/(6n) + π²/(2n) = π²/(3n);
- numéricamente, n(E_0 − n e_∞) por clase (tabla de abajo).

**Consecuencias de (1):**
- **Rigidez.** B = ρ_d = v/2π = 1/d, isótropa: la única forma cuadrática invariante por Weyl en la
  subálgebra de Cartan es la de Killing.
- **Puntos lisos**, los mínimos de la rama de carga Q: s ∈ πQ + 2πΛ_w.
  - n ≡ 0: Q = 0, la red 2πΛ_w.
  - n ≢ 0: Q en la órbita de Weyl de ω_r, el conjunto 2π(½Wω_r + Λ_w). Son los semipesos de la tarea 7.
- **Comprobación de la normalización de d = 2:**
  - n par: E(Θ) − E(0) = Θ²/(4n) = DΘ²/n con D = 1/4 (Shastry–Sutherland 1990; Σ P = 2ΣS·S + n/2);
  - n impar: el punto liso pasa a Θ = π.
- **Heurística de fermiones** (coherente con (1)). Un color con N_a partículas en un anillo de n sitios
  tiene su flujo liso en π(N_a − 1). La cadena de Jordan–Wigner añade π(n − 1). Queda s_a ≡ πN_a
  (mód 2π y un desplazamiento común). Con n = dq + r, los r colores con q + 1 partículas dan s ≡ π1_S,
  |S| = r.
- **LSM.** La dependencia en n mód d es la de Lieb–Schultz–Mattis en la versión SU(n) de Affleck–Lieb
  (1986), y la de su argumento de giro (Yamanaka–Oshikawa–Affleck 1997): un giro 2πω cambia el momento
  en 2πr/d.

**Comprobación numérica de (1)** (guion, sección 3; energía fundamental sin giro):

| n | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| n(E_0 − n e_∞) | −2,671 | 3,251 | 1,400 | −2,318 | 2,871 | 1,755 | −2,255 | 2,694 | 1,898 | −2,232 |

(1) predice −2π²/9 = −2,193 para n ≡ 0 y +2π²/9 = +2,193 para n ≢ 0 (el término de carga suma 4π²/9).
- n ≡ 0 converge desde abajo.
- n ≢ 0 alterna entre n ≡ 1 y n ≡ 2. Las medias de los pares (4,5), (7,8) y (10,11) son 2,33, 2,31 y
  2,30, y bajan hacia 2,19.

## 4. El dual a orden 1/n² (Resultado)

Se desarrollan el Lema 1 y (1) hasta O(1/n²):
- |tr V|² = d² − d|t − s|²/n² + O(n⁻⁴), con t y s sin traza;
- n/d − G = n ε_d + [π²(d−1)/3 − G̃(s)]/(dn), con ε_d = 1/d − e_∞(d);
- G̃(s) = mín sobre mezclas equilibradas de π²|Q|² + |s − πQ − 2πλ|², por el Lema 1.

Resulta
$$v(n,t)=v_\infty+\frac d{d+1}\,\frac{1}{n^2}\Big[\frac{\pi^2(d-1)}{3d}+H_r(t)\Big]+o(n^{-2}),\qquad
v_\infty=\frac{d\,\varepsilon_d}{d+1},$$
$$H_r(t)=\min_{s,h}\ \max_{Q\equiv\omega_r,\ \lambda\in\Lambda_w}\Big\{A|t-s|^2-B\big[\pi^2|Q|^2+|s-\pi Q-2\pi\lambda|^2\big]-B\,h\cdot Q\Big\},$$
$$A=\frac{d\,\varepsilon_d}{d^2-1},\qquad B=\frac1d .$$

Por tanto
$$\lim n^2[v(n,0)-v(n,t)]=\frac d{d+1}\big[H_r(0)-H_r(t)\big].$$
Cada término del máximo es convexo en s porque A > B (d = 2: A = 0,5909, B = ½; d = 3: A = 0,3887,
B = ⅓). El problema es un SOCP pequeño (`qudit_asint.H_cero`).

**Hipótesis**, la misma que en \cite{anillos} 6.2:
- las deformaciones de dos sitios (D_ab ≠ isótropo, |z_ab| distintos, τ ≠ 0) son estacionarias en el
  punto isótropo y no aportan a O(1/n²);
- en t = 0 el óptimo es exactamente el testigo SU(d) (Proposición de la tarea 6), y el término que las
  acopla con el giro es O(1/n²), así que su ganancia es O(n⁻⁴).

**El campo es la excepción.** Para n ≢ 0 el estado del punto liso tiene una carga O(1), es decir, una
densidad de color desequilibrada O(1/n). El campo gana entonces O(1/n) en λ_max, que es O(1/n²) en v.
Por eso se trata aparte y exactamente.

**Apoyo numérico de la hipótesis** (sección 7):
- n ≡ 0: la ganancia de la anisotropía de dos sitios (tarea 7 frente a la familia) escala como n⁻⁴;
- n ≢ 0: el campo da casi toda la diferencia entre la familia isótropa y el óptimo.

## 5. Consecuencias

### 5.1 n ≡ 0 (mód d)

H_0(t) = mín_s máx_λ [A|t − s|² − B|s − 2πλ|²]; el campo no actúa, porque el fundamental es neutro.

- **Ventana.** El óptimo es liso (un solo λ activo) si y solo si s* = (At − 2πBλ)/(A − B) cae en la celda
  de Voronoi de 2πλ. En t eso es
  $$\text{ventana}=\bigcup_\lambda\Big[2\pi\lambda+\kappa_d\big(\mathrm{Vor}(2\pi\lambda)-2\pi\lambda\big)\Big],\qquad
  \kappa_d=1-\frac BA=1-\frac{d^2-1}{d^2\varepsilon_d}.$$
  La anchura a lo largo de una recta u es ψ_∞(u) = κ_d·ℓ_u, con ℓ_u la distancia desde 0 hasta el
  borde de la celda en la dirección u. Con rigidez anisótropa B_u valdría κ_u = 1 − B_u/A.
  - d = 2: ℓ = π.
  - d = 3: ℓ = 2π/3 en la recta raíz (C es un vértice de la celda hexagonal) y π/√3 en la recta peso
    (M es el punto medio de una arista).
- **Coeficiente.** Fuera de la ventana, el testigo queda anclado en el punto c(t) del borde de la celda, y
  $$\lim n^2[v(n,0)-v(n,t)]=\frac d{d+1}\big[B R_c^2-A|t-c|^2\big],$$
  con R_c = dist(c, 2πΛ_w). En t = c esto da R_c²/(d+1).
  - d = 2, Θ = π: R² = π²/2, que da **π²/6** ✓.
  - d = 3, C: R² = 8π²/9, que da **2π²/9**.
  - d = 3, M: R² = 2π²/3, que da **π²/6**.
  - d general, punto medio de arista: π²(d−1)/(d(d+1)) (d = 4: 3π²/20).
- **Dentro de la ventana**: n²[v(n,0) − v(n,t)] → (d/(d+1))·AB|t|²/(A − B).

### 5.2 n ≢ 0 (mód d)

Es la misma fórmula, con las cargas de la órbita de ω_r y el campo. Sea
G_r(s) = mín_{μ equilibrada} Σ_Q μ_Q[π²|Q|² + dist(s, πQ + 2πΛ_w)²].

- **Coeficiente.** En un punto de anclaje c, el coeficiente vale (G_r(c) − G_r(0))/(d+1).
- **d = 3**, con cargas {q_a} (r = 1) o {−q_a} (r = 2), que tienen las mismas redes: la mezcla equilibrada
  es la uniforme, G̃(s) = π²·⅔ + ⅓Σ_a dist(s, πq_a + 2πΛ_w)², y
  - G̃(0) = 4π²/3, G̃(C) = 8π²/9, G̃(M) = 10π²/9;
  - coeficiente en C: **−π²/9**; en C las tres cargas ya están a la misma distancia y el campo no actúa;
  - coeficiente en M: **−π²/18**; sin campo sería −π²/6;
  - n ≡ 1 y n ≡ 2 son idénticos a este orden.
- **Sin ventana lisa** (Resultado). A orden dominante el óptimo deja siempre al menos 3 cargas activas
  (comprobado en 50 puntos), con corrientes distintas; la parte exacta es la Proposición 2.
- **Desanclaje** (el flujo s* deja un punto fijo):
  - recta peso: anclado en 0 hasta θ/L = 2(1 − κ_3)/3 = **0,5717**; se desliza linealmente hasta M,
    adonde llega en θ/L = 2/3 + κ_3/3 = **0,7142** (anchura κ_3 L); después, anclado en M. El
    deslizamiento ocurre porque, entre 0 y M, G̃ = |s − p̄|² + cte con p̄ = (2/3)M;
  - recta raíz (bisección): anclado en 0 hasta θ/L = (1 − κ_3)/3 = **0,2858**; desliza hasta s = L/3,
    adonde llega en θ/L = 1/3; meseta en s = L/3 hasta θ/L = 1/3 + (1 − κ_3)/3 = **0,6192**; desliza
    de nuevo, pasa por C en θ = C (s* = C) y llega a M en θ/L = 2/3 + κ_3/3 = **0,7142**.

### 5.3 d = 2 (prueba obligatoria)

ε_2 = 2 ln 2 − ½, A = (4/3)(ln 2 − ¼), B = ½ y κ_2 = 1 − 3/(8(ln 2 − ¼)), exacto. Las cargas ±q
comparten la red πq + 2πΛ_w, porque 2πq ∈ 2πΛ_w, así que el campo no actúa.
- **n par:** R² = π²/2, coeficiente +π²/6.
- **n impar:** G(0) = π², G(π) = π²/2, coeficiente (π²/2 − π²)/3 = −π²/6.
- **Ventana:** ψ_∞/π = κ_2 en las dos paridades.

Todo se comprueba numéricamente con `H_cero` (1e-7 y anclaje a ±0,1 % del borde).

**d = 4** (información): κ_4 = 0,12800, v_∞ = 0,86010.

## 6. Comparación numérica

### 6.1 Coeficientes

n²[v(n,0) − v(n,t)], con v(n,0) exacto. «Campo» es la cota superior de la familia del Lema 1 en s = t, que
da una **cota inferior** del coeficiente de v_max, rigurosa salvo por la precisión de los autovalores de
Lanczos (todos los sectores; en n = 12, residuo ‖Hv − Ev‖ < 1e-8 en C y M y reevaluación con otra semilla;
en n ≤ 11 solo arranque aleatorio con k = 4 y 40 vectores, y denso hasta 300 estados). «Iso» es la misma familia con h = 0. «T7» es el
óptimo de la tarea 7 (horquilla 1e-7; con n ≤ 5, primal).

| n | r | C campo | C iso | C T7 | M campo | M iso | M T7 |
|---|---|---|---|---|---|---|---|
| 3 | 0 | 2,853 | 2,853 | 3,743 | 2,250 | 2,250 | 3,600 |
| 4 | 1 | −2,019 | −2,019 | −1,725 | −1,657 | −2,485 | −1,240 |
| 5 | 2 | −0,560 | −0,560 | −0,466 | +0,043 | −1,386 | +0,062 |
| 6 | 0 | 2,500 | 2,500 | 2,858 | 1,915 | 1,915 | 2,568 |
| 7 | 1 | −1,708 | −1,708 | −1,618 | −1,098 | −2,223 | −1,050 |
| 8 | 2 | −0,835 | −0,835 | −0,800 | −0,231 | −1,562 | −0,226 |
| 9 | 0 | 2,427 | 2,427 | 2,588 | 1,846 | 1,846 | 2,136 |
| 10 | 1 | −1,564 | −1,564 | — | −0,959 | −2,099 | −0,936 (sonda) |
| 11 | 2 | −0,943 | −0,943 | — | −0,341 | −1,629 | — |
| 12 | 0 | 2,396 | 2,396 | — | 1,817 | 1,817 | — |
| ∞ | 0 | 2π²/9 = 2,193 | | | π²/6 = 1,645 | | |
| ∞ | ≢0 | −π²/9 = −1,097 | | | −π²/18 = −0,548 | (−π²/6 = −1,645) | |

**Lectura de la tabla:**
- **Convergencia.** Las seis sucesiones de la columna «campo» son monótonas hacia su límite y no lo cruzan
  (comprobado). n ≡ 1 queda por debajo y n ≡ 2 por encima.
- **Medias de pares (4,5), (7,8), (10,11):**
  - C: −1,289, −1,272, −1,253 (límite −1,097);
  - M: −0,807, −0,665, −0,650 (límite −0,548).
- **n ≡ 0 en n = 12:** el cociente con el límite es 1,092 (C) y 1,104 (M). Los qubits van parecido
  (1,17 en n = 12).
- **El campo decide en M con n ≢ 0.** La tarea 7 queda a 0,005–0,05 de «campo» (n ≥ 7) y a más de 1 de
  «iso».

### 6.2 Rigidez finita (n ≡ 0 salvo indicación; B = 1/3 a orden dominante)

| n | B local en 0 (raíz = peso) | B amplitud C | B pendiente C | B amplitud M | B pendiente M |
|---|---|---|---|---|---|
| 3 | 0,5000 | 0,4336 | 0,3581 | 0,4559 | 0,4135 |
| 6 | 0,4033 | 0,3800 | 0,3501 | 0,3881 | 0,3719 |
| 9 | 0,3834 | 0,3689 | 0,3492 | 0,3741 | 0,3638 |
| 12 | 0,3750 | 0,3641 | 0,3489 | 0,3681 | 0,3603 |

**Definiciones de las columnas:**
- **B local:** n ΔE/|s|², con extrapolación de Richardson.
- **B amplitud:** n[E(c) − E(0)]/R_c².
- **B pendiente:** nE'(c⁻)/(2R_c), con la derivada lateral de segundo orden.

**Lectura:**
- La rigidez local en 0 es exactamente isótropa: raíz = peso a 1e-6, por la simetría de Weyl. Su
  cociente con 1/3 es 1,21, 1,15 y 1,125, como en la nota de literatura (§9).
- La pendiente en C casi no cambia con n (≈ 1,047 × ⅓).

**n ≢ 0: rigidez local en el punto liso M** (sin campo):

| n | 4 | 5 | 7 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|
| raíz | 0,2929 | 0,5110 | 0,3170 | 0,4464 | 0,3281 | 0,4201 |
| peso | 0,5303 | 0,2912 | 0,4603 | 0,3227 | 0,4311 | 0,3344 |
| media | 0,411 | 0,401 | 0,389 | 0,385 | 0,380 | 0,377 |

- En M solo queda una simetría Z_2 × Z_2, así que se permiten dos rigideces.
- La anisotropía (máx/mín − 1) vale 0,81, 0,75, 0,45, 0,38, 0,31 y 0,26, es decir, ≈ 3/n, y cambia de
  signo entre n ≡ 1 y n ≡ 2.
- Es un efecto de tamaño finito: τ = 0 por construcción en la energía de Σ P^s. Explica las amplitudes
  0,85–1,6 de los «semipesos» de la tarea 7.

### 6.3 Ventana (n ≡ 0; θ/π, medida desde t = 0)

| n | T7 raíz | T7 peso | iso raíz | iso peso | fracción T7/∞ (raíz, peso) | raíz/peso T7 |
|---|---|---|---|---|---|---|
| 3 | 0 (< 8e-5) | 0 (< 5e-5) | 0,0455 | 0 | — | — |
| 6 | 0,0199 | 0,0174 | 0,0649 | 0,0328 | 0,210, 0,211 | 1,146 |
| 9 | 0,0346 | 0,0286 | 0,0674 | 0,0405 | 0,365, 0,348 | 1,209 |
| 12 | — | — | 0,0682 | 0,0441 | — | — |
| ∞ | 0,0950 | 0,0822 | | | 1 | 2/√3 = 1,155 |

- **Forma de la ventana** (cociente raíz/peso). El límite 2/√3 = 1,155 es un **Resultado** (geometría
  del hexágono: vértice frente a arista). Los datos finitos no lo verifican:
  - el cociente de la tarea 7 es 1,146 (n = 6) y 1,209 (n = 9): con dos puntos, el segundo se aleja del
    límite; es compatible con él, pero no converge visiblemente;
  - la familia isótropa, que con n ≡ 0 es el objeto que la teoría de orden dominante describe
    literalmente (el campo no actúa), da cocientes 1,98, 1,66 y 1,55 (n = 6, 9, 12), lejos de 1,155;
  - su ventana se aplana lejos del límite: 0,0649, 0,0674, 0,0682 en la raíz (incrementos 0,0025 y
    0,0008; **72 %** del límite 0,0950 en n = 12) y 0,0328, 0,0405, 0,0441 en la recta peso (**54 %**
    de 0,0822).
- **Explicación posible (no demostrada) del aplanamiento.** La ventana isótropa la fija la pendiente en
  la cúspide: (2/3)(1 − B_pend/A) reproduce la de la raíz (0,0662, 0,0678, 0,0683), y para llegar a 0,0950
  B_pend tendría que bajar a ⅓. La pendiente en C está casi plana (0,3501, 0,3492, 0,3489). Ajustando
  B_amp = B + CR² y B_pend = B + 2CR² (desglose del revisor):

  | n | B_local en 0 | B (ajuste, C) | CR² (C) | n·CR² (C) | B (ajuste, M) | CR² (M) |
  |---|---|---|---|---|---|---|
  | 6 | 0,4033 | 0,4099 | −0,0299 | −0,180 | 0,4043 | −0,0162 |
  | 9 | 0,3834 | 0,3886 | −0,0197 | −0,177 | 0,3844 | −0,0103 |
  | 12 | 0,3750 | 0,3794 | −0,0153 | −0,183 | 0,3759 | −0,0078 |

  B del ajuste sigue a B_local (+0,3–1,6 %), y el término anarmónico CR² va como ~1/n (n·CR² ≈ −0,18 en
  C). B_pend = B + 2CR² es, por tanto, una cancelación entre la bajada lenta (~1/ln n) de B y un término
  ~1/n que se apaga. Esto no contradice el límite, pero hace previsible que la ventana isótropa no se
  acerque a 0,095 de forma monótona hasta que el término 1/n se apague.
- **Conclusión:** la forma y la anchura límite son compatibles con los datos, pero los datos con n ≤ 12
  no las deciden.
- **Escala.** La tarea 7 está muy por debajo del límite (21 % y 36 %), como en los qubits (12 % en
  n = 6, 33 % en n = 12, 74 % en n = 2048). κ es una diferencia pequeña, 1 − B/A con B/A = 0,857, y un
  exceso del 5–10 % en B_n la reduce mucho.
- **La familia isótropa no predice la ventana finita.** La sobrestima con n ≡ 0, y lo mismo pasa en d = 2
  con Δ = 1:

  | n (d = 2) | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | ψ_iso/π (Δ = 1) | 0,0607 | 0 | 0,0675 | 0,0001 | 0,0717 | 0,0141 | 0,0746 | 0,0236 | 0,0769 | 0,0307 | 0,0787 |
  | tab:ventana (Δ óptimo) | 0,0186 | 0,0173 | 0,0329 | 0,0266 | 0,0432 | 0,0331 | 0,0509 | 0,0381 | 0,0569 | 0,0420 | 0,0615 |

  A estos tamaños, la anisotropía de dos sitios (Δ en qubits) mueve el borde de la ventana a O(1). Solo
  se anula en el límite.

### 6.4 Descomposición del óptimo: campo, anisotropía de dos sitios y τ

τ = 0 es el SDP con z_ab = −r_ab e^{i(s_b − s_a)/n}, r_ab ≥ 0, s = t, y D_ab libres. Se tabula n²Δv
(estados de Clarabel 'optimal_inaccurate'; la cadena T7 ≤ τ0 ≤ campo se cumple a 1e-7):

| n | punto | campo − τ0 | τ0 − T7 (= efecto de τ) |
|---|---|---|---|
| 4 | C | 0,272 | 0,021 |
| 4 | M | 0,417 | 0 |
| 5 | C | 0,004 | 0,090 |
| 5 | M | 0,018 | 0 |
| 6 | C | 0,358 | 0 |
| 6 | M | 0,653 | 0 |
| 7 | C | 0,076 | 0,014 |
| 7 | M | 0,049 | 0 |

**Con n ≡ 0, la anisotropía de dos sitios** (T7 − iso) vale 0,358 → 0,161 (C) y 0,653 → 0,290 (M) de
n = 6 a n = 9. En n = 6 τ no aporta nada (tabla); en n = 9 el SDP τ = 0 no se ha calculado y se
**supone** que τ tampoco aporta en C y M. El cociente es 0,45 y 0,44 = (6/9)², así que la ganancia en v escala como n⁻⁴,
como pide la hipótesis de estacionariedad. Con n ≢ 0, T7 − campo en M vale 0,417 (4), 0,049 (7) y
0,023 (10).

### 6.5 Flujo del testigo con n ≢ 0: familia con campo frente a la tarea 7

s*/L de la familia (s en una malla de 25 puntos de la recta) frente a θ_W/L de la tarea 7, en
θ_i = iL/12:

| n, recta | i = 0..12 (familia / T7) |
|---|---|
| 7, peso | 0/0 ×10, luego 1/0,93, 1/1, 1/1 |
| 8, peso | 0/0 ×7, luego 1/0,89, 1/1 … |
| 5, peso | 0/0 ×6, luego 1/0,94, 1/1 … |
| 8, raíz | 0/0 ×4, 0,38/0,38, 0,38/0,42, 0,38/0,44, 0,38/0,46, 0,67/0,76, 1/0,93 … |
| 7, raíz | 0/0 ×5, 0,21/0,17, 0,25/0,20, 0,25/0,19, 0,67/0,59, 1/0,95 … |

- **Recta peso: anclaje en 0 y salto a M.** La familia reproduce dónde termina el anclaje en 0 en la
  tarea 7, salvo n = 4 (en T7 deriva a partir de 8L/12). El orden dominante lo pone en 0,572 L;
  n = 7 (0,75–0,83 L) y n = 8 (0,50–0,58 L) lo encierran.
- **Recta raíz: meseta intermedia.** El límite la pone en L/3 (para θ/L ∈ [1/3, 0,619]), entre la de
  n = 7 (0,21–0,25) y la de n = 8 (0,375).
- **Cota superior.** v_familia ≥ v_T7 en los 104 puntos (mín −9e-16).

## 7. La anisotropía de amplitudes: ¿tamaño finito o τ?

**1. Amplitudes de la energía** (la comparación de «semipesos» de la tarea 7). τ no interviene: el
hamiltoniano Σ P^s tiene τ = 0 por construcción. Todo es tamaño finito:
- la rigidez en M es anisótropa a O(1/n) (≈ 3/n) y cambia de signo con r (sección 6.2);
- además, la tarea 7 comparaba E_0 con la fórmula sin el campo, que no es lo que ve el dual.

**2. Amplitudes de v.** Domina, con mucha diferencia, el **campo de color**, que no es una anisotropía
de dos sitios sino un término de un sitio:
- con n ≢ 0 da la mayor parte del efecto en M, de O(1/n²) a orden dominante;
- lo que queda (anisotropía de dos sitios, τ incluido) es O(n⁻⁴) con n ≡ 0 y pequeño con n ≢ 0;
- τ en concreto: 0 en M y en C con n ≡ 0; con n ≢ 0 en C, n²Δv ≤ 0,09 (n ≤ 7).

**3. Qué domina al tamaño de la tarea 7** (n = 9, C): de los 0,40 de exceso sobre 2π²/9,
- 0,23 son tamaño finito de la parte isótropa, compatible con ~1/ln n: (exceso de la familia)·ln n vale
  0,55, 0,51 y 0,50 en C y 0,48, 0,44 y 0,43 en M (n = 6, 9, 12), sin ajustar ningún modelo; si es así,
  domina cada vez más;
- 0,16 son anisotropía de dos sitios (∝ n⁻²).

## 8. Lo que no cuadra o no queda decidido

1. **Convergencia lenta.** Con n ≤ 12 ningún límite queda fijado numéricamente.
   - n ≡ 0: 9–10 % por encima en n = 12.
   - Medias de pares con n ≢ 0: 14 % (C) y 19 % (M) por encima.
   - Todas las sucesiones son monótonas. Con n ≡ 0, (exceso)·ln n es casi constante (C: 0,55, 0,51,
     0,50; M: 0,48, 0,44, 0,43), compatible con correcciones 1/ln n, pero no se ha ajustado ningún modelo
     ni verificado la amplitud logarítmica de SU(3) (a_N = 1/N en la nota de literatura).
2. **Alternancia n ≡ 1 / n ≡ 2.** A orden dominante son iguales. A tamaño finito difieren en O(1/n)
   relativo y con signo opuesto en: el coeficiente, n(E_0 − n e_∞), la rigidez en M y los puntos de
   anclaje. La atribuyo a correcciones impares en la carga (un invariante cúbico d_abc Q^aQ^bQ^c de los
   operadores irrelevantes), sin calcularlas.
3. **Ventana finita.** La teoría da el límite y la forma (cociente 2/√3) como Resultado; los datos con
   n ≤ 12 no los deciden.
   - El cociente raíz/peso de la tarea 7 se aleja del límite (1,146 → 1,209 frente a 1,155).
   - La ventana de la familia isótropa se aplana al 72 % (raíz) y al 54 % (peso) del límite en n = 12,
     con cocientes 1,98 → 1,55. La explicación posible es la cancelación entre B_local (~1/ln n) y un
     término anarmónico ~1/n en la pendiente de la cúspide (§6.3), no demostrada.
   - La escala finita no la predice nada de lo calculado: la tarea 7 está al 21–36 % del límite, y ni la
     familia isótropa ni su calibración en d = 2 dan el borde a estos n, porque la anisotropía de dos
     sitios cuenta a O(1) en el borde.
4. **Deslizamiento con n ≢ 0.**
   - El orden dominante predice un desanclaje continuo (recta peso: 0,572 → 0,714 L).
   - La familia con campo, en la malla de s, salta de 0 a M sin pasos intermedios.
   - La tarea 7 muestra valores intermedios (0,89–0,94 L), con τ ≠ 0 en la raíz.
   - No sé si a n grande se abre el deslizamiento continuo, ni si la tarea 7 tiene un paso intermedio
     real o un efecto de τ.
5. **La meseta L/3 de la raíz con n ≢ 0** sale a 0,21–0,25 (n = 7) y 0,375–0,42 (n = 5, 8) en la familia
   y en la tarea 7. Rodea el valor límite, pero con solo dos tamaños por clase.
6. **Contenido de las torres conformes con n ≢ 0.** La ecuación (1) es la de SU(d)_1 con la regla de LSM
   de la clase de carga. Su normalización se comprueba con n(E_0 − n e_∞) → ±2π²/9: las medias 2,33,
   2,31 y 2,30 bajan hacia 2,19 pero no lo alcanzan. Tampoco he leído las torres de Affleck (1988) en el
   original.
7. **La hipótesis de estacionariedad** se apoya en dos tamaños con n ≡ 0 (6 y 9: cociente 0,44–0,45
   frente a (6/9)² = 0,444), y en qubits tampoco está demostrada (problema abierto de \cite{anillos}).
8. **n = 3 (n = d)** es singular: v(3,0) = 1, ventana nula y multipletes grandes. No representa a
   n ≡ 0.
9. **Interior del triángulo 0–C–M.** No se ha estudiado: como en la tarea 7, solo las dos rectas.

## 9. Referencias

- **[anillos]** Este trabajo, holonomia_werner.tex: secciones 6.2 y 6.3, Resultado del coeficiente y
  ψ_∞.
- **B. Sutherland**, Phys. Rev. B **12**, 3795 (1975): solución del modelo SU(N) de intercambio. No leído
  en el original; vía Messio–Mila.
- **L. Messio, F. Mila**, Phys. Rev. Lett. **109**, 205306 (2012): e_∞(N), v = 2π/N, c = N − 1.
- **I. Affleck**, Nucl. Phys. B **265**, 409 (1986) y Nucl. Phys. B **305**, 582 (1988): SU(N)_1 como
  límite de las cadenas SU(N). Títulos sin verificar (nota de literatura).
- **I. Affleck, E. H. Lieb**, Lett. Math. Phys. **12**, 57 (1986): teorema de LSM para SU(n).
- **E. H. Lieb, T. Schultz, D. Mattis**, Ann. Phys. **16**, 407 (1961).
- **M. Yamanaka, M. Oshikawa, I. Affleck**, Phys. Rev. Lett. **79**, 1110 (1997): argumento de giro de
  LSM.
- **B. S. Shastry, B. Sutherland**, Phys. Rev. Lett. **65**, 243 (1990): rigidez exacta, D = 1/4 en
  Δ = 1.
- **N. Laflorencie, S. Capponi, E. S. Sørensen**, Eur. Phys. J. B **24**, 77 (2001): correcciones
  logarítmicas de la rigidez (SU(2)).
- **P. Di Francesco, P. Mathieu, D. Sénéchal**, *Conformal Field Theory* (Springer, 1997): modelos WZW
  y pesos conformes de SU(N)_1.
- Nota de literatura de la tarea 2:
  [2026-11-09-rigidez-literatura.md](file:///home/alex/fisica/pyhton/holonomy/docs/superpowers/notes/2026-11-09-rigidez-literatura.md),
  secciones 4–6, para la verificación de las citas.

# Órdenes clásicos de celda k: planteamiento y primeras pruebas

Notas de trabajo (22 de noviembre de 2026). Continúan `robustez_mps.tex` (en adelante [R]) y
`xxz_anillo_retorcido.tex` ([X]).

## 1. Datos de estado fundamental: un techo que no necesita núcleo

Sea `h` un término de dos sitios, hermítico e invariante por la U(1) de la holonomía, y sea
`H_n(Θ) = Σ_j h_j^Θ` el anillo con el giro en el gauge uniforme. Sean `e_n(Θ) = λ_max(H_n(Θ))/n`,
`e_∞ = lim e_n` y `σ` la marginal de dos sitios de un estado de la cadena infinita que alcanza `e_∞`,
promediada sobre traslaciones y sobre la U(1). Con esos datos, `τ = σ − I/d²` y
`tr(hτ) = e_∞ − tr h/d² =: g > 0`, así que el testigo `h` en la condición (D) del Teorema puente de
[R] da

    p(n,Θ) ≤ 1 + (e_n(Θ) − e_∞)/g .

Es decir, `p* = 1` y `ε_D = (e_n(Θ) − e_∞)⁺/g`. Para datos que son el estado fundamental de algo,
`h` hace de techo, tenga o no núcleo `σ`: es una respuesta parcial al problema «Rango completo» de
[R], que pide un techo que sustituya al del núcleo. Los datos de Néel de [R, Sec. 6] son exactamente
este caso con `h` el XXZ de anisotropía `Δ_∞`: allí la estacionariedad de `Δ_∞` es lo que identifica
la marginal del fundamental con los datos.

Con esto, la robustez de estos datos se reduce, como en [X], a dos cotas de tamaño finito del propio
`h` retorcido:
- **(E)**: `e_n(Θ) − e_∞ ≤ C e^{−n/ξ}`, uniforme en Θ;
- **(M)**: un estado del anillo con marginales a distancia `≤ C e^{−n/ξ}` de las de la cadena, giradas.

## 2. La clase de testigos

`h = h_cl + ε h_q`, con:
- `h_cl` diagonal en la base de carga, con fundamental clásico de periodo k, único salvo las k
  traslaciones, y con coste de defecto positivo (condición de Peierls);
- `h_q` invariante por la U(1), que transporta carga.

**Conjetura de trabajo.** Para |ε| pequeño:
1. **k | n**: sensibilidad `O(e^{−n/ξ})`. El mecanismo es el de [X]: Kirkwood–Thomas alrededor del
   orden clásico, localidad de los coeficientes de Taylor y el lema de Schwarz.
   Por el lado del testigo, se espera un multiplete de k niveles (análogo del doblete de Néel), cuya
   dependencia en Θ venga del túnel entre los k órdenes, que transporta carga fraccionaria.
2. **k ∤ n**: el anillo fuerza un defecto móvil de carga fraccionaria q. Ve el flujo como qΘ, y la
   respuesta es la curvatura de su banda: `R_n ≈ c q²/n²`. La sensibilidad, por la cota de Kohn por
   los dos lados, es `≈ 4R_n/(n·g)`, es decir, del orden de `n^{−3}`.

## 3. Modelo mínimo (k = 3)

Con qubits y solo términos de dos sitios no hay orden clásico de periodo 3 con fundamental único
(un Ising con campo solo da ferro o Néel), así que el primer ejemplo necesita d = 3. Se usa espín 1,
`s_i ∈ {0,1,2}` y `S^z = s − 1`:

    H(Θ) = −Σ_i P_asc(i,i+1) + ε Σ_i (e^{iΘ/n} S⁺_i S⁻_{i+1} + h.c.),

con `P_asc` el proyector sobre los pares `(s, s+1 mod 3)`. El orden clásico es `…0 1 2 0 1 2…`, de
carga 0 por celda. El guion es `scripts/celda_k_flujo.py`.

Un segundo modelo, `scripts/celda_k_llenado.py`, tiene el mismo término clásico pero **llenado
1/3**: la carga es `q = 1` si `s = 0` y `0` si no, y la perturbación hace saltar la partícula,
`ε Σ_{x=1,2} (e^{iΘ/n} |x 0⟩⟨0 x| + h.c.)`. Aquí la carga por sitio es fraccionaria, la situación de
Lieb–Schultz–Mattis de [R, Cor. «LSM con celda»]. Con 3 ∤ n el llenado no cabe en el anillo y el
defecto lleva carga ±1/3 o ±2/3.

## 4. Resultados numéricos (diagonalización exacta, n ≤ 11)

`R_n = max_Θ E_0 − min_Θ E_0` es la respuesta al flujo del testigo. Por la cota de Kohn por los dos
lados, la sensibilidad de los datos es `≈ 4R_n/(n·g)`.

**Modelo de carga entera** (ε = 0,1):

| n | n mod 3 | R_n |
|---|---|---|
| 5 | 2 | 3,5·10⁻³ |
| 6 | 0 | 7,3·10⁻⁵ |
| 7 | 1 | 1,4·10⁻⁴ |
| 8 | 2 | 3,3·10⁻⁵ |
| 9 | 0 | 3,1·10⁻⁷ |
| 10 | 1 | 1,0·10⁻⁶ |
| 11 | 2 | 2,7·10⁻⁷ |

La caída es exponencial en las tres clases de n, también con 3 ∤ n. El defecto existe y es móvil:
en el sector `S^z = 0` su banda tiene anchura ≈ 20ε², y ese sector es el fundamental. Pero es
**neutro**: moverlo no transporta carga y no ve el flujo. Los defectos cargados (`S^z = ±1`) solo se
mueven a orden ε³ y quedan por encima.

**Modelo de llenado 1/3:**

| n | n mod 3 | R_n (ε = 0,1) | n²R_n (ε = 0,1) | n²R_n (ε = 0,3) |
|---|---|---|---|---|
| 5 | 2 | 3,1·10⁻⁴ | 0,0077 | 0,155 |
| 6 | 0 | 1,8·10⁻⁵ | 0,0007 | 0,049 |
| 7 | 1 | 1,9·10⁻³ | **0,0930** | **0,676** |
| 8 | 2 | 1,2·10⁻⁶ | 0,0001 | 0,0137 |
| 9 | 0 | 1,3·10⁻⁷ | 0,00001 | 0,0070 |
| 10 | 1 | 9,4·10⁻⁴ | **0,0938** | **0,687** |

- **n ≡ 1 (mod 3): polinómica.** n²R_n es constante y la sensibilidad de los datos va como n⁻³,
  como en el anillo de Néel impar. En el fundamental, el defecto tiene carga 2/3 y una banda de
  anchura 4t ≈ 3,7ε². Avanza 3 sitios por salto, así que da 3 vueltas antes de volver al punto de
  partida y ve el flujo efectivo 3·(2/3)·Θ = 2Θ: `E_0(Θ)` tiene periodo π, con arcos de parábola
  (malla de 25 valores). La predicción `R_n ≈ tπ²/n²` da 1,87·10⁻³ en n = 7, frente a 1,90·10⁻³.
- **n ≡ 0 (mod 3): exponencial**, como predice la conjetura.
- **n ≡ 2 (mod 3): exponencial en estos tamaños.** En el fundamental (con ε = 0,1) el defecto tiene
  carga 1/3 y es casi inmóvil. Su «banda» no tiene n niveles separados del resto, y su anchura
  *decrece* con n (1,6·10⁻⁵ en n = 8 y 1,3·10⁻⁷ en n = 11). Así que no hay un salto local de orden
  fijo: el defecto de carga 1/3 parece localizado a todos los órdenes, y moverlo exige un proceso
  global. El defecto de carga 2/3 de ese sector sí es móvil (anchura ≈ 1,3ε²), pero queda por
  encima con ε ≤ 0,2.

### 4.1 Por qué el defecto de carga 1/3 no se mueve: una ley de conservación

Guion: `scripts/defecto_tercio.py`. Con n = 3m+2 y m+1 partículas hay exactamente **2n** estados
clásicos con un solo defecto: un enlace con incremento 2, de dos especies según el valor `b ∈ {0, 2}`
que sigue al defecto. El hamiltoniano efectivo de des Cloizeaux sobre ellos (peso clásico del cúmulo
≥ 0,98) da:
- las dos especies no se mezclan a ningún orden (elementos ~10⁻¹⁴);
- el único salto es el de desplazamiento 1, que cambia el valor de **todos** los sitios, porque el
  patrón entero se desplaza. Su orden en ε es 3, 5 y 7 con n = 5, 8 y 11, es decir, 2m+1;
- el salto local de 3 sitios, que solo cambia 3 valores, es exactamente nulo.

El mecanismo es este. La perturbación solo mueve la partícula, `(0,x) → (x,0)`, y nunca intercambia
las dos especies neutras 1 y 2. Por tanto, **la secuencia cíclica de 1 y 2, leída saltándose las
partículas, se conserva** (como la secuencia de espines «comprimida» del modelo t-J con salto
infinito). Con 2m+1 sitios neutros, la alternancia 1 2 1 2 … no cierra, y esa secuencia tiene una
repetición: es el defecto de carga 1/3. Solo puede desplazarse si la secuencia comprimida rota, es
decir, si una partícula da la vuelta entera al anillo cruzando los 2m+1 sitios neutros: de ahí el
orden 2m+1 y la caída exponencial. Los defectos de carga 2/3 tienen un número par de sitios neutros,
ninguna repetición en la secuencia, y se mueven a orden ε².

**Contraprueba.** Se añade un intercambio neutro, `λ Σ (|2 1⟩⟨1 2| + h.c.)`, sin carga ni fase de
flujo (`LAM` en el entorno), que rompe la conservación. Con ε = λ = 0,1:
- el defecto de carga 1/3 salta 3 sitios con |t| = 3,33·10⁻³, igual en n = 8 y en n = 11: es una
  banda local;
- en su sector, n²R_n = 0,0291 y 0,0329 en n = 8 y 11, frente a la predicción `tπ²` = 0,0329. Al dar
  saltos de 3 sitios con carga 1/3, ve el flujo efectivo 3·(1/3)·Θ = Θ;
- el sector de carga −2/3 queda casi degenerado (diferencia ~10⁻⁶), con n²R_n = 0,0330 y el mínimo en
  Θ = π en lugar de 0. El fundamental global es el mínimo de las dos ramas, y su R_n es del orden de
  un cuarto: n²R_n = 0,0146 y 0,0085 en n = 8 y 11, y tiende a 0,033/4.

Así, con una perturbación genérica, también n ≡ 2 da respuesta de orden n⁻² (y sensibilidad de orden
n⁻³). La localización era consecuencia de una simetría no genérica de la perturbación.

### 4.2 Carga entera con perturbación aleatoria

Guion: `scripts/celda_k_aleatorio.py`. El término de enlace es
`h = −P_asc + ε K + δ D`, con:
- `K` una matriz hermítica aleatoria con todos los elementos fuera de la diagonal que conservan la
  carga total (5 amplitudes complejas gaussianas);
- `D` una diagonal aleatoria (9 energías).

El giro va en el segundo sitio. Con `K` = salto `S⁺S⁻` y δ = 0, el guion reproduce el modelo de
carga entera (`R_7` = 1,3543·10⁻⁴). Se registran la carga `S^z` del sector fundamental en Θ = 0, que
con 3 ∤ n es la carga del defecto, y `R_n`.

**ε = δ = 0,1, semilla 1, las mismas 12 muestras con n = 6, …, 11.** En los 24 casos (muestra × clase
de n) el comportamiento lo decide la carga del defecto, sin excepciones:
- **n ≡ 1: 8 cargados y 4 neutros.** En los cargados, n²R_n entre n = 7 y 10 es constante: por
  ejemplo 0,1456 → 0,1509, 0,0427 → 0,0433 y 0,0324 → 0,0328. En los neutros, R_n cae un factor
  100–1000 entre n = 7 y 10.
- **n ≡ 2: 7 cargados y 5 neutros.** En los cargados, n²R_n entre n = 8 y 11 vale, por ejemplo,
  0,0065 → 0,0067, 0,0136 → 0,0129, 0,0230 → 0,0234 y 0,0308 → 0,0335. Una muestra (la 6) aún
  deriva: 0,0407 → 0,0495. En los neutros, R_n cae un factor de unos 100 cada 3 sitios.
- **El ganador depende de n mod 3.** En las muestras 3, 8 y 10 la carga del defecto ganador
  cambia según n ≡ 1 o n ≡ 2. Es lógico: con n ≡ 1 el defecto es una repetición y con n ≡ 2 un salto,
  dos objetos distintos.
- Con 3 | n, la caída es siempre exponencial.

**ε = 0,1, δ = 0 (solo término cinético aleatorio), semilla 2.** El defecto es cargado en 7 de 12
muestras con n = 7 y en 5 de 12 con n = 8. Que ganara el neutro en el modelo de la Sección 3 era
una propiedad de ese modelo, no de la clase. Una advertencia: en la muestra 9 con n = 10 el
fundamental en Θ = 0 es neutro, pero n²R_n = 0,02. Lo más probable es que un sector cargado casi
degenerado gane en otros valores de Θ. «El defecto de menor energía» hay que entenderlo, pues,
uniformemente en Θ.

Conclusión: con carga entera por sitio, que el defecto sea neutro o cargado **no es genérico ni en
un sentido ni en otro**: depende del término y de n mod 3, con fracciones comparables. Lo que sí es
robusto es la regla: polinómica si el defecto que gana está cargado, exponencial si es neutro.

### 4.3 Un criterio a priori: energía del defecto a segundo orden

Guion: `scripts/criterio_defecto.py`. En cada sector de carga `S^z` que contiene estados clásicos de
un defecto (subespacio `P`, energía clásica `E_cl = −(n−1)`), se calcula el hamiltoniano efectivo de
Schrieffer–Wolff a segundo orden,

    H_eff = PVP + PV Q (E_cl − H_cl)⁻¹ Q VP,     V = H − H_cl,

y su autovalor mínimo, `E_def(S^z)`. Recoge el primer orden en la parte diagonal δD, el segundo en la
cinética εK y la banda del defecto a ese orden. Todos los procesos son locales, así que `E_def(S^z)` −
`E_def(S^z′)` no depende de n en cuanto n supera el alcance de los procesos de segundo orden. El
criterio dice que **gana el sector de menor `E_def`**, y, por la Sección 4.2, que la sensibilidad es
polinómica si ese sector está cargado y exponencial si es neutro.

**Resultado.** El criterio se calcula en n = 7 y 8 y se compara con el ganador de la diagonalización
exacta en n = 10 y 11, en Θ = 0:
- ε = δ = 0,1, semilla 1: **24/24** aciertos. Los márgenes son ≥ 0,1, porque domina el primer orden
  en δ.
- ε = 0,1, δ = 0, semilla 2: **24/24** aciertos. Aquí todo se decide a orden ε², y los márgenes del
  criterio van de 1,3·10⁻⁴ a 4,9·10⁻². Los de la diagonalización exacta en n + 3 coinciden con ellos en
  orden de magnitud, en general con un 10–20 % de diferencia. El peor caso (muestra 4, n = 7) tiene
  margen 1,3·10⁻⁴ en el criterio y 6,5·10⁻⁴ en la diagonalización exacta: el tercer orden ya cuenta,
  pero no cambia el ganador.

Así, para perturbaciones pequeñas, **qué defecto gana es un cálculo local de segundo orden**, y con él
la regla «cargado ⇒ polinómica, neutro ⇒ exponencial» se vuelve predictiva: se decide sin
diagonalizar el anillo.

Límites:
- El criterio se ha probado solo con ε = 0,1. Con ε mayor, el tercer orden (del que dependen la banda
  de los defectos cargados del modelo de la Sección 3 y el margen de la muestra 4) puede cambiar el
  ganador cuando los márgenes de segundo orden son pequeños.
- ~~La muestra 9 de δ = 0 con n = 10.~~ Explicada; ver la Sección 4.4.

### 4.4 La muestra 9: hay que comparar mínimos de banda, no Θ = 0

En esa muestra (δ = 0, semilla 2) el sector neutro gana en Θ = 0, pero la respuesta global es
polinómica (n²R_n = 0,02 en n = 10). Por sectores, con 17 valores de Θ:

| sector | n = 7: mín − E_min | n = 7: n²R | n = 10: mín − E_min | n = 10: n²R |
|---|---|---|---|---|
| S^z = +1 | 0 | 0,1332 | 0 | 0,1323 |
| S^z = 0 | 3,6·10⁻⁴ | 0,0066 | 2,0·10⁻⁴ | 0,0002 |
| S^z = −1 | 1,9·10⁻² | 0,0956 | 1,9·10⁻² | 0,0866 |

- El sector cargado +1 tiene el mínimo más bajo, pero su banda tiene el mínimo cerca de Θ ≈ π, no en
  0. Las amplitudes complejas de `K` actúan como un flujo de fondo que desplaza la banda del defecto
  cargado (al neutro no le afecta).
- En Θ = 0, por tanto, el anillo muestrea la banda cargada lejos de su mínimo, y gana el neutro.
  Al variar Θ, el fundamental cambia de sector (la secuencia de ganadores en n = 10 es
  `00011000110000110`).
- La respuesta del sector cargado es polinómica (n²R constante); la del neutro, exponencial
  (1,4·10⁻⁴ → 2,2·10⁻⁶).
- Mientras la curvatura del sector cargado, ≈ 0,13/n², supere su ventaja sobre el neutro (≈ 2·10⁻⁴),
  la R global queda limitada por esa ventaja (R = 2,06·10⁻⁴ en n = 10). Para n mayor, el sector
  cargado gana en todo Θ y n²R_n → 0,13.

**Criterio corregido.** El ganador se decide comparando **el mínimo sobre Θ** de cada sector, que en
la cadena infinita es el mínimo de la banda del defecto en el continuo de momentos. Con esa versión
(`TODO_THETA=1` en `criterio_defecto.py`, 8 valores de Θ), en las 24 muestras con δ = 0 el criterio
en n = 7 y 8 acierta el ganador en n = 10 y 11 en **24/24** casos. Solo cambia de ganador respecto a la
comparación en Θ = 0 la muestra 9 con n ≡ 1, que pasa de neutra a cargada (+1), como exige su
respuesta polinómica.

### 4.5 Hasta dónde llega el criterio: ε mayor

Criterio con el mínimo sobre Θ, calculado en n = 7 y 8 y comparado con el ganador de la
diagonalización exacta en n = 10 y 11, con 12 muestras por conjunto y semillas distintas:

| ε | δ | semilla | segundo orden (SW) | diagonalización exacta en n pequeño |
|---|---|---|---|---|
| 0,1 | 0 | 2 | 24/24 | 24/24 |
| 0,1 | 0,1 | 1 | 24/24 | 24/24 |
| 0,2 | 0 | 3 | 24/24 | 23/24 |
| 0,3 | 0 | 4 | 23/24 | 24/24 |
| 0,2 | 0,2 | 5 | 21/24 | 23/24 |
| **total** | | | **116/120** | **118/120** |

La última columna usa como criterio la diagonalización exacta del anillo pequeño (n = 7 u 8), en vez
del segundo orden. Los fallos son de dos clases distintas:
- **Se rompe el segundo orden** (3 de los 4 fallos de SW): en esos casos la diagonalización exacta en
  n = 7 u 8 ya acierta, con márgenes grandes (0,006–0,065). Son errores de la teoría de
  perturbaciones, no del tamaño. En el peor (ε = 0,3, muestra 8), SW se equivoca en 0,18 en la
  energía del sector −1. Meter el desorden diagonal en los denominadores no lo arregla: con δ = 0,2
  aparecen denominadores casi resonantes y el sector neutro se desploma (−0,30 frente al 0 exacto).
- **Efecto de tamaño con margen diminuto** (los 2 fallos de la diagonalización en n pequeño, y 1 de
  SW): márgenes de 0,9·10⁻³ y 1,1·10⁻³ en n = 10. El mínimo sobre 8 valores de Θ muestrea las
  bandas de forma distinta en n = 7 y en n = 10, y cuando dos sectores están a 10⁻³ el orden se puede
  invertir.

Conclusión: el criterio de segundo orden es fiable en el régimen perturbativo, ε ≲ 0,2 sin desorden
diagonal y ε = δ = 0,1 (96/96). Más allá, la versión práctica es diagonalizar exactamente un anillo
pequeño y comparar los mínimos sobre Θ de cada sector (118/120). Sus fallos solo aparecen cuando dos
defectos están casi degenerados, y entonces el ganador asintótico no se puede leer de anillos con
n ≤ 11 por ningún método.

### 4.6 La regla con ε = 0,3

`celda_k_aleatorio.py` con ε = 0,3, δ = 0 y semilla 4: las mismas 12 muestras de la Sección 4.5, con
n = 5, …, 11. Para las muestras 2 y 6 se calculó además n = 14 en el sector ganador (S^z = −1;
~7·10⁵ estados, unos 3 minutos por muestra).

- **3 | n: exponencial, pero más lenta.** R cae un factor de 6 a 20 entre n = 6 y 9, frente a más de
  100 con ε = 0,1: ξ ≈ 1,1–1,7. El orden se debilita, pero no se funde.
- **n ≡ 1.**
  - Con defecto cargado, n²R entre n = 7 y 10 se mantiene en tres muestras (0,773 → 0,760,
    0,505 → 0,504, 0,457 → 0,408) y baja un 22 % en dos (1,61 → 1,26 y 2,05 → 1,59).
  - Con defecto neutro, n²R cae claramente más rápido (por ejemplo, 0,154 → 0,018 y 1,29 → 0,26),
    aunque en algún caso de forma lenta (2,30 → 0,85).
- **n ≡ 2.**
  - Con defecto neutro, n²R cae siempre: un factor de 3 a 25 entre n = 8 y 11.
  - Con defecto cargado y margen grande (0,09 y 0,2), el comportamiento es ambiguo:

    | | n = 8 | n = 11 | n = 14 |
    |---|---|---|---|
    | muestra 2: n²R | 1,404 | 0,667 | 0,352 |
    | muestra 6: n²R | 3,696 | 2,142 | 1,539 |

    El ajuste exacto `R = c/n² + A e^{−n/ξ}` da c = 0,19 y ξ = 1,90 (muestra 2), y c = 1,27 y
    ξ = 1,81 (muestra 6). ξ es del orden de la longitud del túnel con 3 | n en las mismas muestras
    (1,4 y 1,7). La lectura natural es que la respuesta es la de la banda del defecto cargado (c/n²)
    más el túnel entre los órdenes, que con ε = 0,3 aún no se ha apagado en n ≤ 14. En la muestra 6 el
    exponente local baja (3,71 → 3,37), como corresponde a acercarse a n⁻². En la muestra 2 sube
    (4,34 → 4,65), lo que es compatible con esa lectura solo si el término exponencial aún domina (su
    exponente local efectivo es n/ξ ≈ 5–7). **Con tres puntos y tres parámetros, el ajuste no es una
    prueba.**

Conclusión: con ε = 0,3 la separación entre defectos cargados y neutros sigue siendo visible, pero se
estrecha: el túnel tiene ξ ≈ 2 y contamina R_n hasta n ≈ 14. Para confirmar el límite c/n² en n ≡ 2
hacen falta n ≥ 17, fuera del alcance de la diagonalización exacta (≈ 2·10⁷ estados por sector).

### 4.7 DMRG con flujo (n ≡ 2, muestras 2 y 6 de ε = 0,3)

Guion: `scripts/dmrg_flujo.py` (TeNPy 1.1.1, instalado en `.venv` sin añadirlo a `pyproject.toml`).
Usa un MPO periódico con el enlace largo, la carga S^z conservada (sector −1), χ = 160 y 8 valores
de Θ con arranque en caliente. Reproduce la diagonalización exacta:
- n = 8: las 8 energías a 10⁻⁹;
- n = 11: R = 5,50936·10⁻³ y 1,77056·10⁻² (diagonalización exacta: 5,5094·10⁻³ y 1,7706·10⁻²);
- n = 14: R = 1,79456·10⁻³ y 7,85069·10⁻³ (diagonalización exacta: 1,7946·10⁻³ y 7,8514·10⁻³).

Tarda 0,5 h por muestra con n = 14, 0,9 h con n = 17 y 1,2 h con n = 20.

| n | muestra 2: n²R | muestra 2: n²E″(mín) | muestra 6: n²R | muestra 6: n²E″(mín) |
|---|---|---|---|---|
| 8 | 1,404 | 0,747 | 3,696 | 2,150 |
| 11 | 0,667 | 0,556 | 2,142 | 1,658 |
| 14 | 0,352 | 0,517 | 1,539 | 1,320 |
| 17 | 0,333 | 0,417 | 1,171 | 1,123 |
| 20 | 0,261 | 0,503 | 0,741 | 1,065 |

**R con 8 valores de Θ no es una buena medida a estos tamaños.** E₀(Θ) está hecha de arcos con
cúspides en los máximos, y el mínimo se desplaza con n por el flujo de fondo de las amplitudes
complejas (argmin en 1,5π, 0, 0,75π…). Con paso π/4, la cúspide puede quedar a π/8 de la malla, lo
que subestima R hasta un 23 % y explica los saltos de n²R. La **curvatura en el mínimo**, `E″`, con el
punto mínimo y sus dos vecinos (vértice a ≤ 0,45 pasos de la malla), es más robusta:
- **muestra 2:** n²E″ = 0,56, 0,52, 0,42 y 0,50 para n = 11, …, 20. Es constante dentro del error de
  la diferencia finita;
- **muestra 6:** n²E″ = 1,66 → 1,32 → 1,12 → 1,065. Sigue bajando, pero cada vez menos (cocientes
  0,80, 0,85 y 0,95): se está estabilizando hacia ≈ 1.

Un túnel con ξ ≈ 1,8 haría caer la curvatura un factor de unos 5 cada 3 sitios. Ninguna de las dos
muestras lo hace: **la respuesta al flujo del defecto cargado es polinómica, de orden n⁻²**. La
muestra 2 está ya en su límite (n²E″ ≈ 0,5), y la muestra 6 se estabiliza hacia n²E″ ≈ 1 (1,123 en
n = 17 y 1,065 en n = 20). Así que el ajuste de tres puntos de la Sección 4.6 no daba
buenas constantes, aunque sí la forma correcta. El cociente R/E″ tiende a valores entre 0,5 y 1,
compatible con arcos de periodo 2π/q en Θ con q = 2–3 (π²/(2q²) = 1,23 y 0,55). Falta medirlo bien.

### 4.8 Del testigo a los datos: el dual exacto (en curso)

Hasta aquí todo era la respuesta al flujo del testigo fijo `w₀ = −h₀`. La sensibilidad de los datos
es el mínimo sobre **todos** los testigos U(1) de dos sitios (33 parámetros reales con llenado 1/3).
Guiones:
- `scripts/dual_celda.py`: λ_max suavizado, `T log Σ exp(λᵢ/T)` sobre los 2n autovalores mayores,
  BFGS y T de 10⁻² a 10⁻⁷. Da una horquilla certificada,
  `min F_T − T log k ≤ p ≤ f(w_final)`.
- `scripts/testigo_datos.py`.

Validación con los datos de Néel de [R] (κ = 0,4, n = 8): n²[v(8,0) − v(8,π)] = 0,5727, frente al
0,57 de la tabla de [R], con horquillas de anchura 3·10⁻⁷. BFGS sobre λ_max sin suavizar y los planos
de corte no sirven: la banda del defecto deja los n niveles superiores casi degenerados, y los dos
métodos se atascan con errores de hasta 0,13.

Datos: la marginal del fundamental del modelo de llenado 1/3 en un anillo de 12 sitios (Θ = 0). Con
8 valores de Θ (los primeros resultados solo dan la cota superior en cada Θ):

| ε | λ | n | p | s_datos | s_testigo_fijo |
|---|---|---|---|---|---|
| 0,2 | 0 | 5 | 0,610786 | 2·10⁻¹³ | 6,5·10⁻⁴ |
| 0,4 | 0 | 5 | 0,644460 | 3·10⁻¹³ | 3,4·10⁻³ |
| 0,6 | 0 | 5 | 0,6980–0,6983 | 2,7·10⁻⁴ | 7,1·10⁻³ |
| 0,4 | 0 | 6 | 1,000000002 | 2·10⁻⁹ | 9,5·10⁻⁴ |
| 0,6 | 0 | 6 | 1,00000001 | 7·10⁻⁹ | 3,6·10⁻³ |
| 0,4 | 0 | 7 | 0,740409 | 3·10⁻¹² | 4,5·10⁻³ |
| 0,6 | 0 | 7 | 0,77418–0,77427 | 8,9·10⁻⁵ | 8,0·10⁻³ |
| 0,4 | 0,2 | 5 | 0,651461 | 2·10⁻¹³ | 1,5·10⁻² |
| 0,6 | 0,2 | 5 | 0,7070–0,7073 | 3,3·10⁻⁴ | 2,9·10⁻² |
| 0,4 | 0,2 | 6 | 1,0003–1,0015 | 1,2·10⁻³ | 2,3·10⁻³ |
| 0,4 | 0,2 | 7 | 0,744736 | 5·10⁻¹² | 4,5·10⁻³ |
| 0,6 | 0,2 | 6 | 1,0025–1,0044 | 1,9·10⁻³ | 5,9·10⁻³ |
| 0,6 | 0,2 | 7 | 0,78063–0,78075 | 1,2·10⁻⁴ | 8,1·10⁻³ |

Tres efectos, ninguno de los cuales estaba en el análisis del testigo fijo:
1. **El techo clásico.** Con 3 ∤ n y ε pequeño, el testigo óptimo es casi clásico, no depende de Θ, y
   la sensibilidad de los datos es **exactamente cero**, también con n ≡ 1, donde el testigo fijo daba
   respuesta polinómica. Pasa con y sin núcleo: con ε = 0,4 y n = 7, tanto con λ = 0 como con
   λ = 0,2. Es el análogo del techo de Ising `(n−2)/n` de [R] con n impar. El testigo
   clásico vale `1 − a_cl/n + O(ε²)` y `w₀` vale `1 − E_def/(n g)`. Por esa cuenta, el término O(ε²) del
   clásico debería hacerle perder cuando n ≳ 1/ε², pero los datos no muestran ese cruce. Con ε = 0,6
   (1/ε² ≈ 3), la parte cuántica asoma, pero se queda en un 1–4 % de la respuesta del testigo fijo:
   n³ s_datos = 0,034 en n = 5 y 0,031 en n = 7 con λ = 0, y 0,0407 y 0,0413 con λ = 0,2. Es compatible con una sensibilidad de orden
   n⁻³ con un prefactor diminuto, como si el testigo óptimo fuera una mezcla dominada por el clásico
   con una pequeña fracción de `w₀` que no crece con n. Con dos puntos de clases distintas (n ≡ 2 y
   n ≡ 1) esto no es más que una sugerencia.
2. **El techo por rango.** Con λ = 0, σ tiene núcleo exacto, generado por |11⟩ y |22⟩: la secuencia
   comprimida alterna, y dos sitios neutros vecinos nunca son iguales. Es el mismo tipo de núcleo que
   los datos MPS de [R]. Entonces p ≤ 1 para todo Θ y, con 3 | n, p = 1 a 10⁻⁸ en todo Θ: la
   sensibilidad la fija el estado primal, no el túnel del testigo (≈ 10⁻⁹ frente a 10⁻³).
3. **El caso genérico** (λ = 0,2, sin núcleo, n = 6): la dependencia en Θ sí llega a los datos, con
   s_datos = 1,2·10⁻³ y 1,9·10⁻³ (ε = 0,4 y 0,6), un 51 % y un 32 % de la del testigo fijo. Falta certificarla con la horquilla,
   que el guion ya calcula.

Conclusión provisional: **el paso del testigo a los datos no es automático a tamaño finito.** Los dos
techos (el clásico y el del núcleo) pueden anular la sensibilidad de los datos aunque el testigo
límite tenga una respuesta al flujo polinómica o exponencial. Asintóticamente, si σ tiene un único
testigo límite óptimo (`w₀`, el análogo de la unicidad de Δ_∞ en [R, Prop. 17]), el testigo óptimo
debería converger a él y la regla del testigo volvería a valer. Pero con n ≤ 7 no se ve esa
convergencia en los n incompatibles con la celda, y el diagnóstico de la Sección 4.3 sobre qué
defecto gana dice qué le pasa al testigo, no a los datos. Con 3 | n y datos genéricos, en cambio,
la dependencia en Θ sí llega a los datos (30–50 %).

Lo que falta para cerrar el paso:
- más tamaños de la misma clase (n = 7 y 10, o n = 8 y 11), para ver si la fracción
  s_datos/s_testigo crece. Con el dual actual, n = 8 tarda unas 3 h por punto y n ≥ 9 es inviable:
  haría falta bloquear por momento y carga;
- la descomposición del testigo óptimo (cuánto pesa `w₀` y cuánto la parte clásica), que se puede
  leer directamente del `w` que devuelve el dual.

### 4.9 Anatomía del testigo óptimo: el dual aplana la banda del defecto

Guion: `scripts/testigo_optimo.py`. Descompone el testigo óptimo `w` (normalizado con tr(wτ) = 1 y sin
la parte proporcional a la identidad) frente a `w₀` normalizado igual. `α` es el peso de la parte
fuera de la diagonal de `w₀` (el salto) dentro de `w`. Datos de llenado 1/3, Θ = 0.

| ε | λ | n | α | diagonal de w (pares asc. / desc. / iguales) | niveles superiores del anillo con w |
|---|---|---|---|---|---|
| 0,4 | 0,2 | 5 | 0 | +1,09 / −1,09 / 0 | todos degenerados (10⁻¹⁴) |
| 0,6 | 0,2 | 5 | 0,26 | ≈ +1,1 / ≈ −1,05 / 00: −0,31, 11 y 22: +0,07 | 5 niveles en 9·10⁻⁷ |
| 0,6 | 0,2 | 6 | 0,088 | ≈ +1,0 / ≈ +0,9 / 00: +0,76, 11 y 22: −3,18 | doblete a 5·10⁻³, hueco 0,28 |
| 0,6 | 0,2 | 7 | 0,051 | ≈ +1,08 / ≈ 0 / ≈ −1,1 | 13 niveles en 2·10⁻⁵ |

Con `w₀`, en cambio, la banda del defecto con n = 7 tiene anchura ≈ 0,3–0,4 (niveles entre 0,15 y
0,5 por debajo del máximo).

- **El testigo clásico es topológico.** Con ε = 0,4 y n = 5, `w` es +c en los pares ascendentes, −c en
  los descendentes y 0 en los iguales. Su suma sobre el anillo es c veces el número de vueltas del
  patrón (3·⌊n/3⌋ como máximo): es el análogo del testigo de Ising de [R], diagonal y ciego al flujo.
- **El mecanismo.** El dual minimiza λ_max, y una banda dispersiva sube λ_max en media anchura de
  banda. El testigo óptimo, por tanto, **quita la dispersión al defecto**: reduce el salto (α baja de
  0,26 a 0,05 entre n = 5 y 7) y reajusta las energías de los pares. Con la dispersión desaparece la
  dependencia en Θ, y eso explica el 1–4 % de la Sección 4.8.
- **Por qué sale barato.** Las direcciones que usa actúan sobre pares raros en σ: 11 y 22 pesan
  4,9·10⁻⁴ con λ = 0,2 (y 0 con λ = 0), y 00 pesa 2,4·10⁻³. Cambiar sus coeficientes cuesta en el
  bulto ~n·(peso)·δ, mientras que aplanar la banda gana ~(anchura)·δ, que no es extensivo.
  Estimación del cruce: n₀ ~ 0,3/(5·10⁻⁴·3) ~ 200 sitios. Para n ≫ n₀, el coste extensivo debería
  obligar al testigo óptimo a converger a `w₀` (α → 1), y la regla del testigo pasaría a los datos.
  Con n ≤ 7, α todavía **baja**: estamos en el régimen previo.
- **Con núcleo (λ = 0)**, las direcciones sobre 11 y 22 no cuestan nada en el bulto para ningún n.
  La supresión podría ser permanente, salvo que esas direcciones solas no basten para aplanar la
  banda. La n³·s ≈ 0,03 casi constante de la Sección 4.8 sugiere que no bastan del todo.

**Consecuencia para la regla.** La pregunta de si los datos son sensibles al flujo con n incompatible
no la decide solo el defecto del testigo límite. La decide también **si el espacio de testigos tiene
direcciones baratas en el bulto que actúen sobre el defecto**, y eso depende del espectro de σ
(pesos pequeños o núcleo). En el anillo de Néel de [R] la familia de testigos útiles se reducía por
simetría a un parámetro (Δ), y no había esa libertad. Para una versión rigurosa haría falta, además
de la unicidad del testigo límite, una cota inferior de los pesos de σ, o restringir el resultado a
datos con σ de rango completo y bien condicionado.

### 4.10 Datos genéricos y orden más débil

Guion: `scripts/datos_genericos.py`. `h₀ = −P_asc + ε K + 0,1 D`, con llenado 1/3 y K una
combinación aleatoria de **todos** los términos U(1)-invariantes fuera de la diagonal, incluida la
conversión entre las especies neutras 1 y 2. Los datos salen del anillo de 12 sitios (se comprueba
que el fundamental tiene 4 partículas; la semilla 1 tiene 3 y se descarta). Sensibilidades
certificadas por la horquilla del dual y 8 valores de Θ.

**σ no se puede condicionar bien sin fundir el orden.** El par 00 (dos partículas vecinas) forma él
solo el bloque de carga 2, así que ningún término U(1) lo mezcla con nada, y su peso es el de
encontrar dos partículas juntas: 10⁻³–10⁻⁴ mientras haya orden de celda 3. El autovalor mínimo de
σ queda entre 6·10⁻⁵ y 3·10⁻⁴, también con ε = 0,8. Los pares raros son una propiedad estructural
de los datos ordenados, igual que los espines paralelos vecinos en los datos de Néel de [R].

| semilla | ε | mín. autovalor de σ | n | s_datos | s_testigo_fijo | cociente | α |
|---|---|---|---|---|---|---|---|
| 2 | 0,5 | 3,0·10⁻⁴ | 5 | 2,07·10⁻² | 1,86·10⁻² | 1,11 | 1,19 |
| 2 | 0,5 | | 6 | 7,38·10⁻³ | 1,35·10⁻² | 0,55 | −0,33 |
| 2 | 0,5 | | 7 | 1,44·10⁻² | 1,12·10⁻² | 1,29 | 1,22 |
| 3 | 0,5 | 9,3·10⁻⁵ | 5 | 1,37·10⁻³ | 1,70·10⁻² | 0,08 | 0,81 |
| 3 | 0,5 | | 6 | ≤ 2,2·10⁻⁷ | 1,28·10⁻² | ≈ 0 | 0,77 |
| 3 | 0,5 | | 7 | 7,87·10⁻³ | 1,07·10⁻² | 0,73 | 0,80 |
| 3 | 0,8 | 1,7·10⁻⁴ | 5 | 2,31·10⁻² | 2,66·10⁻² | 0,87 | 0,80 |
| 3 | 0,8 | | 6 | 8,36·10⁻³ | 1,56·10⁻² | 0,54 | 1,06 |
| 3 | 0,8 | | 7 | 1,55·10⁻² | 1,12·10⁻² | 1,39 | 0,81 |

- **Con perturbaciones genéricas, la dependencia en Θ sí llega a los datos.** Con n = 5 (3 ∤ n), el
  cociente es 0,87 y 1,11 en dos de las tres configuraciones, frente al 1–4 % del modelo de llenado
  1/3 con solo salto de partícula. El testigo óptimo conserva la mayor parte de `w₀` (α ≈ 0,8–1,2).
- **La excepción es la configuración más ordenada** (semilla 3 con ε = 0,5, el autovalor mínimo de σ
  más pequeño). Ahí el cociente es 0,08 con n = 5, y con n = 6 la sensibilidad cae por debajo de
  2·10⁻⁷ aunque α = 0,77. Pero con n = 7 (n ≡ 1) el cociente sube a 0,73: la supresión desaparece en
  cuanto n crece un poco, o depende de la clase del defecto (repetición con n ≡ 1, salto con n ≡ 2). Al subir ε a 0,8, la misma semilla pasa a 0,87 y 0,54. Encaja con la
  dependencia en la condición de σ de la Sección 4.9, pero con solo tres configuraciones es una
  tendencia, no una ley.
- **Con 3 | n (n = 6),** los datos ven alrededor de la mitad de la dependencia del testigo fijo (0,54
  y 0,55), salvo el caso más ordenado. Que la cota del testigo fijo sobrestime la sensibilidad real
  es lo esperado: el error primal es de segundo orden en el túnel ([X], Teorema de refinamiento).
- `α` no es un buen resumen cuando el testigo óptimo reordena la parte fuera de la diagonal (α < 0
  con la semilla 2 y n = 6, con cociente 0,55).

- **Con n = 7 (n ≡ 1), las tres configuraciones heredan la dependencia en Θ:** cocientes de 0,73,
  1,29 y 1,39, con α ≈ 0,8–1,2, incluida la más ordenada. El testigo fijo no acota la sensibilidad
  (cocientes > 1): solo la estima. La cota rigurosa es la de Kohn con el testigo óptimo de cada Θ.

**Lectura.** Con perturbaciones genéricas, la dependencia en Θ del testigo llega a los datos con un
cociente de orden 1 en 8 de los 9 casos con n = 5–7. La única excepción clara es el par n = 5–6 de
la configuración más ordenada, y con n = 7 esa misma configuración vuelve a heredarla. La supresión
casi total de las Secciones 4.8–4.9 es propia del modelo de llenado 1/3 con solo salto de
partícula, que tiene un núcleo exacto o casi exacto en σ y pocas direcciones de testigo
competidoras. No es lo genérico. Con esto, la regla «defecto cargado ⇒ sensibilidad polinómica» pasa
a los datos en el caso genérico, con la salvedad de los datos muy ordenados a tamaño pequeño. Lo
que no se puede ver con n ≤ 7 es el exponente (n⁻³) en los datos: para eso harían falta tamaños de
la misma clase separados (7 y 10), inalcanzables con el dual actual.

## 5. Lectura provisional

La conjetura de trabajo de la Sección 2 es falsa tal como está: que la celda no sea compatible con n
no basta para una sensibilidad polinómica. Lo que hace falta es que el defecto de menor energía:
1. **tenga carga** respecto a la U(1) de la holonomía. Con carga entera por sitio puede ser neutro, y
   entonces la caída sigue siendo exponencial. La carga fraccionaria por sitio (la condición de LSM)
   obliga a que el defecto esté cargado;
2. **sea móvil**, con una banda dispersiva de anchura independiente de n. Un defecto cargado pero
   localizado (n ≡ 2 en el modelo de llenado 1/3 sin intercambio neutro) tampoco da potencias de n.
   Pero esa localización viene de una ley de conservación de la perturbación (Sección 4.1), y una
   perturbación genérica la rompe. Lo genérico es que el defecto se mueva; la condición que importa
   de verdad es la primera.

En el anillo de Néel impar se cumplen las dos cosas, y además a orden cero: la pared tiene carga 1/2
y se mueve a primer orden en el salto. Por eso allí no se veía la distinción.

Reformulación: **con datos de estado fundamental de un orden clásico perturbado, la sensibilidad es
polinómica (de orden n⁻³) si y solo si el defecto de menor energía que fuerza el anillo está cargado
y es dispersivo; en otro caso es exponencial.** Para perturbaciones genéricas basta lo primero.
Con carga fraccionaria por sitio (la condición de LSM) el defecto siempre está cargado, y la
sensibilidad es polinómica para todo n incompatible con la celda. Con carga entera por sitio puede
ganar un defecto neutro, y entonces es exponencial aunque k ∤ n. La versión rigurosa del lado exponencial es la de [X],
y la del lado polinómico, la Sección «Anillos impares» de [X], con la ecuación de Bloch de la pared
cambiada por la del defecto.

## 6. Pendiente

- ~~Comprobar que el defecto de carga 1/3 está localizado.~~ Hecho (Sección 4.1): lo localiza la
  conservación de la secuencia comprimida, y un intercambio neutro lo libera. La ley de conservación
  está identificada, pero no demostrada en forma de enunciado; es fácil hacerlo, porque la
  perturbación es un salto de partículas que no reordena las especies neutras.
- ~~Con carga entera: ¿es genérico que gane el defecto neutro?~~ Hecho (Sección 4.2): no; depende
  del término y de n mod 3. La regla «cargado ⇒ polinómica, neutro ⇒ exponencial» se cumple en los
  24 casos con δ = 0,1, donde se comparan dos tamaños de la misma clase. Con δ = 0 solo se ha
  mirado la carga del defecto y n²R_n en n = 10, sin comparar tamaños. Quedan por aclarar la muestra
  6 con n ≡ 2, cuyo n²R_n aún deriva entre n = 8 y 11, y la muestra 9 de δ = 0 (neutra en Θ = 0, con
  n²R_n = 0,02).
- ~~Un criterio a priori de qué defecto gana.~~ Hecho (Sección 4.3): la energía del defecto a
  segundo orden acierta en 48/48. La muestra 9 de δ = 0 se explica comparando mínimos de banda, no
  Θ = 0 (Sección 4.4), y con esa versión acierta 24/24 con δ = 0. Con ε mayor (Sección 4.5), el
  segundo orden acierta 116/120 y la diagonalización exacta en n pequeño, 118/120; el límite de
  validez del segundo orden está hacia ε ≈ 0,2–0,3.
- Con ε = 0,3 (Secciones 4.6 y 4.7), la regla se sostiene: con DMRG hasta n = 20, la curvatura en el
  mínimo del defecto cargado va como n⁻² (muestra 2) o se acerca a ello (muestra 6). Faltan la
  muestra 6 con n = 20, n = 23 y una malla de Θ más fina para medir R sin el sesgo de la cúspide.
  Además, en todos los barridos con 8 valores de Θ, R está subestimado hasta un 23 %, lo que afecta a
  las constantes, no a la clasificación.
- Barrer ε para ver si, con n ≡ 2, el defecto móvil de carga 2/3 pasa a ser el fundamental. Si es
  así, habría una transición en ε entre sensibilidad exponencial y polinómica a n fijo mod 3.
- Pasar del testigo a los datos: calcular `v(n,Θ)` con el dual en la familia de testigos
  U(1)-invariantes cercanos a `h`, al menos para n ≤ 9, y comprobar que la sensibilidad sigue a
  `4R_n/(n·g)`.
- ~~Teoría: enunciar el teorema del lado exponencial.~~ Hecho en parte (Sección 7). Están demostrados:
  - la localidad de los coeficientes de Taylor, el túnel, el enrollamiento y el gauge grande
    (Teorema A);
  - el paso a la cota exponencial y a la robustez, bajo la hipótesis de analiticidad uniforme (A)
    (Teorema B y Corolario C).
  Falta demostrar (A) para la clase (C) (Sección 7.5). En 7.6 se reduce a (S), (G), (A2E\*) y (A2M), con evidencia numérica para (A2E\*) y (G), y con un camino concreto para (A2M) (Kirkwood–Thomas con `Z_k`). En 7.7: en la clase con simetría de reloj (el intercambio SU(k), que incluye el llenado con μ = 1), el
  cuasi-modo `Z_k` funciona y da (A2E\*) demostrado; (A2M) está transcrito y demostrado en 7.8 (Teorema Z, a
  falta solo de (G), con ε < 0). (G) está caracterizada en 7.9 como un hueco de dos defectos sobre el vacío vestido,
  con evidencia numérica estable en n. La teoría de un defecto está en 7.10: el Teorema D (perturbativo, con el hamiltoniano efectivo de
  Bloch) da (G) a todos los órdenes y la sensibilidad `n⁻³` con k ∤ n. **En 7.11, (G) y toda (A)
  quedan demostradas de forma no perturbativa** en la clase con simetría de reloj. Se usa Gershgorin
  para el generador vestido `e^{−G}H'e^{G}`, uniforme en n y comprobado numéricamente hasta n = 12.
  Con eso, el Teorema Z′ es incondicional. **En 7.12, la banda de un defecto con k ∤ n
  es también no perturbativa** (Teorema D′): aislamiento uniforme por Gershgorin en el marco torcido,
  niveles simples por simetría y Schwarz contra el Teorema D. Da la sensibilidad `≍ n⁻³` como teorema.
  Hace falta una simetría que separe las órbitas del defecto, (S_d), que el intercambio SU(k) cumple
  por el contenido. Las dos mitades de la conjetura quedan demostradas para el intercambio SU(k).
  **Carga entera (7.13):** el Lema Z8 demuestra que el vestido de una sola fase no puede funcionar.
  En una isla de otra fase de longitud L, el radio de Gershgorin crece como `(8/3)|ε|L`, y así se ha
  medido. El sustituto es un vestido por fases (Pirogov–Sinai simétrico, con las fases relacionadas
  por T), con la desigualdad clave `ε^{L/2}e^{O(εL)}` sumable. **Paso 1 demostrado** (Lema Z9):
  el Kirkwood–Thomas restringido a contornos cortos contrae uniformemente en n, con residuo
  `(M|ε|)^{ℓ/2}` por sitio. **Paso 2 construido:** con el vestido por fases, el radio de la columna de
  una isla ya no crece con L (≤ 1,45 con ε = 0,1 hasta n = 12, frente a 3,6). **Paredes (Lema Z10,
  condicional a los vacíos de fase (V)):** el radio de la columna queda `≤ η_∂|∂μ|`, sin depender
  del tamaño de los dominios. Con el vestido de pared, `η_∂ ≈ 0,04` con ε = 0,05. **(V), inducción en escalas
  (paso 4):** el esquema está escrito, con escalas `ℓ_m = 2^m ℓ_0`, Z10 como paso inductivo y Peierls
  `(M|ε|e^{4d})^{L/2}`. Numéricamente, el vacío completo tiene constantes uniformes en n con
  ε ≤ 0,1 y existe aun donde Jacobi diverge. **Cambio de base (Lema Z11, esquema):** en la base por fases,
  Gershgorin da hueco positivo y estable de n = 6 a 9 (1,37 → 1,42 con ε = 0,05). En la de una
  fase se degrada y se vuelve negativo con ε = 0,1. **El Lema Z11 está demostrado:**
  serie de Neumann con pesos `e^{σ|Y|}`, cota `‖c − e‖_1 ≤ e^{δ_σ|Y|} − 1`, con δ_σ ∝ d. **Contracción (paso 5):** la del paso 2′
  no hace falta, porque la cota de Z10 es directa. La del paso 4 no puede ser la de Jacobi: su radio
  espectral crece linealmente con n en carga entera (0,40 → 0,51 de n = 6 a 12 con ε = 0,1) y es
  constante en SU(3). Tiene que hacerse en variables por fases, que es el enunciado Lema Z12. **Paso 6 (Pirogov–Sinai):** los polímeros
  tienen que ser contornos generales, no islas; el vacío exacto lo muestra con c = 2, donde dos
  islas cortas dominan. Está escrito el modelo de contornos con la iteración 𝔽, y el Lema Z12 (𝔽
  contrae con constante 1/2, uniformemente en n) está demostrado por piezas a partir de Z9, Z10, Z11
  y Kotecký–Preiss. **Paso 6′:** (R) y (KP) sobran. La
  linealización de Kirkwood–Thomas es `H̃ − E` (Lema Z13). Precondicionando con su diagonal en la
  base por fases, el quasi-Newton resuelve la ecuación completa, así que es exacto sin representación
  de contornos. Su radio espectral es uniforme en n (0,64 con ε = 0,2 en n = 6 y 9, donde Jacobi da
  1,006). **Paso 7:** las dos cotas de Z12′ están
  escritas con constantes, la lineal en la norma de paredes y la cuadrática con el factor `ℓ_{m+1}`.
  **Paso 8 (revisión línea a línea):** la prueba de Z12′ tenía tres pasos no uniformes en n:
  - el término `−RδG` por separado, que es extensivo;
  - el cambio de base, con `e^{ℓ|∂μ|‖δg‖}`;
  - la cota del paso en la norma de Z1.

  Se reparan dentro de una escala con la linealización conexa, una norma de polímeros en
  coordenadas y los Lemas Z11s (soporte) y Z14 (productos conexos sin pérdida). Eso da el Lema Z12″,
  demostrado. Entre escalas se pierde un peso fijo `λ = O(|ε|)`. **Teorema Z″ (carga entera,
  k | n): robustez exponencial demostrada para `n ≤ n_*(ε)`, con `log n_* ≈ 9·10⁻⁶/|ε|`** (para
  `S⁺S⁻`, con `|ε| ≲ 3·10⁻⁶`); sin fases se llega solo a `n ≲ 0,1/|ε|`. **Paso 9:** con ε real, el hueco uniforme
  en *todo* n sale de Nachtergaele–Sims–Young (arXiv:2010.15337, Thm. 6.8 y §8.4.2, ruptura de la
  traslación con contorno periódico). La indistinguibilidad G-simétrica es exacta para los patrones
  clásicos, y la robustez sale subexponencial (Teorema Z‴, constantes sin calcular). **Lo que sigue
  abierto es la uniformidad *exponencial* y compleja en todo n.** El diseño con contornos y KP está
  en 9.1: polímeros = excursiones, vestido de pared completo y una sola pérdida de KP. La inducción
  está **demostrada en espacio-tiempo** (9.5): Lema Z15 (Peierls con escala de tiempo
  `T = ½ ln(1/|ε|)`) y Lema Z16 (Zahradník con fases simétricas, τ fijo y una sola pérdida 2ε′). La
  igualdad de las energías libres truncadas es la traslación con k | n, y no hay que truncar. **Lema Z17**
  (9.6): los contornos que dan la vuelta en espacio son eventos de túnel con actividades
  *circulantes* en las fases. En cada momento p forman un gas escalar en tiempo, y KP da
  `Tr(T^m e^{−βH}) = e^{−βnf_n} Σ_p ω^{pm} e^{−βt_p}(1 + O(e^{−γ_1β/2}))`. Como las multiplicidades
  algebraicas son enteras positivas, esto identifica los autovalores también con ε complejo. **Teorema
  Z⁗: robustez exponencial uniforme en todo n para carga entera con k | n**, con
  `ε_2 ≈ 2,3·10⁻³` para `S⁺S⁻` tras afinar las constantes (9.7: celdas de 2 enlaces, contornos
  4-conexos, ≥ 2 saltos por contorno, conteo exacto de animales, peso KP según la forma; 9.8: integral exacta de la creación y la aniquilación; 9.9: cota de Peierls por denominadores de energía, fila a fila y por segmentos; 9.10: cola por percolación). Es unas 700 veces mejor que el radio de una escala de Kirkwood–Thomas.
  Con contornos en tiempo continuo (9.12–9.13), sin celdas y con denominadores de energía, fusiones y rachas
  contadas:
  - **cotas rigurosas:** `ε_2 = 1,03·10⁻²` para Z15c/Z16 y `8,2·10⁻³` para Z17, que pide además decaimiento en
    tiempo. La cola de la recurrencia está acotada por una supersolución (9.13 d).
  - **estimación:** sin la pérdida de la cola, `≈ 1,2·10⁻²`.

  Están escritos la capa de borde, la cota de prueba de KP continuo con inducción por anchura, Z17 en tiempo
  continuo con constantes, y la cola. Las dos vías son mejores que el radio de una escala de
  Kirkwood–Thomas, y son uniformes en n. Los
  números (paso 6′) siguen indicando que la contracción es uniforme hasta ε ≈ 0,2.
  **Lado k ∤ n con carga entera (7.14, Teorema D″):** la carga U(1) cumple (S_d), con una órbita
  neutra y dos cargadas. El aislamiento por fases (Z7″) usa el vacío de cadena, sin vacío recto. La
  regla queda como teorema: si gana la órbita neutra, la sensibilidad es exponencial; si gana una
  cargada, `n^{−2}`. Con `S⁺S⁻` gana la neutra, que baja más a segundo orden (el primer orden es nulo, 9.14). Numéricamente, `R_n` cae
  de 1,4·10⁻⁴ a 7·10⁻⁹ entre n = 7 y 13 con ε = 0,1, y en este modelo el lado k ∤ n también es
  exponencial. Con un término cinético aleatorio (semilla 2, ε = 0,1) gana una órbita cargada y sale
  la otra rama: `n²R_n` es constante (0,128 / 0,129 / 0,131 con n = 7, 10, 13). Con n ≡ 2 la
  tendencia es la misma (exponentes locales que bajan hacia 2,3), y fijar la constante pediría
  n = 17. La parte 2 de la conjetura depende de la carga de la órbita ganadora. D″ usa el vacío de
  cadena de Z″, así que hereda su rango: está demostrado para `n ≤ n_*(ε)`. **En 9.14 (Teorema D‴)
  se quita esa limitación con contornos en espacio-tiempo.** El defecto es un contorno que da la
  vuelta en tiempo, y su movimiento es un gas unidimensional anclado. La banda queda aislada
  uniformemente en *todo* n, con `ω_Q` independiente de n. Para `S⁺S⁻`, el hueco entre órbitas es
  `(56/3)ε²` (n ≡ 1) o `(38/3)ε²` (n ≡ 2) a segundo orden, así que gana la neutra y la sensibilidad es
  exponencial en todo n. Constantes certificadas con intervalos: `ε_3 ≥ 7,22·10⁻³` (gas de excursiones, con los enlaces del Lema Z18) y `ε_4 ≥ 1,74·10⁻³` (órdenes 2 y 3 exactos con racionales, resto desde 4 saltos). En el
  modelo de carga entera, el Kirkwood–Thomas de *una sola fase* falla por razones de fondo (Z8).
  **Con `Q_c ≡ 0 mod k`, el fundamental nota Θ exactamente en el orden n** (Teorema Z20, 9.15). El
  túnel no transporta carga alrededor del anillo, y el coeficiente de ε^n es una suma de caminos de n
  saltos en el mismo sentido: 4|b| = 62,3 con n = 6 y 237 con n = 9, comprobado con diagonalización
  exacta. Lo mismo vale para el defecto neutro con k ∤ n.

## 7. Teorema con k | n

Esta sección enuncia el lado exponencial. Se demuestra todo salvo una hipótesis analítica, (A), que
es la que en [X] da Kirkwood–Thomas. La parte combinatoria («localidad de los coeficientes de
Taylor») no necesita la estructura de [X], y sale de que `h_cl` es diagonal. Se usa la convención
de mínimo: `E_0(n,Θ)` es la energía fundamental de `H = Σ h`. El testigo de la Sección 1 es `−h`.

### 7.1 Hipótesis y notación

Sitio `C^d`, carga `q : {0,…,d−1} → Z`, `Q` la carga de un sitio y `h = h_cl + ε v`, con:
- `h_cl|ab⟩ = u(a,b)|ab⟩` diagonal;
- `v` hermítico de dos sitios, invariante por la U(1): `⟨a'b'|v|ab⟩ ≠ 0 ⇒ q_a'+q_b' = q_a+q_b`.

**(C) Orden clásico de celda k.** Sea `u_min = min u`. El conjunto de pares mínimos
`G = {(a,b) : u(a,b) = u_min}` está formado exactamente por las k aristas de un ciclo dirigido
`a_0 → a_1 → … → a_{k−1} → a_0`, con los `a_i` distintos. Sea `Δ = min_{(a,b)∉G} (u(a,b) − u_min) > 0`.

En los dos modelos de la Sección 3, `G` es el ciclo `0→1→2→0`, con `u_min = −1` y `Δ = 1`.

En el anillo `Z_n`, `H_n(ε,Θ) = Σ_j [u_j + ε v_j^Θ]`, con `v^Θ = (1⊗V_φ) v (1⊗V_φ)†`,
`V_φ = e^{−iφQ}` y `φ = Θ/n` (el gauge uniforme de `dual_celda.py`). La traslación `T` conmuta con
`H_n(ε,Θ)`, y lo mismo la carga total `Q_tot`. Sea `Q_c = Σ_i q(a_i)` la carga de la celda.

**Lema 0 (clásico).** Las configuraciones con todos los enlaces en `G` existen si y solo si k | n.
En ese caso son exactamente k, `c_r(j) = a_{(j+r) mod k}` con `r ∈ Z_k`, y dos de ellas distintas
difieren en **todos** los sitios. Cualquier otra configuración tiene energía clásica
`≥ n u_min + Δ·#{enlaces fuera de G} ≥ n u_min + Δ`.

*Demostración.* Cada estado del ciclo tiene un único sucesor en `G`, y los que no están en el ciclo
no tienen ninguno. Así, una configuración con todos los enlaces en `G` es `s_j = a_{j+r}`, y para que
cierre hace falta k | n. Además, `a_{j+r} ≠ a_{j+r'}` si `r ≢ r'`. ∎

**El multiplete.** Sea k | n. Con ε complejo y `n|ε|‖v‖ < Δ/2`, dentro de la circunferencia
`|z − n u_min| = Δ/2` hay exactamente k autovalores (con multiplicidad algebraica). Su proyector
espectral `P_n(ε,Θ)` es analítico en ε (Kato) y conmuta con `T` y con `Q_tot`. Sobre
`span{c_r}`, `T` actúa como una permutación cíclica de orden k, así que cada sector `T = e^{ip}`,
con `p ∈ (2π/k)Z_k`, contiene un solo nivel del multiplete. Ese nivel `λ_p(n,Θ,ε)` es analítico en
ε. Se definen:
- `λ̄ = (1/k)Σ_p λ_p = tr(HP)/k`;
- `ρ_n(Θ,ε) = tr_{Λ∖{0,1}} P/k`, que con ε real es la marginal de arista del estado `P/k`. Ese
  estado es invariante por traslación y por la U(1), así que la marginal es la misma en todas las
  aristas.

### 7.2 Teorema A (perturbativo; sin hipótesis analíticas)

Sea k | n. Se escriben `f(ε) = Σ_m f^{(m)} ε^m` y `⌈n/2⌉ = N`.
1. **Extensividad y flujo.** Hay números `e^{(m)}` y matrices de dos sitios `ρ_∞^{(m)}`,
   U(1)-invariantes, que dependen solo de `h`, no de `n`, `Θ` ni `p`, tales que:
   - `λ_p^{(m)}(n,Θ) = n e^{(m)}` para `0 ≤ m ≤ N−1` y todo `p`;
   - `ρ_n^{(m)}(Θ) = (1⊗V_φ) ρ_∞^{(m)} (1⊗V_φ)†` para `0 ≤ m ≤ N−2`.
2. **Enrollamiento.** La media `λ̄^{(m)}(n,Θ)` no depende de Θ para `m ≤ n−1`.
3. **Gauge grande.** `λ_p(n,Θ+2π,ε) = λ_{p+2πQ_c/k}(n,Θ,ε)`. El signo depende de la orientación de
   `T`.

Consecuencias inmediatas, rigurosas con n fijo cuando ε → 0:
- la respuesta al flujo del fundamental y la anchura del multiplete son `O(ε^N)`;
- `E_0(n)/n − E_0(n')/n' = O(ε^{min(N,N')})` para dos múltiplos de k;
- la dependencia en Θ de la media del multiplete es `O(ε^n)`.

**Paso 1 (caminos).** Como `h_cl` es diagonal, el resolvente reducido de `H_cl` y `P_0` son diagonales
en la base de configuraciones. Los coeficientes de orden m de `P`, de `HP` y de la función de
transformación de Kato, `U = (1−(P−P_0)²)^{−1/2}[PP_0 + (1−P)(1−P_0)]`, son combinaciones lineales
de palabras `D_0 V D_1 V ⋯ V D_m`, con `D_i` diagonales y `V = Σ_b v_b^Θ`. Lo mismo vale para su
inversa. El elemento `⟨s'|palabra|s⟩` es una suma sobre **caminos** de m pasos `s = s_0 → ⋯ → s_m = s'`.
En cada paso se aplica `v_b^Θ` en un enlace `b`, y cambian como mucho sus dos sitios. Por la
invariancia U(1), cada paso en `b = (j,j+1)` pasa una carga entera `δ` de `j` a `j+1`, y el giro
multiplica su amplitud por `e^{−iφδ}`. Nada más depende de Θ.

**Paso 2 (túnel; parte de 1).** `A = U^{−1}HU` deja invariante `span{c_r}`, y sobre él sus
autovalores son los del multiplete. Como `U` conmuta con `T`, la matriz `A_{r'r} = ⟨c_{r'}|A|c_r⟩` es
circulante: `A_{r'r} = a_{r'−r}`, con autovalores `λ_p = Σ_r a_r e^{ipr}` y `a_0 = λ̄`. Si
`r' ≠ r`, `A^{(m)}_{r'r}` es una suma sobre caminos de `c_r` a `c_{r'}`. Estas configuraciones
difieren en los n sitios (Lema 0), y cada paso cambia dos como mucho, así que `m ≥ n/2`. Por tanto
`λ_p^{(m)} = λ̄^{(m)}` para `m ≤ N−1`.

**Paso 3 (enrollamiento; parte 2).** `λ̄^{(m)} = a_0^{(m)}` es una suma sobre caminos cerrados de
`c_0` a `c_0`. Sea `J_b` la carga neta que cruza `b`. Como la configuración vuelve a la de partida,
la conservación en cada sitio da `J_{b_{j−1}} = J_{b_j}`, así que todos los `J_b` valen un mismo
entero `W`. La fase total es `e^{−iφ Σ_b J_b} = e^{−iΘW}`. Si `m ≤ n−1`, algún enlace queda sin tocar,
así que `W = 0` y la fase es 1.

**Paso 4 (factorización: solo cuentan los racimos conexos).** Se dan variables independientes `ε_b`
por enlace y se añade `η O_e`, con `O` un operador de dos sitios en una arista `e` (sirve para la
marginal: `∂_η λ̄|_{η=0} = tr(Oρ_n)`, porque `tr(H dP) = 0`). Con n fijo, todo es analítico en un
polidisco. Se dice que dos enlaces son vecinos si `|b−b'| ≤ 2`, como en [X].

*Afirmación:* el coeficiente de `ε^α` (y de `η ε^α`) en `λ̄` es nulo salvo que el soporte de `α`
(más `e`) sea conexo.

*Demostración.* Supóngase que el soporte se parte en `A ∪ B`, y que los conjuntos de sitios que
tocan, `S_A` y `S_B`, están a distancia ≥ 2. Se anulan los `ε_b` fuera de `A ∪ B` y se toma
`X = S_A ∪ S_B ≠ Λ`. El espacio `K_r` de las configuraciones iguales a `c_r` fuera de `X` es
invariante, y los `K_r` son ortogonales entre sí (por el Lema 0, porque hay sitios fuera de `X`).
Ningún enlace toca a la vez `S_A` y `S_B`, así que en `K_r ≅ C^{S_A} ⊗ C^{S_B}` se tiene

    H = n u_min + h_A^{(r)} ⊗ 1 + 1 ⊗ h_B^{(r)} ,

con `h_A^{(r)}` el exceso clásico de los enlaces que tocan `S_A` más `Σ_{b∈A} ε_b v_b`. Por el Lema 0
aplicado a cada arco de `S_A` con frontera `c_r`, el exceso clásico tiene su cero único en
`c_r|_{S_A}`, y todo lo demás está a `≥ Δ`. El complemento de `⊕K_r` también es invariante y no
tiene configuraciones fundamentales. Así, el multiplete es
`{n u_min + μ_A^{(r)}(ε_A) + μ_B^{(r)}(ε_B)}_r`, y `λ̄` no tiene monomios mixtos. ∎

**Paso 5 (periodización y gauge; parte 1).** Un soporte conexo de `w` enlaces abarca como mucho `2w`
sitios consecutivos. Si `|α| = m ≤ N−1`, el arco `I` que lo contiene tiene como mucho `2m ≤ n−1`
sitios. Como en el Paso 4, con un solo bloque, `λ̄_α = (1/k)Σ_r μ_I^{(r)}`. El giro se elimina en el
arco con el gauge diagonal `Γ = ∏_{x∈I} V_φ^{x−x_0}`, que conmuta con `h_cl` y cumple
`Γ v_b^0 Γ† = v_b^Θ` en los enlaces interiores (por la U(1)). Sin giro, el problema es el mismo que en
`Z` con frontera `c_r`. Si `|I| = n−1`, los dos vecinos exteriores coinciden en el anillo, pero
valen lo mismo en `Z` porque k | n. Así, `λ̄_α = κ(α̃)`, el coeficiente del racimo levantado en la
cadena, que es invariante por traslación tras promediar en `r`. Los racimos conexos del anillo con
`2m ≤ n−1` están en biyección con las clases módulo `nZ` de los de `Z`, así que
`λ̄^{(m)} = n Σ_{α̃ conexo, |α̃|=m, min sop = 0} κ(α̃) =: n e^{(m)}`. Junto con el Paso 2, esto da
`λ_p^{(m)} = n e^{(m)}`.

Para la marginal, con `e` en el soporte hay `m+1` enlaces, y la condición es `2(m+1) ≤ n−1`. El
gauge lleva `O_e` a `(1⊗V_φ)† O (1⊗V_φ)` (por la U(1), y basta `O` invariante porque `ρ_n` lo es),
y de ahí sale el giro de `ρ_∞^{(m)}`. ∎

**Paso 6 (gauge grande; parte 3).** `W = ∏_x e^{−i(2π/n) x Q_x}` cumple
`W H_n(Θ) W† = H_n(Θ+2π)`: en el enlace que cierra el anillo sobra `e^{−2πiQ} = 1`. Además,
`T W = W e^{i(2π/n)Q_tot} T`. En el multiplete, `Q_tot = nQ_c/k`, así que `W` lleva el sector `p` al
`p + 2πQ_c/k`. ∎

**Lectura del punto 3.**
- **Carga por sitio no entera** (`Q_c ≢ 0 mod k`, la condición de LSM). El flujo permuta
  cíclicamente los k niveles. `E_0 = min_p λ_p` sigue teniendo periodo 2π, pero en general nota Θ ya
  en el orden del túnel: el túnel «transporta carga `Q_c/k`», como decía la conjetura de la
  Sección 2.
- **Carga entera por celda módulo k.** Cada nivel tiene periodo 2π por separado. Numéricamente, la
  dependencia en Θ del fundamental aparece entonces solo en el orden del enrollamiento (7.4).
  *Demostrado en 9.15 (Teorema Z20), con el coeficiente de ε^n.*

### 7.3 Teorema B (exponencial, bajo la hipótesis A)

**(A) Analiticidad uniforme.** Hay `r_0 > 0`, `B < ∞` y `n_0` tales que, para todo `n ∈ kZ` con
`n ≥ n_0` y toda Θ:
- (A1) con ε real, `|ε| < r_0`, los k niveles del multiplete son los k más bajos de `H_n(ε,Θ)` y
  están separados del resto;
- (A2) `λ_p(n,Θ,·)` y `ρ_n(Θ,·)` se extienden analíticamente a `D = {|ε| < r_0}`, con
  `|λ_p/n − u_min| ≤ B` y `‖ρ_n − ρ_n(0)‖_1 ≤ B` en `D`.

**Teorema B.** Bajo (C) y (A), para ε real con `|ε| < r_0`, `n ∈ kZ` con `n ≥ n_0` y toda Θ:

    |E_0(n,Θ)/n − e_∞(ε)|                         ≤ 2B (|ε|/r_0)^{⌈n/2⌉},
    ‖ρ_n(Θ,ε) − (1⊗V_φ) σ(ε) (1⊗V_φ)†‖_1          ≤ 2B (|ε|/r_0)^{⌈n/2⌉−1}.

Aquí `e_∞` es la energía por sitio de la cadena y `σ = lim_n ρ_n(0,ε)` es la marginal de arista de
un estado fundamental de la cadena, invariante por traslación y por la U(1) (un punto límite de los
estados `P/k`). La primera cota vale también para cada uno de los k niveles del multiplete.

*Demostración.* Por el Teorema A, los coeficientes de `λ_p(n)/n` son `e^{(m)}` para `m < ⌈n/2⌉`. Las
estimaciones de Cauchy dan `|e^{(m)}| ≤ B r_0^{−m}`, así que `e(ε) = Σ e^{(m)}ε^m` es analítica en
`D`. Además, `|λ_p/n − e| ≤ 2B Σ_{m ≥ ⌈n/2⌉} (|ε|/r_0)^m → 0`, así que `e = lim λ_p/n` y
`|e − u_min| ≤ B`. Entonces `f = λ_p/n − e` es analítica en `D`, cumple `|f| ≤ 2B` y tiene un cero
de orden `≥ ⌈n/2⌉`, y el lema de Schwarz ([X]) da la cota. Con ε real, (A1) da `E_0 = min_p λ_p` y
`e = e_∞`. La marginal se trata igual: sus coeficientes coinciden hasta `⌈n/2⌉−2`, `V_φ` es un
unitario fijo, y `ρ_n(0)`, la media de las `|c_r(0)c_r(1)⟩⟨⋯|`, no depende de n y conmuta con
`1⊗V_φ`. ∎

**Corolario C (robustez de los datos de estado fundamental).** Bajo (C) y (A), sean los datos
`σ(ε)` del Teorema B y `g = tr h/d² − e_∞ > 0`. Para k | n, `n ≥ n_0` y ε real con `|ε| < r_0`,

    sup_{Θ,Θ'} |p(n,Θ) − p(n,Θ')| ≤ 2B (1/g + 19 d² r_0/|ε|) (|ε|/r_0)^{⌈n/2⌉} .

*Demostración.* Es la de [X, Cor. «robustez»] con la Sección 1:
- (D): con el testigo `−h`, `ε_D = (e_∞ − E_0/n)⁺/g`, acotado por la primera cota del Teorema B;
- (P): el estado `P/k`, con `ε_P` acotado por la segunda;
- se aplica el Teorema puente con `p* = 1` y `1/r' = 19d²`. La hipótesis `tr_1τ = tr_2τ` se
  cumple porque `σ` es invariante por traslación y las marginales de un sitio, diagonales, conmutan
  con `V_φ`. ∎

### 7.4 Comprobación numérica del Teorema A

Guion: `scripts/orden_kn.py`, malla de 16 valores de Θ. Se mide el exponente efectivo en ε entre
ε = 0,05, 0,1 y 0,2. En el modelo de llenado se añade el intercambio neutro con amplitud ε, para que
el túnel no lo impida la ley de conservación de la Sección 4.1. Predicción: todo es `O(ε^N)`, con
`N = ⌈n/2⌉`, y la media del multiplete depende de Θ solo desde `O(ε^n)`.

Exponente efectivo en ε, entre ε = 0,1 y 0,2 (entre paréntesis, entre 0,05 y 0,1):

| modelo | n | N = ⌈n/2⌉ | anchura del multiplete | R del fundamental | R de la media | `e_n − e_{2n}` | hueco mínimo |
|---|---|---|---|---|---|---|---|
| carga entera (`Q_c = 0`) | 6 | 3 | 4,21 (4,07) | 6,48 (6,17) | 6,58 (6,21) | 4,09 (4,06) | 1,22 |
| carga entera | 9 | 5 | 6,45 (6,11) | 10,2 (9,29) | 10,5 (9,39) | — | 1,04 |
| llenado 1/3 (`Q_c = 1`) | 6 | 3 | 4,00 (4,00) | 4,00 (4,00) | 8,57 (8,20) | 4,00 (4,00) | 1,72 |
| llenado 1/3 | 9 | 5 | 6,02 (6,01) | 6,02 (6,01) | 11,0 (—*) | — | 1,75 |

\* Con ε = 0,05, la R de la media de n = 9 (6·10⁻¹⁴) está al nivel del redondeo.

- **Todo cumple la cota de orden N.** El orden real del túnel es `2n/3` (4 y 6), no `n/2`. Con
  k = 3, la cota del Paso 2 no es óptima. En el XXZ (k = 2) sí lo es: allí el orden es `n/2`.
- **La media del multiplete nota Θ solo en el orden `≥ n`.** En el modelo de carga entera es
  exactamente n (6 y 9), así que la cota del enrollamiento (Paso 3) es óptima.
- **El punto 3 se ve tal cual.**
  - Con `Q_c = 1` (carga 1/3 por sitio), el flujo permuta los niveles, y R tiene el orden del túnel
    (4 y 6) y el valor de la anchura del multiplete.
  - Con `Q_c = 0`, R tiene el orden del enrollamiento (6 y 9), muy por debajo de la anchura.
  - Así, la sensibilidad con k | n tiene dos escalas exponenciales:
    - el túnel, `ε^{2n/3}`, si la carga por sitio es fraccionaria;
    - el enrollamiento, `ε^n`, si es entera.
- **El hueco al resto del espectro se mantiene en ≥ 1 hasta ε = 0,2.** Es coherente con (A1) en
  estos tamaños.

### 7.5 Lo que falta: la hipótesis (A)

(A) es exactamente lo que en [X] dan Kirkwood–Thomas y la contracción de Datta–Kennedy para k = 2:
`E_±` analíticas y acotadas en un disco uniforme, y la localidad de la marginal. Para la clase (C)
no está demostrada aquí.

Es la analiticidad uniforme que da un desarrollo en racimos convergente alrededor del orden clásico.
La teoría de Pirogov–Sinai cuántica la proporciona en esencia para perturbaciones cuánticas de
sistemas clásicos con un número finito de fundamentales periódicos y condición de Peierls:
- N. Datta, R. Fernández y J. Fröhlich, J. Stat. Phys. 84, 455 (1996);
- C. Borgs, R. Kotecký y D. Ueltschi, Commun. Math. Phys. 181, 409 (1996).

En (C), la condición de Peierls es automática con interacción a primeros vecinos. Pero esos
trabajos no la enuncian en la forma (A): anillo finito, flujo, cada nivel `λ_p` por separado, ε
complejo y la marginal.

Dos caminos:
1. **Generalizar Kirkwood–Thomas.** En [X], el cambio de ejes convierte el doblete de Néel en un
   estado producto (`ψ ≡ 1`). Con k ≥ 3 no hay un cambio de base local así. Habría que trabajar en
   cada sector `p` con un ansatz `e^S|c_r⟩` (racimos acoplados), restringido a configuraciones que
   no contienen un dominio de otra fase. Es el «contorno truncado» de Pirogov–Sinai.
2. **Citar y adaptar.** Tomar el desarrollo de Borgs–Kotecký–Ueltschi, a temperatura β y con
   β → ∞, en el toro con condición de contorno girada.

La parte combinatoria (Teorema A) ya está completa, y el paso de (A) a la robustez (Teorema B y
Corolario C) también.

### 7.6 Ataque a (A): lo que hace falta de verdad y lo que dicen los números

**(a) El radio de convergencia no baja con n.** Como `h_cl` es diagonal, la serie de
Rayleigh–Schrödinger de cada nivel `λ_p` sale de una recurrencia barata. En el sector de momento p,
el fundamental clásico `|p⟩` no está degenerado, y la recurrencia es

    ψ^{(m)} = S [V ψ^{(m−1)} − Σ_{j=1}^{m} E^{(j)} ψ^{(m−j)}],   E^{(m)} = ⟨p|V|ψ^{(m−1)}⟩,

con `S` diagonal (guion `scripts/rs_radio.py`, 60 órdenes). Radio de convergencia de `λ_p/n` por el
criterio de la raíz (m ≥ 30):

| modelo | Θ | p | n = 6 | n = 9 | n = 12 | n = 15 |
|---|---|---|---|---|---|---|
| carga entera | 0 | 0 | 0,317 | 0,312 | 0,314 | 0,34 (m ≥ 20) |
| carga entera | π/2 | 0 | 0,319 | 0,312 | 0,314 | |
| carga entera | 0 | 2π/3 | 0,49 | 0,39 | 0,36 | |
| llenado, μ = 1 | 0 | 0 | 0,73 | 0,83 | 0,90 | |
| llenado, μ = 1 | π/2 | 0 | 0,77 | 0,86 | 0,92 | |
| llenado, μ = 1 | 0 | 2π/3 | 0,85 | 0,87 | 0,92 | |
| llenado, μ = 0 | 0 | 0 | 0,94 | 1,06 | 0,93 | |

(En el modelo de carga entera, el salto tiene amplitud 2ε.) En ningún caso baja el radio con n. Lo
que baja (p = 2π/3, carga entera) se acerca al valor del sector p = 0. Es la evidencia directa de
(A2) para la energía: hay un disco uniforme, con `r_0 ≈ 0,31` y `≈ 0,7` respectivamente.

Los mismos coeficientes confirman el Teorema A: `c_m(n)` coincide entre tamaños hasta el orden del
túnel `2n/3`. Por ejemplo, n = 12 y n = 15 coinciden hasta m = 6, y n = 12 se separa en m = 8.

**(b) (A1) en todo Θ sobra: basta un flujo sin signo.** En [X], la cota (D) no sale de la
analiticidad con flujo, sino de Marshall (`e_n(Θ) ≤ e_n(0)`). Aquí se generaliza así.

*Lema D (diamagnético).* Supóngase que hay un Θ* y un unitario diagonal `D` tales que
`D H_n(ε,Θ*) D†` tiene todos sus elementos fuera de la diagonal `≤ 0`. Entonces
`E_0(n,Θ) ≥ E_0(n,Θ*)` para toda Θ.

*Demostración.* El flujo solo cambia las fases de los elementos fuera de la diagonal (Paso 1), no
sus módulos, y la diagonal no depende de Θ. Para un vector unitario ψ, se tiene
`⟨ψ|D H(Θ) D†|ψ⟩ ≥ ⟨|ψ|| D H(Θ*) D† ||ψ|⟩ ≥ E_0(Θ*)`. ∎

La condición es que todo lazo de movimientos tenga el signo de un hamiltoniano estocuástico. El
guion `scripts/sin_signo.py` la comprueba construyendo el gauge sobre un árbol:
- **carga entera:** con ε > 0, Θ* = 0 si n es par y Θ* = π si n es impar (como el Marshall de
  [X] con n impar); con ε < 0, Θ* = 0. Es irreducible en `S^z = 0`;
- **llenado con μ = 0:** el mismo patrón, pero reducible: la secuencia comprimida de 4.1 se
  conserva;
- **llenado con μ = 1:** **no tiene gauge sin signo**. El salto y el intercambio neutro se frustran
  entre sí, y el Lema D no se aplica.

*Lema P (Perron más identidad).* Supóngase que, para ε ∈ (0, r_0), `D H_n(ε,Θ*) D†` es estocuástica
e irreducible en el sector de carga `Q* = nQ_c/k`. Supóngase también que el nivel `λ_{p_0}`, el que es
el fundamental del sector para ε pequeño, se continúa analíticamente a (0, r_0). Entonces el
fundamental del sector es `λ_{p_0}(ε)` en todo (0, r_0): no hay cruces que vigilar.

*Demostración.* Por Perron–Frobenius, el fundamental del sector es simple para cada ε, así que
`E_{Q*}(ε)` es real-analítico en (0, r_0). Cerca de 0⁺ coincide con un nivel del multiplete (Kato).
Ese nivel es `λ_{p_0}`, porque el mínimo de un número finito de funciones analíticas coincide con
una de ellas en un intervalo. La continuación analítica de un autovalor sigue siendo un autovalor.
Por el teorema de identidad, `E_{Q*} = λ_{p_0}` en todo (0, r_0). ∎

**Teorema B′ (sustituye a B, con hipótesis más débiles).** Sean (C), k | n y ε ∈ (0, r_0).
Supóngase:
- **(S)** las hipótesis del Lema P en un Θ*;
- **(G)** el fundamental de `H_n(ε,Θ*)` está en el sector `Q*`;
- **(A2E\*)** solo un nivel en un solo flujo: `λ_{p_0}(n,Θ*,·)/n` es analítica en `D`, con
  `|λ_{p_0}/n − u_min| ≤ B`;
- **(A2M)** la de la marginal, como antes.

Entonces la cota de la marginal del Teorema B se mantiene, y para toda Θ,
`E_0(n,Θ)/n ≥ e_∞ − 2B(ε/r_0)^{⌈n/2⌉}`. Con eso, (D) y el Corolario C valen igual: (D) solo
necesita la cota inferior de `E_0`.

*Demostración.* `E_0(n,Θ) ≥ E_0(n,Θ*)` por el Lema D, `E_0(n,Θ*) = E_{Q*} = λ_{p_0}` por (G) y el
Lema P, y se aplica el lema de Schwarz a `λ_{p_0}/n − e` como en B. Que `e = e_∞` sale de que el
giro en un solo enlace cambia la energía en O(1). ∎

(A1) en todo Θ desaparece. La marginal (P) no necesita que el multiplete sea el más bajo: con ε real,
las k ramas (Rellich) dan un proyector ortogonal de rango k, y `P/k` es un estado.

**(c) Qué queda abierto y por dónde atacarlo.**

| pieza | estado |
|---|---|
| Teorema A (combinatoria) | demostrado |
| Lemas D y P, Teorema B′ ⇐ hipótesis | demostrado |
| (S) | se comprueba modelo a modelo: sí en carga entera y llenado con μ = 0; no en llenado con μ = 1 |
| (G), hueco a los sectores vecinos | numérico: 0,94–1,02 (carga entera) y 1,76 (llenado) con ε = 0,2, n = 6, 9 |
| (A2E\*) | numérico: el radio no baja con n hasta n = 15 |
| (A2M) | sin comprobar, ni numérica ni rigurosamente |

Dos caminos para demostrar (A2E\*) y (A2M):
1. **Kirkwood–Thomas en el marco de reloj.** Con `h_cl = −P_asc` y d = k, la transformada de
   Fourier en cada sitio, seguida del gauge diagonal `∏_j Z_j^{j}` (que cierra en el anillo porque
   k | n), lleva `−P_asc` a `−(1/k) Σ_m (X_j† X_{j+1})^m`. Es una suma de proyectores que conmutan,
   estocuástica, con fundamental uniforme en cada sector de la carga `Z_k` (la estructura exacta
   de Datta–Kennedy en [X], con `Z_k` en lugar de `Z_2`). La ecuación de Kirkwood–Thomas en
   caracteres de `Z_k^n` tiene término lineal `−|∂μ| g(μ)`, como en [X].
   - Las perturbaciones con carga U(1) (`S⁺S⁻`, el salto) **no** conservan la carga `Z_k`, y en
     este marco aparecen como el término de `sin φ` de [X]. Eso da un **cuasi-modo** de la fase
     rota (el `c_0` vestido), con residuo de orden `ε^{n/2}`: los caracteres constantes no nulos
     necesitan soporte Λ.
   - Su marginal, por el desarrollo en polímeros de [X, Teorema «estado de prueba»], daría (P), es
     decir, (A2M), de forma rigurosa para los dos modelos.
   - Hay que rehacer la contracción de Datta–Kennedy con desplazamientos generales `(x, y) ∈ Z_k²`.
     **Corrección (7.7):** esto no es solo cuestión de constantes. Con perturbaciones sin simetría de
     reloj, la contracción falla por una razón de fondo. Funciona en la clase con simetría de reloj,
     que incluye el modelo de llenado con μ = 1.
2. **Para (A2E\*), es decir, la cota inferior de la energía,** el cuasi-modo no basta: da un
   autovalor cerca, pero no que no haya otros por debajo. En el caso sin signo, la vía natural es
   Collatz–Wielandt, `E_0 ≥ min_s (Hφ)(s)/φ(s)` para φ > 0, en la base de configuraciones donde H es
   estocuástica. Pero hace falta un φ positivo y log-local en *esa* base, y la de Kirkwood–Thomas
   es la de Fourier.
   - Hecho que ayuda: alrededor de `c_r`, una isla de otra fase de longitud ℓ cuesta `ε^{ℓ/2}` (hay
     que cambiar sus ℓ sitios), no solo su frontera, como en Peierls térmico. A temperatura cero,
     el problema es de fundamental único alrededor de cada `c_r` hasta el orden n/2. Por eso es de
     esperar un desarrollo en polímeros directo en la base de configuraciones, y es lo que
     sugieren los radios que no bajan.
   - La alternativa es adaptar Borgs–Kotecký–Ueltschi (1996) al toro con flujo.

### 7.7 El cuasi-modo `Z_k`: dónde funciona Kirkwood–Thomas y dónde no

Guiones: `scripts/kt_zk.py` (punto fijo de Kirkwood–Thomas numérico en el anillo completo) y
`scripts/kt_constantes.py` (constantes del lema de contracción).

**El marco.** Con d = k y `a_i = i`, se aplica la transformada de Fourier en cada sitio,
`F_ab = ω^{ab}/√k`, y luego el gauge diagonal `D = ∏_j ω^{−j s_j}`, que cierra en el anillo porque
k | n. En el marco:
- `−P_asc` pasa a `−Σ_j E_j`, con `E_j = (1/k) Σ_m τ_j^m` y `τ_j` el desplazamiento de `(s_j, s_{j+1})`
  por `(1, −1)`;
- los `E_j` son proyectores que conmutan entre sí;
- el vector uniforme es `c_0` y tiene energía −n. Comprobado numéricamente: la otra orientación del
  gauge da el orden descendente.

Con los caracteres `χ_μ(s) = ω^{μ·s}`, `μ ∈ Z_k^n`, se tiene `E_j χ_μ = 1[μ_j = μ_{j+1}] χ_μ`. Se define
`|∂μ| = #{j : μ_j ≠ μ_{j+1}}`. Es la energía clásica de `c_0 + μ`: multiplicar por `χ_μ` en el marco
es desplazar la configuración original en `μ`.

**La ecuación.** Sea `ψ = e^G`, con `G = Σ_μ g(μ) χ_μ`. Un operador de dos sitios en el enlace j se
escribe en el marco como `Σ_{δ,ν} ô_j(δ,ν) χ_ν τ_δ`, con `δ, ν ∈ Z_k^{{j,j+1}}`, y
`(τ_δψ)/ψ = exp G_δ`, con `G_δ = Σ_μ g(μ)(ω^{μ·δ} − 1) χ_μ`. Entonces

    (H'ψ)/ψ = −n + Σ_μ |∂μ| g(μ) χ_μ + 𝒩(g),
    𝒩(g) = −Σ_j (1/k)Σ_m (e^{G_{j,m}} − 1 − G_{j,m}) + ε Σ_j Σ_{δ,ν} v̂_j^Θ(δ,ν) χ_ν e^{G_δ}.

El punto fijo es `g(μ) = −𝒩̂(μ)/|∂μ|` para `μ` no constante. Los k caracteres constantes `c·1` quedan
fuera:
- `c = 0` da la energía `E = −n + 𝒩̂(0)`;
- `c ≠ 0` da el residuo `r_c = 𝒩̂(c·1)`, porque `(H' − E)ψ = (Σ_{c≠0} r_c χ_{c·1}) ψ`.

Con k = 2 es la ecuación de [X].

**Dónde se rompe.** Un desplazamiento `δ` del enlace j toca un `μ` si `μ·δ ≢ 0`. Hay dos casos:
- **Desplazamientos de tipo `τ_j^m`** (`δ_j + δ_{j+1} ≡ 0`): solo tocan los `μ` con `j ∈ ∂μ`. Están
  anclados en el borde, como en [X], y el `|∂μ|` de la norma los paga.
- **Desplazamientos que mezclan sectores** (`δ_j + δ_{j+1} ≢ 0`): tocan todo `μ` con
  `{j, j+1} ∩ sop μ ≠ ∅`. Están anclados en cualquier sitio del soporte.
  - Para una isla `μ = c` en un intervalo de longitud L, que tiene `|∂μ| = 2`, hay L anclajes.
  - En la cota de contracción aparece un factor `|sop μ|` que ninguna norma de este tipo absorbe. La
    entrada concentrada en una isla larga da `‖F(g)‖ ≳ L‖g‖/M`.
  - Físicamente: dentro de una isla de otra fase, las fluctuaciones locales tienen otra forma, y
    el ansatz logarítmico necesita L correcciones conexas. La solución verdadera existe, pero la
    iteración no es contractiva uniformemente en n.
  - En [X], esos términos solo venían del flujo, con amplitud `sin φ ≤ π/n`, y el 1/n compensaba el
    `|X| ≤ n`. Aquí vienen de `v` misma, con amplitud ε.

**Qué perturbaciones tienen solo desplazamientos de tipo τ.** Son las que conmutan con el
desplazamiento global de reloj `X ⊗ X` en la base original: la **clase con simetría de reloj**. Sus
elementos de matriz dependen de las diferencias `b − a`, no de los valores. El ejemplo canónico es
el intercambio `SU(k)`, `v = Σ_{a≠b} |b a⟩⟨a b|`. Conserva la carga U(1) para cualquier `q`, y la
transporta: `q(b) − q(a)` por paso.

**El modelo de llenado con μ = 1 es exactamente el intercambio SU(3):** el salto `(0,x) ↔ (x,0)` y el
intercambio neutro `(1,2) ↔ (2,1)` tienen la misma amplitud. Es el modelo con carga fraccionaria por
sitio (LSM), el más interesante de los dos. El de carga entera (`S⁺S⁻`) no tiene simetría de reloj:
en el marco, su parte que mezcla sectores tiene `Σ|v̂| = 4,44`.

El flujo rompe la simetría de reloj, pero solo por `v^Θ − v`. Para `|Θ| ≤ π` (el periodo 2π lo da el
punto 3 del Teorema A), su norma es `O(1/n)`: es el mecanismo de [X].

**Numérico** (`kt_zk.py`, iteración `g ← g − R̂/|∂μ|` en el espacio completo `3^n`):

| modelo | n | converge hasta (ε ≤ 0,5) | iteraciones con ε = 0,15 | `\|r_c\|` con ε = 0,3 |
|---|---|---|---|---|
| intercambio SU(3), Θ = 0 | 6 / 9 / 12 | 0,5 / 0,5 / 0,5 | 15 / 15 / 15 | 4,5·10⁻³ / 3,1·10⁻⁴ / 2,3·10⁻⁵ |
| intercambio SU(3), Θ = π | 6 / 9 / 12 | 0,5 / 0,5 / 0,5 | 15 / 15 / 15 | 1,5·10⁻³ / 1,0·10⁻⁴ / 7,6·10⁻⁶ |
| carga entera, Θ = 0 | 6 / 9 / 12 | 0,2 / ≈ 0,2 (399 it.) / 0,15 | 47 / 64 / 85 | diverge |
| carga entera, Θ = π | 6 / 9 / 12 | 0,2 / 0,15 / 0,15 | 47 / 70 / 93 | diverge |

- En el intercambio SU(3), nada depende de n, y el residuo cae como `ε^{2n/3}`. La energía del
  cuasi-modo es la media del multiplete con error del orden del residuo (n = 6, comparado con
  diagonalización exacta).
- En el modelo de carga entera, el rango de convergencia se encoge con n, muy por debajo del radio
  0,31 de la serie (7.6). Es la firma de la obstrucción.

**Lema Z1 (contracción, clase con simetría de reloj).** Sean:
- `a = sup_j Σ_{δ,ν} |v̂_j(δ,ν)|` (en la clase, solo hay `δ` de tipo τ);
- `b ≥ n sup_j Σ_{δ,ν} |ŵ_j(δ,ν)|`, con `w = v^Θ − v`, para `|Θ| ≤ π` y n ≥ 4;
- `‖g‖ = sup_x Σ_{μ ∋ x} |g(μ)| |∂μ| (M|ε|)^{−w(μ)}`, con `w(μ)` el menor recubrimiento conexo de
  `sop μ` por enlaces (vecinos si `|b − b'| ≤ 2`, como en [X]);
- `d = (4a + b)/M`.

Si `M|ε| ≤ 1` y

    K = 2(e^{2d} − 1 + 2d e^{2d}) + (2(a+b)/M)(1 + 2d) e^{2d} + ((4a + b)/M) e^{2d} ≤ 1/2 ,

entonces F lleva la bola `‖g‖ ≤ d` en sí misma y es una contracción de constante 1/2, uniformemente
en n ∈ kZ y en `|Θ| ≤ π`.

*Demostración.* Es la del Lema «contracción con flujo» de [X], con `|ω^{μ·δ} − 1| ≤ 2`, que dobla
cada g. Se fija el sitio x de la salida y se separan cuatro tipos de término:
- **Términos de `−P_asc`**, anclados en el borde: si x está en el μ_1 de un producto, la suma sobre j
  la absorbe el `|∂μ_1|` de la norma, y los demás factores, anclados en `j ∈ ∂μ_i`, suman `≤ ‖g‖`
  cada uno. Esto da `Σ_{p≥2} p²(2d)^{p−1}·2/p! = 2(e^{2d} − 1 + 2d e^{2d})`.
- **Términos de `v`**, anclados en el borde, con el factor `ε(M|ε|)^{−1} = 1/M` del enlace añadido:
  - `(2a/M)(1+2d)e^{2d}` si x está en el μ_1;
  - `(4a/M)e^{2d}` si x está en `sop ν` (dos enlaces posibles).
- **Términos de `w`**, anclados en un sitio: si x está en el μ_1, el número de enlaces j que tocan
  `sop μ_1` es `≤ 2|sop μ_1| ≤ 2n`. El `1/n` de `‖ŵ‖` lo compensa, y queda `(2b/M)(1+2d)e^{2d}`. Si x
  está en `sop ν`, queda `4b e^{2d}/(nM) ≤ b e^{2d}/M`.
- **La fuente:** `‖F(0)‖ ≤ 2(a + b/n)/M ≤ d/2`.

La conexión de los recubrimientos es la de [X]: cada factor toca `{j, j+1}`, así que su recubrimiento
contiene `b_{j−1}`, `b_j` o `b_{j+1}`. ∎

Con el intercambio SU(3) salen `a = 2`, `b = 8,09` y `M_* = 359`, es decir, `|ε| < 2,8·10⁻³`
garantizado. Numéricamente, la iteración converge hasta ε = 0,5. Es la misma distancia entre la
constante demostrada y la real que en [X], donde la demostrada es M = 45 y la real M ≈ 2.

**Lema Z2 (cuasi-modo y energías).** En las hipótesis de Z1:
1. `Σ_{c≠0} |r_c| ≤ (d/2)(M|ε|)^{⌈n/2⌉}`. Las salidas constantes no nulas tienen soporte Λ, y
   `w(Λ) ≥ n/2`. Así, `ψ` es un cuasi-modo: `‖(H − E)ψ‖ ≤ (d/2)(M|ε|)^{⌈n/2⌉}‖ψ‖`.
2. `|E/n + 1| ≤ (e^{2d} − 1 − 2d) + (a + b/4)e^{2d}/M`, uniformemente, y los `g(μ)` son analíticos en
   ε (iteradas analíticas, convergencia uniforme).
3. **Con Θ = 0 y simetría de reloj**, H' conserva la carga `Z_k` del marco, porque conmuta con
   `χ_{1}`. Entonces `ψ_N = ψ·1[Σ s ≡ N]` es un **autovector exacto**, con
   `E_N = E + Σ_{c≠0} r_c ω^{cN}`. Son los k niveles del multiplete, analíticos y acotados
   uniformemente en `|ε| < 1/M`. Es (A2E\*) en Θ\* = 0, demostrado, como el Lema «analiticidad» de
   [X]. ∎

**Lo que da para el intercambio SU(3)** (k = 3, llenado 1/3, `Q_c = 1`):
- **(S):** en el sector de contenido `(n/3, n/3, n/3)` de las tres especies, que el intercambio
  conserva por separado, H(0) es irreducible y sin signo tras un gauge, para los dos signos de ε.
  Comprobado con n = 6 y 9; con ε < 0 es estocuástica sin gauge. Así, Lemas D y P con Θ\* = 0.
- **(A2E\*):** el Lema Z2.3.
- **(A2M):** el cuasi-modo, promediado sobre traslaciones, es el estado de prueba de (P). Su marginal
  se controla como en [X, Lemas «polímeros» a «el límite»]. Los cambios son estos, y ninguno
  introduce el factor `|sop|`:
  - el desarrollo de alta temperatura con `tanh` pasa a ser uno de Mayer, `e^{hχ_μ} = 1 + f_μ` con
    `|f_μ| ≤ 2|h|`, cuyos polímeros son colecciones conexas, con actividad la media de `∏ f_μ` sobre
    las s del soporte;
  - `ψ̃` lleva `g̃(−μ)`;
  - la periodización y la covariancia se hacen con el gauge `W` del Teorema A.
- **(G):** sigue siendo hipótesis. Numéricamente, el hueco a los demás sectores de contenido es
  1,73–1,76 con `|ε| = 0,2`, n = 6 y 9. Basta en Θ\* y en el sector de contenido, más la comparación
  con los demás sectores para toda Θ.

Para el intercambio SU(3) con k | n, la robustez exponencial de los datos de estado fundamental queda
así **demostrada salvo (G) y la transcripción de 7.7 (A2M)**, en `|ε| < 2,8·10⁻³`. Es el caso con
carga fraccionaria por sitio, el que importa para la holonomía. El modelo de carga entera, sin
simetría de reloj, sigue necesitando otra herramienta: Pirogov–Sinai, o un desarrollo en la base de
configuraciones (7.6 (c), camino 2).

### 7.8 (A2M) para la clase con simetría de reloj: la marginal del cuasi-modo

En [X], la marginal del estado de prueba se controla con siete lemas: polímeros, periodización con
flujo, localidad, anillo auxiliar, covariancia, el límite y el teorema final. Aquí hacen falta muchos
menos, porque el **Teorema A ya da la localidad y la covariancia gauge para el estado del
multiplete**, sin anillos auxiliares ni coeficientes formales en Z. Basta transcribir los
polímeros, que dan una cota uniforme en un disco, y comparar los coeficientes del cuasi-modo con los
del multiplete con n fijo, que es perturbativo.

Hipótesis de toda la sección:
- clase con simetría de reloj (7.7), con d = k y k | n;
- `|Θ| ≤ π`, que basta porque `p(n,Θ)` tiene periodo 2π: el gauge `W` del Teorema A es un unitario
  producto;
- `M̄ ≥ M_*` (Lema Z1), `d = (4a+b)/M̄`, `θ = 4d`, `M_2 = M̄ e^{4θ}`;
- `g` es el punto fijo con ε y `g̃(μ) := conj(g_{ε̄}(μ))`, ambos en la bola `‖·‖ ≤ d` de
  `‖·‖_{M̄}`.

**La marginal holomorfa.** En el marco, `U = D F^{⊗n}` es un producto de unitarios de un sitio, así
que un operador `O` de la arista `A = {j, j+1}` sigue siendo local, con la misma norma:
`O' = Σ_{δ,ν} ô(δ,ν) χ_ν τ_δ`, con `|ô| ≤ ‖O‖` y como mucho `k⁴` términos. Con
`ψ̃(s) = exp Σ_μ g̃(μ) χ_{−μ}(s)`, que coincide con `ψ̄` si ε es real, se define

    m_O(ε) = Σ_s ψ̃(s)(O'ψ)(s) / Σ_s ψ̃(s)ψ(s) = Σ_{δ,ν} ô(δ,ν) Z[h^δ + λ1_ν]' / Z[h] ,

con:
- `Z[h] = Σ_s exp Σ_μ h(μ) χ_μ(s)` y `h(μ) = g(μ) + g̃(−μ)`;
- `h^δ(μ) = g(μ) ω^{μ·δ} + g̃(−μ)`, que solo difiere de `h` en los μ que tocan `A`;
- `'` la derivada en λ = 0 (sin derivada si ν = 0).

**Lema Z3 (polímeros, versión de Mayer).** Si `|ε| < 1/M_2`, entonces `Z[h] ≠ 0` y
`|m_O(ε)| ≤ B'‖O‖`, con `B' = 4k⁴ e^{8θ}/θ`, uniformemente en n ∈ kZ y `|Θ| ≤ π`.

*Demostración.*
1. **Gas de polímeros.** Con `f_μ = e^{h(μ)χ_μ} − 1` (sobre los μ no constantes; los constantes
   tienen `g = 0`), `Z = k^n Σ_𝒳 E_s ∏_{μ∈𝒳} f_μ(s)`, con `E_s` la media uniforme. La media se
   factoriza sobre las componentes conexas de `𝒳`, con conexión cuando los soportes se cortan,
   porque las coordenadas de soportes disjuntos son independientes. Así, `Z/k^n` es un gas de
   polímeros: colecciones conexas γ de μ distintos, con actividad `w(γ) = E_s ∏_{μ∈γ} f_μ` y
   `|w(γ)| ≤ ∏ 2|h(μ)|` si `|h| ≤ 1/2`. Esto sustituye al desarrollo con `tanh` de [X], que es
   propio de `Z_2`. No hay restricción del tipo `△ = ∅`, y la cota es la misma.
2. **El criterio.** Sea `ĥ = sup_x Σ_{μ∋x} |h(μ)| e^{2θ|sop μ|}`. Si `ĥ ≤ θ/2`, el Lema «árboles» de
   [X] con `u_μ = 2|h(μ)| e^{θ|sop μ|}` y Kotecký–Preiss con `a(γ) = θ|sop γ|` dan la convergencia,
   como en [X], Paso 2. Como `|sop μ| ≤ 2w(μ)`, `|∂μ| ≥ 2` y `|ε| ≤ 1/M_2`, se tiene
   `e^{2θ|sop μ|} ≤ (M̄|ε|)^{−w(μ)}`, así que `Σ_{μ∋x} |g(μ)| e^{2θ|sop μ|} ≤ ‖g‖/2 ≤ d/2`, y lo mismo
   para `g̃`. Por tanto `ĥ(h) ≤ d`, `ĥ(h^δ) ≤ d` (sin factor 3: `|h^δ| ≤ |g| + |g̃|`) y
   `ĥ(h^δ + λ1_ν) ≤ d + |λ| e^{4θ}`. Con `d = θ/4` y `|λ| ≤ λ_0 := (θ/4) e^{−4θ}`, todos cumplen
   `ĥ ≤ θ/2`.
3. **Perturbaciones locales.** Si `h'` difiere de `h` solo en los μ que cortan a `A`, los clústeres
   que cambian son incompatibles con un polímero ficticio de soporte `A`. Kotecký–Preiss da
   `|log Z[h'] − log Z[h]| ≤ 2θ|A| = 4θ`. No hay términos `log cosh` que restar.
4. **La marginal.** Cada cociente es como mucho `e^{4θ}`, y su derivada en λ, por Cauchy, como mucho
   `e^{4θ}/λ_0 = 4e^{8θ}/θ`. Sumando los `k⁴` términos sale la cota. ∎

**Lema A′ (marginales por sector de momento).** Sea k | n y sea `ψ_p` el autovector normalizado del
nivel `λ_p` (ε real pequeño, n fijo). Sea `ρ̄_p` su marginal de arista promediada sobre traslaciones.
Los coeficientes de Taylor de `ρ̄_p` de orden `m ≤ ⌈n/2⌉ − 2` no dependen de p, y por tanto valen
`(1⊗V_φ) ρ_∞^{(m)} (1⊗V_φ)†` (Teorema A, punto 1).

*Demostración.* Sea `Ō = (1/n) Σ_j O_j`, que conmuta con T. La familia `H + η n Ō` conserva el
momento, y en cada sector el nivel `λ_p(ε,η)` es analítico, con `∂_η λ_p|_{η=0} = n tr(O ρ̄_p)` por
Hellmann–Feynman. El Paso 2 del Teorema A (la matriz circulante) vale con pasos de ε y de η, porque
cada uno cambia como mucho dos sitios. Así, los coeficientes de `ε^m η` de los `λ_p` coinciden si
`m + 1 < n/2`. La media sobre p es la marginal del multiplete, `ρ_n`. ∎

**Lema Z4 (coeficientes del cuasi-modo).** Con n y Θ fijos, los coeficientes de Taylor de orden
`m ≤ ⌈n/2⌉ − 2` de `m̄_O = (1/n) Σ_j m_{O_j}` son los de `tr[O (1⊗V_φ) ρ_∞ (1⊗V_φ)†]`.

*Demostración.* Con ε real y pequeño, el multiplete está aislado por un hueco `≥ Δ/2`, y
`E(ε) → −n`. El Lema Z2.1 da `‖(1−P)Ψ‖ ≤ (4/Δ)|r| ‖Ψ‖ = O(ε^{⌈n/2⌉})`. Se escribe
`PΨ = Σ_p α_p ψ_p`. Como `Ō` conmuta con T, no hay términos cruzados entre momentos distintos, y
`m̄_O = Σ_p |α_p|² tr(O ρ̄_p)/‖Ψ‖² + O(ε^{⌈n/2⌉})`. Por el Lema A′, esto es
`tr(O ρ_n(Θ)) + O(ε^{⌈n/2⌉−1})` en el eje real. Las dos funciones son holomorfas cerca de 0, así que
sus coeficientes coinciden hasta el orden `⌈n/2⌉ − 2`. Se concluye con el Teorema A, punto 1. ∎

**Teorema Z5 ((P) para la clase con simetría de reloj).** Sea `σ(ε) = Σ_m ρ_∞^{(m)} ε^m`. Es
holomorfa en `|ε| < 1/M_2`, con `|tr Oσ| ≤ B'‖O‖`. Es el límite de las `m̄_O(n, Θ=0)`: sus
coeficientes se estabilizan y están acotados, así que hay convergencia uniforme en compactos. Para
`|ε| < 1/M_2`, n ∈ kZ y `|Θ| ≤ π`, el cuasi-modo promediado sobre traslaciones, que es un estado con
la misma marginal `ρ_n^{cm}(Θ)` en todas las aristas, cumple

    ‖ρ_n^{cm}(Θ) − (1⊗V_φ) σ (1⊗V_φ)†‖_1 ≤ 2 k⁴ B' (M_2|ε|)^{⌈n/2⌉−1} .

Sale del lema de Schwarz aplicado a `m̄_O − tr[O(1⊗V_φ)σ(1⊗V_φ)†]`, que está acotada
por `2B'‖O‖` y tiene un cero de orden `⌈n/2⌉−1` (Lema Z4), y sumando sobre las `k⁴` unidades
matriciales. ∎

Para el intercambio SU(3): `a = 2`, `b = 8,09`, `M̄ = M_* = 359` (el mínimo de `M̄ e^{16(4a+b)/M̄}`
está por debajo de `M_*`), `θ = 0,179`, `M_2 = 735` y `B' ≈ 7,6·10³`.

**Comprobación numérica** (`scripts/kt_marginal.py`). Se compara la marginal media del cuasi-modo
con Θ = π/2 con la del cuasi-modo sin flujo de 12 sitios, girada:

| ε | n = 6 | n = 9 |
|---|---|---|
| 0,1 | 2,5·10⁻⁸ | 4,2·10⁻¹² |
| 0,2 | 2,8·10⁻⁶ | 6,5·10⁻⁹ |

La caída con n es mucho más rápida que `ε^{⌈n/2⌉−1}`, como pasaba en [X] con el exponente n−1.

**Lo que queda para el intercambio SU(k) con k | n.** Se juntan el Lema Z2.3 ((A2E\*) en Θ\* = 0), el
Teorema Z5 ((P)) y los Lemas D y P. Con ε < 0, H(0) es estocuástica sin gauge, y es irreducible en el
sector de contenido `(n/k, …, n/k)`: los intercambios de vecinos generan todas las permutaciones.
Así que, para `−1/M_2 < ε < 0`:
- **(D)** vale uniformemente en Θ, con `ε_D ≤ 2B(M̄|ε|)^{⌈n/2⌉}/g`;
- **(P)** vale con `ε_P ≤ 2k⁴B'(M_2|ε|)^{⌈n/2⌉−1}`;
- los datos son `σ`, que es la marginal de un estado fundamental de la cadena **si se cumple
  (G)**: entonces el multiplete es el fundamental, `e_KT = e_∞` y el límite de los estados `P/k`
  tiene marginal `σ` (Lema Z4 con Θ = 0).

**Teorema Z (robustez para el intercambio SU(k)).** Sea `h = −P_asc + ε Σ_{a≠b} |b a⟩⟨a b|` en
`C^k`, con cualquier carga `q`, k | n y `−1/M_2 < ε < 0`. Supóngase (G): el fundamental de H(0) está
en el sector de contenido `(n/k, …, n/k)`. Entonces, para los datos de estado fundamental σ,

    sup_{Θ,Θ'} |p(n,Θ) − p(n,Θ')| ≤ C (M_2|ε|)^{⌈n/2⌉−1} ,    C = 2B/g + 38 k⁶ B' .

La constante es la del Teorema puente, `ε_D + ε_P/r'` con `1/r' = 19k²`. B es la cota del Lema Z2.2, `g`
es la de la Sección 1 (no el punto fijo) y `M̄|ε| ≤ M_2|ε|`. Con `ε > 0`, lo mismo vale
si (S) se cumple; numéricamente se cumple con n = 6 y 9 (7.7).

Con esto queda **demostrada, salvo (G), la robustez exponencial con k | n para el caso con carga
fraccionaria por sitio** (llenado 1/3 con μ = 1, `q = (1,0,0)`).

### 7.9 La hipótesis (G): qué dicen los números, por qué falla la vía de [X] y qué falta

**(G) para el intercambio SU(3):** el fundamental de `H(0) = −P_asc + ε Σ_{a≠b}|b a⟩⟨a b|` está en el
sector de contenido `c* = (n/3, n/3, n/3)`. El intercambio conserva el número de cada especie, así
que H es diagonal por bloques en los sectores `(c_0, c_1, c_2)`.

**Números** (`scripts/hueco_sectores.py`, todos los sectores de contenido): hueco
`min_{c≠c*} E_c − E_{c*}` y sector que lo realiza.

| ε | n = 6 | n = 9 | n = 12 |
|---|---|---|---|
| −0,5 | 1,237 (3,1,2) | 1,177 (2,4,3) | 1,171 (5,4,3) |
| −0,3 | 1,553 | 1,576 | 1,592 |
| −0,2 | 1,726 | 1,753 | 1,762 |
| −0,1 | 1,881 | 1,892 | 1,893 |
| +0,1 | 1,889 | 1,893 | 1,893 |
| +0,3 | 1,622 | 1,601 | 1,602 |
| +0,5 | 1,398 | 1,256 | 1,215 |

- El hueco **no baja con n**: se estabiliza ya en n = 9–12. Cuando ε → 0 tiende a 2.
- El sector que lo realiza está siempre **a un cambio de especie** de c\*:
  `c = c* + e_b − e_a`. Un cambio de especie crea dos enlaces malos, como `0 1 2 → 0 1 1`, y de ahí
  el 2 clásico.
- Sus estados bajos son **pares de defectos que se separan** (dos paredes con un dominio de otra
  fase entre ellas). El fundamental del sector es el fondo de un continuo de dos cuasipartículas, no
  un estado ligado.

**Por qué no sirve la vía de [X].** En [X], la energía global sale de Perron en el marco de
Kirkwood–Thomas, que es estocuástico: el autovector positivo es el fundamental de todo un sector de
paridad, con todos los `S^z` dentro, y así se comparan los sectores de carga.
- Aquí, en el marco de reloj, el intercambio SU(3) tiene elementos fuera de la diagonal
  `(2/3)ω^{±1}` y `(1/3)ω^{±1}` en las mismas transiciones que los `−1/3` de `−P_asc`. Ningún gauge
  diagonal los hace no positivos: `scripts/sin_signo.py` aplicado a H' con n = 6, para los dos signos
  de ε, encuentra 3 componentes (los sectores `Z_3`), ninguna sin signo.
- Las fases vienen del gauge `∏ω^{−js_j}`, que hace falta para que `−P_asc` sea estocuástica.
- En la base original sí hay Perron (con ε < 0, sin gauge), pero sector de contenido a sector de
  contenido: no compara sectores.
- La cota diamagnética sobre el marco (sustituir cada elemento por `−|·|`) da una cota inferior
  global, pero con otra densidad de energía `e_abs(ε) ≠ e_KT(ε)`. La pérdida es extensiva y no sirve
  para (G), que necesita precisión O(1).

**Lo que sí está demostrado.**
1. *(G) perturbativa, con n fijo.* Si `n|ε|‖v‖ < Δ/4`, (G) se cumple (Kato). No es uniforme en n,
   así que no basta.
2. *Sobre qué hace falta exactamente.* Con ε < 0, el Lema D da `E_0(n,Θ) ≥ E_0(n,0) = min_c E_c`
   para toda Θ. Así que todo lo que falta es comparar sectores de contenido sin flujo, y no hay
   que controlar Θ en los sectores con defectos.
3. *(G) equivale a un hueco de cuasipartículas.* Por el Lema Z2.3, `E_{c*} = n e_KT(ε) + O(n ε^{n/2})`.
   Así que (G) equivale a: `E_c − n e_KT(ε) ≥ 0` para todo c ≠ c\*. Es decir, la energía de un
   sector con defectos, medida desde el vacío vestido, es positiva uniformemente en n. Hace falta
   precisión O(1) sobre una energía extensiva, y ninguna cota local por bloques la da:
   - la cota de Anderson con bloques de dos sitios da `−n(1+ε²)`, frente a `E_{c*} ≈ −n(1+ε²/3)`;
   - con bloques de longitud L, la pérdida es `O(nε²/L)`, y los bloques no ven la restricción
     global de contenido.

**Lo que falta, y con qué.** (G) es un teorema de **hueco por encima del vacío vestido en los sectores
con defectos**, con dos defectos que se pueden separar. Hay dos piezas:
1. **Un defecto.** La energía de un defecto aislado, `Δ_def(ε) = Δ + O(ε)`, analítica y uniforme en n.
   Es exactamente el problema de k ∤ n, donde el anillo fuerza un defecto, y el análogo de la sección
   «Anillos impares» de [X] (ecuación de Bloch de la pared). En el marco de reloj, un defecto es un
   operador local aplicado al vacío de Kirkwood–Thomas. El ansatz `Σ_x f(x) O_x e^G` es el de
   Datta–Kennedy para estados de una cuasipartícula ([dk-quasi] en [X]). Sirve también para la
   otra mitad de la conjetura de la Sección 2: la sensibilidad `n⁻³` con k ∤ n.
2. **Dos defectos.** Hace falta una cota inferior `E_c ≥ n e_KT + 2Δ_def − (ligadura)`, con la
   ligadura acotada:
   - **defectos lejanos:** son aditivos, por la factorización del Paso 4 del Teorema A, ahora con el
     vacío vestido;
   - **defectos cercanos:** es un problema local finito.
   Con ε < 0, Perron en la base original asegura que el fundamental de cada sector es único y de
   momento 0, así que basta controlar el fondo del continuo en momento total 0.

   La alternativa es la diagonalización por bloques local de Del Vecchio–Fröhlich–Pizzo–Ranallo
   ([dfpr] en [X]), que da justo este tipo de hueco, uniforme en n, para el doblete de Néel del XXZ.
   Habría que adaptarla a k fases relacionadas por traslación.

**Estado del Teorema Z.** Queda condicionado solo a (G), que ahora está caracterizada con
precisión: es el hueco de dos defectos sobre el vacío vestido. Numéricamente se cumple con margen,
`≥ 1,17` hasta `|ε| = 0,5`, y es estable en n. La pieza que falta (un defecto, y luego dos) es la
misma que necesita el lado k ∤ n de la conjetura. Así que el siguiente paso natural sirve a las dos
mitades.

### 7.10 Teoría de un defecto: el teorema perturbativo y los números

Esta teoría sirve a las dos mitades de la conjetura. Con k ∤ n, el anillo fuerza un defecto, y su
banda en el flujo da la sensibilidad polinómica (Sección 4). Con k | n, (G) es el hueco de dos
defectos (7.9).

**La variedad de un defecto.** Clase (C) y k ∤ n. Por el Lema 0, toda configuración tiene al menos un
enlace malo. Las que tienen exactamente uno, en el enlace `b_x` y con `s_{x+1} = a_t`, siguen el ciclo
en todo lo demás: `s_{x+1+i} = a_{t+i}`. Llamo `𝒟` al conjunto de las de exceso mínimo; con
`−P_asc`, todas tienen exceso Δ. Dos observaciones:
- si el defecto se desplaza y deja el volumen intacto, `t` cambia con `x`: `t = t_0 + x`. Para
  volver al punto de partida, el defecto da `k/mcd(k, n)` vueltas, que es el «da 3 vueltas» de la
  Sección 4;
- el tipo del enlace malo es `(a_{t+n−1}, a_t)`. Con k = 3 es un par igual `(a,a)` si n ≡ 1, que
  tiene carga −1/3 en el modelo de llenado, y un par descendente `(a+1, a)` si n ≡ 2, con carga +1/3.

**La herramienta: el hamiltoniano efectivo de Bloch.** Con n fijo y ε pequeño, sean `P(ε)` el
proyector espectral del grupo de niveles que sale de `𝒟` y `P_0` el proyector sobre `span 𝒟`. Sea

    H_B = P_0 H P (P_0 P P_0)^{-1} P_0 .

Es semejante a `H|_{ran P}`, así que tiene los mismos autovalores, y sus coeficientes de Taylor son
sumas sobre caminos (Paso 1 del Teorema A). Lo que lo hace útil es que **se factoriza en sumas
tensoriales**: si `H = h_A ⊗ 1 + 1 ⊗ h_B`, con `h_B` de fundamental no degenerado, entonces
`P = P_A ⊗ P_B`, `(P_0PP_0)^{-1}` se factoriza y

    H_B = H_{B,A} ⊗ P_{0B} + P_{0A} ⊗ μ_B ,

que sobre el espacio bajo es `H_{B,A} + μ_B`, aditivo. Con el operador reducido de Kato, la
función de transformación no se factoriza y esto no sale.

**Teorema D (un defecto, perturbativo).** Sean (C), k ∤ n, Θ real y `0 ≤ m ≤ n/2 − 2`. Entonces

    ⟨d'| H_B^{(m)} |d⟩ = n e^{(m)} δ_{d'd} + K^{(m)}(d', d) e^{−iφ D(d,d')} ,

donde:
- `e^{(m)}` son los coeficientes del vacío del Teorema A, los de la cadena;
- el **núcleo del defecto** `K^{(m)}` es invariante por traslación (solo depende de la posición
  relativa de d y d'), no depende de n ni de Θ, y se anula si los defectos están a más de `2m + 2`
  sitios;
- `D(d,d') = Σ_x x (q(d'_x) − q(d_x))` es el cambio del dipolo de carga, calculado en una ventana
  que no da la vuelta: es la **fase de Peierls** del defecto.

*Demostración* (el esquema del Teorema A, con `H_B` en lugar de `λ̄`).
1. Con variables `ε_b` por enlace, el coeficiente de un multiíndice α cuyo soporte tiene una
   componente B separada (a distancia ≥ 2) del resto y de la región `Y(d, d')` es nulo salvo que
   d = d'. Aquí `Y(d, d')` son los sitios donde d y d' difieren más los enlaces malos de los dos.
   La razón: en el espacio de configuraciones iguales a d fuera de `X = S_A ∪ S_B`, H es una suma
   tensorial con `h_B` no degenerado, y la factorización de `H_B` da `μ_B` en la diagonal y nada
   mixto.
2. Las componentes del tipo B suman la energía del vacío: `Σ_B μ_B^{(m)} = n e^{(m)}` menos los
   racimos que tocarían la región del defecto. Esa resta es local y se mete en `K`. Si la clase no
   tiene simetría de reloj, el vacío alrededor de cada sitio está en una fase que depende del sitio.
   Con k ∤ n, el periodo se corta en el defecto, y el resto también es local.
3. El resto son racimos conexos con `Y(d, d')` (los conectados al defecto), de grado m. Abarcan como
   mucho `2m + 2 < n` sitios, así que se levantan a una cadena con un solo defecto, que existe en Z
   para cualquier tipo. El coeficiente no depende de n.
4. El gauge `Γ = ∏_{x∈I} V_φ^{x−x_0}` en la ventana I quita el giro. Como d y d' tienen cargas
   distintas en la ventana, deja la fase `e^{−iφ D(d,d')}`. ∎

**Consecuencias formales, a todos los órdenes hasta n/2 − 2:**
1. **Energía y banda.** Los niveles bajos del anillo son `n e(ε) + ω(κ, ε)`. Aquí ω es la banda de
   un hamiltoniano de Bloch de alcance finito en cada orden y que no depende de n, y κ está
   cuantizado en un anillo efectivo de `kn/mcd(k,n)` sitios con el flujo de Peierls.
2. **Respuesta al flujo.** Si la banda es dispersiva (`K` con salto no nulo en algún orden) y la
   carga del defecto no es nula, la respuesta es la curvatura de la banda: `R_n ∝ n^{−2}` y
   sensibilidad `∝ n^{−3}`. Es la parte 2 de la conjetura de la Sección 2, ahora con un enunciado
   perturbativo preciso.
3. **Dos defectos.** El mismo teorema, con la variedad de dos defectos, da
   `H_B = n e + K_1 ⊗ 1 + 1 ⊗ K_2 + W`. La interacción W viene de los racimos conectados con los dos
   defectos, así que a separación L tiene grado `≥ L/2` y es `O(ε^{L/2})`. Pero tiene una parte de
   primer orden de alcance corto, porque los pares adyacentes de los dos tipos son degenerados con
   los separados (exceso 2Δ) y se conectan por un solo intercambio. Así, el fondo del sector es
   `n e + Δ_1 + Δ_2 − (ligadura de un problema local finito) = n e + 2Δ − O(ε)`. Con eso, **(G) vale
   a todos los órdenes**.

**Números** (`scripts/defecto_su3.py`, intercambio SU(3), sectores de contenido, malla de Θ de 16
o 24 valores). Se usa `Δ_def = E_0 − n e_ref`, con `e_ref = E_{c*}(12)/12`:

| ε | n ≡ 1: Δ_def (n = 7 / 10 / 13) | n ≡ 2: Δ_def (8 / 11 / 14) | n²R_n, n ≡ 1 | n²R_n, n ≡ 2 | hueco de dos defectos (n = 12) | ligadura |
|---|---|---|---|---|---|---|
| −0,05 | 0,99257 / 0,99257 | 0,99584 / 0,99584 | 0,024 / 0,024 | 0,0083 / 0,0082 | 1,9486 | 0,040 |
| −0,1 | 0,97108 / 0,97111 | 0,98343 / 0,98348 | 0,093 / 0,094 | 0,037 / 0,033 | 1,8933 | 0,061 |
| −0,2 | 0,89422 / 0,89500 / 0,89504 | 0,93402 / 0,93549 / 0,93560 | 0,375 / 0,342 / 0,339 | 0,237 / 0,150 / 0,137 | 1,7620 | 0,069 |
| −0,3 | 0,78706 / 0,79122 | 0,85135 / 0,85918 | 0,85 / 0,74 | 1,23 / 0,50 | 1,5916 | ≈ 0,06 |

(La ligadura es `Δ_def(n≡1) + Δ_def(n≡2) − hueco`.)

- **Δ_def converge en n** con corrección exponencial, como dice el Teorema D: con ε = −0,2,
  `0,89500 → 0,89504`. Su primer coeficiente es de segundo orden: `Δ_def − 1 ≈ −2,9 ε²` (n ≡ 1) y
  `−1,65 ε²` (n ≡ 2), porque un intercambio desde un estado de un defecto siempre crea dos defectos
  más. La anchura de banda también es `∝ ε²`: n²R_n se multiplica por 3,9 de ε = 0,05 a 0,1.
- **n ≡ 1: `n²R_n → 0,339`.** Sensibilidad `n⁻³`, la parte 2 de la conjetura para el intercambio
  SU(3), con carga fraccionaria por sitio: el intercambio neutro de amplitud ε libera el defecto
  de 4.1. Con n ≡ 2, n²R_n aún baja de 11 a 14. El defecto descendente es más pesado, y hacen falta
  tamaños mayores para ver el límite.
- **La ligadura es O(ε)**: 0,040 y 0,061 con ε = 0,05 y 0,1, como predice la consecuencia 3. El
  hueco de (G) es `2Δ − O(ε)`, lejos de cerrarse.
- Con ε > 0, todo coincide en módulo, con el mínimo en Θ = π: es el Marshall del Lema D.

**Lo que queda, ya bien delimitado.**
1. **La versión no perturbativa del Teorema D**: la banda `ω(κ, ε)` analítica y acotada en un disco
   uniforme en n, para pasar de «a todos los órdenes» a cotas con Schwarz, como el Teorema B frente
   al A. En la clase con simetría de reloj, el instrumento natural es el ansatz de una cuasipartícula
   de Datta–Kennedy en el marco de reloj, `Σ_x f(x) O_x e^{G}`, con G el vacío de 7.7 y `O_x` el
   operador local que crea el defecto. Con k ∤ n, el marco tiene un giro `Z_k` en un enlace, que es
   lo que fuerza el defecto.
2. **Con eso**, (G) sale de la consecuencia 3, y el Teorema Z queda incondicional para
   `−1/M_2 < ε < 0`. El lado k ∤ n sale como en [X, «Anillos impares»], con la ecuación de Bloch del
   defecto en lugar de la de la pared.

### 7.11 El hueco no perturbativo: Gershgorin para el generador vestido

Esta sección cierra (G) y toda la hipótesis (A) en la clase con simetría de reloj, sin la teoría
completa de una cuasipartícula. La idea es la del argumento de no degeneración de Datta–Kennedy,
sin positividad.

**El generador vestido.** Sea g el punto fijo de 7.7 y `G = Σ_μ g(μ) χ_μ`. Se conjuga el
hamiltoniano del marco por la multiplicación por `e^{G}`, que es invertible:

    H̃ = e^{−G} H' e^{G},    (H̃φ)(s) = Σ_δ C_δ(s) φ(s+δ),    C_δ = c_δ e^{G_δ}.

Es semejante a H, así que tiene su mismo espectro. La ecuación del punto fijo es
`H̃1 = E + Σ_{c≠0} r_c χ_{c·1}`, y sobre un carácter

    (H̃ − E) χ_μ = χ_μ · [ |∂μ| + F_μ ],
    F_μ = Σ_c r_c χ_{c·1} + Σ_δ (C_δ − C_δ^{(0)}) (ω^{μ·δ} − 1).

Aquí `C^{(0)}` es la parte de `−P_asc` sin vestir, que da exactamente `|∂μ|`. El factor
`ω^{μ·δ} − 1` se anula salvo en los desplazamientos que tocan μ:
- los de tipo τ (los de `−P_asc` y los de v) solo tocan los enlaces de ∂μ;
- los del giro w tocan cualquier enlace, pero tienen amplitud `b/n`.

**Lema Z6 (Gershgorin).** Sean las hipótesis del Lema Z1 (`M̄ ≥ M_*`, `|ε| ≤ 1/M̄`, `|Θ| ≤ π`,
n ∈ kZ, ε complejo incluido), `ρ = Σ_{c≠0}|r_c| ≤ (d/2)(M̄|ε|)^{⌈n/2⌉}` y

    η = 2(e^{2d} − 1) + (2a + b)|ε| e^{2d} + ρ/2 .

Para μ no constante, la suma de los módulos de los coeficientes de Fourier de `F_μ` es como mucho
`η|∂μ|`. Para μ = c·1 con c ≠ 0, es como mucho `ρ + 2b|ε|e^{2d}`, y para μ = 0, como mucho ρ. Por
tanto, si `η < 1/2` y `ρ + 2b|ε|e^{2d} < 1`:
1. H tiene exactamente k autovalores, con multiplicidad algebraica, en `|z − E| ≤ ρ + 2b|ε|e^{2d}`;
2. el resto del espectro está en `Re(z − E) ≥ 2(1 − η)`;

uniformemente en n.

*Demostración.* Con `‖·‖_1` la suma de los módulos de los coeficientes de Fourier, que es
submultiplicativa, `‖G_δ‖_1 ≤ 2 Σ_{μ' anclado} |g(μ')| ≤ 2‖g‖ ≤ 2d`, porque `|∂μ'| ≥ 2` y los pesos
son ≥ 1. Así `‖e^{G_δ}‖_1 ≤ e^{2d}` y `‖e^{G_δ} − 1‖_1 ≤ e^{2d} − 1`. Hay cuatro contribuciones a `F_μ`:
- `−P_asc` vestido: como mucho `(1/k)Σ_m 2(e^{2d} − 1) ≤ 2(e^{2d} − 1)` por cada enlace de ∂μ;
- v: como mucho `2a|ε|e^{2d}` por cada enlace de ∂μ;
- el giro w: como mucho `n · (b/n)|ε| · 2e^{2d} = 2b|ε|e^{2d} ≤ b|ε|e^{2d}|∂μ|`;
- los residuos: `ρ ≤ (ρ/2)|∂μ|`.

Para μ constante, los términos de tipo τ se anulan, porque `c·1·δ = c(δ_j + δ_{j+1}) = 0`. Queda
`ρ` más, con Θ ≠ 0, el giro.

La matriz de `H̃ − E` en la base de caracteres tiene en la columna μ la diagonal
`|∂μ| + F̂_μ(0)` y fuera de ella los `F̂_μ(ν)`, con ν ≠ 0. El teorema de Gershgorin por columnas (vale
en cualquier base, porque es una matriz de un operador lineal en dimensión finita) pone los
autovalores en la unión de discos:
- de radio pequeño alrededor de 0 para los k caracteres constantes;
- contenidos en `|z − |∂μ|| ≤ η|∂μ|`, y por tanto en `Re z ≥ 2(1 − η)`, para los demás.

Las dos familias de discos son disjuntas, así que la primera contiene exactamente k autovalores. ∎

**Lo que da** (clase con simetría de reloj, k | n, `|Θ| ≤ π`, los dos signos de ε):
1. **(A1) para toda Θ**: el multiplete está separado del resto por `≥ 2(1 − η) − (ρ + 2b|ε|e^{2d})`,
   uniformemente en n. Ya no hacen falta ni Perron ni el Lema D.
2. **(A2E) para cada nivel y cada Θ.** El contorno `|z − E(ε)| = 1` separa el grupo del resto en todo
   el disco complejo `|ε| < 1/M̄`, así que el proyector del grupo es holomorfo ahí. Como cada sector
   de momento tiene un solo nivel del grupo, cada `λ_p` también lo es, con
   `|λ_p/n − E/n| ≤ (ρ + 2b|ε|e^{2d})/n`. Con el Lema Z2.2, `|λ_p/n + 1| ≤ B` uniformemente.
3. **(G).** El grupo está aislado en todo el disco, y el contenido conmuta con H. Así que el
   contenido del grupo es el mismo para todo ε, el de ε → 0, que es `c*`. El fundamental está en el
   sector `c*`, para los dos signos de ε.
4. **El Teorema B se aplica tal cual**, con (A) demostrada:
   `|E_0(n,Θ)/n − e_∞| ≤ 2B(M̄|ε|)^{⌈n/2⌉}` para toda Θ. Los datos σ son de estado fundamental,
   porque el multiplete es el fondo del espectro.

Para el intercambio SU(3), con `M̄ = 359` y `|ε| ≤ 1/359`, salen `d = 0,045`, `η ≤ 0,23` y
`ρ + 2b|ε|e^{2d} ≤ 0,05`.

**Teorema Z′ (robustez, incondicional).** Sea h de la clase con simetría de reloj; por ejemplo,
`h = −P_asc + ε Σ_{a≠b}|b a⟩⟨a b|` en `C^k` con cualquier carga `q`. Para k | n y
`|ε| < 1/M_2`, con cualquier signo, los datos de estado fundamental cumplen

    sup_{Θ,Θ'} |p(n,Θ) − p(n,Θ')| ≤ (2B/g + 38 k⁶ B') (M_2|ε|)^{⌈n/2⌉−1} .

Juntando el Teorema A, los Lemas Z1, Z2, Z3, A′, Z4 y Z6, los Teoremas B y Z5 y el Teorema puente,
ya no quedan hipótesis abiertas. Para el intercambio SU(3), con `M_2 = 735`, esto cubre el modelo de
llenado 1/3 con μ = 1, el caso con carga fraccionaria por sitio.

**Comprobación numérica** (`scripts/gershgorin_kt.py`; intercambio SU(3); η y la cota medidos
columna a columna sobre la matriz de `H̃ − E` en la base de caracteres; con n = 12, sobre 3000
columnas al azar, así que η es una cota inferior del máximo real):

| n  | ε     | Θ/π | η     | cota de Gershgorin del hueco | hueco exacto |
|----|-------|-----|-------|------------------------------|--------------|
| 6  | ±0,1  | 0   | 0,27  | 1,53                         | 1,88         |
| 6  | ±0,2  | 0   | —     | 0,87–0,91                    | 1,73–1,76    |
| 9  | −0,2  | 0   | 0,675 | 0,798                        | 1,753        |
| 9  | −0,1  | 0   | 0,278 | 1,521                        | 1,892        |
| 9  | +0,1  | 0   | 0,273 | 1,530                        | 1,893        |
| 9  | +0,1  | 1/2 | 0,299 | 1,425                        | 1,892        |
| 12 | −0,2  | 0   | 0,647 | 0,786                        | —            |
| 12 | −0,1  | 0   | 0,260 | 1,521                        | —            |
| 12 | +0,1  | 0   | 0,255 | 1,529                        | —            |
| 12 | +0,1  | 1   | 0,310 | 1,404                        | —            |

η no crece con n: con Θ = 0 incluso baja ligeramente de n = 9 a 12. La cota del hueco es estable
en n con tres cifras. Con Θ ≠ 0 sube un poco, por el término del giro, que es `O(b|ε|)` y no depende
de n. Con ε = 0,1, que es unas 36 veces el radio garantizado `1/M_*`, la cota ya da el 80 % del hueco
exacto. Con ε = −0,2, η > 1/2 y el lema no se aplica en su forma enunciada. Aun así, la cota columna
a columna sigue siendo positiva (≈ 0,79) y estable en n. El radio real del argumento es mucho mayor
que `1/M_*`; la pérdida está en las constantes de Z1, no en el método.

**El lado k ∤ n** se trata en 7.12.

### 7.12 La banda de un defecto, no perturbativa (k ∤ n)

Esta sección pasa el Teorema D de «a todos los órdenes» a cotas, para la clase con simetría de
reloj. El esquema es el del Teorema B frente al A. Gershgorin, con el mismo vestido de 7.11, aísla la
banda de forma uniforme en n. La simetría la parte en niveles simples y holomorfos. El Teorema D
fija sus coeficientes de Taylor, y Schwarz hace el resto. No hace falta el ansatz de una
cuasipartícula.

**El marco torcido.** Con k ∤ n, el gauge `D = ∏_j ω^{−j s_j}` no cierra. En el enlace de cierre
`(n−1, 0)` queda un giro de reloj `t ≡ −n mod k`, `t ≠ 0`. En la base de caracteres, `−P_asc` sigue
siendo diagonal:

    −P_asc χ_μ = (−n + |∂_tw μ|) χ_μ ,

donde `|∂_tw μ|` cuenta los enlaces rectos con `μ_j ≠ μ_{j+1}` más el de cierre si
`μ_{n−1} − μ_0 ≢ −t`. Como la suma de los saltos alrededor del anillo es `≡ t ≢ 0`, se tiene
`|∂_tw μ| ≥ 1`. Los de una pared, `P = {μ : |∂_tw μ| = 1}`, son exactamente nk: un valor c y la
posición del salto. Son la imagen de la variedad 𝒟 de 7.10. El siguiente nivel es
`|∂_tw μ| = 2`, que existe para todo k ≥ 3 porque t se puede partir en dos saltos no nulos. **El hueco
clásico de la banda al continuo es 1, no 2.**

**El flujo, en un solo enlace.** El gauge diagonal `e^{iφ Σ_x x q(s_x)}` es unitario y lleva `H(Θ)`,
con el giro uniforme, a `H^♯(Θ)`. En `H^♯`, los enlaces de la cadena no tienen giro, y el de cierre
lleva la fase entera `e^{−iΘ(q(b')−q(b))}`. Los dos tienen el mismo espectro. A partir de aquí todo
se hace con `H^♯`. Ganancia: en el volumen ya no hay giro (b = 0), y todo lo que no es de la cadena
vive en un solo enlace.

**El hamiltoniano recto y su vestido.** Sea `H'_0` el operador del marco que tiene en *todos* los
enlaces, el de cierre incluido, el término de dos sitios de la cadena, sin giro de reloj ni flujo.
Es invariante por la traslación del marco, es de la clase con simetría de reloj con normas a y
b = 0, y su clásico tiene paredes `|∂μ|`. El Lema Z1 solo usa esa forma, no que k | n. Da:
- el punto fijo `G_0`, con `‖g_0‖ ≤ d` y `d = 4a/M̄`;
- la energía `E_0`;
- el residuo `ρ` del Lema Z2.

`V_c = H'^♯_tw − H'_0` vive en el enlace de cierre. Ahí `H'^♯_tw` lleva el giro de reloj t y la fase
de flujo. Sea `a'` el máximo sobre Θ de la norma de coeficientes del marco de `v` con el flujo Θ
entero en un enlace. Los caracteres de reloj del giro t solo cambian fases, así que no cambian esa
norma.

**Lema Z7 (aislamiento de la banda).** Sean las hipótesis del Lema Z1 para `H'_0`, con k ∤ n,
`|Θ| ≤ π` y ε complejo, y sean

    η = 2(e^{2d} − 1) + 2a|ε|e^{2d} + ρ/2 ,    η_c = 2(e^{2d} − 1) + (a + a')|ε| e^{2d} .

Si `5η + 2η_c < 1`, entonces, uniformemente en n y en Θ:
1. H tiene exactamente nk autovalores, con multiplicidad algebraica, en
   `|z − E_0 − 1| ≤ r_1 := 2η + η_c`;
2. el resto del espectro está en `Re(z − E_0) ≥ 2 − 3η − η_c`.

*Demostración.* Sea `H̃ = e^{−G_0} H'^♯_tw e^{G_0} = H̃_0 + Ṽ_c`. Para `H̃_0` vale el Lema Z6 con b = 0:
`(H̃_0 − E_0)χ_μ = χ_μ[|∂μ| + F_μ]`, con
- `‖F_μ‖_1 ≤ η|∂μ|` si μ no es constante, y entonces `|∂μ| ≥ 2`, porque en un anillo una sola
  pared recta es imposible;
- `‖F_μ‖_1 ≤ ρ` si μ es constante.

`Ṽ_c` es una suma de desplazamientos δ del enlace de cierre con coeficientes `c_δ e^{G_δ}`, y sobre
un carácter actúa multiplicando: `Ṽ_c χ_μ = χ_μ Σ_δ c_δ e^{G_δ} ω^{μ·δ}`. Esto vale para cualquier δ,
tenga o no simetría de reloj. Separando `e^{G_δ} = 1 + (e^{G_δ} − 1)`:
- en `−P_asc`, la parte «1» es exactamente `|∂_tw μ| − |∂μ|`, porque el flujo no toca la diagonal
  clásica;
- en v, la parte «1» tiene norma ≤ `(a + a')|ε|`;
- el resto tiene norma ≤ `2(e^{2d} − 1) + (a + a')|ε|(e^{2d} − 1)`.

En total, `(H̃ − E_0)χ_μ = χ_μ[|∂_tw μ| + F'_μ]`, con `‖F'_μ‖_1 ≤ η|∂μ| + η_c` o `≤ ρ + η_c` si μ es
constante. Como `|∂μ| ≤ |∂_tw μ| + 1`, Gershgorin por columnas da:
- columnas de P, donde `|∂μ| ≤ 2`: discos en `|z − 1| ≤ 2η + η_c`;
- columnas con `|∂_tw μ| = m ≥ 2`: discos en `|z − m| ≤ η(m + 1) + η_c`, luego
  `Re z ≥ m(1 − η) − η − η_c ≥ 2 − 3η − η_c`.

Las dos uniones son disjuntas si `5η + 2η_c < 1`. ∎

Para el intercambio SU(3), `a = 2` y `a' = 10/3`. Con b = 0, el Lema Z1 da `M_* = 165`. La condición
`5η + 2η_c ≤ 1/2` vale con `M̄ = M_3 = 538`, donde `d = 0,015`, `η ≤ 0,072` y `η_c ≤ 0,071`. Con esa
misma condición, la serie de Neumann de `(z − H̃_QQ)^{−1}` converge en `ℓ¹` con razón ≤ 1/2 en todo
el disco de la banda. La razón por columna, `(η(m+1) + η_c)/(m − 1 − r_1)`, es máxima con m = 2. Así
que la reducción de Feshbach a P está bien definida. No hace falta para el teorema, pero sí para la
comprobación numérica.

**Lema Z4′ (energía por enlace del vacío recto).** Sea `H'_0` invariante por traslación del marco,
de la clase con simetría de reloj y sin giro (b = 0), en un anillo de n sitios, con n cualquiera, k ∤ n
incluido. Sea `ε_b(n) = E_0/n`, con `E_0 = R̂(0)` la energía de Kirkwood–Thomas. En las hipótesis del
Lema Z1:
1. `ε_b(n)` es holomorfa en `|ε| < 1/M̄` y `|ε_b(n) + 1| ≤ B_0`, con
   `B_0 = (e^{2d} − 1 − 2d) + a e^{2d}/M̄`, la cota del Lema Z2.2 con b = 0;
2. sus coeficientes de Taylor de orden `m ≤ ⌈n/2⌉ − 2` son los coeficientes `e^{(m)}` del vacío del
   Teorema A, que no dependen de n;
3. la serie `e_∞ = Σ e^{(m)} ε^m` es holomorfa en `|ε| < 1/M̄`, con `|e_∞ + 1| ≤ B_0`, y

       |E_0 − n e_∞| ≤ 2 n B_0 (M̄|ε|)^{⌈n/2⌉−1} .

*Demostración.*
1. La holomorfía y la cota son las del Lema Z2.2. Su prueba suma por enlaces y usa solo Z1, no que
   k | n.
2. *Localidad de los coeficientes.* Se escribe la ecuación del punto fijo, `R̂(μ) = 0` para μ no
   constante, orden a orden. El término lineal de `−P_asc` es `|∂μ| g(μ)`, así que

       g^{(m)}(μ) = −(1/|∂μ|) Φ_μ^{(m)}[g^{(<m)}] .

   Aquí `Φ_μ^{(m)}` es un polinomio en los coeficientes de orden menor. Cada uno de sus monomios
   procede de un enlace j y es un producto `g^{(m_1)}(μ_1)···g^{(m_p)}(μ_p)`, quizá con un factor ε
   y un carácter ν de v en j, tal que:
   - `Σ μ_i + ν = μ`;
   - cada `μ_i` toca `{j, j+1}`, porque `G(s+δ) − G(s)` solo contiene caracteres con `μ'·δ ≠ 0`, como
     en la prueba de Z1;
   - `Σ m_i (+1) = m`.

   Por la norma de Z1, `g^{(m')}(μ') = 0` si `w(μ') > m'`. Así que el soporte de cada monomio, con el
   enlace j, tiene un recubrimiento conexo con ≤ m + 1 enlaces. Si `m ≤ ⌈n/2⌉ − 2`, ese
   recubrimiento no da la vuelta al anillo, y el monomio se levanta a la cadena de forma única salvo
   traslación. Los soportes van a un arco, la suma de caracteres en `Z_k` se conserva, y `|∂μ|` es
   el mismo en el anillo y en la cadena, porque μ se anula fuera de un arco propio.

   Como los términos de enlace de `H'_0` son los de la cadena en todos los enlaces, la inducción
   en m da `g^{(m)}(μ) = g_ch^{(m)}(μ̂)` para todo μ, donde μ̂ es el levantamiento y `g_ch` el punto
   fijo formal de la cadena. Los dos lados son nulos si `w(μ) > m`. Con el mismo argumento para
   `R̂(0)`, cuyos monomios tocan el enlace j que los produce, se obtiene `E_0^{(m)} = n e_ch^{(m)}`,
   donde `e_ch^{(m)}` es la contribución de un enlace de la cadena.

   Falta ver que `e_ch^{(m)} = e^{(m)}`. Se toma n′ ∈ kZ grande. En el anillo de n′ sitios, `H'` es
   el mismo operador, ahora sí con k | n′, y por lo anterior `E^{(m)} = n′ e_ch^{(m)}`. Por el Lema
   Z2.3, `E + Σ_c r_c ω^{cN}` son los niveles exactos del multiplete, con `r_c = O(ε^{⌈n′/2⌉})`.
   Por el Teorema A, punto 1, sus coeficientes son `n′ e^{(m)}` para m < n′/2.
3. Por 1 y 2, con n → ∞, Cauchy da `|e^{(m)}| ≤ (1 + B_0) M̄^m`. La serie converge en
   `|ε| < 1/M̄`, `ε_b(n) → e_∞` uniformemente en compactos y `|e_∞ + 1| ≤ B_0`. La diferencia
   `ε_b(n) − e_∞` es holomorfa, está acotada por `2B_0` y se anula hasta el orden `⌈n/2⌉ − 2`.
   Schwarz da la cota. ∎

*Comprobación numérica.* Con el intercambio SU(3), `ε_b(n) − ε_b(11)`:

| n | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| ε = −0,2 | −4,7·10⁻⁶ | −1,4·10⁻⁶ | +1,3·10⁻⁷ | −4,2·10⁻⁹ | −1,1·10⁻⁹ | +3,1·10⁻¹¹ | −3,9·10⁻¹² |
| ε = ±0,1 | −3,5·10⁻⁷ | −2,3·10⁻⁸ | +3,1·10⁻¹⁰ | −1,9·10⁻¹¹ | −1,2·10⁻¹² | −1,1·10⁻¹⁴ | ≈ 10⁻¹⁵ |

- La convergencia es exponencial y no distingue n mod 3, como dice el lema.
- El vacío recto con k ∤ n no tiene ninguna corrección `O(1/n)`.
- El decaimiento real, de orden ε^{≈1,5n}, es más rápido que la cota `(M̄|ε|)^{n/2}`.
- Con ε = ±0,1, los dos signos coinciden en todas las cifras, así que numéricamente `ε_b` es par en
  ε. No lo he demostrado.

**Hipótesis (S_d) (simetría que separa las órbitas).** H conserva, para todo Θ, una carga que toma
valores distintos en las k órbitas de 𝒟 bajo la traslación. El intercambio SU(k) la cumple, porque
conserva el contenido, el número de cada especie. Las k órbitas tienen contenidos distintos, que son
las permutaciones cíclicas de un contenido no uniforme, porque k ∤ n.

El reloj global `∏X` *no* sirve con Θ ≠ 0. El giro usa la carga q, que `X` no conserva: con n = 4,
ε = 0,1 y Θ = 0,7, `‖[H, ∏X]‖ ≈ 0,035`. Solo la carga U(1) tampoco basta en general. Con q = (1,0,0)
y n = 7, dos de las tres órbitas tienen la misma carga 2.

**Teorema D′ (banda de un defecto, no perturbativa).** Sean h de la clase con simetría de reloj,
k ∤ n, (S_d), `|Θ| ≤ π` y `|ε| < 1/(4M_3)`, ε complejo incluido. Entonces:
1. **Niveles simples y holomorfos.** Los nk niveles de la banda se etiquetan por los autovalores
   conjuntos de T y de la carga de (S_d). Cada sector contiene exactamente uno, `λ_κ(ε, Θ)`, que es
   holomorfo en ε. Para ε real, la banda es el fondo del espectro.
2. **Banda límite.** Existe una función `ω(κ, ε)` que no depende de n ni de Θ, holomorfa en ε en
   ese disco y real-analítica y periódica en κ, cuya serie de Taylor es la del Teorema D. Se cumple

       | λ_κ(ε, Θ) − n e_∞(ε) − ω(κ − φ q_d, ε) | ≤ 3C_1 (4M_3|ε|)^{⌊n/2⌋−1} .

   Aquí κ recorre el conjunto cuantizado de la consecuencia 1 del Teorema D, y `φ q_d` es el
   desplazamiento de Peierls, con el convenio de la fase `e^{−iφD}` del Teorema D. La constante
   `C_1 = 1 + r_1 + sup_n 2nB_0 2^{−(⌈n/2⌉−1)}` no depende de n.
3. **Respuesta al flujo** (ε real). Sea `R_n = max_Θ E_gs − min_Θ E_gs` y `L = kn/mcd(k, n)`.
   - `R_n ≤ (π²/2) sup_κ|∂_κ²ω| L^{−2} + 6C_1(4M_3|ε|)^{⌊n/2⌋−1}`.
   - Si `q_d ≠ 0` y el coeficiente `ω^{(m_0)}` de menor orden que depende de κ tiene mínimos no
     degenerados, existe `ε_1 > 0` tal que `R_n ≥ c(ε) n^{−2}` para `0 < |ε| < ε_1` y n grande. Para
     el intercambio SU(3), `m_0 = 2`.

   La sensibilidad es entonces `≍ n^{−3}`.

*Demostración.*
1. *Simetría y holomorfía.* T y la carga de (S_d) conmutan con `H(Θ)`. El proyector de Riesz Π del
   disco del Lema Z7, transportado por el gauge unitario, es holomorfo en todo el disco, porque el
   aislamiento es uniforme. Además conmuta con los proyectores de sector.

   Con ε = 0, `H = −P_asc` no depende de Θ. Ahí `Z_n` × (S_d) actúa de forma libre y transitiva
   sobre las nk configuraciones de 𝒟. La traslación es libre porque mueve el defecto, y (S_d) separa
   las órbitas. Así que cada sector conjunto tiene dimensión 1. El rango de `Π Π_κ` es continuo,
   así que vale 1 en todo el disco, y `λ_κ = tr(H Π Π_κ)` es holomorfo. Para ε real, la banda está
   debajo del resto por el punto 2 del lema.
2. *Cota uniforme.* `|λ_κ − E_0| ≤ 1 + r_1`, y por el Lema Z4′,
   `|E_0 − n e_∞| ≤ 2nB_0(M̄|ε|)^{⌈n/2⌉−1}`. En `|ε| ≤ 1/(2M_3)` queda `|λ_κ − n e_∞| ≤ C_1`.

   Antes usaba aquí la versión con giro uniforme de `H_0`. Esa versión tiene un desajuste de giro
   `O(1/n)` en el enlace de cierre que no se va con el gauge del arco, y la cota uniforme no estaba
   justificada.
3. *ω.* Por el Teorema D, para `m ≤ n/2 − 2`, el coeficiente m-ésimo de `λ_κ − n e_∞` es
   `ω^{(m)}(κ − φq_d)`. Es un polinomio trigonométrico de grado ≤ 2m + 2 que no depende de n ni de
   Θ. Cauchy en el radio `1/(2M_3)` da `|ω^{(m)}| ≤ C_1(2M_3)^m` en cada punto alcanzable.

   Los puntos alcanzables, `κ − φq_d`, con todos los n ≥ 2m + 4 del mismo tipo mod k y todo Θ, son
   densos en el círculo, aunque `q_d = 0`, porque el paso `2π/L` tiende a 0. Por continuidad, la
   cota vale para todo κ. Así, `ω = Σ ω^{(m)} ε^m` converge en `|ε| < 1/(2M_3)`, y
   `|ω| ≤ 2C_1` en `|ε| ≤ 1/(4M_3)`. Bernstein, con grado ≤ 2m + 2, da la analiticidad en una banda
   `|Im κ| < y_0(ε)`.
4. *Schwarz.* `f = λ_κ − n e_∞ − ω(κ − φq_d)` es holomorfa en `|ε| < 1/(4M_3)`, está acotada por
   `3C_1` y se anula hasta el orden `⌊n/2⌋ − 2`.
5. *Flujo.* Por 1, 4 y el Lema Z7, `E_gs(n, Θ) − n e_∞` es el mínimo de ω sobre una red de paso
   `2π/L` desplazada por Peierls, salvo el error de 4. La variación de ese mínimo con el
   desplazamiento está acotada por `sup|ω''|(π/L)²/2`. Si `q_d ≠ 0`, al recorrer Θ el desplazamiento
   cubre una fracción fija del paso.

   La cota inferior necesita que el mínimo de ω sea no degenerado. Para ε pequeño, ω está dominada
   por `ω^{(m_0)}` en su parte dependiente de κ, así que el mínimo hereda la no degeneración. ∎

Con 7.11 y 7.12, las dos mitades de la conjetura de la Sección 2 están demostradas sin
perturbaciones para el intercambio SU(k), y en general para la clase con simetría de reloj que
cumpla (S_d) en el lado k ∤ n:
- k | n: robustez exponencial, Teorema Z′;
- k ∤ n: un defecto móvil con banda analítica y sensibilidad `≍ n^{−3}`, Teorema D′.

La cota inferior necesita que la banda sea dispersiva. Para el intercambio SU(3), el Teorema D lo da
en orden ε² (`m_0 = 2`), con el salto de k = 3 sitios que se ve abajo. Ese salto da `cos(3κ + ·)`,
cuyos mínimos son no degenerados.

**Comprobación numérica** (`scripts/banda_kt.py`). Se calcula `H̃ = e^{−G_0}H'_tw e^{G_0}` denso en la
base de caracteres, en el intercambio SU(3). La tabla usa la variante con giro uniforme, en la que
`H_0` es H con el enlace de cierre conjugado por `X_0^r`. Con Θ = 0 coincide con la del Lema Z7. La
versión del lema (`GAUGE=1`) se compara después. Energías medidas desde `E_0`:
- «radio P» es el radio máximo de Gershgorin de las columnas de P;
- «cota Q» es `min(Re diag − radio)` fuera de P;
- `θ_Q` es la razón de Neumann `ℓ¹` en los bordes de la banda.

| n | ε | Θ/π | radio P | cota Q | θ_Q | banda exacta | siguiente | salto de 3 sitios | otros saltos |
|---|---|---|---|---|---|---|---|---|---|
| 5 | −0,05 | 0 | 0,126 | 1,820 | 0,13 | [0,9954, 0,9987] | 1,9933 | 8,4·10⁻⁴ | — |
| 7 | −0,05 | 0 | 0,094 | 1,752 | 0,25 | [0,9926, 1,0020] | 1,9925 | 2,5·10⁻³ | ≤ 1,2·10⁻⁶ |
| 8 | −0,05 | 0 | 0,126 | 1,814 | 0,19 | [0,9958, 0,9992] | 1,9895 | 8,4·10⁻⁴ | ≤ 8,7·10⁻⁷ |
| 5 | −0,1 | 0 | 0,274 | 1,610 | 0,28 | [0,9803, 0,9946] | 1,9730 | 3,4·10⁻³ | — |
| 7 | −0,1 | 0 | 0,214 | 1,468 | 0,53 | [0,9711, 1,0080] | 1,9696 | 9,8·10⁻³ | ≤ 2,2·10⁻⁵ |
| 8 | −0,1 | 0 | 0,275 | 1,579 | 0,42 | [0,9834, 0,9968] | 1,9586 | 3,4·10⁻³ | ≤ 2,7·10⁻⁵ |
| 7 | +0,1 | 0 | 0,212 | 1,479 | 0,52 | [0,9711, 1,0080] | 1,9696 | 9,8·10⁻³ | ≤ 1,2·10⁻⁵ |
| 7 | −0,1 | 1/2 | 0,368 | 1,379 | 0,63 | [0,9716, 1,0100] | 1,9700 | 9,8·10⁻³ | ≤ 3,3·10⁻⁵ |
| 7 | −0,2 | 0 | 0,568 | 0,738 | 1,25 | [0,8942, 1,0320] | 1,8758 | 3,8·10⁻² | ≤ 4,1·10⁻⁴ |

- **Feshbach es exacto.** En todos los casos, `H_eff(λ) − λ` tiene un valor singular ≤ 3·10⁻¹⁴ en cada
  nivel de la banda. Los k anillos de defectos, uno por cada contenido, no se acoplan: ≤ 6·10⁻¹⁵,
  también con Θ ≠ 0, como pide (S_d).
- **Concuerda con 7.10.** El fondo de la banda reproduce el `Δ_def` de la tabla de 7.10: 0,9711 (n = 7)
  y 0,9834 (n = 8) con ε = −0,1, frente a 0,97108 y 0,98343. El salto efectivo depende solo de n mod 3,
  porque coincide entre n = 5 y 8. Es el del tipo de defecto del Teorema D, que no depende de n.
- **El salto dominante es de k = 3 sitios y va como ε².** Se multiplica por 3,9–4,0 de ε = 0,05 a
  0,1, como la anchura de banda de 7.10. Los demás son de orden ≥ 4.
- **La separación de Gershgorin se cumple con margen.** Con ε = −0,1, `θ_Q ≤ 0,63`, aunque está a unas
  80 veces el radio garantizado. Con ε = −0,2, la serie de Neumann en `ℓ¹` deja de converger
  (`θ_Q = 1,25`), pero la reducción, resuelta directamente, sigue siendo exacta, y la cota Q sigue
  siendo positiva.
- **Con el flujo en un solo enlace** (`GAUGE=1`, la construcción del Lema Z7), el espectro es el
  mismo y Feshbach sigue siendo exacto (≤ 1,5·10⁻¹⁴). Los márgenes son parecidos:

  | n | Θ/π | radio P | cota Q | θ_Q |
  |---|---|---|---|---|
  | 7 | 1/2 | 0,378 | 1,314 | 0,69 |
  | 5 | 1 | 0,500 | 1,536 | 0,40 |
  | 7 | 1 | 0,446 | 1,252 | 0,75 |

  Todo con ε = −0,1. Con giro uniforme, n = 5 y Θ = π dan radio 0,568, cota 1,562 y `θ_Q` 0,33.

**Notas de la revisión** (lo que cambió respecto de la primera versión de 7.12):
1. **El reloj global `∏X` no es simetría con Θ ≠ 0.** El giro usa la carga q. Hace falta (S_d), y
   el intercambio SU(k) la cumple por el contenido. Sin (S_d), un sector puede contener varios
   niveles, y el argumento solo controla sus sumas simétricas.
2. **El giro uniforme dejaba un desajuste en el enlace de cierre** de `H_0`, de tamaño `O(Θ/n)`, que
   el gauge del arco no quita. La cota `E_0 − n e_∞ = O(1)` no estaba justificada. Ahora el flujo va
   entero en un enlace, `H'_0` es exactamente invariante por traslación y la cota sale de Z4 y
   Schwarz. Las constantes cambian: `η_c` ya no es ≤ η, la condición es `5η + 2η_c < 1` y
   `M_3 = 538`.
3. **Radio.** ω solo está acotada en un disco menor que aquel en que converge. Schwarz se aplica en
   `|ε| < 1/(4M_3)`, con base `4M_3|ε|`, no `2M_3|ε|`.
4. **Densidad.** El barrido en Θ solo cubre el círculo si `|q_d| ≥ mcd(k,n)/k`. En general se usa la
   unión sobre n, cuyo paso tiende a 0.
5. **Cota inferior de R_n.** Solo vale para ε pequeño, con `ω^{(m_0)}` de mínimos no degenerados.
   La no degeneración de ω se hereda; no se supone.

El argumento de Z4 para `ε_b(n)` está ahora escrito como Lema Z4′. Sigue sin demostrarse la
localidad del salto de Feshbach, que el teorema no usa.

### 7.13 Carga entera: por qué falla el vestido de una fase y qué lo sustituye

El método de 7.11 y 7.12 es Gershgorin en la base de caracteres vestida por el punto fijo de una sola
fase. En el modelo de carga entera (`S⁺S⁻`, `q = (−1, 0, 1)`) no puede funcionar, ni siquiera con un G
exacto. El fallo no está en la contracción de Z1, sino en la base. Esta sección lo demuestra, lo mide
y deja escrito el sustituto: un vestido que depende de la fase local, al estilo de Pirogov–Sinai.

**El mecanismo.** Una isla `μ = c·1_I`, con c ≠ 0 en un intervalo I de longitud L y 0 fuera,
representa la configuración `c_0` con un dominio de la fase `c_c` en I. El vestido `e^{G}` es el de la
fase 0, así que dentro de I viste la fase c con las fluctuaciones de la fase 0. Para que eso sea
inocuo, la perturbación tiene que tratar igual las dos fases. Es justo la simetría de reloj. En
`S⁺S⁻`, cada fase tiene activos dos de cada tres enlaces: el par `(2, 0)` no admite `S⁺S⁻`. Las fases
distintas los tienen en posiciones distintas.

En el marco, el desplazamiento δ del enlace j, dentro de la isla, lleva el factor
`ω^{c(δ_j+δ_{j+1})} − 1`, que es nulo solo en los de tipo τ. Sea

    κ_c^{(j)}(ν) = Σ_δ v̂_j(δ, ν) (ω^{c(δ_j + δ_{j+1})} − 1) ,    K_c(j) = Σ_{ν_j ≠ 0 ≠ ν_{j+1}} |κ_c^{(j)}(ν)| .

Para `S⁺S⁻`, `K_c(j) ∈ {0, 4}` según j mod 3 y c, y suma 8 en cada periodo de tres enlaces. Además,
`κ_c^{(j)}(0) = 0`, porque `S⁺S⁻` no tiene diagonal. Toda la masa está en caracteres de dos sitios.
Para el intercambio SU(3), `K_c ≡ 0`.

**Lema Z8 (obstrucción).** Sea G cualquier vestido del marco con `‖g‖ ≤ d` en la norma de Z1. Sea
A la norma de coeficientes de v, que vale 7,11 para `S⁺S⁻`. La columna de `H̃ = e^{−G}H'e^{G}`
correspondiente a la isla `c·1_I` tiene radio de Gershgorin

    radio ≥ |ε| Σ_{j interior a I} K_c(j) − |ε| A (e^{2d} − 1)(L + 1) − C_∂ ,

donde `C_∂` solo depende de los dos bordes, no de L. Para `S⁺S⁻` el primer término es
`≈ (8/3)|ε| L`. Así que, si `(8/3 − A(e^{2d} − 1))|ε| L > 1 + C_∂`, el disco de esa columna corta el
del vacío.

Ninguna elección de G pequeña separa el multiplete por Gershgorin en esta base cuando
`n ≳ 1/|ε|`. La condición `8/3 − A(e^{2d} − 1) > 0` se cumple para `d < 0,159`, que es el régimen
de Z1.

*Demostración.* Por la fórmula de 7.11,
`(H̃ − E)χ_μ = χ_μ[|∂μ| + Σ_j Σ_δ C_δ(ω^{μ·δ} − 1) + Σ_c r_c χ_{c·1}]`. En un enlace interior j, los
términos de `−P_asc` son de tipo τ y se anulan. La parte «1» del vestido de v da
`ε Σ_ν κ_c^{(j)}(ν) χ_ν`.

Los caracteres de salida `μ + ν`, con ν de dos sitios no nulos en el enlace j, son distintos para
enlaces distintos. Ningún otro término de orden cero llega a ellos, porque los de los bordes salen
cerca de los bordes. Lo único que puede cancelarlos son las partes `e^{G_δ} − 1`, cuya norma total
es como mucho `|ε|A(e^{2d} − 1)` por enlace, más los términos de los bordes. Basta la desigualdad
triangular, restringida a esas salidas. ∎

**Comprobación numérica** (`scripts/isla_entero.py`). Se usa el punto fijo de Kirkwood–Thomas
numérico, que con ε = 0,1 todavía converge. Se dan diag (`H̃_{μμ} − E`) y radio para islas c = 1:

| modelo | n | ε | L = 1 | L = 3 | L = 5 | L = 7 | L = 9 | L = 11 |
|---|---|---|---|---|---|---|---|---|
| `S⁺S⁻` | 9 | 0,05 | 2,003 / 0,24 | 2,017 / 0,74 | 2,023 / 0,82 | 2,030 / 1,08 | — | — |
| `S⁺S⁻` | 9 | 0,1 | 2,012 / 0,61 | 2,068 / 1,71 | 2,093 / 1,90 | 2,120 / 2,45 | — | — |
| `S⁺S⁻` | 12 | 0,1 | 2,012 / 0,61 | 2,068 / 1,72 | 2,093 / 1,91 | 2,120 / 2,55 | 2,176 / 3,62 | 2,201 / 3,46 |
| SU(3) | 9 | 0,1 | 2,010 / 0,26 | 2,010 / 0,48 | 2,010 / 0,48 | 2,010 / 0,47 | — | — |

- **El radio crece linealmente en L**, con pendiente ≈ 0,27–0,3 con ε = 0,1 y la mitad con
  ε = 0,05. Es el `(8/3)|ε|` del lema. Con ε = 0,1, ya supera 1 con L = 3. En el intercambio SU(3)
  se satura en 0,48.
- **La diagonal también crece**, ≈ 0,009 por sitio con ε = 0,1, es decir, como ε². Es la energía
  de segundo orden de las fluctuaciones de la fase 0 impuestas a un dominio de la fase c. Esa
  energía no es física: por la traslación T, que lleva `c_r` a `c_{r+1}`, todas las fases tienen la
  misma densidad de energía. La masa fuera de la diagonal, de orden ε por sitio, es justo lo que la
  corrige en el espectro verdadero.
- Esto explica la tabla de 7.7: el rango de convergencia de la iteración se encoge con n, porque
  las islas largas aparecen cuando n crece.

**Qué lo sustituye: el vestido por fases (programa).** La simetría que relaciona las fases es la
traslación, que es exacta y vale con cualquier perturbación invariante. Con k | n, el vestido
correcto de la fase c es el de la fase 0 trasladado: `G_c = T^c G_0 T^{−c}`, es decir,
`g_c(μ) = g_0(T^{−c}μ)`. El programa, que es Pirogov–Sinai simétrico en una dimensión y a
temperatura cero, tiene tres pasos:

1. **Vacío con contornos cortos.** El punto fijo `g_0` se resuelve solo sobre los caracteres sin
   tramos constantes no nulos de longitud ≥ ℓ_0. En ese espacio, el factor `|sop μ|` de 7.7 está
   acotado por `ℓ_0` en los términos que lo producen. La contracción de Z1 funciona si
   `ℓ_0 |ε| A ≲ M̄^{−1}`·const. Basta ℓ_0 fijo, de orden uno, independiente de n.
2. **Base vestida por fases.** Para cada carácter μ, el anillo se parte en dominios, que son tramos
   constantes de longitud ≥ ℓ_0, y en contornos, que es el resto. El vector de base es
   `b_μ = χ_μ · exp(Σ_dominios G_{c}|_{dominio} + Σ_contornos G_∂)`, donde `G|_D` contiene solo
   caracteres con soporte en D y `G_∂` son correcciones de pared, resueltas también por punto fijo.
   Sobre una isla larga, `b_μ` lleva dentro las fluctuaciones de su propia fase. El término
   `K_c` del Lema Z8 desaparece, porque dentro de la isla la ecuación es la de la fase c, que es la
   trasladada de la de la fase 0.
3. **Gershgorin en la base `{b_μ}`.** El precio es que `{b_μ}` ya no es la imagen de una base de
   caracteres por una sola semejanza `e^{G}`. Es un cambio de base que depende del contorno y que
   es casi multiplicativo. Hay que acotar la matriz del cambio de base, de la forma identidad más
   algo local y pequeño, y su inversa. Es el análogo de los «pesos de contorno» de
   Borgs–Kotecký–Ueltschi: la diferencia entre `b_μ` y `e^{G_0}χ_μ` está localizada en los
   dominios de otras fases, y vale `exp(Σ_D (G_c − G_0)|_D)`. Sus coeficientes son grandes,
   `O(εL)` en exponente, pero el peso de una isla es `ε^{L/2}`.

   La cuestión es si el cociente `(peso de la isla) × e^{O(εL)}` se suma. Como
   `ε^{L/2} e^{CεL} = (ε e^{2Cε})^{L/2}`, se suma para ε pequeño. **Esa es la desigualdad clave**, y
   es la condición de Peierls de PS: el coste de una isla, `ε^{L/2}`, gana al error de vestirla
   mal, `e^{O(εL)}`. Cuantitativamente, con M el radio de Z1, hace falta
   `M|ε| e^{2C|ε|} < 1`.

**Qué daría.**
- (A) para el modelo de carga entera con k | n, y por tanto los Teoremas B y Z′ y la robustez
  exponencial, ahora con la escala del enrollamiento: `ε^{n}` frente a `ε^{2n/3}`, por el Teorema A.
- El análogo del Teorema D′ del lado k ∤ n para los defectos de 4.2. Habría que ver si la carga
  U(1) basta como (S_d), o si un sector tiene varios niveles y hay que tratarlo como un bloque
  (7.12, nota 1). Sin comprobar.

**Estado.**
- Demostrado: el Lema Z8, que es la obstrucción y explica por qué el método de 7.11 no se extiende,
  y el Lema Z9, el paso 1, abajo.
- Construido y comprobado numéricamente: el paso 2, abajo.
- Demostrado, condicional a los vacíos de fase (V): el Lema Z10, en el paso 2′ (paredes).
- Esquema escrito y comprobado numéricamente, con ε ≤ 0,1 y n ≤ 12: (V), mediante la inducción en
  escalas del paso 4. Falta la contabilidad de Pirogov–Sinai y la contracción en la norma con pesos.
- Demostrado: el cambio de base, en el paso 3 (Lema Z11, serie de Neumann con pesos `e^{σ|Y|}`).
  En la base por fases, Gershgorin separa el multiplete, comprobado con n = 6 y 9. El balance está
  al final de 7.13.

#### Paso 1: el vacío con contornos cortos (Lema Z9, demostrado)

Sea `𝒜_ℓ` el conjunto de caracteres cuyos tramos constantes no nulos, contados cíclicamente, miden
menos de ℓ. El **Kirkwood–Thomas restringido** resuelve `R̂(μ) = 0` solo para μ ∈ 𝒜_ℓ no constante.
Para los demás fija `g(μ) = 0`.

**Lema Z9.** Sea `a_nc` la norma de coeficientes de la parte de v que no tiene simetría de reloj,
la de los δ que no son de tipo τ. En la clase (C), con esta norma y las hipótesis de Z1, sea

    K_ℓ = K + (ℓ a_nc / M) (1 + 2d) e^{2d} ,

con K la constante de Z1. Si `K_ℓ ≤ 1/2`, el KT restringido es una contracción de constante 1/2 en
la bola `‖g‖ ≤ d`, uniformemente en n y en `|Θ| ≤ π`. El cuasi-vacío `ψ_ℓ = e^{G}` cumple
`(H' − E)ψ_ℓ = (Σ_{c≠0} r_c χ_{c·1} + Σ_{μ∉𝒜_ℓ} R̂(μ) χ_μ) ψ_ℓ`, con densidad del residuo

    sup_x Σ_{μ ∉ 𝒜_ℓ, μ ∋ x} |R̂(μ)| ≤ C d (M|ε|)^{ℓ/2} .

*Demostración.* La obstrucción de 7.7 está en los desplazamientos que no son de tipo τ. Actúan en
todo enlace que toca `sop μ_1`, y no solo en los del borde. Para `μ_1 ∈ 𝒜_ℓ`, cada tramo no nulo mide
< ℓ y tiene al menos una pared propia, así que `|sop μ_1| + |∂μ_1| ≤ ℓ |∂μ_1|`. Ese número de
anclajes lo absorbe el `|∂μ_1|` de la norma con un factor ℓ, que da el término añadido a K. El resto
es la prueba de Z1.

Para el residuo, un μ ∉ 𝒜_ℓ tiene un tramo de longitud ≥ ℓ, y por tanto `w(μ) ≥ ℓ/2`. Las salidas
de F están acotadas en la norma de Z1, con peso `(M|ε|)^{−w(μ)}`. ∎

Para `S⁺S⁻` (`a = 7,11`, `a_nc = 4,44`, `b = 22,3`), `M_*(ℓ) ≈ 1122 + 9,7 ℓ`:
1142 con ℓ = 2, 1278 con ℓ = 16 y 1730 con ℓ = 64. A ε fijo, el mejor ℓ es de orden `1/|ε|`, y la
densidad del residuo queda en `exp(−c/|ε|)`.


**Números del paso 1** (`scripts/vestido_fases.py` con ℓ₀ = ℓ, `S⁺S⁻`, Θ = 0). Densidad del residuo
excluido e iteraciones. n = 9 y n = 12 coinciden en tres cifras:

| ε | ℓ = 2 | 3 | 4 | 5 | 6 | sin restringir: iteraciones con n = 6 / 9 / 12 |
|---|---|---|---|---|---|---|
| 0,05 | 1,7·10⁻² (9 it.) | 1,0·10⁻² | 1,7·10⁻³ | 2,0·10⁻⁴ | 1,3·10⁻⁴ (13 it.) | 14 / 14 / 14 |
| 0,1 | 7,5·10⁻² (14 it.) | 4,6·10⁻² | 1,6·10⁻² | 3,9·10⁻³ | 2,6·10⁻³ (22 it.) | 26 / 26 / 26 |
| 0,2 | 3,8·10⁻¹ (25 it.) | 2,5·10⁻¹ (40 it.) | — | — | — | 102 / 537 / diverge |

- El número de iteraciones y la densidad del residuo no dependen de n.
- La densidad cae con ℓ en escalones de dos sitios, aproximadamente como ε^{ℓ/2}·const.
- Con ε = 0,2, el KT restringido converge en 25–40 iteraciones para todo n, mientras que el completo
  se frena con n y diverge con n = 12. La restricción quita justo lo que rompía la contracción. Con
  ℓ pequeño, eso sí, el residuo todavía es grande.
- La energía restringida converge a la completa: con ε = 0,05 y ℓ = 6, `E/n` coincide en 9 cifras.

**El paso 1 no basta solo.** Su residuo es local pero extensivo: la columna del vacío tiene masa
`n · e^{−c/|ε|}`. Para Gershgorin hace falta además que las islas largas, las que excluye 𝒜_ℓ, estén
en la base con su propio vestido. Eso es el paso 2.

#### Paso 2: la base vestida por fases (construcción y números)

Para cada fase c, sea `G_c` el punto fijo alrededor de `χ_{c·1}`, es decir, de
`H'_c = χ_{−c·1} H' χ_{c·1}`. Por la traslación (k | n), `g_c(μ) = g_0(T^{−c}μ)`, y todas las fases
tienen la misma E. Numéricamente coinciden en 10 cifras.

Para la isla `μ = c·1_I`, el vector de base es `b_μ = χ_μ e^{G_μ}`, con

    G_μ = Σ_ν [sop ν ⊆ I] g_c(ν) χ_ν + Σ_ν [sop ν ⊄ I] g_0(ν) χ_ν .

Para un carácter general se toma la misma regla por dominios. Los ν que cruzan una pared llevan el
vestido de la fase de fuera, sin corrección de pared.

**Números** (`scripts/vestido_fases.py`, `S⁺S⁻`, Θ = 0). Columna μ de `e^{−G}H'e^{G}`, radio con
`G = G_0` → con `G = G_μ`, islas c = 1:

| n | ε | L = 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| 9 | 0,05 | 0,24 → 0,24 | 0,74 → 0,56 | 0,82 → 0,41 | 1,08 → 0,36 | — | — |
| 9 | 0,1 | 0,61 → 0,61 | 1,71 → 1,39 | 1,90 → 1,08 | 2,45 → 0,92 | — | — |
| 12 | 0,1 | 0,61 → 0,61 | 1,72 → 1,39 | 1,91 → 1,09 | 2,55 → 1,03 | 3,62 → 1,43 | 3,46 → 0,68 |

- **Con el vestido por fases, el radio deja de crecer con L.** Oscila con L mod 3, que es el tipo de
  las dos paredes, entre 0,7 y 1,45 con ε = 0,1. Es proporcional a ε: con ε = 0,05 queda entre 0,25
  y 0,56. Lo que queda está anclado en las paredes. Es la forma `η|∂μ|` que Gershgorin tolera.
- **La diagonal deja de crecer:** 2,011–2,040 para todo L, frente a `2,012 + 0,009 L` con una sola
  fase. Desaparece la energía falsa ε²L.
- Las islas c = 2 se comportan igual: radio por fases entre 0,72 y 1,09 con ε = 0,1 y n = 12.

#### Paso 2′: las paredes (Lema Z10 y el vestido de pared)

**Hipótesis (V) (vacíos de fase).** Para cada fase c hay un vestido `G_c` con la propiedad de
Kirkwood–Thomas *completa*: `R̂_c(ν) = 0` para todo ν no constante, con `‖g_c‖ ≤ d` en la norma de Z1.
- En la clase con simetría de reloj, (V) es el Lema Z1, con `G_c = G_0`.
- En carga entera, (V) es justo lo que falta. Sus islas largas son el objeto de la inducción en
  escalas. El Lema Z9 da la versión restringida.
- Por la traslación, basta construir `G_0`.

**Base por fases, general.** Para cualquier μ, sus **dominios** son los tramos constantes maximales.
Sea

    G_μ = Σ_D Σ_{sop ν ⊆ D} g_{c(D)}(ν) χ_ν + Σ_{ν que cruza una pared} h_μ(ν) χ_ν ,

donde `h_μ` es el **vestido de pared**. Es el punto fijo de `R̂_μ(ν) = 0` en los ν a distancia ≤ R
de una pared con `Δ(ν) = |∂(μ+ν)| − |∂μ| ≥ 1`, dividiendo por Δ. En los demás que cruzan una pared,
`h_μ(ν) = g_0(ν)`. Los ν con `Δ ≤ 0` *mueven* una pared. No se pueden quitar con un vestido, porque
son la dinámica de las paredes, de orden ε.

**Lema Z10 (columna por fases; condicional a (V)).** En las hipótesis de Z1, con (V) y
`‖h_μ‖ ≤ d`, se tiene

    (e^{−G_μ} H' e^{G_μ} − E) χ_μ = χ_μ [ |∂μ| + Σ_{p ∈ ∂μ} F_p + ρ_μ ] ,

donde:
- cada `F_p` tiene coeficientes de Fourier anclados en la pared p, con
  `Σ_{ν: dist(ν,p) = r} |F̂_p(ν)| ≤ C d (M|ε|)^{r}`, y por tanto `‖F_p‖_1 ≤ η_∂ = O(d + |ε|)`;
- `‖ρ_μ‖_1` está acotado por los residuos constantes y por el giro, `O(b|ε| + (M|ε|)^{n/2})`.

Así, el radio de Gershgorin de la columna μ es `≤ η_∂ |∂μ|`, *sin dependencia del tamaño de los
dominios*. Con `η_∂ < 1/2`, el argumento de Z6 da (A1), (A2E) y (G) en la base `{b_μ}`, salvo el
cambio de base (paso 3).

*Demostración.* Sea ν una salida. Su coeficiente en `R̂_μ` es una suma de monomios, cada uno un
racimo conexo que toca el enlace j que lo produce, como en la prueba de Z4′. Hay dos casos:
- **El racimo está dentro de un dominio D.** Entonces el enlace j y todos sus caracteres están en
  D, y el monomio es uno de los de la ecuación de la fase `c(D)`. Como `R̂_{c(D)}(ν) = 0`, la suma de
  estos monomios es menos la suma de los monomios de la fase `c(D)` que salen de D.
- **El racimo sale de D.** Lo mismo vale para los monomios de `G_μ` con caracteres de pared o de
  otro dominio.

En ambos casos queda un racimo que conecta ν con una pared, de peso ≥ dist(ν, pared). La norma de Z1
lo acota por `(M|ε|)^{dist}`. Sumando sobre distancias y paredes se obtiene la cota. Los términos de
`−P_asc` dentro de un dominio son de tipo τ, o equivalentemente son los de la fase, y no dan nada
nuevo. El término de las islas de 7.13, `K_c`, desaparece porque dentro de D la ecuación usada es la
de la fase `c(D)`. ∎

**Números del vestido de pared** (`scripts/vestido_fases.py`, `S⁺S⁻`, Θ = 0). Radio de la columna
de la isla `c·1_I`: por fases → con pared, entre corchetes la parte que mueve la pared, y la
diagonal con pared:

| caso | L = 1 | 3 | 4 | 5 | 6 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| ε = 0,05, n = 9, R = 2, c = 1 | 0,24 → 0,16 [0,13] | 0,56 → 0,16 [0,10] | 0,38 → 0,09 [0,05] | 0,41 → 0,09 [0,05] | 0,56 → 0,16 [0,10] | — | — |
| ε = 0,1, n = 12, R = 2, c = 1 | 0,61 → 0,54 [0,32] | 1,39 → 1,05 [0,38] | 1,01 → 0,53 [0,21] | 1,09 → 0,68 [0,12] | 1,45 → 1,15 [0,22] | 1,43 → 1,10 [0,38] | 0,68 → 0,59 [0,33] |
| ε = 0,1, n = 12, R = 2, c = 2 | 0,61 → 0,54 [0,32] | 1,04 → 0,38 [0,13] | 1,01 → 0,53 [0,21] | 0,95 → 0,41 [0,01] | 1,09 → 0,44 [0,01] | 1,07 → 0,35 [0,08] | 0,72 → 0,48 [0,25] |
| ε = 0,1, n = 9, R = 3, c = 1 | 0,61 → 0,44 [0,33] | diverge | 1,00 → 0,23 [0,23] | 1,08 → 0,24 [0,24] | diverge | — | — |

- **Con ε = 0,05, la pared deja casi solo la parte que la mueve.** El radio con pared es ≤ 0,17
  sobre una diagonal ≥ 1,96, es decir, `η_∂ ≈ 0,04` por pared. Gershgorin se cumple con mucho margen.
- **Con ε = 0,1 y R = 2,** el peor caso (c = 1 y L ≡ 0 mod 3) da 1,15 sobre una diagonal de 1,85,
  justo por encima del umbral 1/2 por pared. Con R = 3, los demás tipos de pared bajan a lo que mueve
  la pared (0,23–0,24), pero ese tipo diverge. Ese par de paredes tiene un denominador casi resonante:
  su energía baja ≈ 0,15 con ε = 0,1, y el camino a la configuración con Δ = 1 es casi degenerado.
  Habría que tratarlo como bloque.
- **La diagonal con pared baja como ε²,** 0,02 por pared con ε = 0,05 y 0,075 con ε = 0,1. Es la
  energía física de la pared, `1 − O(ε²)`, que el vestido de una fase no veía.
- n = 9 y n = 12 dan lo mismo en los L comunes. Nada depende de n ni del tamaño del dominio.

**Qué queda.** Con Z10, el problema de carga entera se reduce a dos cosas:
1. **(V):** el vacío completo de una fase, con sus islas largas vestidas a su vez por fases. Es la
   inducción en escalas. Z10 es exactamente su paso inductivo: en la escala m, las islas se visten
   con los vacíos de la escala m − 1, y Z10 controla su columna con constantes que no dependen de m.
2. **El cambio de base `{b_μ}` (paso 3).**

#### Paso 4: la inducción en escalas para (V)

**Por qué hace falta.** El Lema Z9 resuelve el vacío restringido a 𝒜_ℓ con constante de contracción
`K + ℓ a_nc/M`. Esa constante crece con ℓ, así que no se puede tomar ℓ = n. El Lema Z10 dice qué
pasa en las variables adecuadas: la columna de una isla vestida por fases tiene radio `≤ η_∂|∂μ|`,
sin dependencia de L. La inducción usa Z10 para resolver, escala a escala, el sector de las islas
largas, que es donde Jacobi deja de contraer.

**Escalas.** Sea `ℓ_m = 2^m ℓ_0`, con ℓ_0 el de Z9, y sea `𝒜_m = 𝒜_{ℓ_m}`. La afirmación inductiva:

> **(H_m)** Hay `g^{(m)}` soportado en 𝒜_m, con `R̂(μ) = 0` para todo μ ∈ 𝒜_m no constante, tal que
> `‖g^{(m)}‖_{M_m} ≤ d_m` y la densidad del residuo en `∁𝒜_m` es `δ_m ≤ d_m (M_m|ε|)^{ℓ_m/2}`.

(H_0) es el Lema Z9.

**Paso inductivo (esquema).** Sea `𝒮_m = 𝒜_{m+1} ∖ 𝒜_m`, el conjunto de caracteres cuyo mayor tramo
no nulo mide entre ℓ_m y ℓ_{m+1}.
1. **Cambio de variables por fases.** En 𝒮_m, el coeficiente no se busca directamente. Se escribe
   la parte de `e^{G}` con islas de 𝒮_m como un gas de polímeros. Cada polímero γ es una isla
   `(I, c)` con `ℓ_m ≤ |I| < ℓ_{m+1}`, y su factor es

       φ_γ = a_γ · χ_{c·1_I} · exp((G_c^{(m)} − G_0^{(m)})|_I + h_γ) ,

   donde `G_c^{(m)} = T^c G_0^{(m)} T^{−c}`, `h_γ` es el vestido de pared de Z10 y `a_γ` la amplitud.
   Los polímeros de una misma familia están separados. Las islas anidadas, de escalas ≤ m, ya están
   en `G_c^{(m)}`. Así, `g^{(m+1)} = g^{(m)} + (Mayer del gas)` más correcciones de pared.
2. **Ecuación para `a_γ` y `h_γ`.** La condición `R̂(μ) = 0` en 𝒮_m, escrita en estas variables, es
   la columna de Z10, y su diagonal es `|∂μ| + O(η_∂|∂μ|)`. En el interior de I, `G_c^{(m)}` cumple
   la ecuación de su fase salvo su residuo `δ_m` por sitio. Así, la ecuación es una contracción con
   constante `≤ 2η_∂ + C ℓ_{m+1} δ_m`, independiente de m mientras `ℓ_{m+1}δ_m` sea pequeño.
3. **Peierls.** Por localidad, `|a_γ| ≤ (M_m|ε|)^{|I|/2}`: hacen falta ≥ |I|/2 enlaces para crear
   la isla. Además, `|exp((G_c − G_0)|_I)| ≤ e^{2d_m |I|}`. La actividad del polímero es entonces
   `≤ (M_m|ε| e^{4d_m})^{|I|/2}`, sumable si `M_{m+1}|ε| < 1` con `M_{m+1} = M_m e^{4d_m}`. Mayer
   converge, y las nuevas entradas de g tienen el peso de Z1 con `M_{m+1}`.
4. **Constantes.**
   - `d_{m+1} ≤ d_m + C(M_m|ε|e^{4d_m})^{ℓ_m/2}`;
   - `M_{m+1} = M_m e^{4d_m}`;
   - `δ_{m+1} ≤ d_{m+1}(M_{m+1}|ε|)^{ℓ_{m+1}/2}`.

   Como `ℓ_m` crece geométricamente, `Σ_m (M_m|ε|)^{ℓ_m/2}` converge de forma doblemente exponencial.
   `d_m → d_∞ ≤ 2d_0`, `M_m → M_∞ ≤ M_0 e^{8d_0}` y `ℓ_{m+1}δ_m → 0`. En la escala `ℓ_m ≥ n`, `𝒜_m`
   contiene todo, y (H_m) es (V), con constantes que no dependen de n.

**Lo que no está escrito con detalle.**
- La contabilidad combinatoria del paso 1: el gas de islas con islas anidadas, la exclusión entre
  polímeros de la misma escala y la compatibilidad con las paredes. Es la de Pirogov–Sinai en una
  dimensión y con fases simétricas, sin metaestabilidad ni energías libres truncadas, pero no la he
  transcrito.
- La prueba de que la ecuación del paso 2 contrae *en la norma con pesos*, y no solo columna a
  columna. Hay que repetir Z1 con los anclajes de Z10.
- El caso de pared casi resonante, visto con ε = 0,1 (c = 1, L ≡ 0 mod 3). Para ε pequeño no
  aparece, pero fija el radio real.

**Comprobación numérica de (V)** (`scripts/vacio_islas.py`). Kirkwood–Thomas completo, resuelto con
Jacobi acelerado por Anderson. Sea `I_L` el máximo de |g(μ)| sobre los μ cuyo mayor tramo no nulo mide
L, y «norma» `sup_x Σ_{μ∋x} |g(μ)||∂μ|`:

| ε | n | iteraciones | norma | I_3 | I_6 | I_9 | caída por 3 sitios |
|---|---|---|---|---|---|---|---|
| 0,05 | 9 / 12 | 14 / 14 | 0,2351 / 0,2351 | 2,2·10⁻³ | 1,5·10⁻⁵ | 1,6·10⁻⁷ | 147, 94 |
| 0,1 | 9 / 12 | 18 / 18 | 0,6084 / 0,6111 | 9,2·10⁻³ | 2,6·10⁻⁴ | 1,1·10⁻⁵ | 35, 24 |
| 0,2 | 6 / 9 / 12 | 23 / 29 / 29 | 2,57 / 3,73 / 5,33 | 4,2·10⁻² | 5,4·10⁻³ | 1,2·10⁻³ | 7,8, 4,5 |
| 0,3 | 12 | diverge | — | — | — | — | — |

- **Con ε ≤ 0,1, (V) se cumple con constantes uniformes en n.** La norma es la misma con n = 9 y 12,
  hasta la tercera cifra, y los coeficientes de isla caen exponencialmente en L. Con ε = 0,2 y n = 12,
  Jacobi puro diverge, pero Anderson encuentra el punto fijo. Esto confirma que el problema era del
  método y no de la existencia.
- **Con ε = 0,2, la norma crece con n**, porque la caída por cada tres sitios, 7,8 y luego 4,5, ya no
  compensa la entropía de las islas decoradas. Estamos fuera del régimen de Peierls. Con ε = 0,3
  diverge incluso Anderson. Es coherente con el radio de Rayleigh–Schrödinger ≈ 0,31 de 7.6.
- **La caída por cada tres sitios baja un 35 % entre L = 3→6 y 6→9, igual con los tres ε.** Con
  n ≤ 12 no se distingue si es entropía de las decoraciones o el factor `e^{O(εL)}` de vestir mal el
  interior. El paso 3 de la inducción cuenta con los dos.

#### Paso 3: el cambio de base (Lema Z11, demostrado)

Gershgorin necesita la matriz de H en la base `{b_μ}`, `M = B^{−1} H' B`. Los Lemas Z10 y el vestido
de pared solo dan cada columna en su propio sistema de referencia, el de `e^{G_μ}χ`. Hay que pasar de
ahí a la base común.

**Regla local.** Para un carácter ν, sea `hull ν` el menor arco que contiene `sop ν`. El vestido por
fases asigna:
- `g^{(μ)}(ν) = g_c(ν)` si μ es constante, igual a c, en `hull ν`;
- `g_0(ν)`, o el vestido de pared `h_μ(ν)`, en otro caso.

La asignación depende solo de μ en `hull ν + R`, donde R es el alcance del vestido de pared; R = 0 si
no se usa. Sea `x = M|ε|`. Para σ > 0 con `e^{2σ}x < 1`, sea

    A_σ(x) = Σ_{w≥1} (4w + 3) e^{σ(2w+1)} x^w ,    δ_σ = 2k(2R + 1) d A_σ(x) .

**Lema Z11 (cambio de base).** Suponemos (V), con `‖g_c‖ ≤ d` para cada c en la norma de Z1, y la
misma cota para los `h_μ`. Sean β, γ caracteres e `Y = {y : β_y ≠ γ_y}`. Entonces `{b_μ}` es una
base, y las coordenadas `c(β,γ)` de `χ_γ Φ_β` en ella cumplen

    ‖c(β,γ)‖_1 ≤ e^{σ|Y|} ,    ‖c(β,γ) − e_γ‖_1 ≤ e^{δ_σ|Y|} − 1 ,

siempre que `δ_σ ≤ σ`.

*Demostración.*
1. **Localidad de la diferencia.** `L_{βγ} := G_β − G_γ = Σ_ν (g^{(β)}(ν) − g^{(γ)}(ν)) χ_ν`. Un
   coeficiente es no nulo solo si β y γ difieren en `hull ν + R`, es decir, si `hull ν + R` corta a Y.
   Cada sitio de `hull ν` está a distancia ≤ `|hull ν|` de un punto de `sop ν`, y
   `|hull ν| ≤ 2w(ν) + 1`, porque el recubrimiento de Z1 salta como mucho un enlace. Con la norma de
   Z1, `Σ_{ν∋y, w(ν)=w} |g(ν)| ≤ d x^w`, y se obtiene

       Σ_ν |L_{βγ}(ν)| e^{σ|hull ν|} ≤ Σ_{y∈Y} Σ_{|z−y|≤R} 2k Σ_{ν: z ∈ hull ν} |g(ν)| e^{σ|hull ν|} ≤ δ_σ |Y| .

   El factor 2k cubre `|g^{(β)}| + |g^{(γ)}|`, con cualquiera de las k fases.
2. **Un paso de la serie.**
   `χ_γΦ_β = χ_γΦ_γ e^{L_{βγ}} = b_γ + Σ_λ ê_{βγ}(λ) χ_{γ+λ} Φ_γ`, con
   `e^{L_{βγ}} − 1 = Σ_λ ê_{βγ}(λ) χ_λ`. Cada término nuevo es del mismo tipo, con el par
   `(γ, γ+λ)` y región de cambio `Y' = sop λ`. Si `λ = ν_1 + ··· + ν_p` sale de un producto de p
   factores de L, entonces `sop λ ⊆ ∪ hull ν_i` y `e^{σ|Y'|} ≤ ∏ e^{σ|hull ν_i|}`. Por tanto

       Σ_λ |ê_{βγ}(λ)| e^{σ|sop λ|} ≤ Σ_{p≥1} (1/p!) (Σ_ν |L(ν)| e^{σ|hull ν|})^p ≤ e^{δ_σ|Y|} − 1 .

3. **Inducción.** Sea `T_N(β,γ)` la suma de los módulos de los pesos de los caminos de la serie de
   profundidad ≤ N, que es una cota de las coordenadas truncadas. `T_0 = 1` y
   `T_{N+1}(β,γ) ≤ 1 + Σ_λ |ê_{βγ}(λ)| T_N(γ, γ+λ)`. Si `T_N ≤ e^{σ|Y|}` para todo par, entonces
   `T_{N+1} ≤ 1 + (e^{δ_σ|Y|} − 1) ≤ e^{σ|Y|}`, porque `δ_σ ≤ σ`. Así, `T_N ≤ e^{σ|Y|}` para todo N.
   Los `T_N` crecen con N y están acotados, así que la serie converge absolutamente. El resto de
   profundidad N es una suma de vectores `χΦ` de norma acotada en dimensión finita, con pesos que
   tienden a 0. Por tanto `χ_γΦ_β = Σ_α c(β,γ)_α b_α`, con las dos cotas del enunciado; la segunda
   sale del paso 2.
4. **Base.** Con β = 0, todo `χ_γ e^{G_0}` está en el generado de `{b_μ}`. Como `e^{G_0}` es
   invertible, ese generado es todo el espacio, y hay `k^n` vectores, así que `{b_μ}` es una base. ∎

**Corolario (columnas en la base por fases).** Si la columna de Z10 en su propio sistema de
referencia es `H'b_μ = b_μ[D_μ + Σ_ν f_μ(ν)χ_ν]`, con
`Σ_{ν anclado en p} |f_μ(ν)| e^{σ|sop ν|} ≤ η_∂(σ)` para cada pared p, entonces

    |M_{μμ} − D_μ| + Σ_{μ'≠μ} |M_{μ'μ}| ≤ Σ_ν |f_μ(ν)| e^{σ|sop ν|} ≤ η_∂(σ) |∂μ| .

En efecto, `H'b_μ = D_μ b_μ + Σ_ν f_μ(ν) χ_{μ+ν}Φ_μ`, y por el lema, con `Y = sop ν`, cada término
tiene coordenadas de norma `≤ e^{σ|sop ν|}`. La cota pesada `η_∂(σ)` existe si `e^{2σ}x < 1`, por el
decaimiento `x^{dist}` de Z10. Así, el Gershgorin de Z6 se aplica a M tal cual, con η sustituido por
`η_∂(σ)`.

*Comprobación numérica* (`scripts/coord_fases.py`, n = 6, `S⁺S⁻`, 4000 pares al azar). Máximo de
`‖c(β,γ) − e_γ‖_1` según `|Y|`, y la tasa efectiva `max log(1 + ‖c − e_γ‖_1)/|Y|`:

| ε | \|Y\| = 1 | 2 | 3 | 4 | 5 | 6 | tasa efectiva máx. |
|---|---|---|---|---|---|---|---|
| 0,05 | 0,15 | 0,25 | 0,39 | 0,40 | 0,39 | 0,31 | 0,14 |
| 0,1 | 0,36 | 0,59 | 0,80 | 0,99 | 1,01 | 0,84 | 0,31 |
| 0,2 | 1,12 | 3,05 | 3,05 | 3,05 | 3,06 | 3,39 | 0,75 |

- La cota `e^{δ|Y|} − 1` se cumple. La tasa efectiva crece linealmente con ε, como `δ_σ ∝ d`.
- La norma incluso se satura en `|Y|`: el crecimiento exponencial de la cota es pesimista. Cambiar
  la estructura de dominios en Y cuesta lo que cuestan sus bordes, no su volumen.

**Números** (`scripts/base_fases.py`). Matriz `M = B^{−1}H'B` en la base por fases, sin vestido de
pared, frente a la base de una sola fase, `S⁺S⁻`. Con n = 6 se calculan todas las columnas, en doble
precisión. Con n = 9 se factoriza B en precisión simple y se resuelven 319 columnas: las 3
constantes, todas las islas y 300 al azar.

| n | ε | base | η = máx radio/\|∂μ\| | cota del hueco | hueco exacto | radio de las constantes | cond(B) |
|---|---|---|---|---|---|---|---|
| 6 | 0,05 | una fase | 0,436 | 1,151 | 1,795 | 0,86 | 1,3 |
| 6 | 0,05 | por fases | 0,325 | 1,365 | 1,795 | 2·10⁻⁴ | 1,4 |
| 9 | 0,05 | una fase | 0,622 | 0,799 | 1,814 | 1,4 | — |
| 9 | 0,05 | por fases | 0,296 | 1,418 | 1,814 | 0,15\* | — |
| 6 | 0,1 | una fase | 0,977 | 0,145 | 1,530 | 1,9 | 1,8 |
| 6 | 0,1 | por fases | 0,802 | 0,459 | 1,530 | 5·10⁻³ | 2,0 |
| 9 | 0,1 | una fase | 1,358 | −0,543 | 1,591 | 3,0 | — |
| 9 | 0,1 | por fases | 0,717 | 0,608 | 1,591 | 0,14\* | — |
| 6 | 0,2 | ambas | 2,8–2,9 | < 0 | 0,940 | — | 3,7–4,6 |

\* Suelo de ruido de la precisión simple: el radio de las constantes es el mismo con ε = 0,05 y 0,1, y
con n = 6 en doble precisión es ≤ 5·10⁻³.

- **Con una sola fase, el método empeora con n,** como dice el Lema Z8: η pasa de 0,44 a 0,62 con
  ε = 0,05, y con ε = 0,1 la cota del hueco pasa a ser negativa con n = 9. Los vacíos de las fases
  c ≠ 0, vestidos con la fase 0, tienen radio 0,9–3. Son la isla del Lema Z8 cuando ocupa todo el
  anillo.
- **En la base por fases, η no crece de n = 6 a 9,** incluso baja, y la cota del hueco es positiva
  y estable: 1,37 → 1,42 con ε = 0,05 y 0,46 → 0,61 con ε = 0,1.
- **B está bien condicionada** (cond ≤ 2 con ε ≤ 0,1, n = 6). El cambio de base cuesta un factor de
  orden 1, como dice el Lema Z11.
- Con ε = 0,2 no funciona ninguna de las dos bases. Es coherente con el paso 4: ahí ya no estamos en
  el régimen de Peierls.

#### Paso 5: qué contracción hace falta (y cuál no puede existir)

**El paso 2′ no necesita contracción.** La cota de Z10 es directa. Los coeficientes de `G_μ` dentro de
los dominios son los de los vacíos de fase, y los de pared, aunque sean `g_0`, son datos fijos, no
incógnitas. El residuo de la columna se acota sumando monomios con los pesos de Z1. Los factores
polinómicos de anclaje, como `|sop ν| ≤ 2w + 1`, los absorbe el decaimiento `x^{w}` con holgura,
porque no hay ninguna condición de autoconsistencia que iguale el peso de entrada y el de salida. El
punto fijo de pared del paso 2′ es opcional: solo mejora las constantes. **Queda fuera de la lista de
lo que falta.**

**El paso 4 no puede hacerse con la iteración de Kirkwood–Thomas, con ninguna norma.** Sea J la
linealización de la iteración de Jacobi `g ↦ g − R̂(g)/|∂μ|` en el punto fijo, restringida a los
caracteres no constantes. Si su radio espectral `ρ(J)` es ≥ 1, la iteración no contrae en ninguna
norma, aunque el punto fijo exista.

**Proposición (crecimiento de ρ(J); numérica, con una cota heurística).** En el modelo de carga
entera, `ρ(J)` crece con n. Hay una cota inferior heurística `ρ(J) ≳ ε κ √(L/8)` con `L ~ n`. Sale del
vector de prueba de dos niveles: la isla de longitud L (`|∂| = 2`) acoplada a sus L decoraciones
(`|∂| = 4`) por el término `κ_c` del Lema Z8. El bloque `[[0, aᵀ], [b, 0]]` tiene autovalores
`±√(L a b)`. No está demostrada, porque faltaría descartar cancelaciones con el resto de J.

Numéricamente (`scripts/radio_jacobi.py`, Arnoldi sobre J por diferencias finitas):

| modelo | ε | n = 6 | n = 9 | n = 12 | crecimiento por sitio |
|---|---|---|---|---|---|
| `S⁺S⁻` | 0,05 | 0,196 | 0,216 | 0,241 | 0,0075 |
| `S⁺S⁻` | 0,1 | 0,399 | 0,454 | 0,506 | 0,018 |
| `S⁺S⁻` | 0,15 | 0,622 | 0,717 | 0,810 | 0,031 |
| `S⁺S⁻` | 0,2 | 0,862 | 1,006 | 1,149 | 0,048 |
| SU(3) | 0,2 | — | 0,251 | 0,253 | ≈ 0 |

- En carga entera, ρ crece **linealmente en n**, más deprisa que la cota √L. Los autovalores
  dominantes son reales y negativos, lo que indica una iteración que oscila entre la isla y sus
  decoraciones.
- Extrapolando, ρ = 1 en n ≈ 40 con ε = 0,1 y en n ≈ 110 con ε = 0,05. **Para cada ε fijo, Jacobi
  diverge en anillos suficientemente grandes**, aunque Anderson siga encontrando el punto fijo. Con
  ε = 0,2 eso ya pasa en n = 9, que es la divergencia vista en 7.7.
- En el intercambio SU(3), ρ no depende de n. Es la contracción uniforme del Lema Z1.

**Consecuencia: la contracción tiene que hacerse en las variables por fases.** La única
linealización con radio uniforme es la de las columnas vestidas por fases. Por Z10 y Z11, su parte
fuera de la diagonal es `≤ η_∂(σ)|∂μ|` frente a la diagonal `|∂μ|`. Es decir:

> **Enunciado que falta (Lema Z12, contracción por fases).** Sea 𝔽 la iteración de Jacobi escrita
> en las variables del gas de polímeros del paso 4: los coeficientes cortos de `g_0`, las amplitudes
> de isla `a_γ` y las correcciones de pared, con los interiores vestidos por `G_c = T^c G_0 T^{−c}`.
> Entonces 𝔽 es una contracción de constante `≤ 2η_∂(σ) + O(x^{ℓ_0/2})` en la norma de Z1 con pesos
> `e^{σ|hull|}`, uniformemente en n.

La razón por la que debería valer es la misma que en Z10. En estas variables, la ecuación de una isla
es la de su propia fase, que ya está resuelta por traslación, más términos anclados en las paredes. El
término `κ_c` del Lema Z8, que es el que da el crecimiento de ρ(J), no aparece, porque se ha
absorbido en `G_c − G_0`. Escribir 𝔽 con precisión es escribir la contabilidad del gas de polímeros
del paso 4. **Las dos piezas que faltaban son, en realidad, una sola:** el paso de Pirogov–Sinai en
variables de polímero, con Z12 como su estimación de contracción.

#### Paso 6: el paso de Pirogov–Sinai en variables de contorno (Lema Z12)

**Primero, una corrección: los polímeros son contornos, no islas.** El paso 4 tomaba como polímero
una isla `(I, c)` con su interior vestido por `G_c`. Al orden más bajo de Mayer, eso predice
`g_0(c·1_I + ν)/g_0(c·1_I) = (g_c − g_0)(ν)` para las decoraciones ν del interior.
`scripts/polimero_isla.py` lo compara en el vacío exacto, con n = 12, L = 9 y decoraciones de dos
sitios en el centro:
- con c = 1, signo y orden de magnitud coinciden en todas las decoraciones, aunque falta un factor
  ≈ 2: 0,061 frente a 0,034 y −0,027 frente a −0,033 con ε = 0,05;
- con c = 2 falla en una entrada por un factor 10: −0,34 frente a −0,034.

La razón es que el carácter `c·1_I + ν` admite otras descomposiciones más baratas. Con c = 2 y
ν = (1, 2), la configuración es la de dos islas más cortas y cercanas, separadas por dos sitios, y su
amplitud `ε^{2}·ε^{1,5}` gana a la de la isla larga decorada, `ε^{4,5}·ε`.

Por tanto, **la unidad correcta es el contorno**: un grupo de paredes a distancia < ℓ_0 entre sí, con
su configuración interna arbitraria. Es la de Pirogov–Sinai. Las islas largas son dos contornos (sus
dos paredes) separados por un dominio, no un polímero.

**El modelo de contornos.** Para un carácter μ (la configuración `c_0 + μ`) y ℓ_0 fijo:
- las **paredes** son los enlaces con `μ_b ≠ μ_{b+1}`;
- los **contornos** son las componentes conexas de la relación «paredes a distancia < ℓ_0». Cada
  contorno Γ tiene soporte `S_Γ`, el arco de sus paredes ensanchado en ℓ_0/2 a cada lado, y
  configuración `μ|_{S_Γ}`;
- fuera de los contornos, μ es constante en cada **dominio**, y su valor es la fase del dominio. Los
  valores de frontera de Γ fijan las fases de los dominios contiguos.

**Variables.** Con k | n, los vacíos de fase son `G_c = T^c G_0 T^{−c}`. Una función de onda del
marco de la forma

    ψ = exp( G^{s} + Σ_{Γ} ρ(Γ) χ_{μ_Γ} · ∏_{dominios D contiguos} e^{(G_{c(D)} − G^{s})|_D} )   (en el sentido de Mayer)

queda determinada por los coeficientes cortos `g^s`, soportados en 𝒜_{ℓ_0}, y por las
**actividades de contorno** ρ(Γ), una por contorno con su configuración. Aquí `μ_Γ` es el carácter
del contorno con sus dominios, y «en el sentido de Mayer» quiere decir que `G_0` es el logaritmo del
gas de contornos, calculado por el desarrollo de clusters. El vestido de los dominios se toma de los
`G_c`, que a su vez son trasladados de `G_0`. El sistema es, por tanto, autoconsistente.

**La iteración 𝔽.**
- En los caracteres cortos, μ ∈ 𝒜_{ℓ_0}: `g^s(μ) ← g^s(μ) − R̂(μ)/|∂μ|`, el KT restringido de Z9.
- En cada contorno Γ: `ρ(Γ) ← ρ(Γ) − R̂(μ_Γ)/|∂μ_Γ|`. Es la ecuación de KT en el carácter del
  contorno, con denominador su número de paredes.

Las ecuaciones de los caracteres `μ_Γ + ν`, con ν en el interior de un dominio, no se imponen por
separado. Por traslación, son las ecuaciones de la fase del dominio, que el mismo sistema resuelve.

**Lema Z12 (contracción en variables de contorno).** Sean las hipótesis de Z1 con ℓ_0 fijo, y sea
la norma

    ‖(g^s, ρ)‖ = max( ‖g^s‖_{Z1}, sup_x Σ_{Γ: S_Γ ∋ x} |ρ(Γ)| |∂Γ| e^{σ|S_Γ|} x^{−w(Γ)} ) ,

con `w(Γ)` el recubrimiento de Z1 de las paredes y de los sitios no nulos del contorno. Si
`K_{ℓ_0} ≤ 1/4`, `2η_∂(σ) ≤ 1/4` y `δ_σ ≤ σ`, que son las condiciones de Z9, Z10 y Z11, y si se
cumple la condición de Kotecký–Preiss

    sup_x Σ_{Γ ∋ x} |ρ(Γ)| e^{(σ + a)|S_Γ|} ≤ a    (por ejemplo, a = σ) ,

entonces 𝔽 lleva la bola `‖·‖ ≤ d` en sí misma y es una contracción de constante 1/2,
uniformemente en n. Su punto fijo da (V) con `‖g_0‖_{Z1} ≤ C d`.

*Demostración (esquema por piezas).*
1. **Parte corta.** Es el Lema Z9. La constante es `K_{ℓ_0}`, porque en 𝒜_{ℓ_0} los anclajes por el
   soporte están acotados por `ℓ_0|∂μ|`.
2. **Parte de contornos: el término diagonal.** En `R̂(μ_Γ)`, el término lineal en ρ(Γ) es
   `|∂μ_Γ| ρ(Γ)`, de las paredes de `−P_asc`. Los términos no de tipo τ dentro de los dominios, el
   `κ_c` del Lema Z8, no aparecen. El vestido de cada dominio es el de su fase, y la ecuación de esa
   fase, que se cumple en el punto fijo y en la linealización, por traslación, cancela todo lo que no
   llega a una pared. Este es exactamente el argumento de Z10.
3. **Parte de contornos: lo que no es diagonal.** Hay tres fuentes:
   - los monomios anclados en las paredes de Γ, con cota `η_∂(σ)|∂Γ|`;
   - el cambio de base de Z11, que pasa del sistema de referencia de Γ al de los contornos vecinos y
     cuesta `e^{δ_σ|Y|}` con Y la región de cambio;
   - los términos de Mayer, en los que un contorno incompatible con Γ (solapado o a distancia < ℓ_0)
     interviene. Kotecký–Preiss los acota por `a·e^{−σ|S_Γ|}` relativo, y caben en `1/4`.

   En suma, la constante de Lipschitz de la parte de contornos es
   `≤ 2η_∂(σ) + C_KP a ≤ 1/2`.
4. **Acoplamiento entre las dos partes.** `g^s` entra en la ecuación de los contornos a través de
   `G_c − G^s` en los dominios, y ρ entra en la de los caracteres cortos a través del logaritmo de
   Mayer. Las dos cosas están ancladas en paredes y tienen la cota de 3.
5. **Uniformidad y fuente.** Ninguna constante depende de n. Los contornos que dan la vuelta
   tienen `w ≥ n/2` y quedan en el residuo `r_c` de los caracteres constantes, como en Z2.
   `‖𝔽(0)‖ ≤ d/2` como en Z1, porque la fuente es v. ∎

**Qué no está verificado línea a línea.** Hay tres afirmaciones de estructura, todas estándar en
Pirogov–Sinai, que no he transcrito:
1. **(R) La representación exacta.** Que la ecuación de KT completa para `G_0` es equivalente al
   sistema reducido de 𝔽 sobre `(g^s, ρ)` más la autoconsistencia `G_c = T^c G_0 T^{−c}`. Es decir,
   que las ecuaciones de las decoraciones interiores se siguen de las de la fase. Es la propiedad que
   hace que el modelo de contornos de Borgs–Kotecký–Ueltschi sea exacto. Aquí hay que comprobarla
   con la forma de Kirkwood–Thomas, en el logaritmo.
2. **(KP) La aplicación de Kotecký–Preiss** al gas de contornos con la compatibilidad «a distancia
   ≥ ℓ_0». Es directa en una dimensión.
3. **(L) La linealización.** Que la cancelación de 2 vale también para variaciones de `G_0`, porque
   la ecuación de la fase c varía igual que la de la fase 0 trasladada. Es la parte nueva respecto
   de Pirogov–Sinai térmico.

Lo comprobado numéricamente es esto:
- el punto fijo existe y tiene constantes uniformes en n con ε ≤ 0,1 (paso 4);
- la columna vestida por fases no depende de L (Z10);
- el cambio de base cuesta `e^{δ|Y|}` (Z11);
- Gershgorin en la base por fases separa el multiplete (paso 3);
- la estructura de polímero se ve en el vacío exacto, cualitativa con c = 1, y con contornos en lugar
  de islas para entender c = 2.

#### Paso 6′: (R) y (KP) sobran — Kirkwood–Thomas precondicionado por fases

Al intentar verificar (R) apareció una reformulación que la hace innecesaria, y con ella también
(KP). La idea es no reducir la ecuación a variables de contorno. Se resuelve la ecuación completa,
en todos los caracteres, con un precondicionador construido en la base por fases.

**Lema Z13 (la linealización de Kirkwood–Thomas es el generador vestido).** Sea
`R[G] = (H'e^{G})/e^{G} − E`. Para toda variación δG,

    δR = e^{−G} H' (e^{G} δG) − ((H'e^{G})/e^{G}) δG = (H̃_G − E) δG − R[G] δG ,

con `H̃_G = e^{−G}H'e^{G}`. En el punto fijo, `R = Σ_c r_c χ_{c·1}` es de orden `(M|ε|)^{n/2}`, así que
la linealización `J = H̃ − E` es semejante a `H − E`. En el sector no constante tiene el hueco del
multiplete, y es invertible. ∎

Esto explica el paso 5. Jacobi precondiciona con `diag(|∂μ|)`, la diagonal de J en la base de
caracteres. Por el Lema Z8, J no es diagonalmente dominante en esa base, y `ρ(I − D^{-1}J)` crece con
n. En la base por fases, en cambio, J sí es diagonalmente dominante, por Z10 y Z11.

**La iteración.** Sea `B[G]` la base por fases construida con `G_c = T^c G T^{−c}`, sea
`M[G] = B^{−1}H'B` y sea `D[G]` la diagonal de `M − E` en los caracteres no constantes. El
**quasi-Newton por fases** es

    G_{t+1} = G_t − B D^{−1} B^{−1} R̂[G_t]     (en los caracteres no constantes de la escala m).

- **Exactitud, que sustituye a (R).** Si la iteración converge, `B D^{−1} B^{−1}` es invertible, y el
  límite cumple `R̂ = 0` en *todos* los caracteres de la escala, no solo en un sistema reducido. No
  hay que probar que ninguna representación de contornos es exacta. Los contornos son ahora solo
  una forma de leer las coordenadas `B^{−1}δG`.
- **Sin Mayer, lo que sustituye a (KP).** La incógnita es G directamente, sin ningún logaritmo de
  un gas, así que no hace falta Kotecký–Preiss.

**Lema Z12′ (contracción del quasi-Newton; sustituye a Z12 y a (L)).** Sea `r_t = B^{−1}R̂[G_t]` el
residuo en coordenadas por fases. Entonces

    r_{t+1} = (offdiag(M) D^{−1}) r_t + Q_t ,

donde `Q_t` reúne tres términos:
- los de orden 2 en `δG_t`;
- el cambio de la base, `B[G_{t+1}] − B[G_t]`;
- el término `−R δG` de Z13.

Se tienen dos cotas:
1. **Parte lineal.** `‖offdiag(M)D^{−1}‖ ≤ e^{O(σ)} η'_∂/(1 − η'_∂)` en la norma local con pesos.
   Es Gershgorin por columnas (Z10 y Z11), con entradas ancladas en las paredes de μ. El paso de la
   norma ℓ¹ de columnas a la norma local es el de Z1: una salida μ′ que contiene un sitio x que μ no
   contiene está cerca de una pared de μ, y el peso `x^{dist}` lo paga.
2. **Parte cuadrática.** `‖Q_t‖ ≤ C ℓ_{m+1} ‖r_t‖²`. El factor `ℓ_{m+1}` viene del cambio de base. Un
   cambio δG del vacío cambia el vestido de cada dominio de longitud L ≤ ℓ_{m+1} en `L‖δG‖`, y eso
   es lo que obliga a ir por escalas: dentro de la escala m, los dominios miden menos de `ℓ_{m+1}`.

Con el residuo inicial de la escala m, `‖r_0‖ ≤ δ_m ≤ C x^{ℓ_m/2}`, la constante de contracción es

    q_m ≤ η'_∂/(1 − η'_∂) + C ℓ_{m+1} x^{ℓ_m/2} < 1/2 ,

uniformemente en m y en n. (El paso 8 muestra que esto no se sostiene tal cual: véase 8.1.) Lo que queda es escribir las dos cotas con constantes explícitas, que son
del mismo tipo que las de Z1 y Z11. No queda ninguna afirmación de estructura. ∎ (esquema)

**Comprobación numérica** (`scripts/newton_fases.py`). Radio espectral de la linealización del
quasi-Newton, `I − D^{−1}(M − E)`, frente al de Jacobi (paso 5), en `S⁺S⁻`:

| ε | quasi-Newton n = 6 | quasi-Newton n = 9 | Jacobi n = 6 | Jacobi n = 9 | Jacobi n = 12 |
|---|---|---|---|---|---|
| 0,05 | 0,167 | — | 0,196 | 0,216 | 0,241 |
| 0,1 | 0,335 | 0,345 | 0,399 | 0,454 | 0,506 |
| 0,15 | 0,496 | — | 0,622 | 0,717 | 0,810 |
| 0,2 | 0,643 | 0,641 | 0,862 | 1,006 | 1,149 |

Con n = 6, M es completa en doble precisión y D es exacta. Con n = 9, B se factoriza en precisión
simple, D se toma de la columna local de Z10 y el radio se obtiene con Arnoldi.

- **El quasi-Newton por fases contrae uniformemente en n**: 0,335 → 0,345 con ε = 0,1 y
  0,643 → 0,641 con ε = 0,2. Jacobi crece linealmente.
- **Con ε = 0,2 y n = 9, el quasi-Newton contrae (0,64) donde Jacobi diverge (1,006).**
- El radio crece casi linealmente con ε (0,17 → 0,64 de ε = 0,05 a 0,2). Es el `η_∂ = O(ε)` de Z10.

#### Paso 7: las dos cotas del Lema Z12′, con constantes

> **Nota (paso 8).** La revisión línea a línea encontró tres pasos de la cota 2 que no son
> uniformes en n (8.1). Lo que sigue se conserva como registro. La versión válida está en el paso 8.

**Notación.**
- `x = M̄|ε|`, y A la norma de coeficientes de v en el marco, contando los desplazamientos de tipo τ
  y los demás. Para `S⁺S⁻`, A = 7,11.
- `A_σ(x)` como en Z11, y `d = (4a + b)/M̄` como en Z1.
- Para coordenadas en la base por fases `r = (r_μ)` se usa la **norma de paredes**

      ‖r‖_∂ = sup_y Σ_{μ : y ∈ ∂μ} |r_μ| ,

  donde y recorre los enlaces. Los residuos de Z10 están anclados en las paredes, así que esta es su
  norma natural. Los caracteres constantes, que no tienen paredes, son la dirección del multiplete y
  no se iteran.
- **Constante de pared:**

      η = (2(e^{2d} − 1) + 2A|ε|e^{2d}) (1 + 2d A_{2σ}(x)) e^{8σ} .

  Es la cota de Z10 por pared, con los pesos `e^{2σ|hull|}` que piden la localidad y Z11:
  - el primer factor es el `−P_asc` vestido y v en el enlace de pared, como en Z6;
  - `1 + 2dA_{2σ}` cuenta los caracteres de un dominio que cruzan la pared y llevan el vestido de la
    otra fase;
  - `e^{8σ}` convierte `hull + 1` en pesos de soporte de las salidas.

**Lema Z12′ (contracción del quasi-Newton por fases).** Sean las hipótesis de Z9 (con ℓ_0), de Z11
(con `δ_σ ≤ σ`) y `e^{4σ}x < 1`, y sea η < 1/2. En la escala m, con `ℓ_{m+1} = 2ℓ_m` y
`‖G_t‖ ≤ 2d`, el residuo `r_t = B^{−1}R̂[G_t]` cumple

    ‖r_{t+1}‖_∂ ≤ ( q_lin + C_Q ℓ_{m+1} ‖r_t‖_∂ ) ‖r_t‖_∂ ,

donde

    q_lin = (η/(1 − η)) (1 + coth(σ/2)) ,   C_Q = 8 e^{2d}(1 + A|ε|) κ_B² ,   κ_B = e^{2σ}/(1 − η) .

*Cota 1, la parte lineal* `A_lin = offdiag(M) D^{−1}`.
- **Diagonal y columnas.** Por el corolario de Z11, `D_μ ≥ (1 − η)|∂μ|` y, para cada pared p de μ,
  `Σ_{λ anclado en p} |M_{μ+λ, μ}| e^{σ|hull λ|} ≤ η`.
- **Fijado un enlace y**, la suma sobre los μ′ con `y ∈ ∂μ′` de `|(A_lin r)_{μ′}|` tiene dos partes:
  - **(a) y ya era pared de μ.** La columna entera de μ suma `≤ η|∂μ|/D_μ ≤ η/(1 − η)`, y sumando
    sobre esos μ queda `≤ (η/(1 − η))‖r‖_∂`.
  - **(b) y es una pared nueva, creada por λ anclado en una pared p de μ.** Entonces
    `|y − p| ≤ |hull λ| + 1`, y el peso da `≤ η e^{−σ(|y−p|−1)}`. Sumando sobre p,
    `Σ_p e^{−σ|y−p|} ≤ coth(σ/2)`, y sobre los μ con pared p, `≤ ‖r‖_∂`. Así queda
    `≤ (η/(1 − η)) coth(σ/2) ‖r‖_∂`.
- **En total**, `‖A_lin‖_∂ ≤ q_lin`.

*Cota 2, la parte cuadrática.* El paso es `δG = −B D^{−1} r_t`. Por Z11, con columnas de B de norma
`≤ e^{σ|Y|}`, y por la cota de D, `‖δg‖_{Z1} ≤ κ_B ‖r_t‖_∂`. Los tres términos de `Q_t` son:
1. **Resto de segundo orden de la exponencial.**
   `R[G+δG] − R[G] − JδG = Σ_j Σ_δ C_δ (e^{δG_δ} − 1 − δG_δ)`, con `‖δG_δ‖_1 ≤ 2‖δg‖` y
   `‖C_δ‖ ≤ (1 + A|ε|)e^{2d}` por enlace. Da `≤ 4(1 + A|ε|)e^{2d} ‖δg‖²` por pared, porque los
   enlaces lejos de las paredes se cancelan como en Z10.
2. **El término `−R δG` de Z13.** `≤ ‖r_t‖ ‖δg‖`, también anclado en las paredes del residuo.
3. **El cambio de base.** `B[G + δG] = B[G](1 + Δ)`. En la columna μ, el vestido de los dominios de
   otra fase cambia en `exp((δG_c − δG_0)|_D)`, con `‖·‖_1 ≤ 2‖δg‖ |D|`. En la escala m, cada
   dominio no nulo mide `< ℓ_{m+1}` y tiene dos paredes. Así,
   `‖Δ_μ‖ ≤ e^{ℓ_{m+1}|∂μ|‖δg‖} − 1 ≈ ℓ_{m+1}|∂μ|‖δg‖`, y el `|∂μ|` lo compensa el `D_μ^{−1}`.
   Da `≤ 2e^{2d} ℓ_{m+1} ‖δg‖ ‖r_t‖_∂`.
   Del mismo tipo es el término que aparece porque Z10 se aplica con el iterado `G_t` y no con el
   vacío exacto. En el interior de cada dominio queda el residuo trasladado de la fase, de densidad
   `‖r_t‖`. En la columna μ eso da `≤ ℓ_{m+1}|∂μ|‖r_t‖`, que entra en `A_lin r_t` como un término
   `ℓ_{m+1}‖r_t‖²`.

Sumando, con `ℓ_{m+1} ≥ 1` y `‖δg‖ ≤ κ_B‖r_t‖`, se obtiene `‖Q_t‖_∂ ≤ C_Q ℓ_{m+1}‖r_t‖²_∂`. ∎

**Cierre de la inducción.** En la escala m, el residuo inicial es el de los caracteres que la
escala m − 1 excluía, `‖r_0‖_∂ ≤ δ_m ≤ d x^{ℓ_m/2}`. La contracción vale si

    q_lin + C_Q · 2ℓ_m · d x^{ℓ_m/2} ≤ 1/2 ,

y como `ℓ_m x^{ℓ_m/2}` decrece en m para x < 1, basta con m = 0. El paso total es
`Σ_m Σ_t ‖δg‖ ≤ 2κ_B Σ_m δ_m ≤ 4κ_B d x^{ℓ_0/2}`, que es ≤ d. Por tanto G se queda en la bola
`‖G‖ ≤ 2d`. En la última escala, `ℓ_m ≥ n`, el límite cumple `R̂ = 0` en todo carácter no
constante. Eso es (V), con constantes que no dependen de n.

**Constantes para `S⁺S⁻`** (k = 3, a = A = 7,11, a_nc = 4,44, b = 22,3, ℓ_0 = 4). Optimizando
sobre M̄, σ y x con
- `K_{ℓ_0} ≤ 1/2` (Z9),
- `δ_σ ≤ σ` (Z11),
- `q_lin ≤ 1/4`,
- la parte cuadrática ≤ 1/4,

sale `M̄ = 5,8·10⁴`, σ = 0,10, x = 0,47, `d = 8,8·10⁻⁴`, η = 0,0087, `q_lin = 0,185`,
`δ_σ = 0,097`, `C_Q = 12` y parte cuadrática `1,9·10⁻²`. El **radio garantizado es
`|ε| < 8,1·10⁻⁶`.**

Es pequeño. Lo que más cuesta es el factor `1 + coth(σ/2) ≈ 21` de la parte (b) de la cota 1, que
suma paredes lejanas con un peso débil. Con una norma de paredes que pese también la distancia, se
ganaría casi todo ese factor. La pérdida frente a lo observado (contracción hasta ε ≈ 0,2, paso 6′)
es de unos cuatro órdenes. Es la misma clase de pérdida que con `M_* = 359` en SU(3) y M = 45 en [X].

#### Paso 8: revisión línea a línea del Lema Z12′

Al escribir línea a línea las dos piezas que estaban en esquema (la parte (b) de la cota 1 y el
cambio de base de la cota 2) aparecen tres pasos que **no son uniformes en n**. Dos se reparan
dentro de una escala, cambiando de norma y de formulación (Lemas Z11s, Z14 y Z12″, demostrados).
El tercero no se repara: al pasar de una escala a la siguiente se pierde una cantidad fija de peso,
O(|ε|), y eso limita el número de escalas. El Teorema Z″ queda demostrado para
`n ≤ n_*(ε)`, con `log n_* ≍ 1/|ε|`, no para todo n (8.7).

**8.1 Qué falla en el paso 7.**
1. **El término `−R δG` de Z13 (término 2 de la cota 2).** R y δG son densidades en todo el anillo.
   Su producto tiene, en cada pared y, contribuciones de pares formados por un trozo de R en j y uno
   de δG en j′, con j y j′ arbitrarios. *Fallo concreto:* si R y δG tienen densidad uniforme ρ y ρ′,
   `‖R δG‖_∂ ≥ c n ρ ρ′`. La cota «`≤ ‖r_t‖‖δg‖`, anclado en las paredes» es falsa. Lo único conexo es
   la suma de Z13:

       R[G + δG] − R[G] = Σ_b Σ_{δ en b} C_δ e^{Δ_δ G} (e^{Δ_δ δG} − 1) ,   Δ_δ F(s) = F(s + δ) − F(s) ,

   donde cada término de b solo ve los caracteres de δG que tocan b. La parte lineal es
   `L_G δG = Σ_b Σ_δ C_δ e^{Δ_δG} Δ_δ δG`, que es `H̃ − E − R` de Z13 en una sola pieza. Hay que usar
   `L_G` y no `H̃ − E` y `−R` por separado.
2. **El cambio de base (término 3).** `‖Δ_μ‖ ≤ e^{ℓ_{m+1}|∂μ|‖δg‖} − 1 ≈ ℓ_{m+1}|∂μ|‖δg‖` no es una
   cota uniforme. Con `|∂μ| ∼ n` la exponencial no es lineal, y en `r_{t+1} = B_{t+1}^{−1}B_t r̃_{t+1}`
   no hay ningún `D^{−1}` que compense el `|∂μ|`. *Fallo concreto:* un μ con n/ℓ islas da un factor
   `e^{n‖δg‖}`.
3. **La cota del paso, `‖δg‖_{Z1} ≤ κ_B ‖r‖_∂`.** El vector `b_μ` tiene masa de Fourier
   `e^{θ|sop μ|}`, con `θ = 2Γ_0` el producto de los trozos de `G_c − G_0` en los dominios. La norma de
   Z1 pesa los caracteres con `(M|ε|)^{−w}`. La norma de paredes no controla ninguna de las dos cosas.
   *Fallo concreto:* si r está concentrado en una isla de longitud L, `‖r‖_∂ = |r_μ|`, mientras que
   `‖δg‖_{Z1} ≳ |r_μ| (M|ε|)^{−L/2}/L`.

La parte (b) de la cota 1 sí es correcta, salvo por un detalle. El corolario de Z11 acota la norma
de las coordenadas, no dónde caen, y (b) necesita lo segundo. Eso lo da el Lema Z11s.

**Notación.** En todo el paso 8 se toma R = 0: sin vestido de pared, que solo mejoraba constantes.
- **Densidad del vacío con pesos:** `Γ_w(g) = sup_y Σ_{ν : y ∈ hull ν} |g(ν)| e^{w|hull ν|}`. Para un
  vacío en la bola de Z1, `Γ_w ≤ d A_w(x)`, que es el recuento de Z11, y `δ_w = 2kΓ_w`.
- **σ_Z** es un peso con `δ_{σ_Z} ≤ σ_Z`, la hipótesis de Z11.
- **Base relativa:** `b̃_α = χ_α e^{L_α}`, con `L_α = G_α − G`. Sus trozos son `g_c − g_0` en los
  caracteres ν con hull ν dentro de un dominio no nulo de α, de fase c.
- **Coordenadas** de una función F: `[F] = B̃^{−1}F`.

**Lema Z11s (soporte).** Con R = 0, las coordenadas de `χ_γ Φ_β` solo son no nulas en caracteres α
con `sop α ⊆ sop β ∪ sop γ`. En particular:
- **De función a coordenadas:** las coordenadas de `χ_ν` (β = 0, γ = ν) caen en `sop α ⊆ sop ν`, con
  masa `≤ e^{σ_Z|sop ν|}`.
- **De coordenadas a función:** los caracteres de Fourier de `b̃_α` tienen el hull dentro de
  `Ĥ(α)`, el menor arco que contiene `sop α`, con masa `≤ e^{θ|sop α|}`, donde `θ = 2Γ_0`.

*Demostración.* Por la regla local, un trozo de `L_{βγ}` es no nulo solo si β o γ son constantes e
iguales a algún c ≠ 0 en su hull. Ese hull está entonces dentro de un dominio no nulo de β o de γ,
y por tanto dentro de `U = sop β ∪ sop γ`. Así, cada λ del paso 2 de Z11 cumple `sop λ ⊆ U`, y el par
siguiente `(γ, γ + λ)` tiene `sop γ ∪ sop(γ + λ) ⊆ U`. Por inducción en la profundidad, todo
carácter de la serie queda en U. Las masas son las de Z11 y la de `e^{L_α}`: cada trozo se cuenta en
un sitio de su hull, y `Σ_ν |L_α(ν)| ≤ 2Γ_0 |sop α|`. ∎

**8.2 La norma de polímeros y la parte lineal.** Para coordenadas r en los caracteres no constantes
se usa

    ‖r‖_σ = sup_y Σ_{α : y ∈ Ĥ(α)} |r_α| e^{σ|Ĥ(α)|} .

Es la norma de Z1, con el hull en lugar del recubrimiento, llevada a coordenadas. Se pide a Z10 la
cota por pared con pesos de hull:

    Σ_{ν anclado en p} |f_α(ν)| e^{w|hull(ν ∪ p)|} ≤ η_∂(w) ,

que es la del paso 7 con `2σ → w`:
`η_∂(w) = (2(e^{2d} − 1) + 2A|ε|e^{2d}) (1 + 2d A_w(x)) e^{4w}`.

**Lema (cota L).** Sean τ > 0 y `η' = η_∂(σ + τ + σ_Z) < 1`, con `D_α ≥ (1 − η')|∂α|`. Entonces

    ‖offdiag(M) D^{−1}‖_σ ≤ q_L = (η'/(1 − η')) coth(τ/2) .

*Demostración.* Por Z10, la columna α es `D_α b̃_α + Σ_p Σ_ν f_α(ν) χ_{α+ν}Φ_α`. Por Z11s, cada término
de pared tiene coordenadas en α′ con

    Ĥ(α′) ⊆ hull(sop α ∪ sop ν) ⊆ Ĥ(α) ∪ hull(ν ∪ p) ,

con masa `≤ e^{σ_Z|sop ν|}`. Los dos arcos comparten el sitio de p que está en sop α. Se fija y:
- **Si y ∈ Ĥ(α),** el peso de la salida es `≤ e^{σ|Ĥ(α)|} e^{σ|hull(ν∪p)|}`. Sumando sobre ν y p, la
  columna aporta `≤ η'|∂α| e^{σ|Ĥ(α)|}`. Con `D_α^{−1}` y sumando sobre los α que contienen a y, queda
  `(η'/(1 − η'))‖r‖_σ`.
- **Si y ∉ Ĥ(α),** y está en `hull(ν ∪ p)`, así que `|hull(ν∪p)| ≥ t + 1`, donde
  `t = dist(y, Ĥ(α)) ≥ 1`. El peso τ que sobra da `e^{−τ(t+1)}`. Los α a distancia t contienen uno de
  los dos sitios `y ± t`, así que suman `≤ 2‖r‖_σ`. Sumando sobre t, queda
  `(η'/(1 − η')) · 2/(e^τ − 1)`.

Como `1 + 2/(e^τ − 1) = coth(τ/2)`, queda probado. ∎

(En la norma de paredes del paso 7, la cota 1 también es correcta con Z11s y con η en `σ + σ_Z`. Se
cambia de norma porque la de paredes no sirve para la cota 2.)

**8.3 Lema Z14 (productos conexos).** Sean un enlace b, caracteres `α_1, …, α_p` cuyos soportes tocan
b, y una función local f en b. Para pesos `σ ≥ 4Γ_{σ_Z}` y `s = σ + 4Γ_{σ_Z}`, las coordenadas de
`F = f · Π_i b̃_{α_i}` cumplen

    Σ_{α′} |[F]_{α′}| e^{σ|Ĥ(α′)|} ≤ ‖f‖_{b,s} Π_i ( e^{8Γ_{σ_Z}} e^{σ|Ĥ(α_i)|} ) ,

donde `‖f‖_{b,s} = Σ_κ |f̂(κ)| e^{s|hull(κ ∪ b)|}`. Además, cada salida cumple
`Ĥ(α′) ⊆ U ∪ hull(κ ∪ b)`, con `U = ∪ Ĥ(α_i)`.

*Demostración.* Se desarrolla f en caracteres κ. Cada término es `χ_γ e^{L}`, con `γ = Σα_i + κ` y
`L = Σ_i L_{α_i}`.
1. **Los trozos en conflicto.** `L − L_γ` solo tiene trozos cuyo hull corta la región de conflicto O,
   donde la estructura de dominios de γ no coincide con la de algún α_i: solapes, uniones y
   `sop κ`. En un sitio s, el número de trozos es a lo sumo `mult(s) + 1`. Como los `Ĥ(α_i)` son arcos
   que comparten un sitio de b,

       Σ_{s∈O} (mult(s) + 1) ≤ 2(Σ_i |Ĥ(α_i)| − |U|) + 4p + 2|hull(κ ∪ b)| .

2. **La serie.** La de Z11, con primer paso `e^{L − L_γ} − 1`, da masa `≤ e^{Δ}`, con
   `Δ ≤ 2Γ_{σ_Z} Σ_{s∈O}(mult(s) + 1)`. Por Z11s, las salidas caen en `U ∪ hull(κ ∪ b)`.
3. **El balance.** Como `|U| = Σ|Ĥ(α_i)| − (Σ|Ĥ(α_i)| − |U|)`, el peso de la salida deja una holgura
   `σ(Σ|Ĥ(α_i)| − |U|)`, que paga la parte de Δ proporcional a los solapes si `σ ≥ 4Γ_{σ_Z}`. Lo que
   queda es `e^{8pΓ_{σ_Z}}`, más `4Γ_{σ_Z}|hull(κ∪b)|`, que absorbe el peso s de f. ∎

**No hay pérdida de peso.** El producto de factores conexos se acota en la misma norma σ, como en
Z1, a cambio de una constante por factor.

**8.4 Lema Z12″ (una escala).** En la escala m se tiene un vacío `G_m` con `Γ_w(g_m) ≤ γ_w` para
`w ≤ w_m`. Se congela la base `B̃` construida con `G_m`. Sea P la proyección sobre los caracteres no
constantes de `𝒜_{ℓ_{m+1}}`, y sea `r_0 = [R̂(G_m)]` el residuo, con `P_{m−1} r_0 = 0` y
`‖r_0‖_σ ≤ δ_m`, donde `σ + τ + σ_Z ≤ w_m`. Con `D` la diagonal de `M_m = B̃^{−1} L_{G_m} B̃`, se
define

    Φ(r̄) = P r_0 − P offdiag(M_m) D^{−1} r̄ + P[Q(−D^{−1} r̄)] ,
    Q(c) = Σ_b Σ_{δ en b} C_δ e^{Δ_δ G_m} ( e^{Y_δ} − 1 − Y_δ ) ,   Y_δ = Δ_δ(B̃c) .

Entonces:
- un r̄ es punto fijo de Φ si y solo si el vacío `G_{m+1} = G_m − B̃D^{−1}r̄` cumple
  `P[R̂(G_{m+1})] = 0`;
- si

      q_L + q_I + Lip_m(2δ_m) ≤ 1/2 ,

  Φ lleva la bola `‖r̄‖_σ ≤ 2δ_m` en sí misma y contrae con constante 1/2.

Las constantes son:

    q_I  = 2 ℓ_{m+1} ρ_F /(1 − η') ,
    Lip_m(ρ) = (1 + A|ε|) e^{4e^s γ_s} (e^{8Γ_{σ_Z}} U_s/(1 − η')) [ ℓ_{m+1}(e^z − 1 + z e^z) + coth(τ/2)(e^z − 1) ] ,

con:
- `s = σ + 4Γ_{σ_Z} + τ` y `U_s = 2e^{8e^s γ_s}`;
- `z = e^{8Γ_{σ_Z}} U_s ρ/(1 − η')`;
- ρ_F, la densidad de Fourier del residuo de `G_m` en los caracteres dentro de los dominios, con
  peso σ_Z.

*Demostración.*
1. **El desarrollo es exacto y conexo** (8.1, punto 1):
   `[R̂(G_m + B̃c)] = r_0 + M_m c + [Q(c)]`. Con `c = −D^{−1}r̄`, `P(M_m D^{−1} r̄) = r̄ + P offdiag(M_m) D^{−1} r̄`,
   y `P[R̂] = 0` es exactamente `r̄ = Φ(r̄)`. No hay cambio de base, porque la base está congelada, ni
   iteración de Newton: el paso 6′ es la parte lineal de Φ.
2. **La columna de `M_m`** es la de Z10 más un término interior. Para un enlace b interior a un
   dominio de fase c, `ω^{α·δ} = 1`, porque los δ conservan la carga, y el término es
   `(T^c − 1)ρ_b`. Sumando sobre los b del dominio, los caracteres interiores dan
   `(ω^{−cΣν} − 1) R̂_m(ν)`. Este término es nulo en un vacío exacto (Z10) y ≤ `2|sop α| ρ_F` en
   general. Por Z11s no crece el hull. El recuento `|sop α| ≤ ℓ_{m+1}|∂α|` (Z9) y `D_α^{−1}` dan
   q_I. La parte de pared da q_L (8.2).
3. **Q.** `Δ_δ b̃_α = b̃_α u_{α,δ}`, con `u_{α,δ} = ω^{α·δ} e^{Δ_δ L_α} − 1`. Este factor es no nulo
   solo si b toca sop α, y `‖u‖_{b,s} ≤ U_s`. Cada término de orden p de Q es un producto conexo en b,
   así que se aplica Z14. Fijado y, o bien y está en `Ĥ(α_i)` para algún i, o bien está en
   `hull(κ ∪ b)`.
   - **En el primer caso,** la suma sobre los b que tocan sop α_i da `≤ ℓ_{m+1}|∂α_i|`, que compensa
     el `D^{−1}` de `c_{α_i} = −r̄_{α_i}/D_{α_i}`. Los demás factores tocan b y suman
     `≤ ‖r̄‖_σ/(1 − η')` cada uno, porque `|∂α| ≥ 2` y b tiene dos sitios.
   - **En el segundo caso,** el peso τ de f da `Σ_b e^{−τ|y−b|} ≤ coth(τ/2)`.

   La serie en p da la expresión de `Lip_m`, y la diferencia `Q(c) − Q(c′)` se trata igual,
   factor a factor. ∎

**8.5 Entre escalas: la pérdida de peso.** El nuevo vacío es `G_{m+1} = G_m − B̃D^{−1}r̄*`. Por Z11s,
la corrección tiene densidad controlada solo con pesos `w ≤ σ − θ`, por la masa de `b̃_α`. La escala
m + 1 reconstruye la base con `G_{m+1}`, y para eso necesita `w_{m+1} ≥ σ_{m+1} + τ + σ_Z`. Además, el
residuo pasa de las coordenadas viejas a las nuevas a través de Fourier, y eso cuesta `θ + σ_Z`.
Por tanto

    σ_{m+1} = σ_m − λ ,   λ = θ + τ + σ_Z .

**La pérdida es fija; lo pequeño es el residuo.** Las salidas no resueltas tienen un tramo no nulo
≥ ℓ_{m+1}, así que `Ĥ ≥ ℓ_{m+1}`, y el peso τ que sobra da

    δ_{m+1} ≤ 4 δ_m e^{−τ ℓ_{m+1}} .

Con `ℓ_m = 4^m ℓ_0` y `τ ℓ_1 ≥ ln 16`, `ℓ_{m+1}δ_m` decrece, y la condición de Z12″ se cumple en todas
las escalas si se cumple en la primera. El número de escalas está acotado por el presupuesto de peso:

    N = ⌊(σ_0 − σ_min)/λ⌋ + 1 ,   σ_0 = w_0 − τ − σ_Z ,   w_0 < ¼ ln(1/x) ,
    σ_min = max(4Γ_{σ_Z}, θ + σ_Z) + τ .

El argumento cubre los anillos con `n ≤ n_* = ℓ_0 4^{N−1}`. Como q_L ≤ 1/4 pide `τ ≳ 8η' = O(d)`, y θ
y σ_Z son O(d), se tiene `λ = O(d) = O(|ε|/x)`, y con x fijo, `N ≍ 1/|ε|`.

**Números** (`scripts/constantes_z12.py`, `S⁺S⁻`, optimizando M̄, x y τ):

| ε | M̄ | x | τ | λ | σ_0 | N | n_* |
|---|---|---|---|---|---|---|---|
| 3·10⁻⁶ | 9,6·10⁴ | 0,29 | 0,10 | 0,134 | 0,135 | 1 | 16 |
| 2·10⁻⁶ | 1,4·10⁵ | 0,29 | 0,073 | 0,094 | 0,19 | 2 | 64 |
| 1·10⁻⁶ | 2,9·10⁵ | 0,29 | 0,032 | 0,042 | 0,21 | 5 | 6·10³ |
| 3·10⁻⁷ | 1,1·10⁶ | 0,33 | 8,2·10⁻³ | 0,012 | 0,24 | 21 | 10¹⁴ |
| 1·10⁻⁷ | 3,9·10⁶ | 0,39 | 2,2·10⁻³ | 3,4·10⁻³ | 0,23 | 66 | 10⁴¹·⁶ |
| 1·10⁻⁸ | 3,3·10⁷ | 0,33 | 2,9·10⁻⁴ | 4,0·10⁻⁴ | 0,27 | 669 | 10⁴⁰⁶ |

- **Asintóticamente, `log n_* ≈ 9·10⁻⁶/|ε|`.**
- **Comparación con una sola fase.** Sin fases, Z9 con ℓ = n + 1 (KT completo) y Gershgorin a la Z6
  funcionan mientras la isla del Lema Z8 tenga radio pequeño: `n ≲ 0,1/|ε|`. Las fases y las escalas
  llevan el rango de 1/|ε| a `exp(c/|ε|)`, pero no a todo n. Con ε ≥ 10⁻⁶, la cota de una sola fase
  es mejor (10⁵ frente a 6·10³).
- **Los términos cuadrático y de Peierls no limitan.** Con `ℓ_0 ≥ ln 16/(4τ)`, el residuo inicial
  `δ_0 ≲ d (x e^{2(σ_0+σ_Z)})^{ℓ_0/2}` es astronómicamente pequeño, y `Lip_0(2δ_0) ≪ 1/4`.

**Por qué no se quita la pérdida en este esquema.** Cada paso de base a Fourier o de Fourier a base
cuesta una masa O(d) por sitio:
- θ, de la masa de las islas vestidas;
- σ_Z, de la serie de Z11.

Cualquier esquema que reconstruya la base a partir del vacío en Fourier paga esas masas una vez por
escala. Congelar la base en todas las escalas no sirve: el interior de las islas largas quedaría
vestido con la precisión de la escala 0, y eso es el Lema Z8 otra vez. Para quitar la pérdida, las
correcciones de cada escala tienen que quedarse como polímeros con su propia actividad, sin
desarrollarlas nunca en Fourier. Eso es una representación de contornos de Pirogov–Sinai con
Kotecký–Preiss. El paso 6′ parecía hacerla innecesaria precisamente porque escondía esta pérdida.

**8.6 Qué cambia.**
- **El Lema Z12′ queda sustituido por Z11s, la cota L, Z14 y Z12″** (una escala, demostrados), más
  la contabilidad de pesos de 8.5.
- **La comprobación numérica del paso 6′ sigue valiendo.** El radio espectral medido es el de la
  parte lineal de Φ en la escala más fina, y es uniforme en n hasta ε = 0,2. Lo que no está
  demostrado es que la pérdida de peso entre escalas sea un artefacto de la prueba. Los números
  sugieren que lo es.
- **El radio garantizado del paso 7 (`8,1·10⁻⁶`) ya no vale.** Usaba las tres cotas de 8.1. Con
  las cotas corregidas, el esquema de una escala pide `|ε| ≲ 3·10⁻⁶`, y el rango en n es el de la
  tabla.

#### Paso 9: contornos con Kotecký–Preiss, y un atajo por la literatura

El paso 8 deja una pérdida fija de peso por escala. Para quitarla hay que dejar de reconstruir la
base desde el vacío en Fourier. Antes de construir esa representación de contornos se ha mirado si
la literatura ya cubre parte del problema, y la cubre: el hueco uniforme en n, con ε real (9.2).
La representación de contornos sigue haciendo falta para la cota *exponencial* y para la
analiticidad compleja (9.1, 9.4).

**9.1 Qué tiene que hacer la representación de contornos.** Se resume en tres requisitos y un hueco
abierto.

*Primer requisito: polímeros = excursiones.* En el marco dual (caracteres μ, energía clásica |∂μ|),
el vacío de la fase 0 es un gas de **excursiones**: tramos maximales donde μ ≠ 0, cada uno con su
estructura interna (dominios de otras fases y sus decoraciones). Una excursión γ de longitud L tiene
actividad `|ρ(γ)| ≤ (x e^{κ})^{L/2}`, porque separar sus paredes L sitios cuesta L saltos de orden ε
con denominadores ≥ 1. La decoración del interior de un dominio de fase c es el gas de la fase c, y
por la simetría T (k | n) sus actividades son exactamente las de la fase 0 transportadas: no hay
desajuste interior, que era el Lema Z8.

*Segundo requisito: las paredes se visten por completo.* El vestido de pared tiene alcance ilimitado,
con decaimiento: es el `h_μ` de Z10 con R = ∞. Así, en la ecuación de una actividad solo quedan como
términos fuera de la diagonal los movimientos *físicos* de las paredes (`Δ ≤ 0`), que tienen alcance
acotado y amplitud O(ε). El crecimiento de una excursión regenera entonces la tasa de Peierls
`x^{ΔL/2}` a partir de la x desnuda, no a partir de los pesos del vacío. Esta es la diferencia con el
paso 8. Allí el crecimiento pasaba por los trozos de Fourier de G, cuyo peso estaba degradado, y la
tasa de las actividades nunca podía superar la del núcleo, que a su vez quedaba por debajo de la de
G menos `σ_Z`: un círculo.

*Tercer requisito: una sola pérdida.* Kotecký–Preiss convierte el gas en `G_0 = log ψ_0` (Fourier)
con una única pérdida, `τ → τ − a`. La base por fases, Z10 y Z11 se usan después *una vez*, con ese
vacío, para Gershgorin. Ya no hay reconstrucciones por escala.

*Lo que falta demostrar.* La ecuación de las actividades, por inducción en el tamaño de la
excursión y con τ fijo, tiene que cerrar. En concreto:

    D_γ ρ(γ) = −Σ_{movimientos físicos γ′ → γ} M_{γγ′} ρ(γ′) + (fusiones con excursiones vecinas, por KP) ,

con `D_γ ≥ (1 − η)|∂γ|`, y hay que comprobar que da `|ρ(γ)| ≤ (x e^{κ})^{|γ|/2}` para algún κ fijo.
Es la versión de estado fundamental de la inducción de Zahradník. El control de las fusiones es el
estándar de KP. Lo delicado es el vestido de pared con R = ∞: en 7.13 (paso 2′) tenía un denominador
casi resonante con ε = 0,1 (el par de paredes con energía ≈ 0,15). Con ε pequeño no es un obstáculo,
pero hay que acotarlo. **Este paso no está hecho.**

**9.2 El hueco uniforme en todo n, por Nachtergaele–Sims–Young.**
Nachtergaele, Sims y Young, *Quasi-locality bounds for quantum lattice systems. Part II*
(arXiv:2010.15337; AHP 2021), demuestran la estabilidad del hueco para perturbaciones de modelos
sin frustración con hueco. En la Sección 8 añaden tres tipos de ruptura discreta de simetría. El
tipo (S2) es la ruptura de la invariancia por traslación a un subgrupo (estados periódicos), con
condiciones de contorno periódicas. Su Teorema 6.8, con las modificaciones de §8.4.2, da: si el modelo
es un «uniform perturbation model» (Definición 6.5 y Supuesto 6.7), existe `s_γ > 0`, uniforme en el
volumen, tal que para `0 ≤ s ≤ s_γ`

    diam Σ_1(s) ≤ 2s(δ + ε) ,   gap(H_{Λ_n}(s)) ≥ γ ,

donde `Σ_1` es el racimo que viene del espacio fundamental sin perturbar ((6.27)–(6.29)).

*Comprobación de las hipótesis para carga entera con k | n, en el marco original.*
1. **H_0 sin frustración, de alcance finito y con hueco uniforme.** `H_0 = Σ_j (1 − Π^{asc}_{j,j+1})`
   es una suma de proyecciones clásicas que conmutan. Tiene energía fundamental 0, k estados
   fundamentales (los patrones `s_j = c + j`, con k | n) y hueco 1 en todo volumen y en toda bola
   (hueco local `γ_n(m) = 1`).
2. **Traslación y contorno periódico.** El anillo es periódico. La perturbación `εv`, con el giro Θ
   repartido como `Θ/n` por enlace, es invariante por traslación. La Definición 6.5 admite una
   sucesión `Φ_n` que depende de n con norma F uniforme. G = Z_k es el grupo de traslaciones que la
   ruptura deja libre, y el espacio fundamental se rompe a kZ.
3. **Indistinguibilidad G-simétrica (8.8), exacta.** Sea `A = Σ_{a∈Z_k} ρ_a(B)`, con B soportado en
   una bola de radio r, y sea `P` el proyector sobre los patrones en una bola de radio m > r. Si
   i ≠ j, `P^i A P^j = 0`, porque los patrones i y j difieren en todos los sitios de la corona. Si
   i = j, `P^i A P^i = Σ_a ⟨i+a|B|i+a⟩ P^i`, que no depende de i. Así, (8.8) vale con Ω(r) = 0
   para r ≥ 1. El radio de indistinguibilidad es el máximo, `Λ^p_n = Λ_n`, y `ε_n = 0` en (6.20).
4. **(6.32).** `δ_n = |Λ_n| C(K_n, L_n)`, con `K_n + L_n ≍ n`. Como Ω ≡ 0, C está dominado por la
   función de decaimiento del flujo espectral, que es subexponencial para interacciones de alcance
   finito. Así, `δ_n → 0` más deprisa que cualquier potencia.
5. **Supuesto 6.7.** δ, ε y β son uniformes: β, por el crecimiento polinómico de las particiones en
   1D y el hueco local constante.

**No verificado línea a línea.** Las cotas concretas de `C(K, L)` y de β ((5.13), (6.21)) y el valor de
`s_γ`. Las hipótesis estructurales sí se cumplen, y los puntos 1 y 3 son exactos.

**Consecuencia (Teorema Z‴, carga entera, k | n, ε real).** Existe `ε_1 > 0` tal que, para ε real con
`|ε| < ε_1`, todo `n ∈ kZ` y todo Θ:
- el multiplete de k niveles está separado del resto por un hueco `≥ γ > 0`, uniforme en n y Θ.
  Esto es (A1) y (G) con ε real, **para todo n**;
- la anchura del multiplete es `≤ 2s δ_n → 0`, más deprisa que cualquier potencia de n;
- **robustez subexponencial:** `sup_{Θ,Θ′} ‖p(n,Θ) − p(n,Θ′)‖ ≤ C e^{−c n^{ϑ}}` para algún ϑ ∈ (0, 1].

La robustez sale del principio de las perturbaciones locales (Bachmann–Michalakis–Nachtergaele–Sims,
equivalencia automórfica). Con el giro concentrado en un enlace, H(Θ) y H(Θ′) difieren solo cerca
de ese enlace. El hueco uniforme permite el flujo cuasi-adiabático en Θ, que es cuasi-local con la
misma función de decaimiento, y la marginal de arista, a distancia ≍ n del enlace girado, cambia
`≤ C G(n/2)`. **Este último paso tampoco está escrito línea a línea:** es el lema estándar, aplicado
al proyector del racimo.

**9.3 Cómo encajan las dos vías.**

| vía | ε | n | cota de robustez | estado |
|---|---|---|---|---|
| KT con fases, una escala (Z12″, paso 8) | real y complejo, `|ε| ≲ 3·10⁻⁶` | `n ≤ n_*(ε)`, `log n_* ≈ 9·10⁻⁶/|ε|` | exponencial, `(M|ε|)^{n/2}` | demostrado |
| NSY (Teorema Z‴) | real, `|ε| < ε_1` | todo n ∈ kZ | subexponencial | hipótesis comprobadas salvo constantes |
| contornos con KP en espacio-tiempo (9.5–9.6) | real y complejo | todo n | exponencial | Z15, Z16 demostrados; Z17 escrito (Teorema Z⁗) |

Juntas, las dos primeras dan, para ε real pequeño y **todo n**, sensibilidad exponencial hasta
`n_*(ε) = exp(c/|ε|)` y al menos subexponencial a partir de ahí. Para la conjetura (k | n ⇒
sensibilidad exponencial) solo falta la tercera fila. NSY no la da por dos razones: su flujo
espectral solo existe con ε real (perturbación autoadjunta), y su decaimiento es subexponencial.

**9.4 Lado k ∤ n.** NSY no se aplica directamente: con k ∤ n, el anillo tiene frustración (siempre
hay un defecto), el espacio fundamental clásico es la banda de nk defectos y H_0 no tiene hueco en
la banda. El Teorema D″ sigue limitado a `n ≤ n_*(ε)` hasta que se haga 9.1. *Hecho en 9.14 con
contornos (Teorema D‴), uniforme en n.*

**9.5 La inducción de las actividades, en espacio-tiempo.**

*Por qué se cambia de marco.* En la forma de vector propio de 9.1 no hay nada que haga de «cociente de
funciones de partición del interior». Los denominadores son energías de pared, el interior se ve a
través del vestido, y el vestido de pared de alcance ilimitado es él mismo un problema de
Kirkwood–Thomas con resonancias. En la representación de Dyson en espacio-tiempo (Datta–Fernández–
Fröhlich, Borgs–Kotecký–Ueltschi) todo eso desaparece:
- la «base» son las configuraciones clásicas, siempre;
- el interior de un contorno entra como un cociente `Z_{c′}(V)/Z_c(V)`;
- la igualdad de las fases es una simetría exacta (la traslación con k | n).

La inducción es entonces la de Zahradník para fases simétricas, con τ fijo y **sin pérdida por
escala**.

*Marco.* Se trabaja con H en las variables originales:
- `H_0 = Σ_j (1 − Π^{asc}_{j,j+1})`, diagonal;
- `εV = ε(V_d + V_o)`, invariante por traslación y conservando la carga, con parte diagonal `V_d`
  y parte fuera de la diagonal `V_o` de alcance 1;
- el giro Θ en las fases de los saltos que cruzan un enlace.

Se resta la energía por enlace de los patrones, `e_d = ε⟨c|v_d|c⟩`, que no depende de c por la
traslación. Para ε complejo, las paredes cuestan `Re ≥ 1 − δ`, con `δ = 2A_d|ε|`. El desarrollo de
Dyson da

    Tr(T^m e^{−βH}) = Σ_ω Π_i (−ε v_{b_i}) · exp(−∫_0^β H″_0(s(t)) dt) ,

que es una suma sobre historias s(t), constantes a trozos, con saltos en los instantes `t_i` y los
enlaces `b_i`. El cilindro espacio-tiempo es `Z_n × [0, β)`, y la traslación `T^m` cierra el tiempo con
un desplazamiento m.

**Celdas y contornos.** Una celda es `(j, ℓ)`: el enlace j durante `[ℓT, (ℓ+1)T)`, con escala de tiempo
`T = ½ ln(1/|ε|)`. Una celda es *correcta en la fase c* si su bloque de 3×3 celdas no tiene saltos y
los tres enlaces del bloque siguen el patrón ascendente c todo el tiempo. Las celdas correctas
conexas tienen una sola fase: en espacio, el patrón se propaga por enlaces ascendentes; en tiempo,
no hay saltos. Un **contorno** es una componente conexa (con 8 vecinos) de celdas incorrectas, junto
con su historia. Las etiquetas de fase de sus vecinas deben ser compatibles.

**Lema Z15 (Peierls en espacio-tiempo).** Para ε complejo con `|ε| ≤ ε_P`, el peso de un contorno de
N celdas, integrado sobre los instantes de sus saltos, cumple

    |W(γ)| ≤ e^{−τ N} ,   τ = (1/9) min( ln(1/(3AT|ε|)), (1 − δ)T ) = (1/18) ln(1/|ε|) (1 + o(1)) .

*Demostración.* Si una celda es incorrecta, su bloque tiene un salto, o bien no tiene saltos y
entonces la historia es constante en el bloque y alguna pared dura todo el tiempo 3T. Cada salto
hace incorrectas como mucho 9 celdas, y lo mismo cada pared que dura un tramo T de un enlace. Así,
`N ≤ 9(J + W_t/T)`, con J el número de saltos y `W_t` el tiempo total de paredes. Cada salto aporta
`|ε|A` por su amplitud, y la integral sobre su instante dentro del bloque, con a lo sumo 3 enlaces,
aporta `3T`. Las paredes aportan `e^{−(1−δ)W_t}`. Con `T = ½ ln(1/|ε|)` y `δ` pequeño, el mínimo es el
de la fórmula. ∎

La constante 1/9 sale de contar con poco cuidado y se puede mejorar. Lo que importa es que
`τ → ∞` cuando `ε → 0`, uniformemente en n, β, Θ y m.
En 9.7 está la versión afinada, que es la que fija las constantes: celdas de 2 enlaces sin bloque,
contornos 4-conexos y al menos dos saltos por contorno.

**Lema Z16 (inducción de Zahradník, fases simétricas).** Sean `V(γ)` el contorno junto con su
interior, y `Z_c(V)` la función de partición diluida en V con borde de fase c (contornos γ′ con
`V(γ′) ⊂ V`). Se definen las actividades

    z(γ) = W(γ) Π_{componentes I del interior} Z_{c_I}(I) / Z_{c_ext}(I) ,

donde `c_I` es la fase interior y `c_ext` la exterior. Sea `a > 0` con

    Σ_{γ ∋ 0} |W(γ)| e^{3a|γ|} ≤ a/ν   (Kotecký–Preiss, con ν = Δ + 1) ,

donde Δ es el número de vecinos de la conectividad de los contornos y ν cuenta una celda junto con
sus vecinas. Para `|ε| < ε_2` existe tal a, porque el número de conjuntos conexos de N celdas que
contienen una dada es `≤ (e(Δ−1))^{N−1}` y τ crece sin cota (las constantes, en 9.7). Entonces, para
todo contorno que no dé la vuelta al cilindro,

    |z(γ)| ≤ |W(γ)| e^{2a|γ|} ≤ e^{−(τ − 2a)|γ|} ,

uniformemente en n, β, Θ, m y en ε complejo con `|ε| < ε_2`. Además, `Z_c(V) ≠ 0` y `log Z_c(V)`
viene dado por la serie de racimos convergente.

*Demostración.* Inducción sobre `|V(γ)|`. Hipótesis H(N): la cota vale si `|V(γ)| ≤ N`.
1. **Los interiores usan contornos más pequeños.** Sea γ con `|V(γ)| = N + 1`. Todo contorno γ′ de
   un interior I cumple `V(γ′) ⊊ V(γ)`. Por H(N), sus actividades cumplen la cota, y KP con peso a
   converge en I: `log Z_c(I) = Σ_{racimos X ⊂ I} φ_c(X)`.
2. **Energías libres truncadas iguales.** Sea `f_c^{(N)} = −Σ_{X ∋ 0} φ_c(X)/|X|`, sumando sobre los
   racimos de contornos con `|V| ≤ N` en el plano. La traslación por un sitio lleva el gas de la
   fase c al de la fase c − 1 y conserva los tamaños, así que `f_c^{(N)} = f^{(N)}` para todo c. Es
   el único lugar donde se usa k | n. **No hace falta truncar las actividades**, a diferencia del
   caso general de Zahradník, porque ninguna fase es metaestable.
3. **El cociente solo ve el borde.**
   `log Z_c(I) + f^{(N)}|I| = −Σ_{X ⊄ I, X ∩ I ≠ ∅} φ_c(X)|X ∩ I|/|X| + Σ_{X ⊂ I con contornos que no caben} …`.
   Un racimo que corta a I sin caber en I tiene que cruzar γ o tocarlo, así que es incompatible
   con γ. KP en la forma `Σ_{X ≁ γ} |φ(X)| ≤ a|γ|` acota los dos términos, sumando sobre todos los
   interiores, por `a|γ|`. Así, `Σ_I |log Z_{c′}(I) − log Z_c(I)| ≤ 2a|γ|`.
4. **Se cierra la inducción.** `|z(γ)| ≤ |W(γ)| e^{2a|γ|}`, que es la hipótesis con la que se
   escribió la condición KP. El peso e^{a|γ|} de KP más las dos e^{a|γ|} del cociente dan la
   `e^{3a|γ|}` de la condición.
5. **ε complejo.** Las actividades son analíticas en ε, y las cotas solo usan módulos. ∎

**Comparación con el paso 8.** La pérdida es `2a`, **una vez**: τ′ no depende del tamaño de V. Las
conversiones entre base y Fourier del paso 8 no existen aquí, porque la historia clásica es a la vez
la base y la variable del gas. Las constantes son malas: con el recuento de Z15, `ε_2` sale del
orden de `(8e)^{−18} ≈ 10⁻²⁴`. Afinado en 9.7–9.10: `ε_2 ≈ 2,3·10⁻³`; en tiempo continuo (9.12–9.13), `1,03·10⁻²` riguroso (Z17: `8,2·10⁻³`).

**Lema Z17 (del cilindro al multiplete; plan, hecho en 9.6).** Falta pasar del gas de contornos a (A),
con k | n. Son dos tipos de término:
- **Contornos que dan la vuelta en tiempo** (`|γ| ≥ β/T`): desaparecen cuando β → ∞. Con k | n no
  hay ninguna línea de defecto forzada.
- **Contornos que dan la vuelta en espacio** (`|γ| ≥ n/3`): son el túnel entre fases. Definen una
  matriz k×k `𝒯_{cc′}`, con `|𝒯| ≤ C e^{−τ′n/3}`, que en el sector de momento p da `t_p`.

El enunciado buscado, para ε complejo con `|ε| < ε_2` y β ≥ β_0(n), es

    Tr(T^m e^{−βH}) = Σ_{p} e^{2πi p m/k} e^{−β(n f(ε) + t_p(n,Θ,ε))} (1 + R) ,   |R| ≤ C e^{−β γ′} ,

con f y t_p analíticas. Esto da (A1) y (A2) para la energía: los k autovalores de menor parte real son
`λ_p = n f + t_p`, separados del resto por γ′. La marginal sale igual, insertando el observable de
arista. Es el análisis de tamaño finito de Borgs–Kotecký llevado al cilindro con giro, y en espíritu
es estándar. Pero hay que escribir cómo los contornos que dan la vuelta en espacio se reorganizan en
una exponencial en β (un desarrollo de racimos en la dirección del tiempo para la matriz de túnel),
y ese paso no está hecho.

**Estado tras 9.5.**
- La inducción de las actividades, que era el paso abierto de 9.1, queda **demostrada en espacio-
  tiempo** (Lemas Z15 y Z16), uniforme en n y para ε complejo. No queda ninguna pérdida por escala.
- El paso a (A), el Lema Z17, está en 9.6.
- La forma de vector propio de 9.1 queda como motivación. Ya no hace falta.

**9.6 Lema Z17: del cilindro al multiplete.**

Hipótesis: las de Z16, con k | n, ε complejo y `|ε| < ε_2`. Todas las constantes son uniformes en n,
Θ y ε, y no dependen de β. Se escribe `a/T =: γ_1` para la tasa de decaimiento por unidad de tiempo
que da Kotecký–Preiss (con `T = ½ ln(1/|ε|)` y `a` como en Z16, `γ_1 ≳ τ′/T − o(1) ≥ 1/10`).

**Enunciado.** Existen funciones `f_n(ε)` y `t_p(n,Θ,ε)` (con p ∈ Z_k), analíticas en `|ε| < ε_2`,
con

    |f_n(ε) − u_min| ≤ B ,   |f_n − f_{n′}| ≤ C e^{−a min(n,n′)} ,   |t_p| ≤ C n e^{−(τ′−a) n} ,

tales que, si `λ_p := n f_n + t_p`:
1. **El multiplete aislado.** En el sector de momento `q = p n/k`, `λ_p` es el único autovalor de H
   con `Re E < Re(n f_n) + γ′`, y es simple, donde `γ′ = γ_1/2`. En los demás momentos no hay ningún
   autovalor con `Re E < Re(n f_n) + γ′`.
2. **Con ε real,** los k niveles `λ_p` son los k más bajos de H y están separados del resto por
   `≥ γ′ − 2 max|t_p|`. Esto es (A1) y (G).
3. **La marginal.** La marginal de arista del proyector de Riesz de cada `λ_p`, y su media
   `ρ_n(Θ,ε)`, son analíticas, con `‖ρ_n − ρ_n(0)‖_1 ≤ B`. Junto con 1, esto es (A2).

*Demostración.*

**Paso 1: contornos que dan la vuelta.** En el toro `Z_n × [0, β)`, cerrado en tiempo con `T^m`, cada
contorno es de uno de tres tipos:
- **contraíble;**
- **espacial:** da la vuelta en la dirección del espacio y tiene `≥ n` celdas;
- **temporal:** da la vuelta en tiempo (o en las dos direcciones) y tiene `≥ β/T` celdas.

Los contornos espaciales son disjuntos y cada uno separa el cilindro, así que están ordenados
cíclicamente en tiempo: `γ_1 < ⋯ < γ_N`. Entre dos consecutivos queda un anillo `A_i` cuyas celdas
correctas tienen todas la misma fase `c_i`: el complemento de los contornos contraíbles en un
anillo es conexo. Si hay un contorno temporal, no hay espaciales (los cortaría), y su peso es
`≤ e^{−τ′β/T}`.

**Paso 2: el anillo.** `Z_{c}(A)`, la función de partición de los contornos contraíbles dentro de A,
se desarrolla en racimos por KP (Z16). Con altura h:

    log Z_c(A) = −f_n · n h + b_c(∂_−A) + b_c(∂_+A) + u(A) ,   |u(A)| ≤ C n e^{−γ_1 h} .

Aquí `f_n = −Σ_{X ∋ 0} φ(X)/|X|` se calcula en el cilindro *infinito* de circunferencia n e incluye
los racimos que dan la vuelta espacial. Por la traslación, no depende de c (Z16, punto 2). Por eso
`|f_n − f_∞| ≤ C e^{−a n}`: los racimos que dan la vuelta tienen ≥ n celdas. `b_c(∂)` reúne los racimos que tocan un solo borde, y `u` los que unen los dos. Todos son
analíticos en ε.

**Paso 3: un gas de eventos de túnel en tiempo.** Juntando 1 y 2,

    Tr(T^m e^{−βH}) = e^{−β n f_n} [ Σ_{N≥0} Σ_{γ_1<⋯<γ_N} tr_{ℂ^k}( S^m Π_i K(γ_i) ) Π_i (1 + u_i) + R_t ] .

En esta fórmula:
- `K(γ)_{c′c} = W(γ) e^{b_{c}(γ^−) + b_{c′}(γ^+)} e^{n f_n |γ|_t}` es la actividad del evento γ entre la
  fase c (abajo) y la c′ (arriba), donde `|γ|_t` es la extensión de γ en tiempo;
- `u_i = e^{u(A_i)} − 1` es la interacción entre eventos consecutivos;
- `S` es la permutación de etiquetas que induce T;
- `|R_t| ≤ C n e^{−τ′β/T}` recoge los contornos temporales.

Las cotas son:

    Σ_{γ espacial, empieza en [0,1)} ‖K(γ)‖ ≤ C n e^{−(τ′ − a)n} =: κ_n ,   |u_i| ≤ C n e^{−γ_1 h_i} .

**El punto clave: todas las K(γ) son circulantes en las etiquetas.** Sea γ′ el evento trasladado un
sitio. Entonces `K(γ′)_{c′−1,c−1} = K(γ)_{c′c}`. Agrupando γ con sus traslados, la suma de las
actividades de una clase de traslación es una matriz circulante. Las circulantes conmutan con S y
entre sí, y se diagonalizan a la vez con la transformada de Fourier en c. En el sector p, el gas es
**escalar**, con actividades `K_p(γ) = Σ_d K(γ)_{c+d, c} ω^{pd}`.

**Paso 4: el gas unidimensional.** Para cada p hay un gas en el círculo de longitud β. Sus polímeros
son las cadenas maximales de eventos unidos por las interacciones `u_i`, y su actividad por unidad de
tiempo es `≤ κ_n + C n e^{−γ_1}` (pequeña). KP en una dimensión da

    Σ_{N} Σ_{γ_1<⋯<γ_N} Π K_p(γ_i) Π (1 + u_i) = e^{−β t_p} (1 + R_p) ,   |R_p| ≤ C e^{−γ_1 β/2} .

Aquí `t_p = −Σ_{racimos que empiezan en [0,1)} φ_p` y `R_p` recoge los racimos que dan la vuelta en
tiempo. Su peso es `≤ (max(κ_n, C n e^{−γ_1}))^{β}`, que queda por debajo de `e^{−γ_1β/2}` para
`|ε|` pequeño. Además, `|t_p| ≤ κ_n` y t_p es analítica. Con la traza de `S^m`,

    Tr(T^m e^{−βH}) = e^{−β n f_n} [ Σ_{p ∈ Z_k} ω^{pm} e^{−β t_p} (1 + R_p) + R_t ] .

**Paso 5: de las trazas a los autovalores (multiplicidades enteras).** Sea `Π_q` el proyector sobre el
momento q. Promediando en m,

    F_q(β) := Tr(Π_q e^{−βH}) = e^{−β n f_n} [ δ_{q ∈ (n/k)Z} e^{−β t_{p(q)}} + G_q(β) ] ,
    |G_q(β)| ≤ C e^{−γ_1 β/2}   (β ≥ β_0(n)) .

Por otro lado, H es una matriz finita, así que `F_q(β) = Σ_E m_E e^{−βE}`, con `m_E ∈ {1, 2, …}` las
multiplicidades algebraicas de los autovalores de H en el sector q. (La traza de un bloque de Jordan
no tiene términos polinómicos.) Sean `s = Re(n f_n) + γ_1/2` y

    Δ(β) = Σ_{Re E < s} m_E e^{−βE} − δ e^{−β λ_p} .

Esta función es `O(e^{−βs})`: por las dos expresiones de F_q, es igual a `e^{−βnf_n}G_q − Σ_{Re E ≥ s} m_E e^{−βE}`.
Pero Δ es una suma finita de exponenciales con exponentes distintos y parte real `< s`. Si algún
coeficiente fuera no nulo, la parte con la menor parte real sería una función casi periódica por
`e^{−β Re E_min}`, que no es `O(e^{−βs})`. Así que todos los coeficientes son nulos. Hay exactamente un
autovalor con `Re E < s` si `q ∈ (n/k)Z`, que es `λ_p` con `m = 1`, y ninguno en los demás momentos.
Con `γ′ = γ_1/2`, eso es 1. Con ε real, H es autoadjunta, y 1 da 2.

**Paso 6: la marginal.** Como `λ_p` es simple y aislado en su sector,
`Π_q e^{−βH} = e^{−βλ_p} P_p + O(e^{−βs})`, con `P_p` el proyector de Riesz. Así,

    tr(P_p O) = lim_{β→∞} Tr(Π_q O e^{−βH}) / Tr(Π_q e^{−βH}) .

Se inserta O, el observable de arista, en el tiempo 0. El cociente es una suma de racimos conexos
al punto de inserción: los racimos en espacio-tiempo de Z16 y, en la dirección del tiempo, los
eventos de túnel que pasan por el instante 0. Es analítico en ε, con `|tr(P_p O) − tr(P_p(0) O)| ≤
C‖O‖|ε|^{c}`. La media sobre p da ρ_n. ∎

**Lo que se usa y lo que no.**
- **De Z16:** KP con τ′ fijo para los contornos contraíbles, en regiones arbitrarias (anillos, toros)
  y con ε complejo, y la igualdad de las energías libres por traslación.
- **De k | n:** que la traslación permute las fases (sin una línea de defecto forzada) y que los
  eventos sean circulantes.
- **No se usa:** ninguna hipótesis de no degeneración ni de autoadjunción, salvo en el punto 2. El
  argumento de multiplicidades enteras es el que convierte una asintótica en β en una afirmación
  espectral para ε complejo.

**Grado de detalle.** Los pasos 2–4 son el análisis de tamaño finito de Borgs–Kotecký (1990),
llevado a un toro espacio-tiempo con eventos de túnel matriciales, que aquí son circulantes. Están
escritos con sus cotas, pero no con todas las constantes. El paso 1 (topología de los contornos que
dan la vuelta) y el paso 5 son completos.

**Teorema Z⁗ (carga entera, k | n; robustez exponencial uniforme en todo n).** Para `S⁺S⁻` y, en
general, para la clase (C) con carga entera y k | n, existe `ε_2 > 0` tal que (A) vale con
`r_0 = ε_2`, para todo n ∈ kZ y toda Θ. Por el Teorema B,

    sup_{Θ,Θ′} ‖p(n,Θ) − p(n,Θ′)‖ ≤ C (|ε|/ε_2)^{⌈n/2⌉−1} ,   |ε| < ε_2 ,

uniformemente en n. *Demostración:* Z15, Z16 y Z17 dan (A1) y (A2); el Teorema B hace el resto. ∎

Con el recuento de 9.5, `ε_2 ∼ 10⁻²⁴`. Afinado en 9.7–9.10: **`ε_2 ≈ 2,3·10⁻³` para `S⁺S⁻`** (`8,2·10⁻³` en tiempo continuo, 9.13, limitado por Z17), unas 700
veces mejor que el radio de una escala del paso 8, y uniforme en todo n. Con esto, la ruta de
espacio-tiempo sustituye a la del paso 8 para k | n. El paso 8 sigue siendo útil para lo que
depende del marco de vestido por fases: la base para Gershgorin y el lado k ∤ n.

**9.7 Constantes afinadas** (`scripts/constantes_kp.py`). El recuento de 9.5 era muy grueso, y una
primera versión de esta sección olvidaba el factor de vecindad ν de Kotecký–Preiss, lo que daba un
`ε_2 ≈ 3,7·10⁻⁶` que no es una cota válida. Con ν incluido, cinco mejoras:

1. **Celdas de L enlaces sin bloque.** Una celda es incorrecta si algún salto toca sus L + 1 sitios
   en el tramo, o si alguno de sus enlaces es pared. Un salto vuelve incorrectas a lo sumo q celdas,
   con q = 3 si L = 1 y q = 2 si L ≥ 2. Los enlaces no se comparten entre celdas, así que cada
   celda de pared paga `e^{−T}` por separado.
2. **Contornos 4-conexos (Δ = 4).** Dos celdas correctas en diagonal comparten un sitio y el
   instante de frontera entre sus tramos. Si no hay saltos en ese sitio, el valor es el mismo, así
   que la fase también. Las etiquetas son entonces constantes en las componentes 8-conexas de celdas
   correctas, y por la dualidad 4/8 los contornos pueden tomarse 4-conexos. La entropía baja de
   `(7e)^{N−1}` a `(3e)^{N−1}`, y ν de 9 a 5.
3. **Suma exacta sobre el conjunto S de celdas con saltos.** Las celdas que no son de S ni cubiertas
   por S son paredes durante todo el tramo, así que

       |W_N| ≤ e^{−TN} [ (1 + Y e^{qT})^N − 1 − N(Y − Y_2) e^{qT} ] ,
       Y = y e^y ,   Y_2 = y² e^y/2 ,   y = A|ε| T L .

4. **S no vacío.** Un contorno contraíble sin saltos no existe.
5. **Al menos dos saltos.** Todo contorno contraíble necesita uno que crea las paredes y otro que las
   destruye. Por eso se quita el término de un único salto y se cuenta el de dos saltos en una
   celda, `Y_2`.

KP se toma en la forma de Z16, `Σ_N (e(Δ−1))^{N−1} W_N e^{3aN} ≤ a/ν`, optimizando T y a. Resultados
para `S⁺S⁻` (A = 4, δ = 0):

| L | Δ | ≥ 2 saltos | ε_2 | T | γ_1 = a/T en ε_2 | γ_1 en ε_2/10 |
|---|---|---|---|---|---|---|
| 2 | 8 | no | 1,3·10⁻⁶ | 4,3 | 0,03 | 0,12 |
| 2 | 4 | no | 6,8·10⁻⁶ | 3,4 | 0,04 | 0,14 |
| **2** | **4** | **sí** | **2,2·10⁻⁵** | 3,0 | 0,02 | 0,12 |
| 1 | 4 | sí | 2,5·10⁻⁶ | 2,7 | 0,01 | 0,10 |
| 3 | 4 | sí | 1,5·10⁻⁵ | 3,0 | 0,02 | 0,12 |

6. **Conteo exacto de los animales pequeños** (`scripts/animales.py`). Los conjuntos 4-conexos de N
   celdas que contienen una dada son `c_N = N a_N`, con `a_N` los poliominós fijos. Se cuentan
   exactamente con el algoritmo de Redelmeier hasta N = 13 (`a_13 = 1 903 890`). A partir de ahí se
   usa la cota de árboles: cada conjunto conexo tiene un árbol generador canónico, que es un subárbol
   del recubrimiento universal (la raíz tiene 4 hijos y los demás nodos 3). Por la inversión de
   Lagrange, `r_N = (4/m) C(3m+3, m−1)` con `m = N − 1`, que crece como `(27/4)^N` en lugar de
   `(3e)^{N−1}`. La serie de KP está dominada por los N pequeños, que ahora son exactos.

| L | Δ | ≥ 2 saltos | conteo | ε_2 | T | γ_1 en ε_2 | γ_1 en ε_2/10 |
|---|---|---|---|---|---|---|---|
| 1 | 4 | sí | exacto + árboles | 1,3·10⁻⁵ | 2,3 | 0,015 | — |
| **2** | **4** | **sí** | **exacto + árboles** | **7,2·10⁻⁵** | 2,5 | 0,024 | 0,13 |
| 3 | 4 | sí | exacto + árboles | 4,8·10⁻⁵ | 2,5 | 0,024 | — |

7. **Peso KP según la forma del contorno** (`scripts/animales.py` también escribe la distribución de
   perímetros). En lugar de ν = 5 para todo γ, se toma `a(γ) = a(N + θp)`, con p el número de celdas
   vecinas de γ. Un γ′ incompatible con γ corta `γ ∪ ∂γ`, que tiene N + p celdas, así que KP pide
   `S ≤ a·min_γ (N + θp)/(N + p)`, con `S = Σ_{γ∋0} W(γ) e^{3a(N+θp)}`. Como `p ≤ 2N + 2`, el peor caso
   es N = 1 y el mínimo vale `(1 + 4θ)/5`; θ = 0 es la versión con ν. Para N ≤ 13 se usa la
   distribución exacta de (N, p), y más allá `p = 2N + 2` con la cota de árboles. El cociente de Z16
   cuesta ahora `e^{2a(γ)}`, con la misma prueba.

| θ | 0 | 0,3 | 0,6 | 1 |
|---|---|---|---|---|
| ε_2 (L = 2) | 7,2·10⁻⁵ | 8,2·10⁻⁵ | 8,6·10⁻⁵ | **8,8·10⁻⁵** |

- **`ε_2 ≈ 9·10⁻⁵` para `S⁺S⁻`, uniforme en todo n** (L = 2, Δ = 4, al menos dos saltos, conteo
  exacto, θ = 1, T ≈ 2,5). Es unas 30 veces el radio de una escala de Kirkwood–Thomas del paso 8
  (≈ 3·10⁻⁶), y sin el límite `n ≤ n_*`. La tasa en tiempo γ_1 es pequeña en ε_2 (≈ 0,01), y Z17
  solo necesita que sea positiva.
- **Lo que más ayuda**, desde el recuento de 9.5: la conectividad (×5), los dos saltos (×3), el
  conteo exacto (×3,3) y el peso según la forma (×1,2). Las celdas de L = 2 siguen siendo el óptimo.
- **Lo que queda.** La única mejora de estructura que se ve es usar que, en un tramo sin saltos,
  una fila del contorno con la misma fase a los dos lados tiene al menos dos paredes. Por la
  aritmética mod k, una sola pared no cierra. Eso abarata los pares ligados de vida larga
  (`e^{−2T}` por tramo en lugar de `e^{−T}`), pero no las burbujas de otra fase, que son paredes
  físicas. Pide enumerar los contornos con sus etiquetas, y el factor esperado es ≲ 2. Frente a la
  contracción observada hasta ε ≈ 0,2, la pérdida queda en unos tres órdenes. Lo que falta para
  cerrarla no está en el recuento: está en el propio esquema de celdas y en la cota de Peierls.

**9.8 Peierls con la integral exacta de los extremos.** Z15 cuenta un factor T por cada salto, que
es la integral de su instante a lo largo del tramo. Para dos saltos de cada contorno se puede hacer
mejor:
- **El primero (creación).** Es el más temprano del contorno y está en la primera fila. Entre él y el
  salto siguiente, la configuración difiere del patrón solo en los dos sitios del salto, así que hay
  exactamente `E_min` paredes (`E_min = 3` para `S⁺S⁻`: un salto desde un patrón deja los incrementos
  +2, −1, +2). La integral de su instante da `∫ dt_1 e^{−E_min(t_2 − t_1)} ≤ 1/E_min` en lugar de T.
- **El último (aniquilación).** Lo mismo, con el intervalo que lo precede.

Los dos intervalos son distintos, salvo en un contorno de una sola fila con exactamente dos saltos,
lo que solo ocurre con N ≤ 4 celdas. En un contorno de varias filas con dos saltos, los dos instantes
caen en tramos distintos y la exponencial se separa. Además, la primera y la última fila no pueden
tener celdas de pared: una pared que dura todo el tramo existía antes, o sigue después, y la celda
vecina en tiempo sería del contorno. Así, la celda de la creación es de S y está en la primera fila
(a lo sumo r_f opciones), y la de la aniquilación en la última (r_l opciones). Con `ρ = 1/(E_min T)`:

    |W(γ)| ≤ W_{9.7}(γ) · min(1, κ(γ)) ,   κ = r_f r_l ρ²  (varias filas, o una fila con N ≥ 5),
                                            κ = 2N ρ      (una fila, N ≤ 4) .

El factor 2 del último caso cubre la integral `∫∫_{t_1<t_2} e^{−3(t_2−t_1)} ≤ T/3` frente a `T²/2`.
`scripts/animales.py` escribe la distribución exacta de `(N, p, r_f, r_l, una fila)` hasta N = 13
(`scripts/filas.txt`). Más allá se usa `r_f r_l ≤ N²` con la cota de árboles. Resultados (L = 2,
Δ = 4, al menos dos saltos):

| θ | ε_2 | T | a | γ_1 en ε_2 | γ_1 en ε_2/10 |
|---|---|---|---|---|---|
| 0 | 1,3·10⁻⁴ | 2,65 | 0,040 | 0,015 | 0,11 |
| 1 | **1,5·10⁻⁴** | 2,62 | 0,010 | 0,004 | 0,04 |

- **`ε_2 ≈ 1,5·10⁻⁴` para `S⁺S⁻`, uniforme en todo n:** un factor 1,7 sobre 9.7, y unas 50 veces el
  radio de una escala de Kirkwood–Thomas.
- **Por qué no rinde más.** `ρ ≈ 0,13`, así que el factor por contorno, `r_f r_l ρ²`, es pequeño. Pero
  ε_2 solo depende de él a través de la raíz de la serie, y los contornos que dominan en ε_2 tienen
  filas extremas de 2–3 celdas.
- **Lo que queda en la cota de Peierls:**
  - las filas intermedias con saltos, donde la cota toma energía 0 y en realidad hay `E(t) ≥ 2`,
    salvo en los huecos entre una aniquilación y una nueva creación;
  - las filas sin saltos con la misma fase a ambos lados, que tienen al menos dos paredes.

  Las dos piden seguir la energía a lo largo del contorno, no celda a celda, lo que ya es una cota
  de tipo «denominadores de energía». Frente a la contracción observada hasta ε ≈ 0,2, la pérdida
  queda en unos tres órdenes. (Hecho en 9.9.)

**9.9 Peierls por denominadores de energía, fila a fila** (`scripts/constantes_denom.py`). Las
cotas 9.7–9.8 dan a cada celda con saltos energía 0, y a cada salto el factor T de su instante.
Aquí se sigue la energía dentro de cada fila del contorno (sus celdas en un tramo), y la integral
sobre los instantes se hace de verdad.

*Notación.* Celdas de 2 enlaces y 3 sitios. Un **tramo de fila** es un conjunto maximal de celdas
consecutivas del contorno en una fila. Un sitio está **tocado** si algún salto del tramo de tiempo
lo mueve. Un **segmento** es un conjunto maximal de sitios tocados consecutivos, y tiene al menos
2 porque cada salto mueve dos sitios vecinos. Una **celda de pared** es una celda del contorno sin
sitios tocados.

*Hechos.*
- **(S1) Extremos intocados.** Los sitios de los extremos de un tramo de fila los comparte con una
  celda correcta, y por tanto ningún salto los toca. Una fila de una celda no tiene segmentos: el
  único sitio interior no puede tocarse solo.
- **(S2) Ley de conservación por segmento.** Un sitio intocado tiene valor constante en el tramo de
  tiempo. Así, la suma de las desviaciones `d_j − 1` de los incrementos en los enlaces de un
  segmento (de intocado a intocado) es constante. Si el segmento pasa por E = 0 en algún instante,
  esa suma es 0 mod k, y entonces `E(t) = 0` o `E(t) ≥ 2` en todo el tramo. Si no pasa nunca por
  E = 0, `E(t) ≥ 1` todo el tramo.
- **(S3) Creación y aniquilación.** Un salto desde E = 0 deja exactamente 3 paredes en el segmento
  (incrementos +2, −1, +2). Lo mismo ocurre antes de un salto que lleva a E = 0. No hay dos
  intervalos seguidos con E = 0.
- **(S4) Factorización.** La energía de la fila es la suma de las de sus segmentos más las paredes
  de los enlaces entre sitios intocados, que son constantes. Cada celda de pared tiene al menos
  una. Los instantes de los saltos de segmentos distintos se integran por separado.
- **(S5) Bordes en tiempo.** En la primera fila, cada segmento empieza con E = 0; en la última,
  acaba con E = 0; ninguna de las dos tiene celdas de pared (9.8). Una fila intermedia de una
  sola celda sin saltos tiene al menos 2 paredes: los dos lados son la misma componente del
  complemento, con una sola fase.

*La cota.* Sea un segmento con m saltos. Sus m + 1 intervalos se clasifican en: P con E = 0, X₃
vecinos de uno con E = 0 (con E ≥ 3) y X₂ restantes (con E ≥ 2). La integral sobre los instantes
ordenados en el tramo de tiempo, con los excitados extendidos a [0, ∞), cumple

    J ≤ T^{P−1}/(P−1)! · 3^{−X₃} 2^{−X₂}   (P ≥ 1) ,      J ≤ e^{−2T} T^m/m!   (P = 0) ,

y además, en una fila intermedia, `J ≤ e^{−T} T^m/m!` si el segmento no pasa nunca por E = 0.
Siempre `J ≤ T^m/m!`. Se toma el máximo sobre las sucesiones de intervalos admisibles: sin dos con
E = 0 seguidos, y con el primero (último) con E = 0 en la primera (última) fila. Esto es una cota
para cada historia, así que vale sumada. El peso de un segmento de λ sitios es

    F(λ) = Σ_m (A|ε|)^m · #{sucesiones de m enlaces internos que tocan los λ sitios} · J(m) ,

y el de un tramo de fila sale de una matriz de transferencia sobre sus sitios: segmentos con peso
F, celdas de pared con `e^{−T}` (prohibidas en la primera y la última fila). El peso de una fila
de r celdas se acota por el máximo sobre sus particiones en tramos. Se toma también el mínimo con
la cota de fila entera (E(t) = 0 o ≥ 2 en toda la fila). El peso del contorno es el producto de
sus filas.

*Datos combinatorios.* `scripts/animales.py` escribe el perfil de filas de cada poliominó hasta
N = 13 (`scripts/perfiles.txt`: primera, última e intermedias). Los pesos suman exactamente
Σ c_N = 32 791 330. Más allá se usa la cota de árboles con `h^N`, donde h es el supremo por celda
de `G(r)^{1/r}`. Se calcula hasta r = 40, con la razón `G(40)/G(39)` como tasa asintótica: este
paso es numérico, y la tasa converge con r. La matriz de transferencia se ha comprobado contra la
enumeración directa de los patrones de sitios tocados en tramos de hasta 4 celdas.

*Resultado* (S⁺S⁻, A = 4, KP con `a(γ) = a(N + θp)`):

| θ | ε_2 | T | a | γ_1 en ε_2 | γ_1 en ε_2/3 | γ_1 en ε_2/10 |
|---|---|---|---|---|---|---|
| 0 | 1,9·10⁻³ | 4,1 | 0,034 | 0,008 | 0,07 | 0,11 |
| 1 | **2,0·10⁻³** | 4,1 | 0,010 | 0,002 | 0,02 | 0,04 |

- **`ε_2 ≈ 2·10⁻³` para `S⁺S⁻`, uniforme en todo n:** unas 14 veces el de 9.8 y unas 700 veces el
  radio de una escala de Kirkwood–Thomas. Frente a la contracción observada hasta ε ≈ 0,2, la
  distancia baja de tres órdenes a **dos**.
- **Qué cambia en el mecanismo.**
  - Una fila de una celda solo puede ser de pared.
  - Las filas con saltos pagan `e^{−2T}` o los denominadores 1/2 y 1/3, en vez de energía 0.
  - Las fluctuaciones independientes de una fila larga se factorizan por segmentos (S2, S4). Sin
    esto, la cola diverge ya en ε ≈ 10⁻³ (tasa por celda 0,29 frente a 0,11).
  - El T óptimo sube de 2,6 a 4,1: las celdas de pared son más caras y los saltos ya no pagan T.
- **Lo que queda** (corregido en 9.10). No son las celdas de pared: con T ≈ 4, `e^{−T} ≈ 0,017`,
  mientras que la tasa por celda de las filas es ≈ 0,08–0,12. Lo que limita es la cola N > 13,
  acotada con árboles, que pesa más que toda la parte exacta (9.10).

**9.10 La entropía de la cola, por percolación.** *Diagnóstico* en `ε_2 = 2·10⁻³`, T = 4,1, a = 0,01:

| parte de la serie KP | valor |
|---|---|
| exacta, N ≤ 13 | 4,1·10⁻³ |
| cola N > 13 con árboles, `(27/4)^N h^N` | **5,2·10⁻³** |
| razón real entre N = 12 y 13 | 0,41 |
| razón de la cota de árboles | 0,86 |

La parte exacta está dominada por los contornos de una sola fila de 2–3 celdas (las fluctuaciones
del vacío: un salto y su inverso). La serie real cae como ≈ 4,06·h por celda, con 4,06 la constante
de crecimiento de los poliominós. La cota de árboles usa 27/4.

*Cota por percolación.* Para la percolación de sitios con densidad x,
`Σ_{S∋0} x^{|S|}(1−x)^{p(S)} = P_x(0 abierto, |C| < ∞) ≤ x`, con p el perímetro de sitios. Restando los
términos exactos (N ≤ 13, de `scripts/perimetros.txt`) queda `δ(x) = P_x(13 < |C| < ∞)`. Como el peso
de un contorno con N > 13 es `≤ h^N e^{3a(N+θp)}` y `p ≤ 2N + 2`,

    Σ_{N>13} … ≤ (e^{3aθ}/(1−x))² · δ(x)     si   h e^{3a(1+2θ)} ≤ x(1−x)² .

Se toma el menor x que cumple la condición. δ se calcula con 40 cifras:

| x | 0,10 | 0,15 | 0,20 | 0,25 | 1/3 |
|---|---|---|---|---|---|
| δ(x) | 1,8·10⁻⁷ | 2,1·10⁻⁵ | 4,5·10⁻⁴ | 3,9·10⁻³ | 4,0·10⁻² |

En ε = 2·10⁻³ la cola baja de 5·10⁻³ a ≈ 7·10⁻⁴. Resultados (`scripts/constantes_denom.py`, COLA = 1):

| θ | ε_2 | T | a | γ_1 en ε_2 | γ_1 en ε_2/10 |
|---|---|---|---|---|---|
| 0 | 2,1·10⁻³ | 3,8 | 0,034 | 0,009 | 0,11 |
| 1 | **2,3·10⁻³** | 4,0 | 0,010 | 0,003 | 0,04 |

- **`ε_2 ≈ 2,3·10⁻³` para `S⁺S⁻`, uniforme en todo n.** La mejora es pequeña (×1,15), porque en
  cuanto se quita la cola aparecen dos límites a la vez, ya en ε ≈ 3·10⁻³:
  1. **La tasa h de las filas largas** llega al umbral `4/27 = max x(1−x)²` (h ≈ 0,15). Ninguna cota
     que solo use el perímetro puede pasar de ahí. Haría falta la forma de las filas en la cuenta
     (una fila de r celdas es un solo poliominó, no `(27/4)^r`).
  2. **La parte exacta** ya no cumple KP. La dominan las fluctuaciones del vacío de una fila de
     2–3 celdas, cuyo peso `∼ (A|ε|)² T/3` entra en `Σ z e^{3a|γ|} ≤ a/5` con a pequeño.
- **Lo que queda, en orden de rendimiento esperado:**
  - **Resumar las fluctuaciones del vacío antes de los contornos.** Es un cambio de base de
    segundo orden: el patrón vestido `e^{εS}c`, o la energía efectiva de Borgs–Kotecký–Ueltschi.
    Así, los contornos de una fila corta dejan de existir como actividades y la cuenta empieza
    en ε³.
  - **Contar los contornos con la forma de las filas** para la cola: matriz de transferencia sobre
    filas con la condición de que filas consecutivas se tocan, que es necesaria para la
    conexión.
  - Las dos cosas son cambios del esquema, no de las constantes. Mientras tanto, frente a
    ε ≈ 0,2, la distancia sigue siendo de dos órdenes.

**9.11 Resumar el vacío: qué se consigue y qué no.**

*Paso 1: el gas de contornos de una fila, resumado antes (riguroso, sin ganancia).* Los contornos
de una sola fila no tienen interior, así que `z = W`, y son ≈ 70 % de la parte exacta: las
fluctuaciones del vacío dentro de un tramo. Se integran primero, como un gas de polímeros con su
propio peso KP b:

    σ = Σ_{γ∋0, una fila} |W| e^{b(N+θp)} ≤ b · min (N+θp)/(N+p) .

Los contornos de varias filas ven ese gas solo a través de dos efectos:
- la exclusión fuera de `γ ∪ ∂γ`;
- los términos de borde de los interiores.

Juntos dan un factor `e^{3σ(N+p)}`: es la cota de KP con un conjunto de prueba, como en Z16. Las
energías libres del gas son iguales en todas las fases por la traslación. El KP de los contornos de
varias filas ya no tiene que absorber las fluctuaciones (`scripts/constantes_denom.py`,
VACIO = 1).

*Resultado.* El gas de una fila converge con holgura: `σ ≈ 3·10⁻³` aun en ε = 3·10⁻³. Pero ε_2 no
se mueve:
- θ = 1: factible en 2,3·10⁻³, no en 2,6·10⁻³;
- θ = 0: 2,0·10⁻³, frente a 2,1·10⁻³ sin el paso.

*Por qué: la tasa por celda va como √ε, y es intrínseca a las celdas.*

| ε | 2·10⁻³ | 3·10⁻³ | 4·10⁻³ |
|---|---|---|---|
| h (tasa por celda de las filas largas) | 0,117 | 0,150 | 0,181 |

Al multiplicar ε por 1,5, h se multiplica por 1,28 (√1,5 = 1,22). Viene de los pares que cruzan
el borde entre dos tramos: la creación en una fila (b/3) y la aniquilación en la siguiente (b/3).
El par vale `(b/3)²`, que es correcto, porque `∫∫_{t_1<ℓT<t_2} e^{−3(t_2−t_1)} = 1/9`. Pero vuelve
incorrectas unas 4 celdas, así que paga ~√ε por celda. Esos pares, colocados lado a lado, forman
contornos grandes de varias filas con entropía de poliominó. Con 4,06 por celda (y 4/27 en la cota
por percolación), KP falla cuando `h ≈ 0,15`, es decir, en ε ≈ 3·10⁻³. **Ninguna mejora del
recuento con celdas pasa de ahí.** Hay que dejar de declarar incorrecta una celda por una
fluctuación cerrada.

*Paso 2 (plan): contornos en tiempo continuo.*
- **Definición.** Un contorno es una componente conexa del conjunto excitado: las paredes
  (enlace × intervalo) más los tres enlaces que cambia cada salto, en su instante. Dos puntos están
  conectados si coinciden en tiempo y sus enlaces distan ≤ 1. Las energías y las amplitudes se
  factorizan sobre las componentes.
- **Peierls.** Para un contorno contraíble, el corte a tiempo t es 0 mod k (los arcos del
  complemento unen puntos de la misma componente, con una sola fase), así que `E(t) ≥ 2` entre su
  primer y su último salto. Además, E = 3 tras la creación y antes de la aniquilación. Cada salto
  siguiente cae a distancia ≤ 2 de una pared viva: ≤ 5E elecciones contra el denominador 1/E. El
  factor por salto es ≈ `5A|ε|/(1−3a)`, sin T y sin √ε. Una fluctuación del vacío pesa `b²/9` por
  unidad de tiempo y de enlace.
- **KP.** Con `a(γ) = a(∫E dt + κ·#saltos)` y la versión continua de Kotecký–Preiss (Ueltschi),
  la incompatibilidad empieza siempre en un salto de uno de los dos contornos. La condición queda
  en dos densidades:
  - por salto, la de contornos vivos (volumen espacio-tiempo), `5A_1 ≤ aκ`;
  - por unidad de tiempo de pared, la de saltos, `15A_2 ≤ a`.
- **Lo que falta para que sea un lema.**
  - Contar las **fusiones**: piezas nacidas por separado que luego se tocan. El crecimiento
    ordenado en el tiempo no las ve, y se cuentan como un árbol de piezas, cada una con sus propios
    denominadores, que se factorizan porque las energías se suman.
  - Rehacer Z16 y Z17 en tiempo continuo: las pruebas son las mismas, con sumas en lugar de
    integrales.
- **Estimación** sin fusiones (no demostrada): **ε_2 ≈ 1·10⁻²** (a ≈ 0,07, κ ≈ 0,6). Con las
  fusiones bajará, así que lo realista es `5·10⁻³–10⁻²`: un factor 2–4 sobre 9.10, no un orden.

*Balance de la vía de constantes.* De `10⁻²⁴` (9.5) a `2,3·10⁻³` (9.10), y quizá `∼ 10⁻²` en tiempo
continuo. Frente a ε ≈ 0,2 observado, lo que queda está en las constantes combinatorias de KP: la
entropía ~5 por salto y el factor e de KP, no en la física de las fluctuaciones. Bajarlo de verdad
pide una transformación exacta a segundo orden (Schrieffer–Wolff local, que quita los pares de
creación y aniquilación del desarrollo) o una cota no perturbativa distinta. Para el teorema,
`ε_2 ≈ 2·10⁻³` ya basta. Que sea cuantitativamente cercano a 0,2 es un objetivo aparte.
(Hecho en 9.12: la cota completa, con fusiones, da más que esta estimación, porque las razones por
salto y las condiciones de KP son más finas que las de arriba.)

**9.12 Contornos en tiempo continuo** (`scripts/constantes_continuo.py`).

*Definiciones.* En una historia del desarrollo de Dyson (S⁺S⁻, sin parte diagonal: δ = 0):
- Cada salto v está en un enlace j y un instante t_v, y cambia los incrementos de los enlaces
  j − 1, j, j + 1.
- El **conjunto excitado** es

      X = {(i, t) : i es pared en t} ∪ ⋃_v {j_v − 1, j_v, j_v + 1} × {t_v} .

  Dos puntos de X están unidos si tienen el mismo instante y enlaces a distancia ≤ 1, o si son el
  mismo enlace en instantes contiguos dentro de X.
- Un **contorno** γ es una componente conexa de X. Una **racha** es un conjunto maximal de paredes
  de γ consecutivas en un instante.
- Las paredes se reparten entre contornos y las amplitudes son locales, así que el peso se
  factoriza: `|W(γ)| ≤ (A|ε|)^m ∫ e^{−∫E_γ dt}`, con m el número de saltos y E_γ(t) las paredes de γ.
- En el complemento de X, todos los enlaces son ascendentes y la fase es constante en cada
  componente. Las etiquetas, los interiores y la compatibilidad (distancia ≥ 2) son como en 9.5.

**Lema Z15c (Peierls por denominadores).** Sea γ un contorno que no da la vuelta en tiempo. Ordenando
sus saltos en el tiempo, un **marco** en el instante t es el conjunto de paredes de γ vivas en t que
descienden de un mismo nacimiento: dos paredes están en el mismo marco si están unidas dentro de
`γ ∩ {tiempo ≤ t}`.
1. **Cada marco suma 0 mod k.** Sean A, B, … los marcos que toca un salto en el enlace j (los que
   tienen una pared en `[j−2, j+2]`). Antes del salto, toda pared de los enlaces j−1, j, j+1 es de
   uno de ellos: una pared de otro contorno estaría unida a las marcas del salto. El salto suma
   `(+1, −2, +1)`, de suma nula, y el marco resultante suma lo que sumaban A, B, … Un nacimiento suma
   0. Las desviaciones de las paredes son no nulas, así que **cada marco tiene ≥ 2 paredes** y
   `E_γ(t) ≥ 2` entre el primer y el último salto. *Corrección a la versión anterior:* lo que suma 0
   es el marco, no cada racha. Una racha que separa el exterior de un interior de fase `c′ ≠ c`
   suma `c′ − c` y puede tener una sola pared (ver 9.13 a).
2. **Nacimiento y muerte.** Un marco nace con exactamente 3 paredes en una racha: un salto desde el
   patrón deja (+1, −2, +1), no nulas con k ≥ 3. Un marco que muere en un salto tenía exactamente
   esas 3 paredes, porque el salto solo cambia tres enlaces. Si un marco muere, el contorno se
   desconecta, salvo que sea el único marco. Por eso γ muere desde `E = 3, r = 1, f = 1`.
3. **Integral en el tiempo.** Los intervalos entre saltos consecutivos de γ se integran a [0, ∞).
   Con el peso KP `e^{3a∫E}` incluido, cada intervalo de energía E da `1/((1 − 3a)E)`.
4. **Posiciones.** Un salto que toca un marco A cae en `≤ P_A := E_A + 4r_A` enlaces, donde `r_A` es
   el número de rachas de A, porque `[j−2, j+2]` corta alguna racha. Un salto cambia E en `[−3, 3]` y
   r en `[−2, 2]`. Para r: un comienzo de racha en i depende de los enlaces i − 1 e i, así que solo
   cambian los comienzos en `j−1, …, j+2`, y en cuatro posiciones seguidas caben a lo sumo dos.
5. **Fusiones.** Un salto que no toca ningún marco es un nacimiento separado, y su posición se
   difiere. Un salto que toca dos marcos A, B (o tres) fija su posición relativa a A (`P_A`
   elecciones) y el desplazamiento de B (`P_B`). Sumando sobre los pares de marcos, por la
   desigualdad de Maclaurin,

       Σ_{A<B} P_A P_B ≤ C(f,2)(P/f)² =: e₂ ,   Σ_{A<B<C} P_A P_B P_C ≤ C(f,3)(P/f)³ =: e₃ ,

   con `P = E + 4r` (r el total de rachas). Un salto toca a lo sumo tres marcos, porque las paredes
   de marcos distintos distan ≥ 2 y `[j−2, j+2]` tiene cinco enlaces. *Corrección:* la versión
   anterior usaba `max (E_A+4)(E_B+4)`, que ni elige el par ni cuenta las rachas de cada marco.

Con esto, la suma anclada (primer salto en el enlace 0, `t ∈ [0,1)`) cumple `W_anc ≤ b′ V(3, 1, 1)`,
con `b′ = A|ε| e^{3aκ}`. Aquí V(E, r, f), con f el número de marcos vivos, es el menor punto fijo de

    V(E,r,f) = b′/((1−3a)E) · [ 1_{(E,r,f)=(3,1,1)} + P max V(·,·,f) + V(E+3, r+1, f+1)
                                + e₂ max V(·,·,f−1) + e₃ max V(·,·,f−2) ] ,

con los máximos sobre `E′ ∈ [E−3, E+3]`, `r′ ∈ [r−2, r+2]` y los estados posibles
(`2f ≤ E`, `f ≤ r ≤ E`). La cota usa el máximo sobre los estados alcanzables, y no la suma, porque
cada posición fija el estado siguiente. ∎

**KP en tiempo continuo.** Se toma `a(γ) = a(∫E_γ dt + κ m)` y la versión de Kotecký–Preiss para
polímeros en un espacio con medida (Ueltschi 2004). Un γ′ incompatible con γ empieza a tocarlo en un
salto de uno de los dos (9.13 b):
- en un salto de γ, si γ′ tiene una pared viva en uno de los 5 enlaces cercanos;
- en un salto de γ′, si cae a distancia ≤ 2 de una pared de γ: `≤ E + 4r ≤ 5E_γ` enlaces por
  unidad de tiempo. *Corrección:* antes ponía 3E, con rachas de ≥ 2.

Así

    Σ_{γ′≁γ} |z(γ′)| e^{a(γ′)} ≤ 5 m_γ M_1 + 5 (∫E_γ dt) M_2 ≤ a(γ)   si   5M_1 ≤ aκ  y  5M_2 ≤ a ,

con `M_1 = Σ_anc |W| e^{3a(γ)} ∫E dt` y `M_2 = Σ_anc |W| e^{3a(γ)} m`. Los momentos se acotan con
`x ≤ e^{sx}/(es)`: `M_1 ≤ W_anc(3a + s)/(es)` y `M_2 ≤ W_anc(b′e^s)/(es)`. El factor `e^{2a(γ)}` de
los interiores (Z16) ya está en el `3a`.

*Comprobación.* Para ε pequeño, `W_anc → b²/3`, el par de un enlace: `5,38·10⁻⁶` frente a
`b²/3 = 5,33·10⁻⁶` en ε = 10⁻³.

*Resultado* (S⁺S⁻, A = 4; recuento corregido, con r en el estado):

| E_max (truncamiento de V) | ε_2 | a | κ | s | M_1 | M_2 |
|---|---|---|---|---|---|---|
| 40 | 1,31·10⁻² | 0,020 | 1 | 0,2 | 0,0032 | 0,0037 |
| 60 | 1,28·10⁻² | 0,020 | 1 | 0,2 | 0,0030 | 0,0035 |
| 80 | **1,25·10⁻²** | 0,020 | 1 | 0,2 | 0,0029 | 0,0034 |

- **Estimación: `ε_2 ≈ 1,2·10⁻²` para `S⁺S⁻`, uniforme en todo n.** El valor 1,25·10⁻² es el de E_max = 80
  con V = 0 más allá. Las diferencias entre truncamientos decrecen con razón ≈ 0,7, y el límite
  extrapolado es ≈ 1,2·10⁻². **La cota rigurosa, con la cola acotada (9.13 d), es `ε_2 = 1,03·10⁻²`.**
- **La primera versión daba 1,8·10⁻².** Esa cuenta tenía tres fallos:
  - suponía rachas de ≥ 2 paredes;
  - en las fusiones, ni elegía el par de marcos ni contaba sus rachas;
  - en KP ponía 3E en lugar de 5E.

  Corregidos, el radio sigue unas 5 veces por encima del 2,3·10⁻³ de 9.10 con celdas, y frente a ε ≈ 0,2
  queda algo más de un orden.
- **Por qué rinde.** No hay celdas, así que no hay T, ni √ε por celda, ni entropía de poliominós.
  Una fluctuación del vacío pesa `b²/3` por unidad de tiempo y enlace. La entropía de la forma está en
  las posiciones, y los denominadores la compensan.
- **Qué es numérico.** Nada que afecte a la cota. El punto fijo se calcula por iteración en punto
  flotante, en E ≤ E_max. Para E > E_max se usa la supersolución de 9.13 d), que convierte el
  truncamiento en una cota superior. Después se certifica todo con aritmética de intervalos (9.13 e). La
  rejilla de parámetros solo puede hacer el radio más pequeño de lo que podría ser.
- **Detalles:** en 9.13.

**9.13 Los detalles de 9.12** (`scripts/constantes_continuo.py`, modos `z16` y `z17`; certificado en
`scripts/certifica_continuo.py`).

**a) Las etiquetas: una fase en la capa de borde.** Z16 necesita que cada componente del
complemento de γ tenga una sola fase junto a γ. Las rachas de γ ya no lo necesitan: van por marcos
(Z15c, 1).

*Geometría.* Al enlace i se le asigna la franja `[i−½, i+½] × ℝ`. A cada pared viva en un intervalo
de tiempo se le asigna un rectángulo, y a cada salto en el enlace j, el segmento horizontal
`[j−3/2, j+3/2] × {t_v}`. Sea X̂ la unión de estos conjuntos.
- **Los intervalos de pared son cerrados.** Una pared del enlace i aparece o desaparece en un salto
  en i − 1, i o i + 1, y ese salto marca el enlace i en ese instante. Así, X̂ es cerrado, y en una
  región acotada es una unión finita de piezas convexas cerradas.
- **Las uniones de 9.12 son intersecciones.** Mismo instante y enlaces a distancia ≤ 1, o mismo
  enlace con tiempos contiguos, es lo mismo que cortarse las piezas. Por tanto, los contornos son
  las componentes conexas de X̂, y el γ̂ de un contorno que no da la vuelta es compacto (en el
  levantamiento al plano).
- **En el complemento abierto Ŷ = ℝ² ∖ X̂ la fase es localmente constante.** En el interior de una
  franja, la fase del enlace solo cambia con un salto en i − 1, i o i + 1, que marca i. Sobre la
  recta `x = i + ½`, los enlaces i e i + 1 no son pared y comparten el sitio i + 1, así que tienen la
  misma fase. Por tanto, la fase es constante en cada componente de Ŷ.

*Lema (capa de borde).* Sea U una componente de `ℝ² ∖ γ̂`, y sea `U_δ = {p ∈ U : d(p, γ̂) < δ}`.
Para δ pequeño, `U_δ ⊂ Ŷ` y U_δ es conexo, así que la fase es constante en U_δ. Esa fase es la
etiqueta `c_U`.

*Demostración.*
- **U_δ ⊂ Ŷ.** Los demás contornos son cerrados, disjuntos de γ̂ y, cerca de γ̂, finitos, así que
  distan de γ̂ algo positivo, mayor que δ.
- **U_δ es conexo.** ∂U ⊂ γ̂ es conexo: en la esfera, cada componente del complemento de un compacto
  conexo tiene frontera conexa. Además, ∂U es una unión finita de segmentos. Una frontera poligonal
  conexa se recorre con un camino cerrado que deja U siempre al mismo lado, y U_δ es la franja que
  sigue a ese camino. Esto incluye los puntos de pellizco, donde U toca a γ̂ por dos sectores: el
  camino pasa dos veces por ellos. ∎

*Consecuencias.*
- La etiqueta exterior y las interiores están bien definidas. Así, `z(γ) = W(γ) Π_I Z_{c_I}(I)/Z_{c_ext}(I)`
  tiene sentido.
- Una racha cuyos dos vecinos están en la misma U suma 0. Una racha que separa U de U′ suma
  `c_{U′} − c_U`, y puede tener una sola pared: por ejemplo, el borde de un dominio de fase c + 1, de
  longitud múltiplo de k, dentro de la fase c. Ese caso es el que la versión anterior de Z15c
  excluía sin motivo.
- No hace falta ninguna «conexión de esquinas» aparte. Las piezas de X̂ son cerradas y la
  conectividad es topológica, así que las esquinas de γ̂ dejan U_δ conexo.

**b) KP continuo y la inducción de Z16.**

*KP para polímeros con medida* (Ueltschi, *Cluster expansions and correlation functions*, Moscow
Math. J. 4 (2004)). Sea `dμ(γ)` la medida que suma sobre los datos discretos (enlaces, tipos de
salto) e integra los instantes de los saltos. Sea `≁` una relación de incompatibilidad, medible,
simétrica y reflexiva. Supóngase que, para todo γ,

    ∫ dμ(γ′) |z(γ′)| e^{a(γ′) + d(γ′)} 1{γ′ ≁ γ} ≤ a(γ) .

Entonces el desarrollo en racimos converge y `Σ_{X ≁ γ} |φ(X)| e^{d(X)} ≤ a(γ)`, con
`d(X) = Σ_{γ_i ∈ X} d(γ_i)`. Además, `Σ_{X ∋ γ} |φ(X)| ≤ |z(γ)| e^{a(γ)}`.

*La cota de prueba de 9.12.* Si γ′ ≁ γ, sea t* el primer instante en que γ̂′ y γ̂ se cortan.
Antes de t* no se tocan. Las piezas nuevas aparecen solo en saltos, así que t* es un salto de γ o
de γ′:
- un salto de γ con una pared de γ′ viva a distancia ≤ 2;
- o al revés.

Por traslación, la medida de las traslaciones `(x, τ) ∈ ℤ × ℝ` de un γ′ anclado que ponen una pared
viva en `(i, t)` es `∫E_{γ′} dt`. Las que ponen un salto en `(j, t)` miden m. Esto da las densidades
M_1 y M_2 de 9.12, con los factores 5 y `E + 4r ≤ 5E`.

*Inducción por la anchura.* En tiempo continuo no se puede inducir en `|V(γ)|`, porque el interior
de un contorno contiene contornos con cualquier número de saltos. Se induce en la **anchura
espacial** `w(γ) = máx − mín` de los enlaces de γ̂, que es un entero.

*Lema.* Si γ′ está en un interior I de γ, entonces `w(γ′) ≤ w(γ) − 2`.

*Demostración.* I es acotado y `∂I ⊂ γ̂`. La recta horizontal por cualquier punto de γ′ sale de I
por γ̂ hacia los dos lados, y γ′ no es adyacente a γ. Así que, en ese instante, γ tiene piezas a
distancia ≥ 2 a la izquierda y a la derecha de γ′. ∎

*La inducción.*
- **Hipótesis H(N):** `|z(γ)| ≤ |W(γ)| e^{2a(γ)}` si `w(γ) ≤ N`.
- **Energía libre truncada.** `f^{(N)}` es la densidad, por enlace y unidad de tiempo, de los
  racimos de contornos de anchura ≤ N. Cada racimo se asigna a su primer salto. Por la traslación T,
  `f^{(N)}` no depende de la fase (Z16, punto 2). Además, `|f^{(N)}| ≤ W_anc`.
- **Paso.** Sea γ con `w(γ) = N + 1`. Sus interiores solo contienen contornos de anchura ≤ N − 1,
  así que `Z_c(I)` es la función de partición del gas truncado en I. Entonces

      log Z_c(I) + f^{(N)}|I| = − Σ_{X corta I, X ⊄ I} φ(X) |X̂ ∩ I| / |X̂| ,

  con |·| la medida en enlace × tiempo, que es positiva porque cada contorno tiene `E ≥ 2` durante un
  tiempo positivo. Todo X de la suma corta γ̂ o es adyacente a γ, así que X ≁ γ. Sumando sobre los
  interiores, el total es `≤ Σ_{X≁γ} |φ(X)| ≤ a(γ)` para cada fase. Esto da H(N + 1).

El resto de Z16 no cambia.

**c) Z17 en tiempo continuo.** Los pasos 1–4 de 9.6 cambian así; los pasos 5 y 6 no cambian.

1. **Topología.** En el toro `Z_n × [0, β)` cerrado con `T^m`, un contorno es contraíble, espacial o
   temporal.
   - **Espacial.** Su proyección sobre el círculo de enlaces es suprayectiva. Cada enlace contiene
     entonces una marca de un salto de γ, porque los extremos de un intervalo de pared son marcas.
     Cada salto marca 3 enlaces, así que `m ≥ ⌈n/3⌉`. La recurrencia de Z15c vale tal cual en el
     anillo, porque los marcos y las posiciones son locales.
   - **Temporal.** `E_γ(t) ≥ 1` para casi todo t ∈ [0, β). Con
     `e^{−(1−3a)EΔt} ≤ e^{−sΔt} e^{−(1−3a−s)EΔt}`, el peso total de los contornos temporales es
     `≤ C(n) e^{−sβ}`. Basta para el paso 5, que toma β → ∞ con n fijo.
   - **El orden.** Los contornos espaciales separan el cilindro y son disjuntos, así que están
     ordenados cíclicamente. Las regiones intermedias `A_i` son componentes del complemento, y por
     a) cada una tiene una sola fase en su capa de borde con cada γ_i.
2. **El anillo.** `A_i` ya no es un rectángulo. Se escribe

       log Z_c(A_i) = −f|A_i| + b_c(γ_i^+) + b_c(γ_{i+1}^−) + u(A_i) ,

   donde:
   - `|b_c(γ^±)| ≤ a(γ)`, porque son racimos incompatibles con γ;
   - u reúne los racimos que tocan los dos bordes.

   Para el decaimiento en tiempo se pide la condición de KP con `λ = 3a + c`, es decir con
   `d(γ) = c∫E_γ`. Un racimo que cubre un tiempo h tiene `d(X) ≥ 2ch`, porque `E ≥ 2` en la vida de
   cada contorno. Así, con `h_i ≥ 0` el hueco entre el último salto de γ_i y el primero de γ_{i+1},

       |u(A_i)| ≤ a(γ_i) e^{−2c h_i} .

3. **Los eventos.** `K(γ)_{c′c} = W(γ) e^{b_c(γ^−) + b_{c′}(γ^+)} e^{−f|N(γ)|}`, donde `N(γ)` es lo que γ
   quita a los anillos: sus paredes y los enlaces vecinos, `|N(γ)| ≤ 3∫E_γ`. Con `|f| ≤ W_anc`,

       ‖K(γ)‖ ≤ |W(γ)| e^{2a(γ) + 3W_anc ∫E_γ} .

   La estructura circulante no cambia.
4. **El gas de eventos.** Sus polímeros son las cadenas de eventos unidas por factores
   `u_i = e^{u(A_i)} − 1`, con `|u_i| ≤ |u(A_i)| e^{a(γ_i)}`.
   - **El enlace.** La integral sobre la posición en tiempo del evento siguiente es
     `≤ a(γ_i)e^{a(γ_i)} (1/(e(2c−μ)) + ∫E_{i+1}/2)`, si se pesa con `e^{μh}` (μ < 2c). El segundo
     término viene de los solapes en tiempo, con `ext(γ) ≤ ∫E/2`.
   - **El peso por evento.** Absorbiendo `a e^a ≤ e^{2a}/e` y `∫E/2 ≤ e^{ν′∫E}/(2eν′)`, cada evento
     pesa `≤ |W| e^{4a(γ)} e^{ν∫E} max(1, B̃)`, con `ν = 3W_anc + μ/2 + ν′` y
     `B̃ = e^{−1}(1/(e(2c−μ)) + 1/(2eν′))`.
   - **La suma por unidad de tiempo.** Sumando las n posiciones espaciales y usando `m ≥ ⌈n/3⌉`,

         q := max(1, B̃) · max_n n e^{−σ⌈n/3⌉} · W_anc(A|ε|e^{4aκ+σ}, λ = 4a + ν) .

   - **La condición.** KP en una dimensión, con `a(C) = (μ/2)(1 + ℓ(C))`, donde ℓ es la longitud en
     tiempo de la cadena. Las cadenas incompatibles con C (las que la solapan en tiempo) suman
     `≤ (ℓ(C) + 1) Σ_anc |w| e^{a}(1 + ℓ)`. Con `(1+ℓ) ≤ e^{δℓ}e^{δ}/(eδ)` y `δ = μ/2`, la condición es

         q/(1 − q) ≤ e μ² e^{−μ/2} / 4 .

   - **Lo que da.** `|t_p| ≤ q`, y las cadenas que dan la vuelta en tiempo pesan `≤ e^{−μβ/2}`.
     Así, el `γ_1` de 9.6 pasa a ser `min(μ, 2s)/2`, independiente de n.

Se toman `μ = c` y `ν′ = c/2`. El script busca `(c, a, κ, s, σ)`.

| | ε | parámetros | q | cota 1D |
|---|---|---|---|---|
| radio de Z17, truncado (estimación) | 8,3·10⁻³ | c = 0,5, s = 0,2, σ = 0,5 | — | — |
| radio de Z17, con la cola de 9.13 d) (riguroso) | **8,2·10⁻³** | — | — | — |
| ε = 2·10⁻³ (riguroso) | | c = 0,5, a = 0,001, κ = 1, s = 0,2, σ = 0,5 | 2,7·10⁻⁴ | 0,13 |
| ε = 5·10⁻³ (riguroso) | | c = 0,2, a = 0,003, κ = 1, s = 0,2, σ = 0,5 | 1,5·10⁻³ | 0,025 |

El gas de eventos tiene margen (q ≪ cota). Limita la condición de KP con decaimiento en tiempo
`λ = 3a + c`; con el conteo por clases de 9.13 d), la condición de la cola ya casi no resta (8,2 frente a
8,3·10⁻³).

**d) La cola de V: una supersolución.** V es el menor punto fijo de un operador monótono T. Si U ≥ 0
cumple `T(U) ≤ U`, entonces `V ≤ U`, porque `V = lim Tⁿ(0) ≤ U`. Se construye U así:
- **Para E > E_max:** `U = K ρ^E (DE)^f`. El factor `(DE)^f` paga las fusiones: `e₂/E` crece como E, y
  cada marco de más lo compensa con un factor DE.
- **Para E ≤ E_max:** U es el punto fijo numérico de T con ese borde.

*Conteo de posiciones por clases.* La cota gruesa `P = E + 4r` posiciones, todas con el peor `ρ^{ΔE}`,
desperdicia mucho: el cambio de energía de un salto depende de cuántas paredes hay en su ventana. Sea w
el número de paredes del marco en la ventana `[j−1, j+1]` del salto. Entonces:
- `ΔE ∈ [3 − 2w, 3 − w]` (cada pared de la ventana se borra o se mueve, y las demás se crean);
- las posiciones con w = 0 (ΔE = +3, crecimiento) están junto a los extremos de las rachas: `n₀ ≤ 2r`;
- las de w ≥ 1: `n₁ + n₂ + n₃ ≤ E + 2r`;
- cada pared está en exactamente tres ventanas: `n₁ + 2n₂ + 3n₃ = 3E`.

El crecimiento pesa entonces `Σ_w b_w n_w` con `b_w = max_{ΔE ∈ [3−2w, 3−w]} U(E+ΔE)/U(E)`, y su máximo
sobre `(n₁, n₂, n₃)` es un programa lineal en dos restricciones, que se resuelve enumerando sus vértices
(seis, según `N = E + 2r` frente a `3E/2` y `3E`). El mismo conteo se usa en la recurrencia interior
(E ≤ E_max), donde baja W_anc un 0,3 %, y en la cola.

*La cola, analíticamente (E ≥ E_max + 4).* Se usan:
- `(E′/E)^g ≤ e^{(E′−E)g/E}`, con `g ≤ f ≤ E/2`;
- `x = r/E ≤ 1` y `P = E(1 + 4x)`;
- `e₂ ≤ P²/2` y `e₃ ≤ P³/6`.

Con ellos, `T(U)/U ≤ b′ S/(1 − λ)`, donde, con `q = ρe^φ` y `M = max(q³, q^{−3}, ρ^{−3})`,

    S = sup_{φ ≤ 1/2, x ≤ 1} [ 2x q³ + L(q, x) + ρ³D(1 + 3/E_1)e^{3φ} + (1+4x)²M/(2D) + (1+4x)³M/(6D²) ] ,

y `L(q, x)` es el máximo del programa lineal por unidad de E. El mínimo en ρ, D es `S = 19,5`, en ρ = 0,46
y D = 20,25 (sin las clases era 28,9, en ρ = 0,775 y D = 4,55). La condición de la cola es
`b′ ≤ (1 − λ)/19,5`.

*La costura (E_max < E ≤ E_max + 3), numéricamente.* Se comprueba `T(U) ≤ U` estado a estado. K se
elige como el doble del máximo de `V/(ρ^E(DE)^f)` en `E_max − 2 ≤ E ≤ E_max`, y se cuadruplica si hace
falta. Con E_max = 40 basta el primer K.

*Lo que cuesta.* El valor de U en (3,1,1) coincide con el truncado en todas las cifras: la cola apenas
pesa. La condición `b′S/(1−λ) ≤ 1` es más fuerte que la convergencia real, pero con las clases ya no
limita mucho. El radio riguroso queda en **`ε_2 = 1,03·10⁻²`** (Z16; a = 0,017, κ = 1, s = 0,1) y
**`8,2·10⁻³`** (Z17), frente a las estimaciones truncadas de ≈ 1,2·10⁻² y 8,3·10⁻³. Sin las clases eran
7,4·10⁻³ y 6,0·10⁻³.

*Dos ideas que no sirven.* Llevar el marco mayor en el estado no ayuda a la cola: su peor régimen son
muchos marcos pequeños, donde el marco mayor no es especial. Contar los nacimientos «a distancia» (cada
marco nuevo con el factor de su posición) no admite un potencial estable de la forma `ρ^E(DE)^f`: el
factor de posición crece con E más deprisa de lo que D puede compensar.

**Balance de 9.12–9.13.**
- **Z15c y Z16:** cota rigurosa `ε_2 = 1,03·10⁻²` para `S⁺S⁻`, uniforme en n (estimación ≈ 1,2·10⁻²).
- **Z17:** pide además decaimiento en tiempo y el gas de eventos. Cota rigurosa `ε_2^{Z17} = 8,2·10⁻³`
  (estimación ≈ 8,3·10⁻³).
- **El Teorema Z⁗** vale con `|ε| < 8,2·10⁻³`, unas 3,6 veces el 2,3·10⁻³ de 9.10 con celdas.

**e) El certificado con intervalos** (`scripts/certifica_continuo.py`). Todo lo que entra en T es ≥ 0 y
T es monótono, así que basta llevar el extremo superior de cada intervalo. No hace falta que la
iteración sea exacta: basta comprobar, con redondeo hacia arriba, que la U calculada es supersolución.
1. **Redondeo dirigido.** Cada suma y cada producto de arreglos va seguido de `nextafter(·, +∞)`. El
   resultado redondeado al más cercano está a ≤ ½ ulp del exacto, así que el siguiente número de coma
   flotante es una cota superior, también en la zona subnormal. Los máximos y los desplazamientos son
   exactos. Los coeficientes de los vértices del LP son enteros o medios enteros pequeños, también
   exactos. `e₂` y `e₃` se acotan con la misma regla.
2. **Escalares** (`b′`, `1 − λ`, `e`, `e^s`, las condiciones de KP y de Z17) con `mpmath.iv`: se usa el
   extremo superior de lo que se cota y el inferior de los denominadores y de las cotas que hay que
   respetar.
3. **La U que se comprueba.** En E ≤ E_max, U es el punto fijo numérico de `(1 + η)T`, con η = 10⁻⁹. T
   es lineal en el prefactor, así que el margen relativo es η, muy por encima del error de redondeo
   (unas 30 ulp). La iteración se detiene cuando el cambio relativo es ≤ 10⁻¹³ en cada estado. En
   E > E_max, U es exactamente `Kρ^E(DE)^f`: ρ, D y K son números en coma flotante, es decir racionales
   exactos. Se calculan cotas superior e inferior por productos redondeados hacia arriba y hacia abajo.
4. **Comprobación.** `T_sup(U_sup) ≤ U` en todos los estados con E ≤ E_max, y `≤ U_inf` en la costura.
5. **La cola.** `S` se acota en 4000 trozos de φ ∈ [0, ½] con intervalos. En cada trozo,
   `q^d ≤ max(q(φ₀)^d, q(φ₁)^d)` y `e^{3φ} ≤ e^{3φ₁}`. Basta x = 1, porque el integrando crece con x: el
   LP gana vértices con b ≥ 0, y los demás términos son crecientes. Sale `S ≤ 19,4654` (en la rejilla,
   19,4645). Se comprueba `b′S/(1−λ) ≤ 1` con intervalos.
6. **Monotonía en ε.** Con los parámetros fijos, U sigue siendo supersolución si el prefactor baja, y
   todas las condiciones son monótonas en |ε|. Así, el certificado en ε₂ vale para todo |ε| ≤ ε₂.

Los parámetros se buscan en coma flotante (9.12 y 9.13 c) y se certifican en el último ε de la bisección.
Ambos pasan a la primera:

| | ε certificado | parámetros | margen |
|---|---|---|---|
| Z15c / Z16 | **1,033·10⁻²** | a = 0,01737, κ = 1, s = 0,1 | 5M₁ = 0,0150 ≤ aκ = 0,0174; 5M₂ = 0,0164 ≤ a |
| Z17 | **8,198·10⁻³** | c = 0,2, a = 0,01244, κ = 1, s = 0,1, σ = 0,1 | q = 0,0106; q/(1−q) ≤ 0,0246 |

W_anc certificado y en coma flotante difieren en una parte en 10⁹ (el η). Con esto los radios de 9.12–9.13
son teoremas, con la cola acotada y el redondeo controlado.

**9.14 El lado k ∤ n con contornos: el Teorema D‴** (`scripts/orbitas_segundo_orden.py`).

El Teorema D″ (7.14) usa el vacío de cadena de Z″ y hereda su rango, `n ≤ n_*(ε)` (9.4). Aquí se
cambia ese vacío por los contornos en espacio-tiempo de 9.5–9.6 y 9.12–9.13. El defecto que impone
k ∤ n pasa a ser un contorno que da la vuelta en tiempo, y su movimiento es un gas unidimensional en
el tiempo, como los eventos de túnel de Z17. Z16 no usa k | n para los contornos contraíbles: sus
energías libres truncadas son cantidades del plano, iguales por traslación en `Z × ℝ`. Por eso vale
igual en el anillo con k ∤ n.

Hipótesis: h de la clase (C), carga U(1) conservada, tres órbitas de un defecto con cargas distintas
(7.14), ε complejo con `|ε| < ε_3`, `|Θ| ≤ π`, n con k ∤ n. Las constantes no dependen de n, Θ ni β.

**Paso 1: el contorno del defecto.** Con el marco de 9.12 (X = paredes ∪ los 3 enlaces de cada salto,
contornos = componentes conexas de X̂):
- **La suma de las paredes de un contorno es constante en el tiempo.** Un salto suma
  `(+1, −2, +1)` a los enlaces de su ventana, y esas paredes son del contorno del salto (9.12, Z15c,
  punto 1). Esto vale también si el contorno da la vuelta en espacio. Un contorno cuya proyección en
  el tiempo es un intervalo nace sin paredes, así que suma 0 mod k en todo instante.
- **Las paredes de cada instante suman `−n ≢ 0 mod k`**, porque `Σ_j (s_{j+1} − s_j − 1) = −n`. Así que
  en cada instante hay un contorno con suma no nula. Ese contorno no puede nacer ni morir, así que da
  la vuelta en tiempo. Se le llama Γ.
- **El sector de un defecto.** Sea `t` un *instante libre* si la sección de Γ en t es una sola pared.
  En un instante libre, la configuración fuera de Γ y de los contornos contraíbles es el patrón de un
  defecto `(b, c)`, y la carga Q fija c, porque las tres órbitas tienen cargas distintas. Entre dos
  instantes libres consecutivos, Γ hace una *excursión*: un intervalo de tiempo en el que su sección
  tiene ≥ 2 paredes, más los instantes de sus saltos. La excursión lleva `(b, c)` a `(b + r, c)`.
- **Las excursiones mueven el defecto en múltiplos de k.** Una excursión que no da la vuelta en
  espacio cambia la configuración solo en su ventana. Allí, los sitios de `(b, b + r]` pasan de la
  fase de la derecha a la de la izquierda. La carga que cambia es la suma de `q_izq − q_der` en esos
  r sitios. Es 0 si k | r, porque cada celda suma 0, y no es 0 si k ∤ r, porque las dos fases son
  distintas. (Con `S⁺S⁻` y k = 3, se ha comprobado en las seis combinaciones de n mod 3 y c: los
  desplazamientos de 1 y 2 sitios cambian la carga en ±1 o ±2.) Así que **r ∈ kZ**. En el anillo,
  como `mcd(k, n) = 1`, los pasos de k recorren todo `Z_n`. La traslación global T, que lleva
  `(b, c)` a `(b + 1, c)`, es una simetría pero no un movimiento local.
- **Todo lo demás es caro.** Una segunda componente que da la vuelta en tiempo, o un Γ sin instantes
  libres, tiene ≥ 2 paredes en todo instante (salvo los de salto). Cada pared de más cuesta
  `Re ≥ 1 − δ` por unidad de tiempo. Estas historias pesan `≤ C(n) e^{−β(Re(n f_∞ + ω_Q) + γ_2)}`,
  con `γ_2 = 1 − δ − O(a)`. El factor C(n) cuenta las posiciones y no importa, porque β → ∞ a n fijo.

**Paso 2: el complemento de Γ.** Como en Z16, la actividad de Γ incluye los cocientes
`Z_{c_I}(I)/Z_{c}(I)` de sus interiores. Fuera de Γ queda una región conexa con una sola fase c en
las coordenadas desenrolladas, y sus contornos contraíbles forman el gas de Z16. KP da

    log Z_c(fuera de Γ) = −f_∞ (nβ − |Γ̂|) + Σ_{piezas p de Γ} b_Q(p) + Σ_{p < p′} u(p, p′) + u_∘ ,

donde:
- `f_∞` es la energía libre del plano, la misma para toda fase (Z16, punto 2);
- `b_Q(p)` reúne los racimos que tocan una sola pieza (un tramo libre o una excursión). En un tramo
  libre de duración ℓ vale `ℓ·b_Q + O(1)`. `b_Q` es la densidad de una pared recta infinita, que
  solo depende de c y de n mod k. El `O(1)` corrige los extremos del tramo, y se pasa a las
  excursiones vecinas como un factor acotado;
- `Σ_{p < p′} u(p, p′)` abrevia los racimos que tocan dos o más piezas. No son interacciones de
  pares: al exponenciar, cada racimo es un enlace entre las piezas que toca, y los polímeros del gas
  de 4 son los conjuntos de piezas unidos por racimos, como en el paso 4 de Z17. Un racimo que toca
  piezas separadas por un tiempo d pesa `≤ C e^{−γ_1 d}`;
- los racimos que dan la vuelta en *tiempo* pesan `≤ C n e^{−γ_1 β}` y van a R;
- `u_∘` reúne los racimos que dan la vuelta en espacio. Tocan Γ por los dos lados, así que tienen
  ≥ n − w celdas, con w la anchura de Γ, y `|u_∘| ≤ C β e^{−a(n−w)}` por unidad de tiempo.

**Paso 3: Peierls y fase de Θ.** Con el giro repartido como `Θ/n` por enlace, H(Θ) conmuta con la
traslación T, y cada salto que pasa carga por un enlace lleva `e^{±iΘ/n}`.
- Un contorno contraíble vuelve a su configuración sin dar la vuelta, así que su flujo neto es 0 y no
  ve Θ.
- Una excursión que desplaza el defecto r sitios, con r ∈ kZ, mueve los sitios de `(b, b + r]` de una
  fase a la otra. El flujo neto es `Σ_j j·Δq_j` en la ventana, y vale `rQ` (salvo el signo, que es
  convención). Por celda es la diferencia de dipolo entre las dos fases, que es kQ. Así que la
  excursión lleva `e^{−iΘQr/n}`. Comprobado con k = 3 en los seis casos de (n mod 3, c): el dipolo de
  una celda es −3, 0 o 3, igual a 3Q.
- Solo ven Θ de otro modo los objetos que dan la vuelta en espacio, que pesan `≤ e^{−c n}`.

**Paso 4: el gas unidimensional.** Las piezas forman una sucesión cíclica en tiempo, y la actividad
de una excursión solo depende de su forma y de r, por la traslación. En el sector de momento q y carga
Q, cada excursión es un número, `K_q(ε) = Σ_{r ∈ kZ} K_r e^{−i(q − ΘQ/n)r}`, y el gas es escalar,
como en el paso 3 de Z17. Por eso ω_Q tiene periodo `2π/k` en κ. Los momentos del anillo,
`κ ∈ (2π/n)Z`, recorren ese periodo con paso `2π/(kn)`, porque `mcd(k, n) = 1`. La referencia es el tramo libre, `e^{−ℓ(1 + d_Q − b_Q)}`, con `d_Q` la parte diagonal
de εV en la pared. KP en una dimensión, igual que en el paso 4 de Z17, da

    Tr(Π_Q Π_q e^{−βH}) = e^{−β(n f_∞ + ω_Q(q − ΘQ/n) + δ_{Q,q})} (1 + R) + (paso 1) ,

con:
- `|R| ≤ C e^{−γ_1β/2}`;
- `|δ_{Q,q}| ≤ C e^{−c n}`;
- `ω_Q(κ) = 1 + d_Q − b_Q − (racimos del gas de excursiones en la cadena Z, con símbolo en κ)`.

ω_Q se calcula en la cadena infinita, así que no depende de n salvo por n mod k (el tipo de pared).
Además, `K_r` decae exponencialmente en |r|, así que ω_Q es analítica en κ en una banda
`|Im κ| < c′`, y en ε con `|ε| < ε_3`, con `|ω_Q − 1| ≤ B`. Las diferencias entre el anillo y la
cadena son las excursiones y los racimos que dan la vuelta en espacio, de tamaño ≥ n, y están en δ.

La condición de KP del gas de excursiones no lleva un factor n: las excursiones están ancladas al
defecto. La actividad por unidad de tiempo es `≤ Σ |εA|^{saltos} e^{−(1−δ)(duración)}`, sumando sobre
las excursiones ancladas, y se acota como `W_anc` de 9.12. Es pequeña para |ε| pequeño, y eso da
`ε_3 > 0`. Las constantes están más abajo.

**Paso 4′: los enlaces entre excursiones (Lema Z18).** Los racimos del vacío que tocan dos o más
excursiones las acoplan. Aquí se escribe ese desarrollo y se da su cota.

*Referencia.* Para cada excursión e, sea `D_e` la unión de su región y de los dos tramos rectos del
defecto (en b y en b + r) durante su ventana de tiempo. Sea Γ₀ la línea del defecto con cada excursión
sustituida por el tramo recto en su posición de entrada. Entonces:
- **La referencia es exacta en el tiempo.** Los racimos que no tocan Γ₀ suman `β b_Q`, salvo
  correcciones en los puntos donde Γ₀ cambia de posición, que están dentro de algún D_e.
- **Coeficientes.** Sea `c(X) = 1[X ≁ Γ] − 1[X ≁ Γ₀] ∈ {−1, 0, 1}`. Si X toca Γ fuera de toda ventana
  de excursión, es incompatible con las dos y `c(X) = 0`. Así, `c(X) ≠ 0` solo si X toca algún D_e,
  y no toca la pared libre en ningún otro sitio.
- **Dependencia local.** Sea A(X) el conjunto de excursiones e con `X ≁ D_e`. Si X tocase la pared
  durante la ventana de otra excursión e″, tocaría `D_{e″}`, y e″ estaría en A(X). Por eso `c(X)`
  solo depende de las excursiones de A(X), y también
  `Φ(A) := Σ_{X : A(X) = A} c(X) φ(X)`.

*Desarrollo.* El peso relativo de Γ es

    e^{−Σ_{X≁Γ} φ(X)} = e^{−β b_Q} Π_e e^{−Φ({e})} Π_{|A| ≥ 2} (1 + ψ(A)) ,   ψ(A) = e^{−Φ(A)} − 1 .

- `e^{−Φ({e})}` es parte del vestido de e.
- Al desarrollar el último producto, los polímeros del gas de 4 pasan a ser **conjuntos enlazados**:
  excursiones unidas por familias conexas de A's. Su actividad es
  `K(P) = Π_{e∈P} K(e) Σ_{familias conexas} Π ψ(A)`, y dos polímeros son incompatibles si alguna
  excursión de uno se solapa en el tiempo con una del otro.
- KP con `a(P) = Σ_{e∈P} (μ₀ + μ₁ dur_e)` se reduce a la condición para una excursión, con la
  actividad `|K(e)| G(e)`. Aquí `G(e)` suma los conjuntos enlazados que cuelgan de e (desarrollo en
  árbol).

*La cota de G.* Sean m₀ y m₁ la densidad y el momento en la duración del gas (con `e^{a(e)}` y el
propio vestido, que incluye G).
- **Un enlace.** Un racimo X que toca e y otra excursión e′ obliga a que e′ empiece en una ventana de
  longitud `|τ(X)| + dur_{e′}`, donde τ(X) es la extensión de X en el tiempo. Integrar e′ da
  `≤ m₀|τ(X)| + m₁`. Con varias, el factor es `e^{m₀|τ| + m₁} − 1`.
- **El tamaño temporal del racimo.** Para integrar |τ(X)| hace falta el KP del vacío con decaimiento
  en el tiempo c (el de Z17, `λ = 3a + c`): da `Σ_{X≁D_e} |φ(X)| e^{c∫E(X)} ≤ a(D_e)`, y
  `|τ(X)| ≤ ∫E(X)/2`. Con `x e^{m₀x} ≤ e^{2cx}/(e(2c − m₀))`:

      G(e) ≤ e^{L X_e} ,   L = e^{m₁} ( m₁ + m₀/(e(2c − m₀)) ) e^{0,05} ,   X_e = 5 m_e M₁ + 5 (∫E_e + dur_e) M₂ .

  El factor `e^{0,05}` cubre el `e^{|φ(X)|}` de `|e^{−Φ} − 1| ≤ |Φ| e^{|Φ|}`, porque
  `|φ(X)| ≤ W_anc < 0,05`. Hace falta que `2c > m₀`. ∎

*Consecuencia para las cuentas.* El vestido de una excursión pasa a ser
`(3 + L)(5mM₁ + 5∫E M₂) + (2 + L)·5·dur·M₂`. El 3 son los interiores (×2) y el contacto. El 2 son `b_Q`
en la referencia y los racimos que tocan los tramos rectos de `D_e`. Este último término **faltaba**
en la versión anterior, que ponía `η = 5M₂` en vez de `10M₂`. L se fija de antemano, y se comprueba
después que el L que piden los m₀ y m₁ calculados no lo supera.


**Paso 5: multiplicidades enteras.** Es el paso 5 de Z17 en el sector (Q, q). Con
`s = Re(n f_∞ + ω_Q) + γ′` y `γ′ = min(γ_1/2, γ_2)/2`, hay exactamente un autovalor con `Re E < s`,
y es simple. Es `λ_{Q,q} = n f_∞ + ω_Q(q − ΘQ/n) + δ_{Q,q}`. En los sectores de carga que no son de
una órbita, no hay ninguno.

**Teorema D‴ (banda de un defecto, carga entera, k ∤ n, uniforme en n).** Con las hipótesis de
arriba, para **todo** n con k ∤ n:
1. **Niveles simples y holomorfos.** Los nk niveles de la banda se etiquetan por (Q, κ), con
   `κ ∈ (2π/n)Z`. Son simples, holomorfos en ε y aislados por `γ′`, independiente de n. Con ε real,
   son el fondo del espectro.
2. **Banda límite.** `|λ_{Q,κ} − n f_∞ − ω_Q(κ − ΘQ/n)| ≤ C e^{−c n}`, con ω_Q independiente de n
   (en cada clase de n mod k), analítica en κ y en ε. Así, el hueco entre órbitas,
   `g_orb = min_{Q ≠ Q*} min_κ ω_Q − min_κ ω_{Q*}`, no depende de n por construcción: la hipótesis
   del punto 3 de D″ pasa a ser una propiedad de las funciones ω_Q.
3. **Regla neutro/cargado.**
   El fundamental está en la órbita Q* si g_orb supera la variación de ω_{Q*} en la red de
   momentos (`O(n^{−2})`) más `2Ce^{−cn}`. Esto vale para `n ≥ n_0(ε)`.
   - Si `Q* = 0`, ω_0 no ve Θ y `R_n ≤ 2 max|δ| ≤ C e^{−c n}`. Mejor: con Q = 0, las excursiones y los
     contornos contraíbles no llevan fase, así que la dependencia en Θ solo viene de objetos que dan
     la vuelta en espacio alrededor de Γ. Tienen `≥ N_n := ⌈(n−1)/3⌉` saltos, porque cada salto
     marca 3 enlaces y Γ solo cubre uno sin saltos. (En 9.15 se afina a N_n = n.) Así, la parte de `λ_{0,κ}` que depende de Θ es
     analítica, está acotada por 2B y se anula hasta el orden N_n − 1. El lema de Schwarz da
     `R_n ≤ 4B(|ε|/ε_3)^{N_n}`. El exponente es más pobre que el `⌊n/2⌋ − 1` de D″; los números
     (≈ ε^{0,7n}) indican que el real es mayor.
   - Si `Q* = ±1` y el mínimo de `ω_{Q*}` es no degenerado, `R_n ≍ n^{−2}`: el desplazamiento
     `ΘQ/n` recorre hasta 1,5 pasos de la red `2π/(kn)`.

*Demostración.* Pasos 1–5, y para 3, Schwarz y el argumento de D′. ∎

**La regla para `S⁺S⁻`, a segundo orden.** El primer orden es nulo en las tres órbitas: `PVP = 0` en el
subespacio de un defecto, porque ningún salto mueve el defecto sin crear paredes. A segundo orden,
`ω_Q = 1 + ε² w_Q(κ) + O(ε³)`. Aquí `w_Q` es el hamiltoniano efectivo `−PV(H_0 − E_0)^{−1}VP`, menos la
energía del vacío, `e^{(2)} = −4/3` por enlace. Es el mismo en n = 10, 13 y 16, y en 11, 14 y 17:

| n mod 3 | `w_{±1}` | `w_0(κ)` | `min_κ w_0` | `g_orb / ε²` a segundo orden |
|---|---|---|---|---|
| 1 | −2 (plana) | `−10 − (32/3) cos 3κ` | −62/3 | 56/3 ≈ 18,7 |
| 2 | −4 (plana) | `−6 − (32/3) cos 3κ` | −50/3 | 38/3 ≈ 12,7 |

La forma de `w_0` coincide con los autovalores en todos los n de la tabla, con error < 10⁻¹³. El
defecto neutro salta 3 sitios a segundo orden, con amplitud `−(16/3)ε²`. Es el paso de k sitios de
arriba, y el periodo 2π/3 de ω_Q. Los cargados no saltan a este orden.

Así, `g_orb = (56/3)ε² + O(ε³)` o `(38/3)ε² + O(ε³)`. Como ω_Q es analítica con `|ω_Q − 1| ≤ B` en
`|ε| < ε_3`, y con el resto acotado (abajo) existe `ε_4 > 0` (certificado: ≥ 1,74·10⁻³; por Cauchy salía ≈ 1,1·10⁻⁵) tal que, con `0 < |ε| < ε_4`,
gana la órbita neutra. **Para `S⁺S⁻`, el lado k ∤ n con carga entera tiene sensibilidad exponencial,
uniformemente en n.**

Los números de 7.14 lo confirman:
- A ε = 0,025, la diagonalización exacta da `Δ_0 = 0,98722`, frente a `1 − (62/3)ε² = 0,98708`, y
  `Δ_{±1} = 0,99858`, frente a `0,99875` (n = 10).
- Con n = 11 da `0,98964` frente a `0,98958`, y `0,99743` frente a `0,9975`.

**Lo que se usa y lo que no.**
- **De Z16:** KP para los contornos contraíbles, que no usa k | n.
- **De Z17:** el gas unidimensional en el tiempo y el argumento de las multiplicidades enteras.
- **Ya no se usan:** ni el vacío de cadena de Z″, ni Z7″, ni la cota de Gershgorin.

**Revisión.** Al releer la prueba se corrigieron cinco puntos:
- la conservación de la suma de paredes no necesita que el contorno sea contraíble en espacio;
- las excursiones solo mueven el defecto en múltiplos de k. Faltaba este punto, y de él salen el
  periodo de ω_Q y la red de momentos;
- los términos que unen piezas de Γ son de muchos cuerpos, y los tramos libres tienen corrección de
  extremos;
- con Q* = 0, el orden de la dependencia en Θ se obtiene directamente, sin el Teorema D;
- el fundamental está en la órbita Q* solo para n ≥ n_0(ε), por la discretización de κ.

**Grado de detalle.** Los pasos 1 y 5 son completos. Los pasos 2–4 siguen a Z16 y Z17, con estas
diferencias: una pieza anclada en lugar de eventos que dan la vuelta en espacio, y un gas escalar por
la traslación en lugar de circulantes. Los enlaces entre excursiones están en el paso 4′ (Lema Z18),
y las constantes, abajo, certificadas.

**Las constantes** (`scripts/constantes_defecto.py`).

*ε_3: el gas de excursiones.* Una excursión se cuenta con los marcos de 9.12, más el marco del defecto.
El marco del defecto tiene suma no nula, así que tiene ≥ 1 pared y no muere. El estado es (E, r, f),
y la excursión acaba al volver a (1, 1, 1):

    V(E,r,f) = b′/den(E) · [ crec + V(E+3, r+1, f+1) + e₂ max V(f−1) + e₃ max V(f−2) ] ,   V(1,1,1) = 1 ,
    den(E) = (1 − λ)E − (1 + η) − μ .

Aquí:
- `crec` lleva las clases de posición de 9.13 d), y las fusiones son las de 9.12;
- la referencia es el tramo libre: una pared, energía 1;
- `λ = 3a + c` son los pesos de KP del vacío y de los interiores;
- `η = 2a` cubre los racimos del vacío que tocan los tramos libres, tanto `b_Q` como los enlaces;
- μ es el peso de KP del gas unidimensional, `a(exc) = μ(duración + 1)`.

Desde un instante libre, el primer salto deja E′ ∈ [2, 4], porque `(+1, −2, +1)` sobre una pared aislada
deja ≥ 2 paredes. También puede nacer un marco aparte, que debe fundirse después. Así, la densidad de
excursiones por unidad de tiempo es

    m₀(μ) = e^μ b′ [ 5 max_{E′∈[2,4], f=1} V + V(4, 2, 2) ] .

Dos excursiones son incompatibles si se solapan en el tiempo. Con `a(exc) = μ₀ + μ₁·duración`, KP en
una dimensión pide `m₀ ≤ μ₁` y `m₁ ≤ μ₀`, donde:
- m₀ es la densidad, con peso `e^{a(exc)}`;
- m₁ es el momento en la duración. Se calcula con su propia recurrencia, `U = V/den + T(U)`, que
  resulta de derivar `1/den`. (La primera versión usaba `μ₀ = μ₁` y `m₁ ≤ m₀(μ+s₁)/(e s₁)`, que
  infla el momento.)

Los parámetros del vacío son los certificados para Z17 en 9.13 e): a = 0,01244, κ = 1, c = 0,2. Valen
para todo `|ε| ≤ 8,198·10⁻³`. Con ellos:

| | ε | μ₀ | μ₁ | m₀ | m₁ |
|---|---|---|---|---|---|
| radio | **ε_3 = 7,82·10⁻³** | 0,40 | 0,113 | 0,113 | 0,385 |
| | 3·10⁻³ | 0,014 | 0,008 | 8,0·10⁻³ | 0,013 |
| | 1·10⁻³ | 0,002 | 0,002 | 8,4·10⁻⁴ | 1,3·10⁻³ |

Con la versión anterior (μ₀ = μ₁ y momento por `e^{sx}`) salía `ε_3 = 4,86·10⁻³`, igual con E_max = 40
y 60. Ahora ε_3 queda cerca del 8,2·10⁻³ de Z17, que es el techo, porque los parámetros del vacío son
los de Z17.
- **Certificado** (`scripts/certifica_defecto.py`, abajo): `ε_3 ≥ 7,22·10⁻³`, con los enlaces del
  Lema Z18.
- Los enlaces entre piezas por racimos del vacío están en el Lema Z18 (paso 4′).

*Una paridad.* Cada salto `S⁺S⁻` cambia `Σ_j j s_j` en ∓1. Una excursión que desplaza el defecto r ∈ 3Z
lo cambia en rQ (paso 3). Así, el número de saltos es ≡ rQ mod 2:
- **Q = 0:** todas las historias tienen un número par de saltos, así que **ω_0 es par en ε**;
- **Q = ±1:** un desplazamiento de 3 sitios pide un número impar de saltos. Los defectos cargados no
  se mueven a segundo orden (de ahí la banda plana) y empiezan a moverse en tercer orden.

Comprobado enumerando los procesos de segundo orden en el anillo de 7 sitios: la neutra salta ±3, y
las cargadas solo vuelven a sí mismas.

*Tercer orden exacto.* `H₃ = PV R V R V P` (R el resolvente reducido) en el subespacio de un defecto,
con n = 10, 13 (n ≡ 1) y 11, 14 (n ≡ 2):
- **neutra:** `w₃ = 0`, como pide la paridad;
- **cargadas:** `w₃(κ)` es un salto de 3 sitios, `(32/3) cos(3κ + φ)` con n ≡ 1 y `(40/9) cos(3κ + φ)`
  con n ≡ 2. Es el primer orden en que se mueven, de acuerdo con la paridad.

Como H₂ es plana para las cargadas, H₃ es el término siguiente del hamiltoniano efectivo. El vacío no
tiene tercer orden, también por la paridad.

*ε_4: gana la neutra, sin Cauchy.* Los términos de orden 2 y 3 son exactos, así que basta acotar todo lo
que tiene ≥ 4 saltos. Se acota con mayorantes de la misma recurrencia, separadas por el número de saltos
(`scripts/constantes_defecto.py`, modo `eps4b`). En cada ε se usa el `a` mínimo con las condiciones de
KP del vacío (de 1,4·10⁻⁴ en ε = 10⁻³ a 6,6·10⁻⁴ en 2,3·10⁻³), de modo que todo escala bien con ε.
El resto se reparte así:
- **R1:** excursiones con ≥ 4 saltos propios, vestidas;
- **R1v:** el vestido de las excursiones de 2 y 3 saltos. Un racimo del vacío que toca un objeto con m
  saltos y pared integrada ∫E pesa en total `≤ 5mM₁ + 5∫E M₂` (9.12). El vestido es la exponencial
  de tres veces eso: dos por los interiores y una por el contacto. El tramo libre lleva `e^{5M₂}`. Se
  acota `|e^X − 1| ≤ X e^X`, con `∫E ≤ 4·duración` o `7·duración`;
- **R2a:** contornos del vacío con ≥ 4 saltos que tocan la pared (5 posiciones por unidad de tiempo);
- **R2b:** racimos de ≥ 2 contornos que tocan la pared: `|z|(e^{a(γ)} − 1) ≤ a(γ)|z|e^{a(γ)}`;
- **R3:** racimos de ≥ 2 excursiones: `≤ μ₀m₀ + μ₁m₁`, con el KP de dos parámetros.

La neutra gana si `g₂ε² − c₃|ε|³ > 2·resto`, con `c₃ = 32/3` o `40/9`. En κ = 0 la neutra está en su
mínimo, `1 + w_0(0)ε² + resto`, y las cargadas están por encima de `1 + w_{±1}ε² − c₃|ε|³ − resto`:

| n mod 3 | **ε_4** | hueco `g₂ε² − c₃ε³` | 2·resto | R1 | R1v | R2a | R2b | R3 |
|---|---|---|---|---|---|---|---|---|
| 1 | **2,3·10⁻³** | 1,0·10⁻⁴ | 1,0·10⁻⁴ | 1,2·10⁻⁵ | 2,5·10⁻⁵ | 5,6·10⁻⁷ | 3,7·10⁻⁶ | 8,8·10⁻⁶ |
| 2 | **1,9·10⁻³** | 4,7·10⁻⁵ | 4,7·10⁻⁵ | 5,6·10⁻⁶ | 1,2·10⁻⁵ | 2,7·10⁻⁷ | 1,8·10⁻⁶ | 4,1·10⁻⁶ |

Con Cauchy salía ε_4 ≈ 1,1·10⁻⁵, así que la mejora es de unas 170 veces. Lo que limita es el vestido
de las excursiones cortas (R1v), que es la cota más gruesa. Los números de 7.14 indican que la neutra
gana al menos hasta ε = 0,2.

**El certificado** (`scripts/certifica_defecto.py`). Es el de 9.13 e), con la recurrencia de las
excursiones:
1. **Supersolución.** U es el punto fijo numérico de `(1 + 10⁻⁹)T` en E ≤ E_max, con la forma exacta
   `Kρ^E(DE)^f` en la cola (los mismos ρ = 0,46 y D = 20,25). Se comprueba `T_sup(U) ≤ U`, con redondeo
   hacia arriba, en E ≤ E_max y en la costura.
2. **La cola.** `den(E) ≥ ((1−λ) − c₀/E_1)E`, con `c₀ = 1 + η + μ₁` y `E_1 = E_max + 4`. El marco del
   defecto puede tener una sola pared, así que `φ = f/E ≤ ½ + 1/(2E_1)`. Con ese rango, `S ≤ 19,8309`
   (frente a 19,4654 en el vacío), y se comprueba `b′S/((1−λ) − c₀/E_1) < 1` con intervalos.
3. **Cubos de saltos.** Son composiciones finitas de T_sup aplicadas a U (los que cuentan «≥ J») o
   a 0. En la cola se acotan por U, porque cada cubo es ≤ V_todas ≤ U.
4. **Momento en la duración.** `Y ≥ V/den + T(Y)` tiene su propia supersolución, con cola
   `K_Y ρ^E(DE)^f` y `K_Y(1 − θ) ≥ K/((1−λ)E_1 − c₀)`, donde θ < 1 es la razón de la cola.
5. **El vacío.** M₁ y M₂ salen de `cond_cert`; `W_{≥4}`, de los cubos sobre la supersolución del vacío;
   y las de R2b, de `W_cert` (9.13 e).
6. **Monotonía.** Con los parámetros fijos, todo es monótono en |ε|, así que el certificado en ε vale
   para todo |ε| menor.

Con el Lema Z18 (enlaces, paso 4′) y L = 1:

| | ε certificado | margen | L pedido |
|---|---|---|---|
| ε_3 | **7,22·10⁻³** | `m₀ ≤ 0,0969 ≤ μ₁ = 0,0987`, `m₁ ≤ 0,3459 ≤ μ₀ = 0,350` | 0,69 ≤ 1 |
| ε_4, n ≡ 1 | **2,09·10⁻³** | hueco ≥ 8,1·10⁻⁵ > 2·resto ≤ 8,1·10⁻⁵ (a = 5,6·10⁻⁴) | 0,021 ≤ 1 |
| ε_4, n ≡ 2 | **1,74·10⁻³** | hueco ≥ 3,9·10⁻⁵ > 2·resto ≤ 3,8·10⁻⁵ (a = 3,8·10⁻⁴) | 0,015 ≤ 1 |

Para ε_4, el vacío se toma con KP con decaimiento en el tiempo c = 0,02, que el lema necesita
(`2c > m₀`). Sin los enlaces, y con el término de los tramos rectos que faltaba, salían 7,82·10⁻³,
2,29·10⁻³ y 1,91·10⁻³. La pérdida, del 8–9 %, viene sobre todo del vestido (3 + L) en vez de 3 y de
`η = (2 + L)·5M₂`. En ε_4, L = 1 sobra mucho: el L pedido es 0,02. Los valores certificados coinciden
con los de coma flotante en una parte en 10⁹.

**w₂ y w₃, exactos** (`scripts/orbitas_exactas.py`). Se suman con fracciones los caminos de 2 y 3
saltos que vuelven al subespacio de un defecto, con denominadores enteros. Las amplitudes de salto,
`h(r) = ⟨T^r p|H|p⟩`, son las mismas en n = 10, 13 (n ≡ 1) y 11, 14 (n ≡ 2):

| n mod 3 | órbita | h₂(0) | h₂(±3) | h₃(±3) |
|---|---|---|---|---|
| 1 | neutra | −10 | −16/3 | 0 |
| 1 | cargadas | −2 | 0 | 16/3 |
| 2 | neutra | −6 | −16/3 | 0 |
| 2 | cargadas | −4 | 0 | 20/9 |

El vacío da e₂ = −4/3 por enlace y e₃ = 0, ya restados. Así, `w_0 = −10 − (32/3) cos 3κ` (n ≡ 1) y
`−6 − (32/3) cos 3κ` (n ≡ 2), `w_{±1} = −2` y −4, y `w₃ = (32/3) cos 3κ` y `(40/9) cos 3κ`, todos
exactos. Su independencia de n es la del paso 4: los procesos de ≤ 3 saltos no dan la vuelta al
anillo.

**Con esto, ε_3 y ε_4 son teoremas**, al mismo nivel que los radios de Z⁗ (9.13 e). El conteo es el
de 9.12–9.13, los enlaces son los del Lema Z18, y la aritmética está certificada.

**9.15 El orden en que el fundamental nota Θ con carga entera** (`scripts/orden_theta.py`).

Pregunta de 7.2 y de 6: con carga entera por celda módulo k, ¿nota Θ el fundamental solo en el orden
del enrollamiento, ε^n, y no en el del túnel? Los números de 7.4 lo indicaban (exponente 6 y 9 con
n = 6 y 9). Aquí se demuestra, y se da el coeficiente.

*Flujo y dipolo.* Con el giro repartido como `Θ/n` por enlace, H(Θ) conmuta con T, y un camino de
saltos lleva `e^{iΘJ/n}`, donde J es el flujo neto de carga (saltos en un sentido menos en el otro).
Sea `D = Σ_{j<n} j q_j` el dipolo con el corte en el enlace (n−1, 0). Un salto por un enlace interior
cambia D en lo mismo que J. Uno por el enlace del corte lo cambia en `J ∓ n`. Así, para un camino de
a a b, `J ≡ D(b) − D(a) mod n`.

**Lema Z19 (los estados de P no se distinguen por el dipolo).** Sea P el subespacio degenerado:
- **(a)** los k patrones, si k | n;
- **(b)** los estados de un defecto de una órbita de carga Q, si k ∤ n.

Entre dos estados de P, `ΔD ≡ 0 mod n` en (a) si y solo si `Q_c ≡ 0 mod k`, y en (b) si y solo si
`Q = 0` (con `mcd(Q, n)` en general: Θ entra a través de `ΘQ/n`).

*Demostración.* En los dos casos, los estados de P son trasladados unos de otros: `b = T^d a`. Con el
corte fijo, `D(T^d a) − D(a) = d Q_tot − n·(carga de los d sitios que cruzan el corte)`, donde Q_tot es
la carga total.
- En (a), `Q_tot = nQ_c/k`, así que `d Q_tot ∈ nZ` para todo d si y solo si `k | Q_c`.
- En (b), `Q_tot = Q`.

(Comprobado con k = 3, n = 6, 9 y 12: `ΔD mod n = 0` entre todos los pares de patrones.) ∎

**Teorema Z20 (orden del enrollamiento).** Sea h de la clase (C), con carga U(1).
1. **k | n y `Q_c ≡ 0 mod k`.** Para cada momento p, los coeficientes de Taylor de `λ_p(n, Θ, ε)` de
   orden < n no dependen de Θ. El de orden n es `e^{iΘ} b_p + e^{−iΘ} b′_p`, donde b_p (y b′_p) es la
   suma, con la fase de momento, de los caminos de n saltos todos en el mismo sentido que van de P a
   P sin pasar por P:

       b_p = Σ_d B(0 → d) ω^{pd} ,   B(a → b) = Σ_{caminos} Π (amplitudes) Π_{intermedios} 1/(E_0 − E(m)) .

   Con ε real, `b′_p = b̄_p`, y la dependencia en Θ del nivel es `2 Re(b_p e^{iΘ}) ε^n + O(ε^{n+1})`.
   Con la analiticidad uniforme de Z⁗ (`|λ_p − n f_n| ≤ B` en `|ε| < ε_2`), el lema de Schwarz da

       sup_{Θ,Θ′} |λ_p(Θ) − λ_p(Θ′)| ≤ 2B (|ε|/ε_2)^n ,   y lo mismo para E_0 = min_p λ_p (ε real).

2. **k ∤ n, órbita neutra.** Lo mismo para `λ_{0,κ}`, con `b = Σ_r B(p₀ → T^r p₀) e^{iκr}` y el radio
   ε_3 de D‴: `R_n ≤ 2B(|ε|/ε_3)^n`. Esto mejora el exponente `⌈(n−1)/3⌉` del punto 3 de D‴.

*Demostración.* Los coeficientes de Taylor de λ_p coinciden con los de la serie de perturbaciones en el
sector de momento p, donde P tiene dimensión 1. El de orden m es una suma de productos de elementos de
matriz a lo largo de caminos de m saltos de P a P, que pueden volver a P por el camino (los términos de
Kato con proyectores intermedios). Cada tramo entre dos visitas a P tiene, por el Lema Z19,
`J ≡ 0 mod n`, y `|J| ≤` número de saltos `< n`. Así, `J = 0` y el tramo no ve Θ. Con m = n, solo
dan `|J| = n` los caminos de n saltos en el mismo sentido. Esos caminos no vuelven a P, porque un tramo
tendría `0 < |J| < n`. Por eso el término de Θ es exactamente el «sin retorno» `⟨P|V(RV)^{n−1}|P⟩`,
restringido a esos caminos. Para la cota: `λ_p(Θ) − λ_p(Θ′)` es analítica en `|ε| < ε_2`, está
acotada por 2B y se anula hasta el orden n − 1. ∎

*Coeficientes exactos* (`S⁺S⁻`, amplitud 2 por salto, con fracciones):

| caso | n | B | `4|b|` del fundamental |
|---|---|---|---|
| k \| n | 6 | `B(0→0) = −5008/1215`, `B(0→±1) = −464/81` | 62,314 (p = 0) |
| k \| n | 9 | `B(0→0) = 130564/10935`, `B(0→±1) = 259048/10935` | 237,28 (p = 0) |
| defecto neutro | 7 | `Σ_r B_r = 45422/729 + …` (todas positivas) | 1808,0 (κ = 0) |
| defecto neutro | 8 | todas negativas | 4110,1 (κ = 0) |

*Comprobación con diagonalización exacta.* Se calcula `R/ε^n`, con R el rango del fundamental en
24 valores de Θ:

| caso | n | ε = 0,01 | ε = 0,02 | ε = 0,05 | predicción `4|b|` |
|---|---|---|---|---|---|
| k \| n | 6 | 62,42 | 62,73 | 64,92 | 62,31 |
| k \| n | 9 | (redondeo) | (redondeo) | 253,1 | 237,3 |
| defecto neutro | 7 | 1800,7 | 1782,7 | 1663,5 | 1808,0 |
| defecto neutro | 8 | 4130 (redondeo) | 4062,1 | 3840,9 | 4110,1 |

A ε → 0 el cociente tiende a la predicción. Las diferencias a ε = 0,05 son del orden de ε²
relativo, y con n = 9 y ε ≤ 0,02, R (≈ 10⁻¹³) está en el nivel del redondeo.

*Lectura.*
- **Con carga entera, el fundamental nota Θ exactamente en el orden n.** No lo hace en el orden del
  túnel (2n/3 con k = 3), aunque el multiplete se desdobla en ese orden: cada nivel tiene periodo 2π
  por separado, y el túnel no transporta carga alrededor del anillo.
- **Con k ∤ n pasa lo mismo si gana la órbita neutra.** Con una órbita cargada, Θ entra en el orden
  del primer salto del defecto (tercer orden con `S⁺S⁻`), a través de `ΘQ/n`, y da el `n^{−2}` de D‴.
- **El «ε^{0,7n}» aparente de 7.14 es este orden n**, con coeficientes que crecen con n: 4|b| pasa de
  62 a 237 de n = 6 a 9, y de 1808 a 4110 de n = 7 a 8. A ε = 0,1 se mezclan las dos cosas.
- **La marginal.** El mismo argumento vale para `tr(P_p O)` con O local que conserva la carga: el
  operador O mueve a lo sumo `q_O` unidades por el enlace (q_O ≤ 2 en una arista de espín 1). Su
  dependencia en Θ empieza en el orden `≥ n − q_O`. Para pasar esto a la robustez del Corolario C
  habría que rehacer el teorema puente comparando Θ con Θ′ directamente, en vez de con el volumen
  infinito. En 9.16 se ve que el puente no puede hacerlo, y que para estos datos la pregunta cambia.

**9.16 La sensibilidad de p(n,Θ) por SDP: una cota local que no ve Θ** (`scripts/sdp_sensibilidad.py`).

*Por qué el puente no da el orden n.* El Teorema puente de [R] encierra p(n,Θ) entre una cota dual y una
primal, cada una comparada con el volumen infinito:
- **la dual**, con el testigo −h, varía en Θ en `O(ε^n)` por Z20;
- **la primal**, con el fundamental, varía en `O(ε^{n−2})`.

Pero la anchura de la ventana es el desajuste de tamaño finito entre la marginal del anillo y los datos,
que es del orden del túnel y no depende de Θ. Como p puede estar en cualquier punto de la ventana, la
diferencia en Θ solo queda acotada por esa anchura, que es la cota del Corolario C. Para comparar Θ con
Θ′ directamente veo dos vías:
- **por el dual exacto,** `p(n,Θ) = min_h F_h(Θ)` sobre testigos invariantes por traslación, con un
  denominador que no depende de Θ. Habría que controlar el testigo óptimo, y nada impide que sea
  crítico;
- **por el primal,** transportando un estado óptimo de Θ′ a Θ con el gauge `e^{−i(Θ−Θ′)Σ x Q_x/n}`.
  Solo falla la arista del corte, en la parte de τ que transfiere carga, y al promediar sobre
  traslaciones el error es O(1/n).

*El SDP.* Para decidir si la mejora es siquiera cierta, se resuelve el SDP de p(6,Θ). Los datos σ son
la marginal del fundamental de H(0) en el anillo de 12 sitios, al mismo ε. Basta un estado invariante
por U(1) y por traslación, porque los datos lo son en el gauge uniforme. Así, ω se parte en bloques de
carga y momento (dimensión ≤ 25), y basta la marginal de una arista. Las relaciones lineales que impone
la invariancia, como `tr₁ρ = tr₂ρ`, dejan el sistema de igualdades con rango deficiente. Se quitan con
una factorización QR con pivoteo; sin eso, el método de punto interior (Clarabel) falla.

| ε | p_loc | p(6, 0) | p(6, π/2) | p(6, π) | rango en Θ |
|---|---|---|---|---|---|
| 0,10 | 1,0000457 | 1,0000457 | 1,0000457 | 1,0000457 | 3·10⁻⁸ (tolerancia) |
| 0,20 | 1,0018490 | 1,0018490 | 1,0018491 | 1,0018490 | 1·10⁻⁷ (tolerancia) |
| 0,22 | 1,0036697 | 1,0036697 | 1,0036697 | 1,0036697 | 5·10⁻⁹ |
| 0,24 | 1,0072921 | 1,0072922 | 1,0071868 | 1,0067698 | 5,2·10⁻⁴ |
| 0,26 | 1,0137891 | 1,0126093 | 1,0105178 | 1,0080642 | 4,6·10⁻³ |
| 0,28 | 1,0231334 | 1,0162630 | 1,0097783 | 1,0033214 | 1,3·10⁻² |
| 0,30 | 1,0337356 | 1,0172690 | 1,0045594 | 0,9930411 | 2,4·10⁻² |

(Desde ε = 0,24 el solver da «optimal_inaccurate».)

*Lectura.*
- **La cota local.** Los datos σ tienen rango completo, pero su autovalor mínimo es muy pequeño. Está
  en |00⟩ y |22⟩, los sectores de carga ±2 de la arista, que tienen dimensión 1: es la probabilidad de
  dos espines iguales en los extremos, ≈ 5·10⁻⁶ a ε = 0,1 y 2·10⁻⁴ a ε = 0,2. La positividad de la
  marginal da `p ≤ p_loc = (1/9)/(1/9 − λ_min(σ))`. Es el techo por rango de [R] con λ_min en lugar de 0.
  El testigo es `−|00⟩⟨00| − |22⟩⟨22|`, y **no depende de Θ**, porque el giro no actúa en los sectores
  de dimensión 1.
- **Hasta ε ≈ 0,23 (n = 6), p(6,Θ) = p_loc para toda Θ.** La sensibilidad es **cero**, no `ε^n`: el
  óptimo está fijado por una cota local ciega al flujo.
- **Por encima, mandan las restricciones globales.** La cota local deja de alcanzarse, primero con Θ
  ≠ 0, y la sensibilidad crece deprisa: 5·10⁻⁴ a ε = 0,24 y 2·10⁻² a ε = 0,30.
- **Consecuencia.** Para estos datos, la pregunta del orden n de la sensibilidad de p queda
  desplazada. Mientras se alcance la cota local, la robustez es exacta. La cuestión de fondo pasa a
  ser **si p(n,Θ) = p_loc para ε < ε_c, con ε_c uniforme en n** (no: ver 9.17). Eso pide construir un estado del
  anillo, para toda Θ, con marginales exactas y sin peso en |00⟩ ni en |22⟩. El puente no basta: con
  p* = p_loc, su lado primal pierde `ε_P/r′ ≈ λ_min`.
- **Qué no se ha comprobado.** n = 9 no cabe con el método de punto interior: los bloques llegan
  a ~350. Habría que usar un método de primer orden (SCS), que es más lento e impreciso.

**9.17 ¿p(n,Θ) = p_loc para todo n? No: la cota de energía la desplaza desde n = 9**
(`scripts/sdp_piezas.py`). Intentar la prueba lleva a dos lemas exactos que sirven para todo n, a una
construcción que pierde O(1/ℓ) y, al final, a un contraejemplo.

*Lema Z21 (anclaje).* Sea ω un estado del anillo a Θ = 0, invariante por U(1), con un sitio j₀ en un
estado clásico puro |v⟩, es decir, `ω = |v⟩⟨v|_{j₀} ⊗ η`. Entonces el estado `U ω U†`, con el gauge
abierto `U = ⊗_j V^{(j−j₀) mod n}` y `V = e^{−iΘQ/n}`, tiene en cada arista la marginal
`(1⊗V) ρ_e (1⊗V)†`, sin error.

*Prueba.*
- En una arista interior (j, j+1), U actúa como `V^a ⊗ V^{a+1} = (V^a ⊗ V^a)(1 ⊗ V)`. Por la
  invariancia U(1), ρ_e conmuta con `Q⊗1 + 1⊗Q`, y el primer factor no actúa.
- Las dos aristas que tocan j₀ son productos `|v⟩⟨v| ⊗ (diagonal)`. Las deja fijas cualquier unitario
  diagonal, y en particular el del gauge y el giro `(1⊗V)`. ∎

Más en general basta que **ningún ciclo de coherencias del soporte de ω enrolle el anillo**: entonces
existe un potencial g(x) con `g(x′) − g(x)` igual a la carga transferida en cada par coherente de una
arista, y `U_g = e^{−iΘ g/n}` hace el giro exacto. El anclaje es el caso más simple de esto.

*Lema Z22 (pegado).* Una *pieza* de longitud ℓ es una familia de estados {η_v}, v = 0, 1, 2, invariantes
por U(1), de ℓ − 1 sitios libres entre anclajes |v⟩ y |v+ℓ mod 3⟩. Sea K_ℓ el promedio sobre v de la
suma de sus ℓ marginales de arista. Si n ∈ 3Z se escribe `n = Σ ℓ_i` y `ℓ_i ρ ∈ K_{ℓ_i}` para todo i,
entonces ρ es la marginal (con giro, para toda Θ) de un estado invariante por traslación del anillo n.

*Prueba.* Las piezas se concatenan compartiendo anclajes: el estado es un producto a través de cada
anclaje, y cada arista conserva su marginal. Se promedia sobre v, que reparte el valor inicial, y sobre
traslaciones. El giro es el Lema Z21. ∎

Con ℓ ∈ {4, 5, 6} se cubre todo n ∈ 3Z, n ≥ 6 (9 = 4 + 5, y el resto por suma). Así,
**`p(n,Θ) ≥ min_ℓ p_pieza(ℓ)` para toda Θ**, donde `p_pieza(ℓ) = max{p : ℓ ρ_p ∈ K_ℓ}` es un SDP
finito que no depende de n.

*Lo que da el SDP de piezas (ε = 0,1; datos de 9.16).*

| ℓ | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| p_loc − p_pieza(ℓ) | 7,0·10⁻² | 3,4·10⁻² | 2,3·10⁻² | 1,7·10⁻² |
| sin imponer las coherencias | 4·10⁻⁴ | 0 | 0 | 0 |
| s_max a ε = 0,2 (escala de coherencia alcanzable, diagonal fija) | — | 0,67 | 0,75 | 0,79 |

(A ε = 0,1, s_max(4) = 0,70.)

La pérdida es ≈ 0,1/ℓ y viene **toda de las coherencias**: las dos aristas que tocan un anclaje no
pueden llevar ninguna, y el resto no puede compensarlo. La diagonal sola se alcanza ya con ℓ = 4. Por
tanto, el anclaje da una cota inferior uniforme en n y en Θ, pero no el valor exacto. Cualquier corte
que desfase una arista cuesta O(1/n) de coherencia.

*La cota de energía (el contraejemplo).* Sea h(Θ) el término de arista de H(Θ), y sean
`e_R = E₀(12)/12` la energía por sitio de los datos y `ē = tr h/9 = −1/3`. Por la covarianza gauge,
`tr(h(Θ) ρ_p(Θ)) = e_R + (p − 1)(e_R − ē)`. Para cualquier estado del anillo n, esto es ≥ E₀(n,Θ)/n.
Por tanto:

  **p(n,Θ) ≤ p_E(n,Θ) := 1 + (e_R − E₀(n,Θ)/n)/(ē − e_R).**

Esta cota depende de Θ exactamente por E₀(n,Θ), es decir, en `O(ε^n)` (Z20). Con k | n, el túnel
baja la energía por sitio de un anillo pequeño, así que p_E > 1 si n < 12 y p_E < 1 si n > 12.

| ε | p_loc | p_E(6, 0) | p_E(6, π) | p_E(9, 0) | p_E(15, 0) |
|---|---|---|---|---|---|
| 0,1 | 1,0000457 | 1,000679 | 1,000661 | **1,0000278** | 0,9999985 |
| 0,2 | 1,0018490 | 1,010858 | 1,009359 | **1,0017060** | 0,9995620 |

- **n = 6:** p_E ≫ p_loc, y la cota local manda: es lo que vio el SDP de 9.16. El umbral ε ≈ 0,23
  de n = 6 no es p_E (que sigue muy por encima), sino otra restricción global.
- **n = 9:** p_E(9,0) < p_loc ya a ε = 0,1. Por tanto, **p(9,Θ) < p_loc**, y la igualdad
  `p(n,Θ) = p_loc` falla para todo n ≥ 9 (p_E decrece con n). La respuesta a la pregunta de 9.16 es
  **no**.
- **Lectura.** El techo local `λ_min(σ)` es del orden del túnel del anillo de referencia. En el
  anillo pequeño, el túnel propio es mayor y el techo local se alcanza, ciego a Θ. Desde n = 9, el
  túnel del anillo es menor que el de los datos, y manda la cota de energía. Esa cota sí ve Θ, en
  orden ε^n con el coeficiente de Z20:
  `|p_E(n,Θ) − p_E(n,Θ′)| = |E₀(n,Θ) − E₀(n,Θ′)|/(n(ē − e_R))`.
- **Qué queda abierto.** No se sabe si p(n,Θ) alcanza p_E. Si se alcanzara, la sensibilidad sería
  exactamente la de E₀, de orden ε^n. El estado que la alcanzaría tendría que ser el fundamental de
  H(Θ), y su marginal no es de la forma ρ_p(Θ). Por eso es de esperar `p < p_E`, con un testigo óptimo
  distinto de h. 9.18 responde: la diferencia es de segundo orden.

**9.18 p(n,Θ) = p_E(n,Θ) + O(ε^{2n−4}): la sensibilidad es la de la energía**
(`scripts/susceptibilidad.py`).

*Planteamiento.* Se toman datos autoconsistentes, σ = σ_n(0), la marginal del fundamental del propio
anillo. Así p(n,0) = 1 exactamente: lo alcanza el fundamental, y es el único estado con esa marginal,
porque tiene la energía mínima. Para Θ ≠ 0, sea D(Θ) la marginal del fundamental de H(Θ) sin el giro,
menos σ.

*Dos hechos medidos.*
- **D es de orden ε^{n−2}, no ε^n.** Es el orden de la marginal ya conocido. Pendiente log-log entre
  ε = 0,05 y 0,1: 4,2 en n = 6 y 7,3 en n = 9. Por eso el argumento de mezcla (absorber D con un
  estado interior) solo da `|p(n,Θ) − 1| = O(ε^{n−2})`.
- **Pero p sigue a p_E.** SDP de 9.16 con NREF = n = 6 y tolerancias 10⁻¹²; el error del solver es
  ~4·10⁻⁷, visible porque viola la cota exacta p ≤ p_E en Θ = 0.

| ε | Θ | 1 − p_E | 1 − p (SDP) | p_E − p | \|D\|_max | p_E − p predicho |
|---|---|---|---|---|---|---|
| 0,10 | π/2 | 8,94·10⁻⁶ | 9,76·10⁻⁶ | 8·10⁻⁷ (≈ error) | 2,28·10⁻⁴ | 1,7·10⁻⁶ |
| 0,10 | π | 1,79·10⁻⁵ | 2,08·10⁻⁵ | 3,0·10⁻⁶ | 3,22·10⁻⁴ | 4,5·10⁻⁶ |
| 0,15 | π/2 | 1,17·10⁻⁴ | 1,74·10⁻⁴ | 5,6·10⁻⁵ | 1,37·10⁻³ | 3,9·10⁻⁵ |
| 0,15 | π | 2,35·10⁻⁴ | 4,12·10⁻⁴ | 1,8·10⁻⁴ | 1,93·10⁻³ | 8,0·10⁻⁵ |
| 0,20 | π/2 | 7,40·10⁻⁴ | 1,38·10⁻³ | 6,4·10⁻⁴ | 5,08·10⁻³ | 4,1·10⁻⁴ |
| 0,20 | π | 1,48·10⁻³ | 3,17·10⁻³ | 1,7·10⁻³ | 7,20·10⁻³ | 6,2·10⁻⁴ |

  `(p_E − p)/|D|²_max`, calculado con los valores de la tabla, es 29, 48 y 33 en Θ = π y 15, 30 y 25 en
  π/2 (ε = 0,10, 0,15 y 0,20; el de π/2 a ε = 0,10 está al nivel del error del solver). El exceso es
  **cuadrático en D**: el cociente no tiene tendencia en ε.

*El mecanismo.* El conjunto de marginales alcanzables es convexo, y σ_n(Θ) está en su borde con normal
h(Θ). Que tr(h ρ) cambie mueve el estado a lo largo de la normal, y eso es exactamente p_E. Lo
tangencial, D, solo cuesta en segundo orden. Si se perturba el fundamental, ψ → (1 + Σ x_a O_a)ψ con
`O_a = Σ_j o_a(j,j+1)`, la marginal se mueve linealmente en x, pero la energía solo cuadráticamente.
El coste por sitio de un desplazamiento tangencial δm es `δe = ¼ δmᵀ χ⁺ δm`, con la susceptibilidad
estática `χ_ab = n⁻¹ Re⟨φ_a|(H − E₀)⁻¹|φ_b⟩`, `φ_a = (1 − P₀)O_a ψ₀`. Así:

  **p(n,Θ) = p_E(n,Θ) − δe/(ē − e) + (orden superior),   δm = (p_E − 1)τ − D(Θ).**

La columna «predicho» es esa fórmula. Acierta el orden y la escala |D|² con un factor de 0,5 a 2,7.
A ε ≥ 0,15 el SDP queda por debajo de la predicción cuadrática; eso apunta a términos de orden superior,
porque χ tiene direcciones muy rígidas (autovalores ~10⁻⁶), y no está aclarado.

*Uniformidad en n.* χ (19×19, sobre la base real de las 9×9 invariantes por U(1)) tiene exactamente 5
direcciones nulas: identidad, carga, las dos telescópicas (`A⊗1 − 1⊗A`, que fija tr₁ = tr₂) y la
propia h, porque Hψ₀ = E₀ψ₀. Son justo las que el desplazamiento no necesita, o la normal. El resto del
espectro converge en n:

| ε | n | χ_min no nulo | χ_max |
|---|---|---|---|
| 0,1 | 6 / 9 / 12 | 2,49 / 1,54 / 1,45 ·10⁻⁶ | 8,7 / 7,3 / 7,1 ·10⁻² |
| 0,2 | 6 / 9 | 1,61 / 1,03 ·10⁻⁴ | 0,24 / 0,21 |

Los otros patrones (separados en ε^n) tienen momento 2πp/3 ≠ 0, y los O_a invariantes por traslación
no los acoplan con ψ₀. Por eso χ no tiene el `ε^{−n}` que sugeriría el hueco del sector.

**Teorema Z23 (condicional; demostrado sin condiciones para n ≥ 12 en 9.24, con O(ε^{2n−8}) en lugar de O(ε^{2n−4})).** Datos autoconsistentes, k | n, ε real y pequeño. Supóngase:
- **(H_χ)** `χ ≥ χ₀ > 0` en el complemento de las 5 direcciones nulas, con χ₀ independiente de n;
- control de los términos de orden superior de ψ_x con cotas de agrupamiento uniformes.

Entonces `p_E(n,Θ) − C|D(Θ)|² ≤ p(n,Θ) ≤ p_E(n,Θ)`, y por tanto:

  **p(n,0) − p(n,Θ) = (E₀(n,Θ) − E₀(n,0))/(n(ē − e)) + O(ε^{2n−4}).**

El primer término es el de Z20, de orden ε^n con el coeficiente exacto b_p. Para n ≥ 5, 2n − 4 > n.
**La sensibilidad de p en Θ es exactamente de orden ε^n, con coeficiente conocido.**

*Prueba (esquema).*
- La cota superior es la de 9.17, y es rigurosa.
- Para la inferior, se resuelve la parte tangencial `m(x) − σ_n(Θ) = δm` por la función implícita.
  Con (H_χ), queda `|x| ≤ C|D|`, y la energía sube `O(n|D|²)`.
- La componente normal se ajusta eligiendo p, lo que pierde `δe/(ē − e) = O(|D|²)`.
- El estado `ψ_x` es invariante por traslación y por U(1), y está en el sector del fundamental. ∎

*Qué falta para que sea incondicional.*
- (H_χ) está medida (n = 6, 9, 12), no probada. Probarla pide la expansión en racimos de 9.5–9.6 para
  la susceptibilidad: es una correlación conexa integrada en el tiempo, que el formalismo de contornos
  ya controla.
- El resto de la función implícita es del mismo tipo.
- Con datos de volumen infinito σ_∞, el mismo argumento da `p(n,Θ) = p_E(n,Θ) + O(|σ_n(Θ) − σ_∞|²)`.
  Eso exige además acotar la corrección de tamaño finito de la marginal. *Corregido en 9.27:* no es
  O(ε^{n−2}), sino O(ε^{2n/3−1}), del orden del túnel (medido por ED).
- Con los datos de 9.16 (NREF = 12) en n = 9 el teorema predice `p(9,Θ) ≈ p_E(9,Θ)` (D ~ ε⁷). No se
  ha comprobado: el SDP de n = 9 no cabe con Clarabel.

*Consecuencia para 9.16–9.17.* El cuadro completo para σ fijo es:

  `p(n,Θ) ≈ min{ p_loc, p_E(n,Θ) − O(|D|²) }`.

- En anillos pequeños manda el techo local, ciego a Θ.
- En los grandes manda la energía, que ve Θ exactamente en el orden ε^n de Z20.
- El orden n buscado en el punto 1 es el verdadero, y lo da el lado dual con el testigo h. El puente
  no lo veía porque su lado primal pierde en primer orden lo que en realidad cuesta en segundo.

**9.19 Hacia (H_χ): reducción a expectativas locales** (`scripts/susceptibilidad.py`, `matrices=True`).

Notación: `X = [φ_a]`, `φ_a = (1 − P₀)O_a ψ₀`, `A = H − E₀`. Las tres matrices por sitio son
- `V = X*X/n`, la varianza estática;
- `K₁ = X*AX/n = (2n)⁻¹⟨[O_a,[H,O_b]]⟩`, el doble conmutador;
- `K₃ = X*A³X/n = (2n)⁻¹⟨[C_a,[H,C_b]]⟩`, con `C = i[H,O]`.

Así, `χ = X*A⁻¹X/n`. (Escribo K₁ y no D para no confundirlo con el D(Θ) de 9.18.)

*Lema Z24 (exacto, todo n).* En el complemento de las 5 direcciones nulas:

  **χ ⪰ K₁ K₃⁻¹ K₁**   y   **χ ⪰ V K₁⁻¹ V**.

*Prueba.* Son la desigualdad de Cauchy–Schwarz matricial (complemento de Schur de una matriz de Gram
semidefinida positiva):
- la primera, para `Y = A^{−1/2}X` y `Z = A^{3/2}X`, con `Y*Z = X*AX`;
- la segunda, para `A^{−1/2}X` y `A^{1/2}X`.

φ_a tiene momento 0 y es ortogonal a ψ₀, y los otros patrones del multiplete tienen momento ≠ 0. Por
eso φ_a está en el rango de A, y `A⁻¹` está definido allí. Las direcciones nulas de χ, K₁, K₃ y V son
las mismas: las de φ_a = 0. ∎

*Cota de K₃ (rigurosa, uniforme en n).*
- `C = Σ_j c_j` con c_j de alcance 4 y `‖c_j‖ ≤ 6‖h‖‖o‖`.
- `[H, c_j]` tiene alcance 6 y norma `≤ 10‖h‖‖c_j‖`.
- A lo sumo 9 términos c_k solapan con él.

Así, `K₃ ⪯ 3240 ‖h‖³ ‖o‖²`. Es muy holgada: numéricamente `K₃,max ≈ 4,5–6,3` para ε ≤ 0,1.

*Corolario.* **(H_χ) se reduce a (H_K): `K₁(n,ε) ⪰ κ(ε) > 0` en el complemento, uniformemente en n.**
Entonces `χ ⪰ κ²/c₃`. K₁ es una matriz 19×19 de expectativas en el fundamental de operadores de
alcance ≤ 4. Ya no intervienen la resolvente, el hueco ni las correlaciones a distancia.

*Medidas* (el cociente es lo que recupera la cota de Z24; n = 9 y 12 coinciden en 2–3 cifras):

| ε | n | χ_min | (K₁K₃⁻¹K₁)_min | cociente | K₁,min | K₁,min/ε⁴ |
|---|---|---|---|---|---|---|
| 0,01 | 9 / 12 | 5,94 / 5,93 ·10⁻¹¹ | 5,71 / 5,71 ·10⁻¹¹ | 0,96 | 9,33·10⁻¹⁰ | 0,093 |
| 0,02 | 9 / 12 | 1,00 / 1,00 ·10⁻⁹ | 0,95 / 0,95 ·10⁻⁹ | 0,95 | 1,53·10⁻⁸ | 0,096 |
| 0,05 | 9 / 12 | 5,23 / 5,11 ·10⁻⁸ | 4,70 / 4,60 ·10⁻⁸ | 0,90 | 7,0·10⁻⁷ | 0,11 |
| 0,10 | 9 / 12 | 1,54 / 1,45 ·10⁻⁶ | 1,27 / 1,20 ·10⁻⁶ | 0,82 | 1,7·10⁻⁵ | 0,17 |

La cota de Cauchy–Schwarz con la varianza, `V K₁⁻¹ V`, recupera el 99 % de χ_min (ε = 0,01–0,04). Las
excitaciones que crea cada O_a son casi de un solo modo.

*Estructura graduada.* Las 14 direcciones del complemento se reparten por órdenes en ε, igual en χ, V
y K₁:
- **8 de orden 1:** coherencias que crean un salto, de energía ~3. Aquí `V ≈ 1/6`, `K₁ ≈ 1/2` y
  `χ ≈ 1/18`.
- **2 de orden ε²:** las funciones diagonales que distingue un solo salto sobre el patrón.
  - Un salto en una arista (0,1) o (1,2) cambia tres aristas a descendentes:
    `Δ₁o = o(10) + o(21) + o(02) − o(01) − o(12) − o(20)`.
  - Un salto inverso en (2,0) da `1111`: `Δ₂o = 3o(11) − o(01) − o(12) − o(20)`.
  - Medido: `χ ≈ 0,111 ε²`, `V ≈ 0,445 ε²`, `K₁ ≈ 1,78 ε²`.
- **4 blandas de orden ε⁴:** las que solo distinguen configuraciones a dos saltos del patrón, como
  |00⟩ y |22⟩. Medido: `χ_min ≈ 0,006 ε⁴` y `K₁,min ≈ 0,093 ε⁴`.

*Por qué K₁ es positiva: forma de Dirichlet.* Para o diagonal, `[P_asc, O] = 0`, y

  `K₁(o) = (2n)⁻¹ Σ_{x∼x′} (−ε t_{xx′} ψ₀(x)ψ₀(x′)) (O(x) − O(x′))²`,

sumando sobre pares de configuraciones unidas por un salto. Cuando el signo de los saltos se puede
eliminar (H estocuástico), todos los pesos son ≥ 0 y ψ₀ > 0 en su sector (Perron–Frobenius). Entonces
K₁(o) = 0 solo si O es constante en el soporte, es decir, en una dirección nula. Los pesos
determinan los órdenes:
- un par (patrón, un salto) pesa `ε · ε = ε²`, y ve Δ₁ y Δ₂;
- un par (un salto, dos saltos) pesa `ε · ε · ε² = ε⁴`, y ve las blandas.

Esto da positividad estricta para cada n, pero no todavía uniforme.

*Uniformidad en n: el argumento.*
1. **Analiticidad.** ψ₀(ε) es analítica en un disco complejo `|ε| < ε_a`, uniformemente en n, por el
   aislamiento del multiplete de Z⁗ y el proyector de Riesz. Así K₁(n,ε) es analítica con
   `|K₁| ≤ M` en el disco.
2. **Coeficientes universales.** El coeficiente de orden m de K₁(n,·) no depende de n si
   `m < n − c`. Los diagramas de menos de n saltos no ven el anillo, como en Z19, y los observables
   tienen alcance ≤ 4.
3. **Resto.** Por Cauchy, truncando en el orden K con `n > K + c`, el resto es
   `≤ 2M(|ε|/ε_a)^{K+1}/(1 − |ε|/ε_a)`, uniforme en n.
4. **Positividad graduada.** Con K = 7, la matriz truncada es definida positiva por bloques (1, ε², ε⁴)
   si lo son sus coeficientes principales en la base graduada. Numéricamente lo son, con
   K₁,min/ε⁴ → 0,093.
5. **Conclusión.** (H_K), y con ella (H_χ), valen para `|ε| < ε_χ` y `n ≥ K + c`. Los n pequeños
   (6, 9) se comprueban directamente.

*Estado.*
- **Demostrado:** Z24, la cota de K₃, la reducción a (H_K) y la positividad de K₁ para cada n en el
  régimen estocuástico.
- **Con argumento pero sin constantes:** la uniformidad (pasos 1–5). Faltan tres cosas:
  - los coeficientes principales exactos de la parte blanda (orden ε⁴) en aritmética racional, con
    perturbaciones de Rayleigh–Schrödinger sobre un patrón, como en `orbitas_exactas.py`;
  - M y ε_a explícitos a partir de Z⁗;
  - el signo de los saltos con ε > 0 y n impar, donde H no es estocuástico de forma evidente; el
    argumento analítico no lo necesita.
- **ε_χ esperable.** Con ε_a del orden de ε₂ ≈ 8·10⁻³ y la parte blanda de orden ε⁴, el resto de
  Cauchy obliga a `ε_χ ≪ ε_a`. Como en Z⁗, el resultado será cualitativamente uniforme, con un
  radio explícito pequeño.

**9.20 Los coeficientes exactos de K₁: positividad graduada en racionales** (`scripts/k1_racional.py`).

*Método.*
- El patrón 0,1,2,… se viste por Rayleigh–Schrödinger exacto, en fracciones, hasta orden 4. Se usa
  normalización intermedia, y los denominadores son las energías enteras de P_asc.
- Con eso se forma `K₁(ε) = n⁻¹⟨ψ|O_a(H − E)O_b|ψ⟩/⟨ψ|ψ⟩` como polinomio racional en ε, truncado en
  orden 4.
- Por debajo de orden n no se alcanzan otros patrones, así que es la serie del anillo y de la cadena
  infinita.
- Base racional de los operadores de arista:
  - |x⟩⟨x| (9);
  - |x⟩⟨y| + |y⟩⟨x| (5, bloque Re);
  - i(|x⟩⟨y| − |y⟩⟨x|) (5, bloque Im).
  Re e Im se desacoplan porque ψ es real.
- En el bloque Re se quitan |01⟩, |02⟩, |10⟩, |12⟩ y |20⟩. Lo que queda es transversal a las 5
  direcciones nulas (rango 9 comprobado), incluida h(ε) = −P_asc + εT, porque su componente en P_asc
  es −1 para todo ε.
- Reducción graduada de Schur: en el orden más bajo k presente, se separan imagen y núcleo de la matriz
  principal M_k. Se forma `C − ε^k Bᵀ A⁻¹ B` como serie exacta y se repite en el núcleo.

*Resultado (exacto).*
- `E/n = −1 − (4/3)ε² − (872/405)ε⁴ + O(ε⁶)`. El orden 2 es el ya conocido.

| bloque | orden | direcciones | matriz principal |
|---|---|---|---|
| Re | 1 | 4 | identidad |
| Re | ε² | 1 | 32768/729 |
| Re | ε⁴ | 4 | S₄ (4×4 racional, abajo) |
| Im | 1 | 4 | identidad |
| Im | ε² | 1 | 32768/729 |

Total: 8 + 2 + 4 = 14, todas las del complemento; no queda ninguna sin determinar en orden ≤ 4.

`S₄` (en la base del núcleo que da sympy):
```
[[ 98776/7971615,     -2680544/39858075,    6157448/119574225,     14728/7971615   ],
 [ -2680544/39858075,  84631936/199290375, -193278112/597871125,  -750752/39858075 ],
 [ 6157448/119574225, -193278112/597871125, 913521304/1793613375, -6682216/119574225],
 [ 14728/7971615,     -750752/39858075,    -6682216/119574225,    161944/7971615   ]]
```
Sus menores principales dominantes (Sylvester) son todos positivos:

  98776/7971615,  193298432/261508830075,  135174431965184/694882570819440375,
  2⁴⁸/13677373641439044901125.

**S₄ es definida positiva, exactamente.** Sus autovalores en esta base son ≈ 6,7·10⁻⁵, 2,4·10⁻³, 0,16
y 0,80.

*Comprobaciones.*
- **No depende de n:** n = 9 y n = 12 dan la misma S₄ y el mismo E₄, número a número.
- **Coincide con la numérica.** La truncación exacta en orden ε⁴, pasada a la base ortonormal, da
  `K₁,min ≈ 0,092 ε⁴`; la numérica (9.19) da 0,093.

| ε | autovalores pequeños de K₁: exacto O(ε⁴) / numérico (n = 12) |
|---|---|
| 0,01 | 9,22/9,33 ·10⁻¹⁰; 4,444/4,451 ·10⁻⁹; 8,04/8,08 ·10⁻⁹; 6,001/6,008 ·10⁻⁸; 1,7781/1,7781 ·10⁻⁴ |
| 0,02 | 1,47/1,53 ·10⁻⁸; 7,11/7,15 ·10⁻⁸; 1,30/1,33 ·10⁻⁷; 9,60/9,64 ·10⁻⁷; 7,1155/7,1155 ·10⁻⁴ |

- **n = 6 es especial.** La serie de orden 4 es degenerada: dentro de 4 saltos se alcanza otra
  configuración con la energía del patrón. Hay que tratarlo aparte (ED, 9.18–9.19).

**Teorema Z25.** Sea k = 3, n ∈ 3Z con n ≥ 9. Supóngase que:
- el fundamental es analítico en `|ε| < ε_a` con `|K₁(n,ε)| ≤ M`, uniformemente en n (por Z⁗ y el
  proyector de Riesz).

Entonces existe `ε_χ = ε_χ(M, ε_a) > 0` tal que, para `|ε| < ε_χ` y todo n ≥ 9,

  **K₁(n,ε) ⪰ κ ε⁴ en el complemento de las nulas,  κ > 0 independiente de n,**

y por Z24, `χ ⪰ κ²ε⁸/c₃`: **(H_χ) vale**. Con la cota de K₃ medida en lugar de c₃, y la de Cauchy–Schwarz
con V, que es casi exacta, el orden real es χ_min ≈ 0,006 ε⁴.

*Prueba.*
1. Los coeficientes de Taylor de K₁(n,·) hasta orden 4 son los racionales de arriba para todo n ≥ 9.
   Un diagrama de orden ≤ 4, con observables de alcance ≤ 4, no da la vuelta al anillo ni alcanza otro
   patrón, como en Z19. n = 9 y 12 lo confirman.
2. El resto se acota por Cauchy: `‖K₁ − T₄‖ ≤ 2M(|ε|/ε_a)⁵/(1 − |ε|/ε_a)`, uniforme en n.
3. **Lema de positividad graduada.** Sea `K(ε) = T₄(ε) + R(ε)`, donde las matrices principales de la
   reducción de Schur de T₄ son definidas positivas en los órdenes 0, 2 y 4 y cubren todo el espacio.
   Entonces `T₄ ⪰ 2κε⁴` para |ε| pequeño, y `K ⪰ κε⁴` si `‖R‖ ≤ κε⁴`.
   La prueba aplica la congruencia por bloques `diag(1, ε, ε²)` en la base graduada: la matriz
   reescalada tiende a `diag(M₀, M₂, S₄) ≻ 0` cuando ε → 0.
4. Por 2, `‖R‖ ≤ κε⁴` vale si `|ε| ≤ ε_χ ∼ (κ/2M)·ε_a⁵`. ∎

*Lo que queda.*
- **M y ε_a explícitos** a partir de Z⁗. Es la única constante no calculada.
- **ε_χ será muy pequeño.** Con ε_a del orden de ε₂ ≈ 8·10⁻³, el factor ε_a⁵ lo hace minúsculo. Se
  mejoraría llevando la serie racional a orden 6 o 7, o acotando el resto directamente con los
  contornos en lugar de con Cauchy.
- **n = 6.** Basta la ED de 9.19, o una reducción graduada degenerada.

*Consecuencia.* Con Z25, la hipótesis (H_χ) de Z23 queda demostrada para `|ε| < ε_χ`, módulo las
constantes de analiticidad de Z⁗. En ese régimen, la sensibilidad de p(n,Θ) en Θ es exactamente
`(E₀(n,Θ) − E₀(n,0))/(n(ē − e)) + O(ε^{2n−4})`. Falta aún el control de orden superior de la
función implícita de Z23, que es del mismo tipo: analiticidad y Cauchy.

**9.21 El radio ε_χ explícito.**

*Qué da Z⁗ y qué falta.* Z⁗ certifica la convergencia de Kotecký–Preiss del gas de contornos en
espacio-tiempo para ε complejo con |ε| < ε₂ = 8,2·10⁻³. De ahí salen la analiticidad y las cotas de
los **autovalores** λ_p, no las de los vectores propios. Pero K₁ es una matriz de **expectativas
locales**: `K₁,ab = ⟨k_ab⟩` con `k_ab = ½Σ_{l,m}[o_a(0),[h_l,o_b(m)]]`, que vive en ≤ 6 sitios. Una
expectativa local, en la representación de espacio-tiempo, es un cociente Z_A/Z con una inserción en
t = 0:

  `⟨A⟩ = Σ_{x,y} A_{xy} ρ_{yx}`,   `ρ_{yx} = Z(y | x)/Z`,

donde Z(y | x) fija los valores y (arriba) y x (abajo) en los r sitios del soporte. Esto da el lema que
hace falta.

*Lema Z26 (inserción; esquema — **retirado en 9.22**: tiene dos agujeros; lo sustituye el Lema Z26 por el autovalor).* Para |ε| < ε_a ≤ ε₂ y A con soporte en r sitios,
`|ρ_{yx}(ε)| ≤ e^{2rμ}`, con μ la masa de KP por sitio a través de un instante, y por tanto
`|⟨A⟩_ε| ≤ ‖A‖_{ℓ¹} e^{2rμ}`.

*Prueba (esquema).*
- log Z(y|x) − log Z es la suma de los racimos que tocan el corte.
- El corte añade, como mucho, contornos de frontera con el mismo peso por unidad de duración.
- KP con la función a(γ) certificada acota esa suma por r·μ en cada lado del corte. ∎

Tomo `μ = 5M₂`, con M₂ la masa certificada por `cond_cert`, porque la condición de Z16 es
`5M₂ ≤ a`:

| ε_a | M₂ (certificado) | e^{2·6·5M₂} |
|---|---|---|
| 4·10⁻³ | 7,39·10⁻⁴ | 1,045 |
| 6·10⁻³ | 1,83·10⁻³ | 1,116 |
| 8·10⁻³ | 3,46·10⁻³ | 1,231 |

*M explícito.* `|K₁,ab(ε)| ≤ M_ab = (N₀ + |ε| N₁)_ab e^{60M₂}`, con N₀ y N₁ las normas ℓ¹ de las
partes de orden 0 y 1 en ε de k_ab. Se calculan exactamente en dispersa sobre 7 sitios: máximos 378 y
2646. En la base ortonormal de W, la norma de Frobenius de la matriz de cotas es
**F = 1058 / 1144 / 1278** para ε_a = 4 / 6 / 8 ·10⁻³.

*La serie a orden 6.* `k1_racional.py 12 6` y `15 6`.
- **Universalidad.** n = 12 y n = 15 dan los mismos coeficientes en todas las entradas, en los órdenes
  0 a 6.
- **Energía.** `E/n = −1 − (4/3)ε² − (872/405)ε⁴ − (12646/1215)ε⁶ + O(ε⁸)`.
- **Casos degenerados.** n = 9 es degenerado a orden 6 (como n = 6 a orden 4): dentro de 6 saltos
  alcanza otra configuración con la energía del patrón.

Por tanto, la serie universal de orden K vale para:

| K | n |
|---|---|
| 4 | n ≥ 9 |
| 6 | n ≥ 12 |
| 8 | n ≥ 15 |

La regla es K < 2n/3. Es justo cuando un camino de K saltos puede rehacer el patrón desplazado: n = 6
falla en orden 4, n = 9 en orden 6 y n = 12 en orden 8.

*Orden 8 (`scripts/k1_serie_rapida.py`).*
- **El método.** Es la misma serie, pero en coma flotante con vectores dispersos sobre las
  configuraciones a ≤ K + 2 saltos del patrón: 6,2·10⁵ en n = 15 y 3,8·10⁶ en n = 18.
- **Validación.** Reproduce los coeficientes exactos de orden ≤ 6 con un error ≤ 4·10⁻¹² (n = 12, 15).
- **Universalidad.** n = 15 y n = 18 coinciden en todas las entradas hasta el orden 8, con
  diferencias ≤ 6·10⁻¹⁰ sobre coeficientes de hasta 1,5·10³, es decir, el redondeo.
- **Energía.** `E/n = … − 10,40823ε⁶ − 58,14187ε⁸` (el de ε⁶ es −12646/1215).
- **El límite.** El orden 10 pediría n ≥ 18 a profundidad 12, más de ~50 GB; no cabe.

*La cota inferior del polinomio.* λ_T(ε) es el autovalor mínimo de T_K(ε) en W, con el producto de
Hilbert–Schmidt, calculado con 100 dígitos (mpmath) desde los racionales exactos. En coma flotante se
pierde por debajo de ε ~ 10⁻⁴: allí aparecía un falso 0,025.

| ε | 10⁻¹⁴ | 10⁻¹⁰ | 10⁻⁷ | 10⁻⁵ | 10⁻³ | 10⁻² | 3·10⁻² | 10⁻¹ |
|---|---|---|---|---|---|---|---|---|
| λ_T/ε⁴ (K = 4) | 0,04066 | 0,04066 | 0,04066 | 0,04066 | 0,04066 | 0,04083 | 0,04216 | 0,0536 |
| λ_T/ε⁴ (K = 6) | 0,04066 | 0,04066 | 0,04066 | 0,04066 | 0,04066 | 0,04110 | 0,04458 | 0,0780 |

El límite 0,04066 es el autovalor mínimo de la parte graduada exacta (9.20), en la base de W. Tomo
`κ = 0,04` en `(0, 0,1]`.

*El radio.* Por Cauchy, `‖K₁ − T_K‖ ≤ F r^{K+1}/(1 − r)` con `r = |ε|/ε_a`. La condición
`‖K₁ − T_K‖ ≤ (κ/2)ε⁴` da `K₁ ⪰ (κ/2)ε⁴ = 0,02 ε⁴` en W, y por Z24 (H_χ). Es decir:

  **ε_χ^{K−3} = (κ/2) ε_a^{K+1} (1 − ε_χ/ε_a)/F.**

| ε_a | K = 4 (n ≥ 9) | K = 6 (n ≥ 12) | K = 8 (n ≥ 15) |
|---|---|---|---|
| 4·10⁻³ | 1,9·10⁻¹⁷ | 6,8·10⁻⁸ | — |
| 6·10⁻³ | 1,4·10⁻¹⁶ | 1,7·10⁻⁷ | 1,1·10⁻⁵ |
| 8·10⁻³ | 5,1·10⁻¹⁶ | 3,2·10⁻⁷ | **1,84·10⁻⁵** |

Con K = 8, `λ_T/ε⁴ = 0,04066` en todo (10⁻¹⁰, 10⁻³) (100 dígitos, órdenes 7–8 en coma flotante). Los
órdenes 7–8 en coma flotante no estorban: su error relativo es ~10⁻¹², y en ε ~ 10⁻⁵ su
contribución es ~10⁻³⁵. Recortar las normas ℓ¹ al soporte real no cambia F, porque los k_ab ocupan de
verdad 6 sitios.

**Resultado.** Con Z26, para n ∈ 3Z y

  **|ε| < ε_χ = 1,84·10⁻⁵ (n ≥ 15),   3,2·10⁻⁷ (n = 12),   5·10⁻¹⁶ (n = 9),**

se tiene:
- `K₁ ⪰ 0,02 ε⁴` en el complemento de las nulas;
- `χ ⪰ (0,02 ε⁴)²/c₃`, es decir, (H_χ).

n = 6 se trata aparte.

*Qué es riguroso y qué no.*
- **Exacto:**
  - los coeficientes hasta orden 6 (fracciones);
  - su igualdad para n = 12 y n = 15;
  - los de orden 7–8, en coma flotante validada (≤ 4·10⁻¹² frente a los exactos), iguales para
    n = 15 y n = 18;
  - la positividad graduada (Sylvester, 9.20);
  - las normas ℓ¹ y las masas M₂ (certificadas con intervalos en 9.13 e).
- **Comprobado en rejilla, no en intervalo:** `λ_T ≥ 0,04 ε⁴` en (0, 0,1]. Lo que falta es una cota de
  Lipschitz entre los puntos de la rejilla. Por debajo de ~10⁻⁵ lo da la estructura graduada exacta,
  porque λ_T/ε⁴ tiende a un límite positivo.
- **Esquema:**
  - el Lema Z26 y la elección μ = 5M₂. Es la pieza que falta escribir con detalle: los contornos de
    frontera de la inserción no están en 9.5–9.13;
  - la universalidad para todo n por encima del umbral (K < 2n/3), que se apoya en el argumento de Z19 y
    en las comprobaciones n = 12, 15, 18.

*Cómo mejorarlo.* ε_χ es pequeño por tres razones, cada una con su remedio:
- **El orden K.** Hecho hasta K = 8 (de 3,2·10⁻⁷ a 1,84·10⁻⁵). K = 10 daría ~10⁻⁴, pero no cabe en
  memoria con este método.
- **Las normas ℓ¹.** Son crudas (F ~ 10³). Con la estructura de carga se pueden afinar mucho.
- **Cauchy.** Acotar el resto directamente con los contornos, que decaen como `(|ε|/ε₂)^{K+1}` con un
  prefactor de orden 1 en vez de F, daría un ε_χ del orden de 10⁻⁴.

**9.22 Lema Z26: χ es analítica y acotada, por el autovalor de H + tO** (`scripts/k1_serie_rapida.py`
modo `chi`, `k1_racional.chi_series`).

*Por qué se cambia el enunciado de 9.21.* La versión de Z26 con inserción tiene dos agujeros:
- **Las dos mitades no se separan.** Las paredes que cruzan t = 0 fuera de la ventana son a la vez de
  la mitad de antes y de la de después, así que `G₊·G₋` no acota.
- **El marco que nace en la inserción no suma 0 mod 3.** Eso rompe el «≥ 2 paredes por marco» y la
  regla de muerte de Z15c.

Se sustituye por un enunciado sobre **autovalores**, que es lo que Z⁗ controla, y se aplica
directamente a χ. Así deja de hacer falta pasar por K₁ y Z24.

*Notación.* o = Σ_a c_a o_a, con c real y de norma 1 en la base de Hilbert–Schmidt. `O = Σ_j o(j,j+1)`,
`H_t = H + tO`, y `λ₀(ε,t)` es el autovalor del fundamental en el sector de momento 0. Para ε real,

  `χ(c) = (1/n)⟨φ|(H − E₀)⁻¹|φ⟩ = −(1/2n) ∂²_t λ₀(ε,t)|_{t=0}`

(segundo orden de un autovalor simple y aislado; O conserva el momento). Para ε complejo, la segunda
igualdad define la continuación analítica.

**Lema Z26.** Sean `α = √2`, `d = √2` y `ε_ef = ε_a + αt′/4`. Supóngase que los certificados de Z16 y Z17
(9.13 e) pasan en ε_ef, con todas las λ desplazadas en `δ = d·t′`. Entonces, para todo n ∈ 3Z, todo
c de norma 1 y todo ε complejo con |ε| ≤ ε_a:

  **|χ(c; ε)| ≤ M_χ := (2W̄ + 2q/n)/t′²,**

donde `W̄ = W_anc(4ε_ef e^{3aκ}, 3a + δ)` y q es la cota del gas de eventos de Z17, ambos en esos
parámetros. Además, χ(c; ·) es analítica en |ε| < ε_a.

*Prueba.*
1. **Gauge telescópico de la parte diagonal.**
   - Sea ō la media de o_d sobre los pares ascendentes (01), (12) y (20). Las desviaciones
     `dev(a) = o_d(a, a+1) − ō` suman 0, así que existe g con `g(a+1) − g(a) = dev(a)`.
   - Se define `o′_d(x, y) = o_d(x, y) − ō − (g(y) − g(x))`. En el anillo,
     `Σ_j (g(s_{j+1}) − g(s_j)) = 0`, así que `H_t = H + tO′ + t n ō` exactamente.
   - Ahora `o′_d = 0` en toda arista ascendente: comprobado, < 2·10⁻¹⁶.
   - En las paredes, `|o′_d| ≤ d‖c‖` con d = √2, la norma máxima de las funcionales `c ↦ o′_d(x)`.
   - Por tanto, el coste de cada pared es `Re ≥ 1 − d|t|`, y el de los patrones es exactamente 0. La
     traslación sigue permutando las fases.
2. **Los saltos de O son de la clase de Z15c.** La parte fuera de la diagonal de o mueve δ = 1 o 2
   unidades de carga en un enlace. Los incrementos de las tres aristas cambian en `(+δ, −2δ, +δ)`.
   - **El marco suma 0** (punto 1 de Z15c).
   - **Un nacimiento desde el patrón deja 3 paredes:** (2,2,2) si δ = 1 y (0,0,0) si δ = 2.
   - **Un marco solo muere desde 3 paredes,** porque el salto cambia 3 aristas.
   - **Cada arista de la ventana cambia en una cantidad ≢ 0 mod 3,** así que valen las clases
     `ΔE ∈ [3 − 2w, 3 − w]` de 9.13 d).
   - **Amplitud.** En cada posición, la suma sobre los tipos de salto de |amplitud| es la suma por
     columnas de la parte fuera de la diagonal: `≤ 4|ε| + α|t|`. Las dos entradas de una columna usan
     coordenadas disjuntas de c, así que α = √2.

   Así, la recurrencia V(E, r, f), la cola y KP valen con `A|ε| → 4ε_ef` y con las paredes a coste
   `1 − δ`, es decir, λ → λ + δ en cada denominador `(1 − λ)E`.
3. **Z17 para H_t.** Con los certificados, los pasos 1–6 de 9.6 dan, para |ε| ≤ ε_a y |t| ≤ t′:
   - λ₀ es simple, aislado y analítico en (ε, t);
   - `λ₀/n = (−1 + tō) + c_n(ε,t) + t₀/n`, con `|c_n| ≤ W̄` (la densidad de racimos, 9.13 b) y
     `|t₀| ≤ q`.
4. **Cauchy en t.** El coeficiente de t² de `G(t) = λ₀/n` cumple
   `|G₂| ≤ sup_{|t|=t′} |G(t) − G(0) − tō| / t′² ≤ (2W̄ + 2q/n)/t′²`, y `χ = −G₂`. Es analítico en ε
   porque λ₀ lo es en (ε, t). ∎

*El certificado* (`scripts/certifica_tO.py`, parámetros de Z17: c = 0,2, a = 0,01244, κ = 1, s = σ = 0,1; con
intervalos). Pasa en toda la rejilla con ε_ef ≤ 8,06·10⁻³. Por ejemplo:

| ε_a | t′ | ε_ef | δ | W̄ | q | M_χ (n ≥ 15) |
|---|---|---|---|---|---|---|
| 3·10⁻³ | 1,2·10⁻² | 7,24·10⁻³ | 0,017 | 3,4·10⁻⁴ | 8,3·10⁻³ | 12,4 |
| 6,5·10⁻³ | 4·10⁻³ | 7,91·10⁻³ | 0,0057 | 4,0·10⁻⁴ | 9,9·10⁻³ | 132 |
| 7·10⁻³ | 3·10⁻³ | 8,06·10⁻³ | 0,0042 | 4,1·10⁻⁴ | 1,0·10⁻² | 244 (218 con n ≥ 18) |

Domina el término del túnel, 2q/n. La cota q de 9.13 c) es gruesa: no usa que t₀ decae como e^{−cn}.

*La serie de χ.* Se resuelve `(H − E)ξ = φ` orden a orden: `(H₀ − E₀)ξ_m = φ_m − Tξ_{m−1} + Σ_k E_k ξ_{m−k}`,
con `⟨ψ|ξ⟩ = 0`. La compatibilidad (componente del patrón nula) es exactamente `⟨ψ|φ⟩_m = 0`, y se
comprueba en cada orden. ξ llega a K + 2 saltos, así que la serie es universal si `K + 2 < 2n/3`.

| comprobación | resultado |
|---|---|
| serie O(ε⁶) frente a la χ numérica (n = 12) | 4–5 cifras a ε = 0,01; 3 a 0,03 |
| exacta (fracciones, n = 12, orden 4) frente a coma flotante (n = 15) | 7·10⁻¹⁵ |
| n = 15 frente a n = 18, órdenes ≤ 6 | ≤ 2·10⁻¹³ |
| n = 18, orden 8 (3,8·10⁶ configuraciones, 10 GB, 3 min) frente a orden 6 | ≤ 2·10⁻¹³ |

*Positividad graduada de χ, exacta* (fracciones, n = 12, orden 4). Los menores de Sylvester de las
matrices principales son todos positivos:

| bloque | orden | dir. | menores de Sylvester |
|---|---|---|---|
| Re | 1 | 4 | 1/729, 1/531441, 1/387420489, 1/282429536481 |
| Re | ε² | 1 | 8/729 |
| Re | ε⁴ | 4 | 264227/102036672000, 334670779/18828635765400000000, 22221153917/2084647712458321125000000, 4096/7915146783240188021484375 |
| Im | 1 | 4 | 1/729, … (como Re) |
| Im | ε² | 1 | 8/729 |

En W, con el producto de Hilbert–Schmidt (100 dígitos): `λ_T/ε⁴ = 0,002972` en (10⁻¹², 10⁻³), y crece
por encima. Se toma `κ_χ = 0,9 · 0,002972 = 0,00267`.

**Teorema Z27 ((H_χ) con radio explícito).** Sea n ∈ 3Z y ε real con:

  **|ε| < ε_χ = 1,2·10⁻⁵ (n ≥ 18, K = 8; ε_a = 7·10⁻³, t′ = 3·10⁻³),
  1,7·10⁻⁷ (n = 15, K = 6; ε_a = 6,5·10⁻³, t′ = 4·10⁻³).**

Entonces `χ(c) ≥ (κ_χ/2) ε⁴ = 1,3·10⁻³ ε⁴` para todo c ∈ W de norma 1.

*Prueba.* Para cada c, la función `χ(c; ·)` es analítica con `|χ| ≤ M_χ` (Z26). Sus coeficientes hasta
orden K son los universales. Por Cauchy,
`χ(c; ε) ≥ cᵀT_K(ε)c − M_χ r^{K+1}/(1 − r) ≥ λ_T(ε) − M_χ r^{K+1}/(1 − r)`, con `r = ε/ε_a`. El radio
es el de `M_χ r^{K+1}/(1 − r) = (κ_χ/2)ε⁴`. ∎

*Qué es riguroso.*
- **Riguroso:**
  - Z26, con los certificados de Z16 y Z17 en intervalos y los parámetros desplazados;
  - el gauge y las constantes α y d (cálculo exacto de normas);
  - la positividad graduada de χ (Sylvester en fracciones).
- **Numérico validado:**
  - los coeficientes de orden 5–8, en coma flotante, con error relativo ~10⁻¹³. En ε ~ 10⁻⁵ pesan
    ~10⁻³⁰, muy por debajo del margen;
  - `λ_T ≥ κ_χ ε⁴`, comprobado en una rejilla y no en intervalos. Debajo de ~10⁻³ lo fija la estructura
    graduada exacta.
- **Argumento:** la universalidad de los coeficientes para todo n con `K + 2 < 2n/3`. Un camino de
  menos saltos no da la vuelta ni alcanza otra configuración degenerada. Está comprobada entre n = 15 y
  18 hasta orden 6. Para orden 8, n = 21 no cabe en memoria.
- **No cubiertos aquí:** n = 6, 9 y 12. Ahí la serie que hace falta (orden ≥ 4 con K + 2 < 2n/3) no
  existe o es demasiado corta. Se cierran en 9.23 (Z28, Z29), junto con mejores radios para n = 15 y 18.

*Comparación con 9.21.* La vía de K₁ daba 1,84·10⁻⁵, pero dependía de la inserción, que no estaba
probada. La vía del autovalor da 1,2·10⁻⁵ sin ese lema. Cuesta poco porque la cota de M_χ (~200) es
mucho mejor que la de F (~1300), lo que compensa el κ menor de χ (0,0027 frente a 0,04). Cómo seguir
mejorando:
- acotar t₀ con su decaimiento en n, en lugar de q, lo que bajaría M_χ unas 10 veces: ×1,6 en ε_χ;
- llegar a orden 10, lo que pide n ≥ 21 a profundidad 12 y no cabe;
- mejores parámetros de Z17 para H_t.

**9.23 Los n pequeños: Bauer–Fike con la serie de cada n** (`scripts/k1_serie_rapida.py` modo `chi3`,
`k1_racional` con `TRES = True`, `scripts/radio_chi_bf.py`).

*Por qué hace falta otra cosa.* Con n fijo, la serie universal se rompe cuando un camino de K + 2
saltos rehace el patrón desplazado, es decir, cuando K + 2 ≥ 2n/3. Pero para n fijo eso no es una
degeneración nueva: las configuraciones degeneradas son los otros dos patrones. En el sector de momento
0 hay una sola combinación de los tres.

*RS en el sector de momento 0.* Se parte de `ψ₀ = Σ_c |c⟩`, la suma de los tres patrones, con
normalización intermedia `ψ_m(c) = 0` para m ≥ 1. Así:
- la componente en los patrones de cada lado derecho es la misma para los tres; se comprueba, y da
  E_m;
- para ξ, la compatibilidad pide que se anule en los tres patrones;
- la ortogonalidad reparte `⟨ψ|ξ⟩ = 0` por igual entre ellos.

*Comprobaciones.*
- **Regresión.** Con n = 15 y K = 6 reproduce la serie de un solo patrón (6·10⁻¹⁴).
- **Frente a la ED.** La serie de orden 8 coincide con la χ por ED en 5–6 cifras en n = 6, 9 y 12
  (ε = 0,01), y en 4–5 cifras a ε = 0,03.
- **Dónde se separan de la universal.** n = 12 en el orden 6 y n = 15 en el orden 8: justo en
  K + 2 = 2n/3.
- **Energía.** n = 6 tiene `E₄/n = −2632/405`, frente al universal −872/405 (el exceso es el túnel en
  orden 4 = 2n/3).
- **Un fallo que se corrigió.** En la versión en fracciones, un `0/3` entero daba 0.0 en coma flotante
  y contaminaba la serie exacta; ahora es `Fraction`.

*Positividad graduada, exacta* (fracciones, orden 4, tres patrones). En n = 6, 9 y 12, todos los
menores de Sylvester son positivos en los bloques Re (órdenes 1, ε², ε⁴: 4 + 1 + 4 direcciones) e Im
(1, ε²: 4 + 1).

| n | menor mínimo en ε⁴ |
|---|---|
| 6 | 4,1·10⁻²¹ |
| 9 | 5,2·10⁻²² |
| 12 | 5,2·10⁻²² |

La serie exacta coincide con la de coma flotante a ≤ 1,4·10⁻¹⁴.

**Lema Z28 (analiticidad para n fijo, Bauer–Fike).** En el sector de carga 0 y momento 0, el hueco de
H₀ = −P_asc sobre los patrones es 2.

*Prueba del hueco.* Los incrementos del anillo suman 0 mod 3. Con una sola pared de incremento d ∈
{0, 2}, la suma sería `(n − 1) + d ≢ 0` (n ≡ 0 mod 3). Así que hay al menos 2 paredes. Además, el
nivel −n tiene una sola combinación de momento 0.

Sea `V = εT + tO` con `‖T‖ ≤ 2√2·n` (la norma del bloque de carga 2 de un enlace es 2√2) y
`‖O‖ ≤ n‖o‖ ≤ n`. Si `2√2·n|ε| + n|t| < 1`, entonces, por Bauer–Fike (H₀ es normal en el sector) y por
continuidad del recuento:
- λ₀(ε,t) es el único autovalor en `|z + n| < 1`;
- es analítico, con `|λ₀ + n| ≤ ‖V‖`.

Por Cauchy en t, `|χ(c; ε)| ≤ M_n = (4√2·ε_a + t′)/t′²` para |ε| ≤ ε_a, con
`t′ = 1/n − 2√2·ε_a`. Es elemental: no usa contornos. ∎

*El radio* (K = 8; órdenes ≤ 4 exactos, 5–8 en coma flotante validada; κ = 0,9·min λ_T/ε⁴). Se
optimiza en ε_a:

| n | λ_T/ε⁴ | ε_a | t′ | M_n | **ε_χ(n)** |
|---|---|---|---|---|---|
| 6 | 0,003196 | 4,78·10⁻² | 3,14·10⁻² | 306 | **3,6·10⁻⁴** |
| 9 | 0,002972 | 3,19·10⁻² | 2,09·10⁻² | 459 | **1,6·10⁻⁴** |
| 12 | 0,002972 | 2,39·10⁻² | 1,57·10⁻² | 612 | **8,9·10⁻⁵** |
| 15 | 0,002972 | 1,91·10⁻² | 1,26·10⁻² | 765 | **5,7·10⁻⁵** |
| 18 | 0,002972 | 1,59·10⁻² | 1,05·10⁻² | 918 | **4,0·10⁻⁵** |

Con K = 6 los radios son de 25 a 40 veces menores (n = 6: 1,4·10⁻⁵; n = 12: 2,2·10⁻⁶).

*Cómo encaja con Z27.* ε_a ~ 1/(2√2n) gana a los ~7·10⁻³ de Z⁗ mientras n ≲ 50. Con la serie
universal (n ≥ 18), Bauer–Fike da ε_χ(n) ≈ 4,0·10⁻⁵·(18/n)^{2,2}, que cae por debajo de los 1,2·10⁻⁵
de Z27 hacia n ≈ 30.

**Teorema Z29 ((H_χ) para todo n).** Para todo n ∈ 3Z con n ≥ 6 y todo ε real con
**|ε| < 1,2·10⁻⁵**, se cumple `χ(c) ≥ 1,3·10⁻³ ε⁴` en W (κ ≥ 0,00267/2). Por separado, el radio vale:
- 3,6·10⁻⁴ (n = 6), 1,6·10⁻⁴ (n = 9), 8,9·10⁻⁵ (n = 12), 5,7·10⁻⁵ (n = 15), por Z28;
- `max(1,2·10⁻⁵, ε_χ^{BF}(n))` para n ≥ 18, por Z27 y Z28.

*Qué es riguroso.* Lo mismo que en 9.22:
- Z28 es elemental;
- la positividad graduada es exacta;
- los coeficientes de orden 5–8 y la cota `λ_T ≥ κε⁴` son numéricos validados (rejilla, no
  intervalos);
- la universalidad para n ≥ 18 se apoya en el argumento de los caminos.

**Con esto, (H_χ), la hipótesis de Z23, queda demostrada para todo n.** Lo que falta para que Z23 sea
incondicional es el control del orden superior de la función implícita (9.18).

**9.24 Z23 sin condiciones (n ≥ 12): Newton–Kantorovich con las cotas de analiticidad**
(`scripts/z23_kantorovich.py`).

*La construcción (sustituye al esquema de 9.18).* Con datos autoconsistentes, σ = σ_n(0):
- **La familia.** Sea ψ_x el fundamental (momento 0) de `H_x(Θ) = H(Θ) + Σ_{a∈W} x_a O_a(Θ)`, con
  `o_a(Θ) = (1⊗V)o_a(1⊗V)†`, de modo que H_x(Θ) es la versión girada de H_x(0).
- **Su marginal.** Por Hellmann–Feynman, la marginal sin giro de ψ_x tiene coordenadas
  `m_W(x) = ∇e(x)` y `m_h(x) = e(x) − x·∇e(x)`, con `e = λ₀/n`. Las 4 coordenadas triviales (traza,
  carga, telescópicas) coinciden solas.
- **Lo que hay que resolver.** `ρ_p(Θ)` es alcanzable con un estado puro si
  `∇e(x) = σ_W + (p − 1)τ_W` y `e − x·∇e = σ_h + (p − 1)τ_h`. Son 15 ecuaciones con 15 incógnitas
  (x ∈ W, de dimensión 14, y p).
- **Por qué funciona.** La hessiana en 0 es `−2χ(Θ)`, invertible en W por Z29. La columna de p es τ,
  cuya componente normal es `τ_h = e_R − ē ≈ −2/3 ≠ 0`.

*Las cotas, todas por analiticidad* (Z26 en general; Bauer–Fike, Z28, para n fijo). En el polidisco
`|ε| ≤ ε_a`, `|x| ≤ t′` (x complejo) se tiene `|e(x) − (−1 + x·ō)| ≤ A_e`, con
`A_e = W̄ + q/n` (Z26) o `A_e = 2√2ε_a + 2t′` (Bauer–Fike). Por Cauchy, en la bola |x| ≤ t′/2:
- **Hessiana:** `‖Hess e‖ ≤ 16A_e/t′²`.
- **Tercera derivada:** `‖∂³e‖ ≤ L := 216A_e/t′³`. Es la derivada direccional, `48A_e/t′³`, por el
  factor 4,5 de polarización de una forma trilineal simétrica.
- **D, por la estructura de corrientes (Z19–Z20 con O):** `f(ε, x) = e(x; Θ) − e(x; 0)` solo tiene
  monomios `ε^k z^l` con `k + 2l ≥ n`. En la serie de Kato, la fase es `e^{iΘJ/n}` con `J ∈ nZ`, y
  `|J| ≤ k + 2l`, porque un salto doble de O aporta 2. Así, `∂_z f|₀` empieza en `ε^{n−2}` y, por
  Cauchy en las dos variables,

    `|D_W| ≤ (2A_e/t′) r^{n−2}/(1 − r)`,  `|p_E − 1| ≤ 2A_e r^n/((1 − r)|τ_h|)`,  `r = |ε|/ε_a`.

  Es el orden ε^{n−2} medido en 9.18 (pendientes 4,2 y 7,3).
- **χ(Θ) frente a χ(0):** `|χ(Θ) − χ(0)| ≤ 4A_e r^{n−4}/(t′²(1 − r))`, despreciable frente a κ′ε⁴ si
  n ≥ 9.

*Newton–Kantorovich.*
- **El sistema.** `g(x) = ∇_W e(x) − σ_W − (p − 1)τ_W`, con `g(0) = d = D_W − (p − 1)τ_W`,
  `β = ‖J⁻¹‖ ≤ 1/(2κ″ε⁴)` (`κ″ = κ′ − |χ(Θ) − χ(0)|/ε⁴`, κ′ = 1,3·10⁻³) y L como arriba.
- **La conclusión.** Si `β²L|d| ≤ ½` y `2β|d| ≤ t′/2`, hay una solución única x* con `|x*| ≤ 2β|d|`.
- **El p.** La ecuación de energía fija p por un punto fijo contractivo: p entra en d solo a través
  de `(p − 1)τ_W`, y `∂m_h/∂x = −xᵀHess` es O(|x*|).
- **El déficit.** Por la identidad
  `e(x) − e(0) − x·∇e(x) = −∫₀¹ s xᵀ Hess(sx) x ds`, con `H₀x = −d + O(L|x|²/2)`:

    `0 ≤ p_E − p ≤ (½|x*||d| + 0,6L|x*|³)/|τ_h| = O(|D|²/ε⁴) = O(ε^{2n−8})`.

**Teorema Z23 (sin condiciones para n ≥ 12).** Con datos autoconsistentes, para n ∈ 3Z y ε real con:

| n | radio | vía |
|---|---|---|
| ≥ 21 | ε_χ(n) (≈ 1,2–2,8·10⁻⁵) | Z26 o Bauer–Fike |
| 18 | **5,0·10⁻⁶** | Bauer–Fike: ε_a = 1,57·10⁻², t′ = 1,11·10⁻² |
| 15 | 6,9·10⁻⁸ | Bauer–Fike |
| 12 | 2·10⁻¹⁵ | Bauer–Fike |

se tiene, para toda Θ,

  **p(n,Θ) = p_E(n,Θ) − δ_n(Θ),   0 ≤ δ_n ≤ C ε^{2n−8},**

  **p(n,0) − p(n,Θ) = (E₀(n,Θ) − E₀(n,0)) / (n(ē − e_R)) + O(ε^{2n−8}).**

El primer término es exactamente el de Z20, de orden ε^n con coeficiente `2Re(b₀(e^{iΘ} − 1))`. En
los bordes de los rangos, `δ_n ≤ 6·10⁻⁸⁶` (n = 18). *Corregido en 9.27:* el «10⁻⁶⁴» que se daba aquí era
la cota **superior** de |p_E − 1| que imprime el guion, que no sirve para comparar; el término principal
vale ~6·10⁻⁹³ en el borde, por debajo de la cota de δ, y solo la supera para ε ≤ 1,1·10⁻⁶. El
estado que alcanza p es puro: el fundamental de H(Θ) + Σx*_a O_a(Θ).

*Por qué n = 6 y 9 quedan fuera, y qué los cerraría.* La condición de Kantorovich compara
`|D| ~ ε^{n−2}` con `κ′²ε⁸/L`: el ε⁴ de las direcciones blandas de χ entra al cuadrado. Hace falta
n − 2 > 8.
- **Lo que dicen los números.** Para n = 6, el SDP de 9.18 muestra `p_E − p ≈ 30|D|²`. Es el
  comportamiento cuadrático esperado, sin el factor 1/ε⁴, lo que sugiere que D apenas tiene componente
  en las direcciones blandas.
- **Lo que haría falta.** Un Newton graduado: normas con pesos (1, ε, ε²) en los tres bloques, y una
  cota de las componentes blandas de D por la misma estructura de corrientes. Las direcciones blandas
  distinguen configuraciones a dos saltos, y una corriente que da la vuelta las ve en orden más alto.
  No está hecho.

*Qué es riguroso.* Lo mismo que en Z26–Z29: analiticidad, Cauchy y Newton–Kantorovich son
rigurosos; la estructura de corrientes es el argumento de Z19–Z20; las cotas de χ son las de Z29. Las
constantes son muy holgadas: los radios para n = 12 y 15 son artefactos de usar ε⁴ dos veces.

**9.25 n = 6 y 9: el Newton graduado** (radios mejorados en 9.26; `scripts/newton_graduado.py`, `scripts/marg_k0.py`,
`scripts/kantorovich_graduado.py`).

*La idea.* En variables reescaladas `u = S y`, con `S = diag(1, ε, ε²)` sobre los bloques duro (8),
intermedio (2) y blando (4), el jacobiano `Ĵ = S⁻¹(2Bᵀχ B)S⁻¹` es de orden 1. La condición de
Kantorovich es `h = ‖Ĵ⁻¹‖·L̂·‖Ĵ⁻¹S⁻¹Bᵀd‖ ≤ ½`, con L̂ la tercera derivada reescalada. (d es un
covector, así que en la base B sus componentes son Bᵀd.)

*La base explícita.*
- **Duras:** los 4 saltos desde el patrón, (01↔10), (12↔21), (02↔20) y (11↔20), cada uno en Re e Im.
- **Intermedias:** (02↔11), en Re e Im.
- **Blandas:** para cada `o_d ∈ {|00⟩, |11⟩, |21⟩, |22⟩}`,

  `B_s = o_d + ε·Σ_y (Δ_y/3)·(salto de tipo y)`,

  con `Δ₁ = o(21) + o(10) + o(02)` (saltos (01)→(10) y (12)→(21)) y `Δ₂ = 3o(11)` (salto inverso
  (20)→(11)). Los o_d se anulan en las aristas ascendentes.
- **Por qué la corrección.** Con ella, la amplitud de cada excitación de un salto pasa a
  `2ε(1 + Δy/3)` y su energía a `3 + Δy`. El orden ε² de e queda `−(4ε²/3)(1 + Δy/3)`: **exactamente
  lineal** en y. Es un cambio local de la escala de tiempo, el análogo local de la dirección nula h.

*Lo medido con ED* (Θ = π; tercera derivada por la fórmula de tercer orden, validada con una integral
de Cauchy en t):
- **Ĵ.** Sus autovalores están en [0,0044, 0,77], estables al bajar ε.
- **L̂.** Todos los bloques son ≲ ε⁻²: sss, mss, mms y hhs van como ε⁻²; hss, hhm, hmm y hms como ε⁻¹.
- **D (n = 6).** `D_duro ≈ 3,9ε⁴`, `D_interm ~ ε⁵`, `D_blando ~ ε⁶`.
- **n = 9.** h = 0,011 a ε = 0,04. (Para la ED de n = 9 hace falta proyectar en momento 0 con los
  3 vectores del multiplete; una versión con 4 vectores daba basura.)

**Lema Z30 (soporte ponderado).** Sea `F(ε, y) = e(ε, B y)` y sea G = F menos su polinomio de Taylor
de grado ≤ 2 en y. Con pesos ε: 1, y_duro: 2, y_interm: 1, y_blando: 0, todo monomio
`ε^k y_h^a y_m^b y_s^c` de G tiene peso `k + 2a + b ≥ 4`.

*Prueba.* En la serie de Kato, con la parte diagonal blanda dentro de H₀′, cada monomio es una suma de
caminos cerrados desde el patrón. Las ε y las y duras e intermedias son movimientos. Las y blandas
diagonales solo cambian denominadores. Por casos, para grado ≥ 3 en y:
- **a ≥ 2:** el peso es ≥ 4.
- **a = 1:** hace falta al menos otro movimiento. Si es una sola ε, el término de dos movimientos es
  `x_h·2ε(1 + Δy/3)/(3 + Δy) = const` en y_s, y se cancela. Así que k + b ≥ 2.
- **a = 0, b ≥ 1:** el salto intermedio no toca el patrón. Desde una excitación de un salto lleva a
  distancia 2, de modo que k ≥ 3 si b = 1, y k ≥ 2 si b ≥ 2.
- **a = b = 0, c ≥ 3:**
  - k = 0: el patrón no nota y_s;
  - k = 2: es la identidad lineal de arriba;
  - k impar: no puede cerrar, porque cada salto simple cambia el dipolo en 1;
  - así que k ≥ 4. ∎

*Constantes (Bauer–Fike con la parte diagonal blanda en H₀′).*
- **Polidisco P:** |ε| ≤ ε_a, |y_h| ≤ R_h, |y_m| ≤ R_m, |y_s| ≤ R_s.
- **Hueco:** el de H₀′ es ≥ 2(1 − R_s). Cada pared se mueve como mucho |y_s|, porque los o_d viven en
  pares distintos.
- **Perturbación:** `‖V‖/n ≤ 2√2ε_a + 8R_h/√2 + 2R_m/√2 + ε_a R_s·8/3 =: A`, y se pide `nA < 1 − R_s`.
- **Cotas de F:** `|F + 1| ≤ A` y `|G| ≤ 121A` en P (120 monomios de grado ≤ 2, por Cauchy).
- **Schwarz en una variable.** `λ ↦ G(λε_a, λ²Y_h, λY_m, y_s)` se anula a orden 4, así que
  `|G| ≤ 121A(ε/ε_a)⁴` en el polidisco `{|y_h| ≤ (ε/ε_a)²R_h, |y_m| ≤ (ε/ε_a)R_m, |y_s| ≤ R_s}`.
- **Tercera derivada.** En u, ese polidisco tiene radios `≥ ρ := ε² min(R_h/2ε_a², R_m/2ε_a, R_s/2)`.
  Cauchy con radio ρ y polarización (4,5) dan `L̂ ≤ 27·121A(ε/ε_a)⁴/ρ³ = O(ε⁻²)`.
- **d por bloques, con corrientes (Z19–Z20).** `d_i ≤ (2A/R_i) r^{n−cur_i}/(1 − r)`, donde
  cur = 2 para el salto doble (02↔20), 1 para los demás saltos y 0 para los blandos. A eso se suma el
  término `(p − 1)τ`, de orden rⁿ.
- **Ĵ.** Se evalúa con la serie de χ (exacta hasta orden 4, en coma flotante validada hasta 8; 60
  dígitos). Se le restan el resto de Cauchy de χ (Z28) dividido por ε⁴ y la diferencia
  `Ĵ(Θ) − Ĵ(0)` por bloques, acotada con corrientes.

**Teorema Z23 para n = 6 y 9.** Con datos autoconsistentes:

| n | parámetros (ε_a, R_h, R_m, R_s) | rango | h en el borde | déficit p_E − p |
|---|---|---|---|---|
| 6 | (0,02, 10⁻³, 10⁻², 0,3) | **0 < ε ≤ 1,67·10⁻¹⁴** | 0,25 | ≤ 5·10⁻⁸⁷ |
| 9 | (0,03, 10⁻³, 10⁻³, 0,1) | **0 < ε ≤ 1,76·10⁻⁷** | 0,46 | ≤ 10⁻⁶² |

En ese rango, `p(n,0) − p(n,Θ) = (E₀(n,Θ) − E₀(n,0))/(n(ē − e_R)) + (déficit)`, con el déficit muchos
órdenes por debajo del término principal (~ε^n).

*Sobre el rango de n = 6.* El script certifica desde 2·10⁻¹⁹. Por debajo, la evaluación a 60 dígitos de
Ĵ, con entradas divididas por ε⁴ y cancelaciones de orden ε⁸, pierde precisión. Pero todas las
condiciones mejoran como ε² al bajar ε, y `Ĵ(ε) → Ĵ(0)`, la matriz graduada exacta, definida positiva
(Sylvester, 9.23). Así que el rango se extiende a (0, 1,67·10⁻¹⁴].

*Por qué los radios son tan pequeños.* Ninguna cota se optimizó:
- el `121A` de G;
- el ε_a⁻⁴ de Schwarz;
- el ρ³ isótropo de Cauchy, cuando cada bloque podría usar su propio radio;
- la cota de Θ en Ĵ por el peor bloque.

La ED indica que el Newton graduado funciona de verdad hasta ε ~ 10⁻³ (n = 6) y ~ 10⁻¹ (n = 9).

*Qué es riguroso.* El Lema Z30 (conteo de movimientos más dos identidades exactas de dos
movimientos), Bauer–Fike, Schwarz, Cauchy y Kantorovich. Lo heredado:
- los coeficientes de χ de orden 5–8 están en coma flotante validada;
- las cotas de D usan la estructura de corrientes de Z19–Z20;
- la cota (p − 1)τ usa |τ_W| ≤ 1, sin factor de B (B = 1 + O(ε)).

**Con esto, Z23 vale para todo n ∈ 3Z, n ≥ 6**, con un radio en ε que depende de n:
1,7·10⁻¹⁴ (6), 1,8·10⁻⁷ (9), 2·10⁻¹⁵ (12), 6,9·10⁻⁸ (15), 5·10⁻⁶ (18) y ≈ ε_χ(n) para n ≥ 21. Los
de n = 12 y 15 vienen del Kantorovich sin graduar de 9.24; el graduado los mejoraría.

**9.26 Constantes optimizadas: series en ε con colas de Cauchy** (`scripts/serie_respuesta.py`,
`scripts/kantorovich_series.py`).

*Dónde se perdía.* Comparadas con la ED, las cotas de 9.25 pierden un factor ~10⁹ en L̂ y ~10⁸ en d.
La causa es que cada cota de Cauchy en el polidisco paga potencias de ε_a⁻¹ (ε_a ≲ 1/(3n), por
Bauer–Fike). Las mejoras baratas no bastan:
- |G| ≤ 4A en lugar de 121A: cada parte homogénea en y está acotada por A, por Cauchy en una variable
  λ ↦ F(ε, λy);
- polarización de Banach (constante 1 en espacios de Hilbert) en lugar de 4,5;
- pesos por bloques en la norma.

Todo eso solo da ×100 (n = 6) y ×7 (n = 9).

*La idea.* Las cantidades evaluadas en y = 0 son funciones de ε sola:
- el gradiente d(ε, Θ);
- el jacobiano Ĵ(ε, Θ);
- la tercera derivada ∂³F̂(ε, Θ).

Sus series se calculan hasta un orden K alto. La cota de Cauchy se usa solo para la cola de orden > K,
que lleva un factor (ε/ε_a)^{K+1}.

*Las series* (`serie_respuesta.py`). Rayleigh–Schrödinger en todo el sector Sz = 0, desde la suma de
los tres patrones (no degenerada en momento 0), con giro uniforme y operadores girados o_a(Θ). Para
cada orden k da:
- **gradiente:** `⟨O_a⟩/n`;
- **hessiana:** `−2χ`, por los vectores de respuesta ξ_a = (H − E)⁻¹φ_a;
- **cúbica:** por la regla 2n + 1 de Wigner, `∂³e = 2Re Σ⟨ξ_a|O_b − ⟨O_b⟩|ξ_c⟩/n` sobre las tres
  elecciones del índice central.

Se pasa a la base B(ε) = B₀ + εB₁ orden a orden. Se usaron K = 36 (n = 6) y K = 40 (n = 9).

*Comprobaciones:*
- **Contra la ED** (n = 6, ε = 0,02, Θ = 0 y π): la serie sumada coincide en energía, gradiente,
  hessiana y cúbica con errores de 10⁻¹⁶, 10⁻¹⁷, 2·10⁻¹⁴ y 10⁻¹², respectivamente.
- **Estructura en Θ.** Los coeficientes son polinomios trigonométricos con frecuencias enteras
  `|w| ≤ (k + Σcur)/n`: es la regla de corrientes de 9.24, verificada orden a orden. Se recuperan
  exactamente con M > 2w_max valores de Θ (M = 16 para n = 6, 12 para n = 9). El ruido fuera de la regla
  es ≤ 10⁻¹⁴ relativo y se anula por el teorema.
- **Ceros graduados.** Los coeficientes que el Lema Z30 y la positividad graduada anulan (Ĵ: k < e_i + e_j;
  cúbica: k < 4 − 2a − b) salen ≤ 10⁻¹⁵ relativos hasta el orden 36–40. Es una comprobación numérica
  de Z30 a todo orden calculado.

*Las cotas* (en cada intervalo [ε_l, ε_r] de razón 1,01, desde 10⁻³⁰; cada potencia de ε se evalúa en
el extremo que la maximiza, de modo que se cubre el continuo):
- **Ĵ.**
  - Valor en ε_l, menos la derivada de la serie por (ε_r − ε_l).
  - Parte de Θ acotada por `2Σ_{w≠0}|ĉ_w|`.
  - Cola: `|coef_k(∂_i∂_j F)| ≤ (1 + δ_ij)A/(R_iR_j ε_a^k)`.
  - Todo en norma de Frobenius.
- **d.**
  - Parte de Θ de la serie del gradiente, más su cola `2A/R_i·(ε/ε_a)^{K+1}/(1 − r)`.
  - Más `|p − 1||τ_W|`, con `|p − 1| ≤ |ΔE/n|/0,6` (ΔE por su serie y su cola) y τ_W calculado.
- **∂³F̂ en 0.** Serie (Frobenius ≥ norma trilineal), parte de Θ y cola `m·A/(R_iR_jR_k ε_a^k)`, con
  m = 6, 2 o 1 según cuántos índices se repitan.
- **Variación en la bola de radio 2η.** Cuarta derivada por Cauchy, con el Schwarz ponderado de Z30
  (|G| ≤ 4Aλ⁴), radio τ = min_i ρ_i/w_i − 2η y polarización de Banach: `C₄ = 96Aλ⁴/τ⁴`.
  Entonces `L = ‖∂³F̂(0)‖ + 2C₄η`.
- **Con la serie de ∂⁴** (opción `cuarta`). Fórmula 2n + 1 de Wigner de cuarto orden:
  - ζ_cd = R(O_c − ⟨O_c⟩)ξ_d;
  - T_abcd = −½⟨ξ_a|O_b − ⟨O_b⟩|ζ_cd + ζ_dc⟩ − E₂_ab⟨ξ_c|ξ_d⟩;
  - ∂⁴E es la suma de T sobre las 24 permutaciones.

  Se validó en un juguete contra diferencias finitas en mpmath (10⁻¹⁴) y contra la ED para n = 6 (4·10⁻¹¹).
  Ceros graduados (k < max(0, 4 − 2a − b)) ≤ 2·10⁻¹⁶ y ruido fuera de la regla de corrientes ≤ 2·10⁻¹⁵.
  Cola de Cauchy `m·A/(∏R_i·ε_a^k)` con m = ∏ m_i!. En la bola, la quinta derivada va por Cauchy,
  `C₅ = 480Aλ⁴/τ⁵`, y `L = ‖∂³F̂(0)‖ + 2η‖∂⁴F̂(0)‖ + 2C₅η²`.

**Teorema Z23 para n = 6, 9 y 12 (radios mejorados).** Para n = 12 se usa la serie con K = 30 y M = 8. Con datos autoconsistentes y para todo Θ real:

| n | (ε_a, R_h, R_m, R_s), pesos (w_m, w_s) | rango certificado | h en el borde | déficit |
|---|---|---|---|---|
| 6 | (0,015, 10⁻⁴, 8·10⁻³, 0,4), (1, 1), con ∂⁴ en serie | **0 < ε ≤ 8,7·10⁻⁵** (1,17·10⁻⁵ sin ∂⁴; antes 1,7·10⁻¹⁴) | 0,49 | ≤ 8·10⁻²⁸ |
| 9 | (0,03, 10⁻⁴, 4·10⁻³, 0,1), (1, 1), con ∂⁴ en serie | **0 < ε ≤ 1,18·10⁻²** (5,1·10⁻³ sin ∂⁴; antes 1,8·10⁻⁷) | 0,34 | ≤ 8·10⁻²¹ |
| 12 | (0,02, 2·10⁻⁵, 10⁻³, 0,1), (0,3, 0,3) | **0 < ε ≤ 4,1·10⁻³** (antes 2·10⁻¹⁵, 9.24) | 0,50 | ≤ 10⁻²⁹ |

El tramo (0, 10⁻³⁰] se cubre como en 9.25: todas las cotas mejoran al bajar ε y Ĵ → Ĵ(0) > 0
(Sylvester exacto, 9.23).

*Qué limita ahora.*
- **n = 9.** Sin ∂⁴ en serie (ε ≤ 5,1·10⁻³) la cuarta derivada por Cauchy, `2C₄η ≈ 7·10⁹`, superaba a
  `‖∂³F̂‖ ≈ 10⁶`. Con ∂⁴ en serie el óptimo pasa a ε_a = 0,03, R_s = 0,1, y el radio a 1,18·10⁻². Queda a
  un factor ~8 de la ED (~0,09). El corte es abrupto: h salta de 0,34 a > 1/2 en un intervalo de razón 1,01.
  Con ε/ε_a ≈ 0,4, la cola de Cauchy de d hace crecer η rápido. La bola de radio 2η se acerca entonces
  al borde del polidisco, τ → 0, y el término de C₅ ∝ τ⁻⁵ explota.
- **n = 6.** Sin ∂⁴ en serie lo limitaba solo C₄: `2C₄η ≈ 3,6·10¹³` frente a `‖∂³F̂‖ ≈ 8·10⁹`. Con
  ∂⁴ en serie el radio sube ×7,5, hasta 8,7·10⁻⁵. En el borde el término de variación
  (`2η‖∂⁴F̂‖ + 2C₅η² ≈ 1,1·10¹⁰`) sigue dominando a `‖∂³F̂‖ ≈ 1,4·10⁸`, ahora por la cota de Cauchy de ∂⁵.
- **Cómo seguir.**
  - Para acercarse a la ED (~3·10⁻³) haría falta un dominio de analiticidad mejor que el de Bauer–Fike
    (ε_a ~ 0,1).

*Qué es riguroso.* Lo mismo que en 9.25, salvo que los coeficientes de las series se calculan en coma
flotante (doble precisión) y están validados de tres formas:
- contra la ED;
- contra los coeficientes racionales exactos (los ceros graduados, a 10⁻¹⁵);
- por la regla de corrientes.

La positividad de Ĵ en cada intervalo usa el valor en ε_l y la cota de la derivada, no una rejilla.

**9.27 Revisión de los problemas abiertos** (`scripts/marginal_tamano.py`, `scripts/radio_convergencia.py`,
`scripts/b0_flotante.py`; recogido en `sensibilidad_celda3.tex`).

*1. n = 18: el término principal no se ve en el borde.* Con `scripts/b0_flotante.py` (coma flotante; sin
cancelaciones, porque todos los sumandos tienen el signo (−1)^{n−1}; reproduce n = 15):

  `B(0→0) = −404,4966`, `B(0→±1) = −2396,3969`, **`4|b₀| = 20789,16`** (n = 18).

El término principal del rango en Θ es `4|b₀|ε^n/(n(ē − e_R))`. En el borde:

| n | ε_n | cota de δ_n | término principal |
|---|---|---|---|
| 6 | 8,7·10⁻⁵ | 8·10⁻²⁸ | 7·10⁻²⁴ |
| 9 | 1,18·10⁻² | 8·10⁻²¹ | 2·10⁻¹⁶ |
| 12 | 4,1·10⁻³ | 10⁻²⁹ | 3·10⁻²⁷ |
| 15 | 6,9·10⁻⁸ | 5·10⁻¹⁰⁸ | 1,6·10⁻¹⁰⁵ |
| 18 | 5,0·10⁻⁶ | 6·10⁻⁸⁶ | **6·10⁻⁹³** |

En n = 18, δ/principal ∝ ε^{n−8} = ε^{10}, y la cota de δ queda por debajo de la mitad del principal solo para
**ε ≤ 1,1·10⁻⁶**. Lo mismo, peor, en n = 21–30 (n = 21: 2·10⁻⁸⁰ frente a ~10⁻⁹²). El teorema sigue valiendo
como desigualdad; lo que no se certifica en `(1,1·10⁻⁶, 5·10⁻⁶]` es que p vea Θ.

*2. La corrección de tamaño finito de la marginal es del orden del túnel.* ED, σ_n frente a σ_{n′} mayor
(`scripts/marginal_tamano.py`), pendiente log-log de `|σ_n − σ_{n′}|_max` entre ε = 0,05 y 0,1:

| n | referencia | pendiente | 2n/3 − 1 | n − 2 (lo esperado en 9.18) |
|---|---|---|---|---|
| 6 | 12 | 3,08 | 3 | 4 |
| 9 | 12 | 5,06 | 5 | 7 |
| 12 | 15 | 7,12 | 7 | 10 |

Por componentes: coherencias de un salto ε^{2n/3−1}; diagonales y salto doble (02↔20) ε^{2n/3}; intermedias
(02↔11) ε^{2n/3+1}. En variables graduadas, todo es O(ε^{2n/3−2}). Es el cruce entre patrones vestidos: un
operador de arista ahorra un salto del túnel. No depende de Θ.

*Consecuencia para los datos σ_∞.* Con el esquema cuadrático, la parte del déficit que no depende de Θ es
O(ε^{4n/3−4}) y no afecta a la sensibilidad. La que depende es el término cruzado
O(ε^{2n/3−2}·ε^{n−2}) = O(ε^{5n/3−4}), menor que ε^n desde n = 9. Se espera, pues, la misma sensibilidad de
orden ε^n. Falta la prueba (con σ_∞, p(n,0) ≠ 1 y el Newton no parte de un fundamental exacto).

*3. El radio de convergencia de λ₀(ε).* RS en el sector Sz = 0 desde la suma de los tres patrones, hasta orden
120 (`scripts/radio_convergencia.py`): `|E_m|^{−1/m} ≈ 0,30` (n = 6) y `0,29` (n = 9) en m = 120, bajando muy
despacio. El dominio de analiticidad es 10–20 veces mayor que el ε_a de Bauer–Fike; lo que limita los radios
es la cota, no la física. Es una estimación, no una cota.

*4. Universalidad de χ.* La prueba de extensividad del Teorema de caminos (gauge telescópico de la parte
diagonal de o, como en Z26, y variables ε_b, t_b por enlace) da que el coeficiente de `ε^k t^l` de λ₀/n es
universal si `k + l ≤ ⌈n/2⌉ − 1` (Corolario de universalidad del .tex). Para χ a orden 8 eso es n ≥ 21. **No**
cubre la igualdad con la serie de n = 18 a orden 8, que es la que usan los radios de n = 21–30: ahí sigue
haciendo falta el recuento fino del túnel (2n/3 saltos simples). Con el salto doble, que vale por dos en
Σ|Δs|, ese recuento no es inmediato; la numérica (n = 15 se separa de la universal justo en K + 2 = 2n/3) dice
que el salto doble no adelanta el túnel, pero no está demostrado.

*5. (F) uniforme es la pieza que falta.* ε_F(n) cae deprisa (2,1·10⁻³ en n = 15, 4,5·10⁻⁵ en n = 18). Es de
esperar: las correcciones relativas al túnel son O(nε²). Un radio uniforme pide `a_{±1} = t_n ε^{2n/3}·exp(n·O(ε²))`
con signo fijo, por ejemplo con el gas de eventos de Z17.

*6. Detalle.* El Lema clásico da al menos 2 paredes en todo sector de carga (la suma de incrementos no usa la
carga). Los sectores cargados tienen hueco 2, y `radio_F.py` ya no exige ε ≤ 1/(4√2n): basta ε < 1/(2√2n).
Con eso, n = 6 pasa de 2,9·10⁻² (tope) a **3,08·10⁻²** (limitado por la serie). Los demás ε_F no cambian.

**9.28 Datos de otro anillo y de volumen infinito** (`scripts/serie_datos.py`, `scripts/kantorovich_datos.py`,
`scripts/newton_datos_ed.py`, `scripts/certifica_Fx.py`; problema abierto 3).

*Planteamiento.* Datos σ = σ_N (marginal del fundamental del anillo N > n, Θ = 0) o σ_∞, e_R = tr(hσ). La cota
superior p ≤ p_E (9.17) vale para cualquier σ. Para la inferior, el Newton graduado de 9.25–9.26 solo cambia en
el término independiente:

  `d = [grad_n(Θ) − grad_n(0)] + d_fs + (p − 1)τ_W`,   `d_fs = grad_n(0) − σ_W`,

con d_fs independiente de Θ y |p − 1| ≤ |e_n(Θ) − e_R|/|τ_h|. d_fs sale de la serie de σ_N
(`serie_datos.py`: RS sin vectores de respuesta, la marginal por traslación; reproduce grad[0] de
`serie_respuesta.py` a 4·10⁻¹⁶ relativo; cabe N = 15).

*Estructura de d_fs.* Sus coeficientes son nulos hasta el orden 2n/3 − 2 (n = 9 frente a 12: ≤ 6·10⁻¹⁶;
n = 12 frente a 15: ≤ 3·10⁻¹⁵), como la ED de 9.27. Demostrado: hasta ⌈n/2⌉ − 2 (Corolario de universalidad);
solo esos se ponen a cero en la certificación.

*Lema del testigo.* Sea x* la solución del Newton y supóngase **(F_x)**: el nivel de momento 0 de
`H_{x*}(Θ) = H(Θ) + Σ x*_a O_a(Θ)` es su fundamental. Entonces **p(n,Θ) = p*(Θ) exactamente**. Prueba: con el
testigo `w* = h + Σ x*_a o_a`, por la covariancia gauge, todo ω admisible cumple
`tr(ω H_{x*}) = n(c + p·s)`, con `s = tr(w*τ) = τ_h + x*·τ_W < 0`, y `tr(ω H_{x*}) ≥ λ₀(H_{x*}) = n(c + p*s)`;
así p ≤ p*. ∎ (Vale igual con datos autoconsistentes: el δ_n del Teorema principal es el déficit exacto.)

*Consecuencia.* p(Θ) − p(0) = [p_E(Θ) − p_E(0)] − [δ(Θ) − δ(0)], con δ = Ψ(x*)/|τ_h|,
`Ψ(x) = e(0) − e(x) + x·∇e(x)`. La variación se acota por
`|Δδ| ≤ [(|Ĵ| + 2ηL)·2η·|Δx| + (2η)²|ΔĴ|/2]/|τ_h|`, `|Δx| ≤ β/(1 − 2h)·(|Δd| + 2η|ΔĴ| + |Δp||τ|)`, con |ΔĴ| en
la bola por Taylor (parte de Θ de la tercera en 0, cuarta por la cota de L − L₀). Escala: término cruzado
`|d_fs|·|Δd| ~ ε^{2n/3−2}·ε^{n−2} = ε^{5n/3−4}`, menor que ε^n desde n = 9. Sin el lema, la horquilla
[p_E − δ, p_E] tiene anchura ε^{4n/3−4}, que no es menor que ε^n para n ≤ 12: **el lema es imprescindible**.

*n = 6 queda fuera:* h ~ ε^{2n/3−4} = ε⁰.

*Resultados* (`kantorovich_datos.py`; en cada fila, los mejores parámetros del polidisco entre los del .tex y los de la rejilla gruesa del guion; Kantorovich en intervalos de razón 1,01 desde 10⁻³⁰):

| n | datos | Kantorovich vale en | déficit en el borde | Δδ < principal/2 hasta |
|---|---|---|---|---|
| 12 | σ₁₅ | ε ≤ 1,26·10⁻³ | ≤ 1,6·10⁻²⁸ | **ε ≤ 1,3·10⁻⁴** |
| 12 | σ_∞ | ε ≤ 4,42·10⁻⁹ | ≤ 2,5·10⁻⁵⁰ | ε ≤ 5,5·10⁻¹⁶ |
| 9 | σ₁₂ | ε ≤ 4,31·10⁻⁵ (con ∂⁴ en serie) | ≤ 4,3·10⁻³⁰ | **ε ≤ 2,3·10⁻⁷** |
| 9 | σ_∞ (de N = 15, k₀ = 6) | ε ≤ 2,03·10⁻⁸ | ≤ 1,1·10⁻⁴³ | ninguno (≥ 10⁻³⁰) |
| 9 | σ_∞ (de N = 12, k₀ = 4) | sin rango | — | — |

*Parámetros* (ε_a, R_h, R_m, R_s, w_m, w_s): (0,0225; 10⁻⁴; 4·10⁻³; 0,15; 1; 1) para n = 9 con σ₁₂;
(0,03; 10⁻⁴; 4·10⁻³; 0,1; 1; 1), los del .tex, para n = 9 con σ_∞ (la rejilla da 1,87·10⁻⁸: elige por el borde
grueso); (0,015; 4·10⁻⁵; 2·10⁻³; 0,15; 0,9; 0,9) para n = 12. Con los del .tex para todos (como en la primera
versión) los bordes eran 3,60·10⁻⁵, 4,88·10⁻⁴ y 1,25·10⁻⁹, y la última columna 1,9·10⁻⁷, 4,7·10⁻⁵ y
7,0·10⁻¹⁷: **la diferencia era solo la elección de parámetros** (las primeras ejecuciones los fijaban a mano; sin
argumentos el guion recorre la rejilla). Los radios dependen bastante de ella y la rejilla no optimiza.

Para n = 12 con σ₁₅, la cota de Δδ escala como ε¹⁶ = ε^{5n/3−4} (comprobado entre 10⁻⁶ y 10⁻⁴): es el término
cruzado, no un artefacto.

*σ_∞.* Se toman los coeficientes de σ₁₅ hasta k₀ = ⌈15/2⌉ − 2 = 6 (universales por el Corolario) y una cola
uniforme en N de Z26 (9.22): `|coef_k(grad_∞)| ≤ (2W̄/t′)/ε_u^k`, `|coef_k(e_∞)| ≤ W̄/ε_u^k`, ε_u = 7·10⁻³,
W̄ = 4,1·10⁻⁴, t′ = 3·10⁻³. Es riguroso módulo Z26 (certificado con intervalos), pero la cola (ε/7·10⁻³)⁷ es tan
gruesa que los radios son ridículos. Mejorarlos pide más órdenes universales (un anillo mayor, o demostrar la
universalidad hasta el orden del túnel, 2N/3 − 2 = 8) y un dominio uniforme mayor.

*ED* (`newton_datos_ed.py 9 12`, Newton de 15 ecuaciones resuelto por ED, residuo ≤ 10⁻¹⁴):

| ε | p_E(π) − p_E(0) | p*(π) − p*(0) | Δδ | Δδ/Δp_E | δ(0) | |x*|_max |
|---|---|---|---|---|---|---|
| 0,05 | 8,20·10⁻¹¹ | 5,96·10⁻¹¹ | 2,24·10⁻¹¹ | 0,27 | 3,4·10⁻⁸ | 0,54 |
| 0,08 | 6,19·10⁻⁹ | 3,32·10⁻⁹ | 2,87·10⁻⁹ | 0,46 | 1,2·10⁻⁶ | 1,25 |
| 0,12 | 2,91·10⁻⁷ | 1,09·10⁻⁷ | 1,82·10⁻⁷ | 0,63 | 2,1·10⁻⁵ | 2,4 |

(F_x) se cumple en todos (el nivel de momento 0 es el más bajo; el cuarto nivel está a ≥ 1,5). δ(0) escala como
ε^{7,2–7,5}, cerca del ε^{4n/3−4} = ε⁸ esperado. **A ε moderado la sensibilidad de p con datos σ₁₂ es claramente
menor que la de la energía** (73 %, 54 % y 37 % de ella): Δδ/Δp_E decrece al bajar ε, aproximadamente como ε,
más despacio que el ε² del recuento. La identificación con la energía es asintótica y lenta en n = 9.

*Certificación de (F_x)* (`certifica_Fx.py`). Weyl no basta (‖Σx*O‖ ~ 10⁻⁷ frente a ~10⁻²⁴). Complemento de
Schur sobre P = span(patrones), con V_d = parte diagonal de Σx*O (al resolvente, que sigue diagonal) y
V = εT + V_o la no diagonal: `H_ef(E) = P(H₀+V_d)P + Σ_{j≥2} PV(R(E)V)^{j−1}P`, `R(E) = Q(E − H₀ − V_d)⁻¹Q`.
Por traslación H_ef es circulante, λ_k = c₀ + c₁ω^k + c₂ω^{2k}; los niveles son las raíces de E = λ_k(E), y si
λ_k > λ₀ en toda la ventana, E_k > E₀ (E − λ_k(E) creciente). Principal: caminos de orden D = 2n/3 con x = 0,
E = E₀, exactos: hueco **244,148 ε⁶** (n = 9; ED: 244,39 a ε = 0,01) y **1654,34 ε⁸** (n = 12). Independiente de
Θ: |principal| = su mayorante (misma fase en todos los caminos mínimos), y a Θ = π/4 eso fuerza corriente neta
nula. Resto: mayorante entrada a entrada con A = ε|T| + Σ|x_o||O_o|, denominadores rebajados en
δ = 2‖V_d‖ + ‖A‖, hasta orden D + 12, más cola por normas. Cotas de x* de la bola de Kantorovich:
|u_i| ≤ 2ηW_i, y = S⁻¹u, x = B(ε)y (las blandas, y ~ u/ε², quedan ≲ 10⁻⁵ en el borde).

| n | datos | (F_x) certificada en | peor error/hueco |
|---|---|---|---|
| 9 | σ₁₂ | ε ≤ 4,3·10⁻⁵ | 4·10⁻³ |
| 9 | σ_∞ | ε ≤ 2,03·10⁻⁸ | 8·10⁻⁴ |
| 12 | σ₁₅ | ε ≤ 1,26·10⁻³ | 0,18 |
| 12 | σ_∞ | ε ≤ 4,4·10⁻⁹ | 2·10⁻⁵ |

Cada rango es todo el de Kantorovich con los parámetros de la fila correspondiente de la tabla de resultados. Con esto, **la sensibilidad de la tabla es
incondicional** (módulo Z26 para σ_∞). Fuera del multiplete basta Weyl. No comprobado en el guion: la monotonía
de E − λ_k(E) (inmediata, |λ_k'| = O(‖V‖²)).

*Qué falta.*
- **σ_∞ con radios razonables:** universalidad hasta el orden del túnel y un dominio uniforme mayor.
- **n = 15, 18:** no cabe la serie de N = 18; sin graduar, la cola de d_fs solo es O(ν^{⌈n/2⌉−1}) demostrada,
  insuficiente frente al β ~ ε⁻⁴.

#### Balance para la carga entera

**Teorema Z″ (carga entera, k | n; corregido en el paso 8).** Por los pasos 1–6 y 8, existen ε_0 > 0
y c > 0 tales que, con `|ε| < ε_0` y `n ≤ n_*(ε)`, valen (A1), (A2E), (G) y (A2M) con constantes que
no dependen de n. Aquí `log n_*(ε) ≥ c/|ε|`; para `S⁺S⁻`, `ε_0 ≈ 3·10⁻⁶` y `c ≈ 9·10⁻⁶` (tabla de
8.5). En ese rango valen también el Teorema B, el Teorema Z5 con la marginal y la robustez
exponencial, con la escala del enrollamiento: `sup_{Θ,Θ'} |p(n,Θ) − p(n,Θ')| ≤ C(M|ε|)^{⌈n/2⌉−1}`,
y con exponente efectivo n por el Teorema A.

**Complemento (paso 9, Teorema Z‴).** Con ε real, el hueco (A1) + (G) vale para *todo* n ∈ kZ,
por el Teorema 6.8 y §8.4.2 de Nachtergaele–Sims–Young. La anchura del multiplete y la robustez son
subexponenciales en n. Las hipótesis estructurales están comprobadas; las constantes, no.

**Cerrado en 9.5–9.6 (Teorema Z⁗):** la uniformidad exponencial y compleja para *todo* n, por
contornos en espacio-tiempo con Kotecký–Preiss (Z15, Z16) y el análisis de tamaño finito del túnel
(Z17), con `ε_2 ≈ 2,3·10⁻³` para `S⁺S⁻` (9.7–9.10), o `8,2·10⁻³` con contornos en tiempo continuo (9.12–9.13: cota rigurosa con la cola acotada; 1,03·10⁻² para Z16, 8,2·10⁻³ con Z17, que limita). En tiempo continuo, Z17 tiene sus constantes explícitas (9.13 c), y los radios están certificados con
intervalos (9.13 e). La pérdida de peso entre escalas de 8.5 sigue en la vía de Z″, pero Z⁗ no la
necesita: los contornos con Kotecký–Preiss la sustituyen. La
versión anterior de este teorema (paso 7, `ε_0 = 8,1·10⁻⁶`, uniforme en n) se apoyaba en las tres
cotas falsas de 8.1.

Estado de cada pieza:

| pieza | estado |
|---|---|
| Lema Z8 (obstrucción del vestido de una fase) | demostrado y medido |
| Lema Z9 (vacío restringido, paso 1) | demostrado |
| Lema Z10 (columna por fases, paso 2′) | demostrado, condicional a (V) |
| (V) por inducción en escalas (paso 4) | demostrado para `n ≤ n_*(ε)` (paso 8); números uniformes en n con ε ≤ 0,1 |
| contracción del paso 2′ | no hace falta: la cota de Z10 es directa |
| contracción del paso 4 | la de Jacobi no existe: ρ(J) crece con n |
| Lema Z12 (contracción en variables de contorno, paso 6) | sustituido por Z12′ |
| Lema Z13 (la linealización de KT es H̃ − E) | demostrado |
| Lema Z12′ (contracción del quasi-Newton por fases, pasos 6′ y 7) | **la prueba del paso 7 tiene tres pasos no uniformes en n** (8.1); sustituido por Z12″; el radio espectral observado sigue uniforme en n hasta ε = 0,2 |
| Lema Z11s (soporte de las coordenadas, 8.1) | demostrado |
| cota L en norma de polímeros (8.2) | demostrada; `q_L = η'/(1−η') coth(τ/2)` |
| Lema Z14 (productos conexos sin pérdida de peso, 8.3) | demostrado |
| Lema Z12″ (contracción en una escala, 8.4) | demostrado con constantes explícitas |
| Teorema Z‴ (paso 9.2, vía Nachtergaele–Sims–Young) | hueco uniforme en todo n con ε real; robustez subexponencial; hipótesis comprobadas, constantes no |
| contornos con KP en espacio-tiempo (Z15, Z16; paso 9.5) | demostrados: Peierls e inducción de Zahradník con fases simétricas, τ fijo, ε complejo, uniforme en n; `ε_2 ≈ 2·10⁻³` con celdas de 2 enlaces, contornos 4-conexos, conteo exacto de animales, peso KP según la forma (9.7), integral exacta de los extremos (9.8) denominadores de energía por segmentos (9.9) y cola por percolación (9.10); en tiempo continuo, `ε_2 = 1,03·10⁻²` riguroso, ≈ 1,2·10⁻² estimado (Z15c con marcos y rachas, 9.12; capa de borde, KP continuo, inducción por anchura, supersolución de la cola y certificado con intervalos, 9.13) |
| Lema Z17 (del cilindro al multiplete, 9.6) | escrito: gas de eventos de túnel circulantes, KP en tiempo, multiplicidades enteras; en tiempo continuo (9.13 c), pasos 1–4 con constantes explícitas, radio riguroso 8,2·10⁻³ (certificado con intervalos, 9.13 e) |
| **Teorema Z⁗ (robustez exponencial uniforme en todo n, k \| n)** | **Z15 + Z16 + Z17 + Teorema B**; `ε_2 ≈ 2,3·10⁻³` para `S⁺S⁻` (`8,2·10⁻³` en tiempo continuo, 9.13) |
| **Teorema D‴ (banda de un defecto, k ∤ n, uniforme en n; 9.14)** | escrito: defecto como contorno que da la vuelta en tiempo, gas de excursiones ancladas, multiplicidades enteras; para `S⁺S⁻` gana la neutra, `g_orb = (56/3)ε²` o `(38/3)ε²` a segundo orden; **demostrado**, con `ε_3 ≥ 7,22·10⁻³` y `ε_4 ≥ 1,74·10⁻³` certificados (enlaces por el Lema Z18; w₂, w₃ exactos con racionales); ω_0 es par en ε y las cargadas se mueven desde el tercer orden |
| **Teorema Z20 (orden del enrollamiento; 9.15)** | demostrado: con `Q_c ≡ 0 mod k` (y en la órbita neutra con k ∤ n), Θ entra exactamente en el orden n; coeficiente exacto, comprobado con diagonalización exacta (n = 6, 7, 8, 9); `R_n ≤ 2B(|ε|/ε_2)^n` |
| sensibilidad de p(n,Θ) (9.16) | el puente no da el orden n; numéricamente (SDP, n = 6) p(6,Θ) = p_loc, la cota local de |00⟩ y |22⟩, ciega a Θ, hasta ε ≈ 0,23: sensibilidad **cero**; **no es uniforme en n** (9.17): la cota de energía p_E(n,Θ), que ve Θ en orden ε^n, queda por debajo de p_loc desde n = 9; el anclaje (Z21, Z22) da `p ≥ p_pieza`, ciega a Θ, con pérdida ≈ 0,1/ℓ; **Z23** (9.18, condicional a (H_χ), medida para n = 6, 9, 12): p = p_E − O(ε^{2n−4}), sensibilidad exactamente de orden ε^n con el coeficiente de Z20; (H_χ) reducida por Z24 (9.19) a la positividad uniforme de K₁, matriz de expectativas locales, y **Z25** (9.20): coeficientes racionales exactos hasta ε⁴, positividad graduada exacta (Sylvester), iguales para n = 9 y 12; (H_χ) vale para |ε| < ε_χ; **9.21:** ε_χ = 1,84·10⁻⁵ (n ≥ 15, serie de orden 8, universal n = 15 = 18), 3,2·10⁻⁷ (n = 12), 5·10⁻¹⁶ (n = 9), módulo el Lema Z26 de inserción (esquema, retirado); **Z26 + Z27** (9.22): por el autovalor de H + tO, con Z16/Z17 certificados en los parámetros desplazados, (H_χ) vale con χ ≥ 1,3·10⁻³ε⁴ para |ε| < 1,2·10⁻⁵ (n ≥ 18) y 1,7·10⁻⁷ (n = 15); **Z28 + Z29** (9.23): Bauer–Fike con la serie de cada n en momento 0 cierra n = 6, 9, 12 (ε_χ = 3,6·10⁻⁴, 1,6·10⁻⁴, 8,9·10⁻⁵): **(H_χ) vale para todo n ∈ 3Z con |ε| < 1,2·10⁻⁵**; **9.24: Z23 sin condiciones** por Newton–Kantorovich para n ≥ 12 (|ε| < 5·10⁻⁶ si n = 18, ≈ ε_χ(n) si n ≥ 21; 6,9·10⁻⁸ si n = 15; 2·10⁻¹⁵ si n = 12): p = p_E − O(ε^{2n−8}), sensibilidad exactamente la de Z20; **9.25: n = 6, 9 por Newton graduado** (Lema Z30 de soporte ponderado, base blanda explícita o_d + ε(Δ/3)·salto): ε ≤ 1,7·10⁻¹⁴ (n = 6), 1,8·10⁻⁷ (n = 9). **Z23 vale para todo n ∈ 3Z, n ≥ 6**; **9.26: series en ε con colas de Cauchy**: ε ≤ 8,7·10⁻⁵ (n = 6), 1,18·10⁻² (n = 9), ambos con ∂⁴ en serie, y 4,1·10⁻³ (n = 12); **9.27:** en n = 18 el término principal solo domina a δ para ε ≤ 1,1·10⁻⁶; σ_n − σ_∞ = O(ε^{2n/3−1}) (no ε^{n−2}); radio de convergencia de λ₀ ≈ 0,3; universalidad de χ demostrada entre n ≥ 21, no con n = 18 |
| paso entre escalas (8.5) | pierde un peso fijo `λ = O(|ε|)` por escala: cubre `n ≤ n_*(ε)`, `log n_* ≍ 1/|ε|`; **uniforme en n: abierto** |
| polímeros | son contornos, no islas: el vacío exacto lo muestra con c = 2 |
| Lema Z11 (cambio de base, paso 3) | demostrado (serie de Neumann pesada); comprobado con n = 6 |
| Gershgorin final | comprobado numéricamente con n = 6 y 9 (base por fases, cota del hueco > 0) |

### 7.14 El lado k ∤ n con carga entera: órbitas neutras y cargadas (Teorema D″)

El Teorema D′ (7.12) usaba la simetría de reloj en tres sitios:
1. el aislamiento de la banda (Lema Z7, con el vestido de una fase);
2. la simetría que separa las órbitas, (S_d), por el contenido;
3. la energía del vacío recto (Lema Z4′), que en carga entera ni siquiera tiene sentido. Los
   términos de enlace del marco tienen periodo k en j, y con k ∤ n no cierran en un anillo invariante.

Con lo construido en 7.13 se sustituyen las tres.

**(S_d) por la carga U(1).** Una configuración con un defecto, con el enlace malo en b y
`s_{b+1+i} = c + i`, tiene carga total `Q = Σ_{i<n} q(c + i)`. Solo depende de c, y la traslación
conserva c. Las tres órbitas, una por cada c, tienen

    n ≡ 1:  Q = q(c) ∈ {−1, 0, 1} ;     n ≡ 2:  Q = q(c) + q(c + 1) ∈ {−1, +1, 0}  (c = 0, 1, 2),

todas distintas. `S⁺S⁻` y el flujo conservan Q, así que (S_d) vale para todo Θ. Hay una órbita
**neutra** y dos **cargadas**, con Q = ±1.

**El vacío de cadena.** Por el Teorema Z″, los vacíos (V) de los anillos con k | n′ tienen
constantes uniformes en `n′ ≤ n_*(ε)`. Tras el paso 8, la uniformidad solo está demostrada en ese
rango. Sus coeficientes son holomorfos y acotados, y los de orden bajo se estabilizan (Z4′): dos
anillos `n′ < n″ ≤ n_*` dan coeficientes que difieren en `O(x^{n′/2})` en los caracteres de hull
`≤ n′/2`. Se toma como vacío de cadena `g_c^∞ := g_c^{(n′)}`, con n′ el mayor múltiplo de k que no
pasa de n_*. Es un punto fijo de Kirkwood–Thomas en la cadena salvo errores `O(x^{n′/2})`, cumple la
cota de Z1 y `g_c^∞ = T^c g_0^∞ T^{−c}`, con energía por enlace `e_∞ + O(x^{n′/2})`. Para
`n ≤ n_*`, esos errores quedan por debajo de los del Teorema D″. Que exista el límite n′ → ∞ es la
misma cuestión abierta que la uniformidad de Z″ en todo n.

**Lema Z7″ (aislamiento de la banda, carga entera).** En el anillo torcido (k ∤ n), cada carácter μ
se parte en dominios, que son los arcos entre paredes torcidas, cada uno con su fase física. La base
por fases viste cada dominio con `g_c^∞` restringido a él, transportado por la identificación física
del anillo, que es invariante por traslación. Con `η' = η_∂(2σ)` de Z10 y Z11, y `η' < 1/3`,
uniformemente en n:
1. hay exactamente nk autovalores en `|z − n e_∞ − 1| ≤ η' + C n x^{n/2}`;
2. el resto está en `Re(z − n e_∞) ≥ 2(1 − η')`.

*Demostración.* Una columna con un defecto, `|∂_tw μ| = 1`, tiene un solo dominio: el anillo cortado
en el defecto. `g_c^∞` cumple Kirkwood–Thomas exactamente en el interior. Lo que queda está anclado
en el defecto, con cota η′ por Z10 y Z11, más los caracteres de la cadena que no caben en un
intervalo de longitud n, de peso `x^{n/2}`. Los enlaces interiores aportan exactamente `e_∞` cada uno,
así que la referencia es `n e_∞` y no hace falta el vacío recto. Las columnas con `|∂_tw μ| = m ≥ 2`
tienen radio `≤ η' m`. Z10, Z11 y Z12′ son locales y no usan k | n. Gershgorin, como en Z7. ∎

**Teorema D″ (banda de un defecto, carga entera, k ∤ n).** *(Versión uniforme en todo n, con contornos:
Teorema D‴, 9.14.)* Sea `h = −P_asc + ε S⁺S⁻` (o, en general,
una perturbación de la clase (C) que conserve la carga U(1) y cuyas tres órbitas tengan cargas
distintas). Sean `|Θ| ≤ π`, `|ε| < ε_0′` y `n ≤ n_*(ε)`, donde `ε_0′` es el radio de Z7″, del
orden del `ε_0` del Teorema Z″, y `n_*` es el rango del paso 8. Entonces:
1. **Niveles simples y holomorfos.** Los nk niveles de la banda se etiquetan por `(Q, κ)` y son
   simples y holomorfos. Para ε real, la banda es el fondo del espectro.
2. **Banda límite por órbita.**

       | λ_{Q,κ}(ε, Θ) − n e_∞(ε) − ω_Q(κ − ΘQ/n, ε) | ≤ C (4x)^{⌊n/2⌋−1} ,

   con `ω_Q` independiente de n y de Θ, y con la serie de Taylor del Teorema D. El desplazamiento de
   Peierls es `ΘQ/n`. Un salto de r sitios dentro de una órbita es una traslación rígida, y en una
   ventana que no da la vuelta cambia el dipolo en `−rQ`, porque las celdas tienen carga entera
   `Q_c = 0`.
3. **Regla neutro/cargado.** Sea `Q*` la órbita de menor `min_κ ω_Q`, separada de las demás por un
   hueco g_orb > 0 independiente de n.
   - Si `Q* = 0`, entonces `R_n ≤ C′ (4x)^{⌊n/2⌋−1}`: **sensibilidad exponencial**.
   - Si `Q* = ±1` y el mínimo de `ω_{Q*}` es no degenerado, entonces `R_n ≍ n^{−2}`:
     **sensibilidad polinómica**. El desplazamiento `Θ/n` recorre un paso entero de la red `2π/n`.

*Demostración.* Es la del Teorema D′, con tres cambios:
- Z7″ en lugar de Z7;
- la carga U(1) como (S_d);
- `n e_∞` como referencia en lugar del vacío recto, de modo que la cota uniforme
  `|λ − n e_∞| ≤ 2 + η′` es directa.

Teorema D, Cauchy, densidad (con n → ∞ si Q = 0) y Schwarz, igual que allí. En el punto 3, con
Q* = 0, `ω_0` no depende de Θ, y la única dependencia es el error de 2. ∎

**Números** (`scripts/orbitas_entero.py`, `S⁺S⁻`, malla de 12 valores de Θ,
`e_ref = E_0(12, Q = 0)/12`):

| ε | n | Δ_def, Q = 0 | Δ_def, Q = ±1 | R_n, Q = 0 | n²R_n, Q = ±1 |
|---|---|---|---|---|---|
| ±0,1 | 7 / 10 / 13 (n ≡ 1) | 0,8133 / 0,8181 / 0,8187 | 0,9665 / 0,9684 / 0,9686 | 1,4·10⁻⁴ / 1,0·10⁻⁶ / 7,0·10⁻⁹ | 0,208 / 0,078 / 0,056 |
| ±0,1 | 8 / 11 (n ≡ 2) | 0,8431 / 0,8443 | 0,9533 / 0,9553 | 3,3·10⁻⁵ / 2,7·10⁻⁷ | 0,213 / 0,053 |
| 0,2 | 7 / 10 / 13 | 0,379 / 0,394 / 0,403 | 0,745 / 0,785 / 0,806 | 9,1·10⁻³ / 7,8·10⁻⁴ / 6,2·10⁻⁵ | 4,27 / 4,07 / 3,88 |
| 0,2 | 8 / 11 | 0,423 / 0,432 | 0,725 / 0,765 | 4,8·10⁻³ / 4,1·10⁻⁴ | 3,31 / 2,92 |

- **(S_d) se ve directamente.** Las dos órbitas cargadas son degeneradas entre sí, por la simetría que
  combina la conjugación de carga `s → 2 − s` con la reflexión `j → −j`, que deja `−P_asc` invariante.
  La neutra está separada. Los dos signos de ε dan lo mismo.
- **Gana la órbita neutra.** Todas las órbitas bajan en *segundo* orden: el primer orden es nulo,
  porque ningún `S⁺S⁻` mueve un defecto sin crear paredes (9.14). La neutra baja mucho más:
  `Δ_0 = 1 − (62/3)ε² + O(ε³)` con n ≡ 1 y `1 − (50/3)ε² + O(ε³)` con n ≡ 2, frente a
  `Δ_{±1} = 1 − 2ε²` y `1 − 4ε²`. A ε = 0,1 los términos de orden superior ya pesan: el hueco entre
  órbitas es ≈ 0,15, frente a 0,19 a segundo orden. Es independiente de n, como pide el punto 3.
  *(Corregido en 9.14: aquí se leía un primer orden, `1 − 1,85|ε|`, a partir de los datos con
  ε = 0,1 y 0,2, que no lo distinguen.)*
- **Por tanto, `R_n` es exponencial** para el fundamental: un factor de unas 140 cada tres sitios con
  ε = 0,1 (≈ ε^{0,7n}), y de unas 12 con ε = 0,2.
- **Las órbitas cargadas, que son excitadas, tienen respuesta polinómica.** Con ε = 0,2,
  `n²R_n ≈ 4,3 → 3,9`, casi constante. Con ε = 0,1 todavía baja (0,21 → 0,056), porque su banda es
  de orden ε² y el régimen asintótico llega más tarde.

**Lectura.** Para `S⁺S⁻` uniforme, el lado k ∤ n con carga entera es **exponencial**. La parte 2 de
la conjetura de la Sección 2 (sensibilidad polinómica con k ∤ n) no se da en este modelo: el defecto
que gana es neutro. Es la regla «cargado ⇒ polinómica, neutro ⇒ exponencial» de 4.2, ahora como
teorema para ε pequeño. Qué órbita gana lo decide el orden más bajo en que cada una se mueve. Eso es
el criterio de 4.3, «energía del defecto a segundo orden», que acertaba 48/48. Con `S⁺S⁻` la
diferencia aparece en segundo orden (9.14). En los modelos de 4.2 con perturbación aleatoria puede ganar
una órbita cargada, y entonces el Teorema D″ da `n^{−2}`.

#### Perturbación cinética aleatoria: cuando gana el defecto cargado

Guion: `scripts/orbitas_aleatorio.py`. Usa el término `termino(rng, ε, 0)` de `celda_k_aleatorio.py`
(Sección 4.2, semilla 2, ε = 0,1, δ = 0), que es de la clase (C) y conserva la carga U(1). Para cada
muestra y cada n calcula las energías de defecto por sector de carga, la órbita ganadora `Q*`, el
hueco entre órbitas `g_orb` y `R_n`. Con `QSOLO=q` calcula solo el sector `q` en una malla fina de Θ
(`NTHETA` valores). La malla gruesa de 12 valores no basta: con una banda cargada de anchura ≈ ε²,
el máximo en Θ cae entre nodos, y `n²R_n` sale con saltos del 20–40 %.

**Cribado.** Con n = 7 y 8 ganan órbitas cargadas en varias muestras (como en 4.2). Se siguen las
muestras 1 y 7 con n ≡ 1, y las 1 y 3 con n ≡ 2:

| muestra | n | Q* | g_orb | R_n (malla fina) | n²R_n |
|---|---|---|---|---|---|
| 1 | 7 / 10 / 13 | −1 | 0,010 | 2,62·10⁻³ / 1,29·10⁻³ / 7,74·10⁻⁴ | 0,128 / 0,129 / 0,131 |
| 7 | 7 / 10 / 13 | +1 | 0,015 | 2,70·10⁻³ / 1,34·10⁻³ / 7,49·10⁻⁴ | 0,133 / 0,134 / 0,127 |
| 1 | 8 / 11 / 14 | −1 | 0,024 | 4,33·10⁻⁴ / 1,52·10⁻⁴ / 8,77·10⁻⁵ | 0,028 / 0,018 / 0,017 |
| 3 | 8 / 11 / 14 | −1 | 0,023 → 0,022 | 1,01·10⁻³ / 2,71·10⁻⁴ / 1,54·10⁻⁴ | 0,064 / 0,033 / 0,030 |

(Malla de 48 valores de Θ con n ≡ 1 y de 36 con n ≡ 2.)

- **Hipótesis del punto 3 de D″.** La órbita ganadora es la misma en todos los n de cada serie, y
  `g_orb` no depende de n a la precisión mostrada. Es pequeño (≈ ε²), pero está separado de cero.
- **n ≡ 1: `n^{−2}` limpio.** Exponentes locales `−d log R/d log n`: 1,98 y 1,95 (muestra 1), 1,98 y
  2,20 (muestra 7). `n²R_n` es constante al 3–5 %.
- **n ≡ 2: compatible con `n^{−2}`, sin tanta limpieza.** Exponentes locales 3,3 → 2,3 (muestra 1)
  y 4,1 → 2,3 (muestra 3). Bajan hacia 2, y `n²R_n` cambia solo un 6–9 % de n = 11 a 14. Lo
  compatible es `R_n = a n^{−2} + (corrección que decae más deprisa)`, con la corrección todavía
  visible en n = 8. No es una caída exponencial: R_n baja un factor 1,7 de 11 a 14, frente al
  factor ≈ 140 del caso neutro (`S⁺S⁻`). Para fijar la constante haría falta n = 17 (3¹⁷ estados),
  fuera de alcance con este código.
- Con n ≡ 2 la constante es 4–8 veces menor que con n ≡ 1. Con n ≡ 1 el defecto es una repetición y
  con n ≡ 2 un salto (4.2), así que `ω_{Q*}` y su curvatura en el mínimo son otras.

**Lectura.** En el lado k ∤ n con carga entera, ambas ramas del Teorema D″ se dan dentro de la misma
clase: la exponencial con `S⁺S⁻` (gana la neutra) y la polinómica `n^{−2}` con un término cinético
aleatorio en el que gana una cargada. La parte 2 de la conjetura (sensibilidad polinómica con k ∤ n)
es, pues, cierta para estos datos y falsa para `S⁺S⁻`. Lo que la decide es la carga de la órbita
ganadora, no k ∤ n por sí solo.

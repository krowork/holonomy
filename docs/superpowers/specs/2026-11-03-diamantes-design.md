# Diseño: holonomía y pegado de perspectivas de Werner en cadenas de diamantes

Fecha: 2026-11-03. Tercer artículo de la serie, después de los anillos
(`~/fisica/holonomia_werner.tex`) y las escaleras (`~/fisica/holonomia_escaleras.tex`).

## Intención y porqué

Estudiar las cadenas de diamantes `K₄−e` con el mismo método que los anillos y las
escaleras: holonomía de los ejes como invariante gauge, `v_max` como función de la
holonomía, cotas generales, dual y simetrías, y numérica certificada al crecer el tamaño.

El programa de fondo de Alex es físico: reconciliar las descripciones de dos observadores
cosmológicos cuyos horizontes son pantallas holográficas "sin más allá". El diamante es el
modelo mínimo de dos perspectivas que comparten una región. **Esa interpretación no entra en
este artículo**; queda para un artículo posterior. Este es técnico, con el mismo tono,
estructura y nivel de rigor que el de escaleras.

### Hecho de partida

Cada diamante contiene dos triángulos, y en el anillo de 3 `v_max = 1/3` para toda holonomía
(\cite{anillos}, Proposición *triángulo*). Por la cota de subgrafos (\cite{escaleras}),
**`v_max ≤ 1/3` en cualquier grafo con un triángulo**, para cualesquiera datos. En el
diamante simple la cota se alcanza siempre (comprobado con el primal completo en 11 puntos,
3 de ellos con rotaciones aleatorias en las 5 aristas: 1/3 a 1e-8). Así que el diamante
simple no tiene estructura en la holonomía.

La estructura aparece al **pegar diamantes por una punta**: el qubit compartido hace que
`v_max` baje de 1/3 y dependa de la holonomía. Valores de exploración (datos abelianos,
fases iguales en cada diamante, gauge ad hoc, no invariantes):

| Cadena | sin holonomía | (π/2, π/2) | (π, π) | f = 0, ψ = π |
|---|---|---|---|---|
| Abierta, 2 diamantes | 1/3 | 0,32921 | 0,32590 | — |
| Cerrada, k = 2 | 1/3 | 0,32514 | 0,32505 | 0,32106 |

La pregunta central del artículo es, por tanto, **cuánto y cómo baja `v_max` por debajo del
techo 1/3** según la holonomía y la longitud de la cadena. A diferencia de las escaleras, el
techo es fijo y lo que se mide es el déficit.

## Objetivos

1. **Diamante simple** `K₄−e`: proposición "`v_max = 1/3` para toda holonomía", con la cota
   (subgrafos + triángulo) y un estado explícito que la alcance, más un guion que la
   compruebe en puntos aleatorios. Es una sección corta, no un estudio numérico.
2. **Cadena abierta de 2 diamantes** (7 qubits, 10 aristas, 4 ciclos, unidos por una punta).
   Es el análogo de la 2×3 abierta en el artículo de escaleras:
   - mapas abelianos;
   - con datos no abelianos: los mismos invariantes (θ_A, θ_B, γ) en cada diamante, y
     **mínimo sobre los 3 parámetros de orientación relativa** entre los dos diamantes; luego
     mínimo global sobre (θ_A, θ_B, γ);
   - el punto donde esté el mínimo: valor, si es mínimo local estricto, testigo y unicidad,
     como en la sección del punto de rotaciones perpendiculares de escaleras.
   Nota: en ciclos impares θ = π es el caso *menos* frustrado (no el más, como en los
   cuadrados), así que no se presupone que el mínimo esté en el punto de Klein.
3. **Cadena cerrada de k diamantes** (3k qubits, 5k aristas, 2k+1 ciclos), con datos
   abelianos uniformes: flujos (f_A, f_B) en los dos triángulos de cada diamante y la
   holonomía ψ de la vuelta, como (f, ψ) en la escalera periódica.
   - dual reducido por U(1), traslación de un diamante y las reflexiones que conserven los
     datos; contraste con el primal en k = 2 y 3;
   - generación de columnas con cotas certificadas para k grande, y el límite k → ∞ del
     déficit `1/3 − v_max` en el punto que resulte relevante.
4. **Artículo** `~/fisica/holonomia_diamantes.tex`, con el preámbulo de los otros dos.
   Cita \cite{anillos} y \cite{escaleras} por número de teorema y no repite sus
   demostraciones. Incluye la tabla de "Estado de las afirmaciones y problemas abiertos" y
   la de guiones, como en escaleras.

## No objetivos

- La interpretación cosmológica u holográfica (observadores, horizontes, pantallas).
- Diamantes sin diagonal (cuadrados unidos por las puntas), red jerárquica, redes 2D.
- Cadena abierta de 3 diamantes con holonomías no abelianas (10 qubits; el primal completo no
  cabe). Como mucho, con datos abelianos y primal U(1) si cabe en memoria.
- Cadenas cerradas no abelianas.
- Certificación con aritmética de intervalos (el nivel es el de escaleras).

## Antiobjetivos (fracaso aunque "funcione")

- Un número del artículo que no salga de un guion del repo, o que el guion no compruebe con
  `OK`/`FALLO`.
- Reescribir o reformular resultados de los artículos anteriores en lugar de citarlos.
- Una ejecución que agote la memoria de la máquina (30 GB; la app de Claude corre en ella).
  Cada guion pesado lleva `ulimit -v`, se ejecuta solo y el README da tiempo y memoria pico.
- Resultados aproximados donde se puede tener el SDP: Alex prefiere ejecuciones en serie y
  exactas a paralelas y aproximadas (el tiempo no es restricción, la memoria sí).

## Restricciones

- Código en `~/fisica/pyhton/holonomy` (remoto `krowork/holonomy`), en español, con la
  convención del repo: commits directos a `main`, mensaje en español con cuerpo y trailer
  `Co-Authored-By`.
- `~/fisica` no es un repo git: copiar el `.tex` antes de ediciones grandes.
- Las figuras salen de `scripts/figures.py` y se copian a `~/fisica`.
- El README y los docstrings citan las secciones y teoremas del artículo; si el `.tex` se
  reordena, se actualizan.
- Memoria: `ulimit -v` en todo cálculo pesado; medir el pico con `/usr/bin/time -f "%M KB"`.

## Decisiones de Alex

- Tercer artículo independiente, técnico, "como los demás".
- Empezar por los casos más sencillos.
- La interpretación queda para un artículo posterior.
- La familia es la de cadenas de `K₄−e` (con diagonal), no diamantes sin diagonal, sabiendo
  que el techo es 1/3.

## Decisiones mías (revisables)

- **Geometría y numeración.** Diamante simple: puntas 0 y 3, centrales 1 y 2; aristas
  0→1, 0→2, 1→2, 1→3, 2→3; triángulos A = (0,1,2) y B = (1,3,2). Cadena: la celda m tiene la
  unión j_m y las centrales u_m, w_m, con aristas j_m→u_m, j_m→w_m, u_m→w_m, u_m→j_{m+1},
  w_m→j_{m+1}. Abierta de k: 3k+1 sitios; cerrada: j_k = j_0, 3k sitios.
- **Datos no abelianos "uniformes" en la cadena de 2**: mismos invariantes (θ_A, θ_B, γ) por
  diamante; los 3 parámetros de orientación relativa se minimizan, no se fijan.
- **Memoria de la cadena de 2 no abeliana** (primal de 128×128): medido el 2026-11-03, no
  cabe (Clarabel aborta al pasar de 20 GB de memoria virtual en un solo punto). Se usa el
  dual con generación de columnas y cotas certificadas (aceptado ya en escaleras),
  contrastado con el primal U(1) en los puntos abelianos.
- **Título provisional**: "Holonomía y pegado de perspectivas de Werner en cadenas de
  diamantes".
- **Código**: módulo nuevo `holonomy/diamond.py` (geometría, fases abelianas en un gauge
  fijo, flujos de triángulo y `DiamondDual` para la cadena cerrada, siguiendo `LadderDual`).
  Guiones `scripts/diamond*.py`, tests en `tests/`, sección en el README. Se reutilizan
  `sdp.primal`, `sdp.primal_u1_graph`, la búsqueda de `ladder_open3_min.py` y
  `ladder_perp_min.v_at`, y la cota de estrella de `tree_star.py`.
- **Flujo uniforme en la cadena cerrada**: se exploran f_A = f_B y f_A = −f_B.
- **Orden de trabajo**: diamante simple → cadena abierta de 2 → cadena cerrada, cada fase con
  su guion y su sección del artículo antes de pasar a la siguiente.

## Lo que queda al constructor

- Las demostraciones nuevas (estado explícito del diamante, simetrías del grupo de
  automorfismos, bloques del dual de la cadena cerrada) y cuáles se heredan citando.
- El gauge concreto de las fases y la parametrización de la orientación relativa.
- Mallas, tolerancias, reinicios y qué punto de la cadena cerrada se usa para el límite.
- Hasta qué k se llega en la cadena cerrada, según la memoria medida.
- La estructura fina del artículo, siempre que siga la de escaleras.

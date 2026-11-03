# Cadenas de diamantes: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Código, datos y texto del tercer artículo, "Holonomía y pegado de perspectivas de
Werner en cadenas de diamantes", al nivel del de escaleras.

**Architecture:** Un módulo de geometría (`holonomy/diamond.py`), un dual general por
generación de columnas para grafos pequeños con datos no abelianos (`holonomy/gdual.py`) y
la maquinaria de bloques (U(1) + traslación) de `LadderDual` extraída a `holonomy/periodic.py`
para reutilizarla en `DiamondDual`. Cada resultado del artículo sale de un guion
`scripts/diamond*.py` con salida `OK`/`FALLO`.

**Tech Stack:** Python ≥ 3.11, NumPy, SciPy, CVXPY + Clarabel, pytest, LaTeX.

**Spec:** `docs/superpowers/specs/2026-11-03-diamantes-design.md`

**Ejecutor por defecto:** `general-purpose`, un subagente nuevo por tarea con revisión entre
tareas (subagent-driven). Excepciones indicadas en cada tarea.

## Global Constraints

- Todo en español: código, comentarios, docstrings, mensajes de commit, artículo.
- Commits directos a `main`, mensaje en español con cuerpo y trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git add` selectivo, nunca `-A`.
- Todo cálculo de más de ~1 GB se ejecuta bajo `(ulimit -v <KB>; ...)` y se mide con
  `/usr/bin/time -f "%M KB"`; los guiones pesados se ejecutan de uno en uno, nunca en paralelo
  con otro pesado. La máquina tiene 30 GB y la app de Claude corre en ella.
- **El primal completo no abeliano de 7 qubits no cabe**: medido el 2026-11-03, Clarabel
  pide más de 20 GB de memoria virtual en un punto de la cadena de 2 y aborta. Para datos no
  abelianos con 7 qubits se usa `GraphDual.v_dual_cg` (tarea 3).
- Tolerancias de Clarabel: `CLARABEL_OPTS` de `scripts/_common.py` (1e-10).
- Cada guion: docstring con qué comprueba, tiempo y memoria; `Checker` de `_common`; escribe
  sus datos en `data/` con `holonomy.report.write_csv`; reanudable si tarda más de ~30 min
  (patrón de `ladder_open3_min.py`, evaluaciones en un CSV que no se repiten).
- Los guiones de exploración que tardan más de ~5 min no entran en `run_all.sh`.
- `~/fisica` no es repo git: copiar `holonomia_diamantes.tex` a `holonomia_diamantes.tex.bak`
  antes de cada edición grande.
- No reformular resultados de \cite{anillos} ni \cite{escaleras}: se citan por número.

## Review Focus

1. **Convenios de orientación.** Un triángulo recorrido al revés da el flujo con el signo
   cambiado sin que nada falle. Esperado: los flujos que devuelve `triangle_fluxes` son los
   que se pasaron a `abelian_phases`, y la holonomía no abeliana de cada triángulo, calculada
   con `paulis.holonomy` sobre el ciclo, es la que se pasó a `chain_rotations` (tests en la
   tarea 1).
2. **Cota inferior inválida en la generación de columnas.** Si el testigo toca la caja,
   la cota inferior deja de valer. Esperado: estado `en_caja` y el guion lo reporta como
   `FALLO`, nunca como resultado (test en la tarea 3).
3. **Estado no óptimo del solver escondido.** Esperado: toda línea de resultado lleva
   `chk.status_note(status)` (revisión en las tareas 4, 6 y 7).
4. **Memoria.** Un guion lanzado sin `ulimit` puede tumbar la sesión. Esperado: el uso en el
   docstring incluye el `ulimit` y el README da el pico medido (tareas 4–8).
5. **Regresión en escaleras al extraer `periodic.py`.** Esperado: los tests de escalera pasan
   sin cambios y `LadderDual(4).v_dual_sdp` da el mismo valor que antes a 1e-10 (tarea 5).

---

### Tarea 0: Registrar diseño y plan

**Ejecutor:** la sesión principal (no hace falta subagente).

- [ ] **Paso 1:** `git add docs/superpowers/specs/2026-11-03-diamantes-design.md docs/superpowers/plans/2026-11-03-diamantes.md`
- [ ] **Paso 2:** commit "Diamantes: diseño y plan del tercer artículo", con cuerpo y trailer.

---

### Tarea 1: Geometría de las cadenas de diamantes

**Files:**
- Create: `holonomy/diamond.py`
- Test: `tests/test_diamond.py`

**Interfaces:**
- Produces:
  - `chain_edges(k: int, periodic: bool = False) -> list[tuple[int, int]]`. Celda m: unión
    `j = 3m`, centrales `u = 3m+1`, `w = 3m+2`, siguiente unión `j' = 3m+3` (o `0` si
    `periodic` y `m = k-1`). Aristas de cada celda en este orden:
    `(j,u), (j,w), (u,w), (u,j'), (w,j')`. Con `k = 1` abierta da
    `[(0,1),(0,2),(1,2),(1,3),(2,3)]`, el diamante simple. `periodic` exige `k >= 2`
    (`ValueError` si no).
  - `n_sites(k, periodic=False) -> int`: `3k+1` o `3k`.
  - `KINDS = ("ju", "jw", "uw", "uj", "wj")` y `edge_kinds(k, periodic=False) -> list[str]`.
  - `abelian_phases(k, fA, fB, periodic=False, psi=0.0) -> list[float]`, en el gauge
    simétrico: `(j,u) = fA/2 + s`, `(j,w) = -fA/2 + s`, `(u,w) = 0`, `(u,j') = fB/2 + s`,
    `(w,j') = -fB/2 + s`, con `s = psi/(2k)` si `periodic` y `0` si no. `fA`, `fB` escalares
    o secuencias de longitud k.
  - `triangle_fluxes(k, phis, periodic=False) -> tuple[list, list, tuple | None]`:
    `fA_m = φ(j,u) + φ(u,w) − φ(j,w)` (ciclo j→u→w→j),
    `fB_m = φ(u,j') − φ(w,j') − φ(u,w)` (ciclo u→j'→w→u) y, si `periodic`, las holonomías de
    las dos vueltas `(Σ φ(j,u)+φ(u,j'), Σ φ(j,w)+φ(w,j'))`, que con el gauge anterior son
    `ψ ± Σ(fA+fB)/2`.
  - `chain_rotations(k, pairs, periodic=False) -> list[np.ndarray]`, matrices R_e en el orden
    de `chain_edges`, con `pairs = [(H_A, H_B), ...]` una por celda: identidad en
    `(j,u), (j,w), (u,j')`; `R(u,w) = H_A`; `R(w,j') = (H_B H_A)^T`. Así la holonomía de A
    (`R_ju R_uw R_wj`) es `H_A` y la de B (`R_uj' R_j'w R_wu`) es `H_B`, ambas en el marco de
    `j_m`, y todos los marcos de unión coinciden por el árbol j→u→j'. Solo abierta
    (`periodic=True` da `NotImplementedError`: las cadenas cerradas no abelianas son no
    objetivo).
  - `relative_pairs(tA, tB, g, Q) -> list[tuple]`: los dos pares de la cadena de 2 con los
    mismos invariantes, `H_A = R_z(tA)`, `H_B = R_y(g) R_z(tB) R_y(g)^T` (como
    `ladder_perp_min.v_at`), el segundo conjugado por `Q ∈ SO(3)`.

- [ ] **Paso 1: tests que fallan** en `tests/test_diamond.py`:
  - `test_diamante_simple_aristas`: `chain_edges(1) == [(0,1),(0,2),(1,2),(1,3),(2,3)]`,
    `n_sites(1) == 4`, `len(chain_edges(3, periodic=True)) == 15`, `n_sites(3, True) == 9`.
  - `test_flujos_ida_y_vuelta` (Review Focus 1): para `k=3`, `fA=[0.3,-1.1,2.0]`,
    `fB=[0.7,0.2,-0.4]`, `psi=0.9`, periódica, `triangle_fluxes` devuelve `fA`, `fB` y
    `(0.9 + Σ(fA+fB)/2, 0.9 − Σ(fA+fB)/2)` a 1e-12; abierta, `None` en la tercera posición.
  - `test_holonomia_no_abeliana` (Review Focus 1): con dos pares aleatorios
    (`scipy.spatial.transform.Rotation.random`, semilla fija), la holonomía de cada triángulo
    de cada celda, calculada con `paulis.holonomy` sobre el ciclo (aristas recorridas al revés
    traspuestas), es `H_A` y `H_B` a 1e-12.
  - `test_abeliano_coincide_con_rotaciones`: `chain_rotations` con `H = R_z(f)` y
    `sdp.primal` coincide con `sdp.primal_u1_graph(abelian_phases(...))` en el diamante simple
    con `fA=0.8, fB=-1.3` a 1e-7.
  - `test_periodica_k1_rechazada`: `pytest.raises(ValueError)`.
- [ ] **Paso 2:** `.venv/bin/python -m pytest tests/test_diamond.py -v` → FAIL (módulo no existe).
- [ ] **Paso 3:** implementar las funciones de arriba en `holonomy/diamond.py`, con un
  docstring de módulo como el de `ladder.py` que fije la numeración, el gauge y los ciclos.
- [ ] **Paso 4:** pytest de `tests/test_diamond.py` y de `tests/test_core.py` → PASS.
- [ ] **Paso 5:** commit "Diamantes: geometría, flujos de triángulo y gauge de árbol".

---

### Tarea 2: El diamante simple, v_max = 1/3

**Files:**
- Create: `scripts/diamond_single.py`
- Modify: `scripts/_common.py` (`edge_mismatch`), `run_all.sh` (añadir `diamond_single` a `FAST`)

**Interfaces:**
- Consumes: `diamond.chain_edges`, `diamond.chain_rotations`, `sdp.primal`.

- [ ] **Paso 1:** guion con comprobaciones:
  1. primal completo = 1/3 (≤ 1e-7) en (0,0,0), (π,π,π/2), 6 puntos (θ_A, θ_B, γ) al azar y
     6 juegos de rotaciones aleatorias en las 5 aristas (semilla fija);
  2. en esos puntos, el estado óptimo tiene marginales nulas y las 15 correlaciones de cada
     arista a 1e-6 (añadir a `_common.edge_mismatch` un parámetro `edges=None`; con `None`,
     el anillo como ahora).
- [ ] **Paso 2: estado explícito** (time-box: 2 h de trabajo del subagente). Buscar un estado
  que alcance 1/3 para toda holonomía: inspeccionar rango y espectro de los óptimos del
  paso 1 y probar construcciones a partir del estado del triángulo
  (`scripts/tri.py`, \cite{anillos} Prop. *triángulo*). Si se encuentra, añadir la
  comprobación 3: el estado explícito tiene las correlaciones de cada arista a 1e-12 en los
  mismos puntos. Si no se encuentra en el time-box, no añadirla y anotar en el informe de la
  tarea "alcance solo numérico" (el artículo lo dirá así en la tabla de estado).
- [ ] **Paso 3:** `.venv/bin/python scripts/diamond_single.py` → todas `OK`, código 0.
- [ ] **Paso 4:** commit "Diamantes: el diamante simple tiene v_max = 1/3".

---

### Tarea 3: Dual general por generación de columnas

**Files:**
- Create: `holonomy/gdual.py`
- Test: `tests/test_gdual.py`

**Interfaces:**
- Produces: `class GraphDual(n: int, edges: list[tuple[int, int]])` con
  - `v_dual_sdp(Rs, solver="CLARABEL", **kw) -> tuple[float, np.ndarray, str]` (SDP
    completo, solo para n ≤ 6);
  - `v_dual_cg(Rs, y0=None, tol=1e-8, r0=6, add=4, box=50.0, maxiter=200, rmax=40, **kw)
    -> tuple[float, np.ndarray, str, float]`, mismo contrato que `LadderDual.v_dual_cg`:
    `(cota superior, testigo, estado, diferencia de cotas)`, estado en
    `{"optimal", "en_caja", "no_converge"}`;
  - `witness_spread(Rs, v_opt, eps, U=None, **kw) -> list[tuple[float, float]]`.
- Testigo: `W(y) = Σ_e Σ_ab y_{e,ab} σ_a^i σ_b^j` (9 coeficientes por arista, sin términos de
  un sitio: la inversión temporal \cite{escaleras} los anula en un testigo promediado);
  normalización `N(y) = Σ_e Σ_ab y_{e,ab} (−R_e)_{ab}`; `v = min λ_max(W)/N`.
  Los operadores σ_aσ_b se construyen una vez como matrices dispersas de 2ⁿ.
- `witness_spread` con `U` (los subespacios finales de `v_dual_cg`) resuelve sobre las
  restricciones comprimidas: el conjunto factible es mayor, así que el rango obtenido es una
  cota superior del verdadero (rango → 0 prueba unicidad; finito no prueba lo contrario).
  Documentarlo en el docstring.

- [ ] **Paso 1: tests que fallan** en `tests/test_gdual.py`:
  - `test_anillo4`: `GraphDual(4, ring_edges(4)).v_dual_sdp([Rz(θ/4)]*4)` = `f4(θ)` de
    `test_core.py` a 1e-7 para θ = 0, 1.0, π.
  - `test_diamante_un_tercio`: en el diamante simple con 2 juegos de rotaciones aleatorias,
    `v_dual_sdp` y `v_dual_cg` = 1/3 a 1e-7, estado `optimal`.
  - `test_cg_igual_sdp_n6`: en la escalera `ladder_edges(3)` (n = 6, el mayor caso con
    primal completo) con `(R_z(π), R_x(π))` en el gauge de `tree_gauge_open`:
    `v_dual_cg` = `v_dual_sdp` = `sdp.primal` a 1e-7.
  - `test_cadena2_abeliana`: cadena abierta de 2, `abelian_phases(2, π/2, π/2)`,
    `v_dual_cg` = `sdp.primal_u1_graph` a 1e-7 (contraste a 7 qubits sin primal completo).
  - `test_caja_reportada` (Review Focus 2): con `box=1e-3`, `v_dual_cg` devuelve estado
    `"en_caja"`.
- [ ] **Paso 2:** pytest → FAIL.
- [ ] **Paso 3:** implementar `GraphDual`. Reutilizar `LadderDual.top_eig` (importarlo, no
  copiarlo) y seguir la estructura de `LadderDual.v_dual_cg` (poda a `rmax`, SVD para
  ortonormalizar). Arranque `y0`: Heisenberg rotado, `y_{e} = −R_e` aplanado.
- [ ] **Paso 4:** pytest de los dos ficheros de tests → PASS. Medir el pico de
  `test_cadena2_abeliana` con `/usr/bin/time` y anotarlo en el docstring de `v_dual_cg`.
- [ ] **Paso 5:** commit "Diamantes: dual general por generación de columnas".

---

### Tarea 4: Cadena abierta de 2, mapas abelianos y orientación relativa

**Files:**
- Create: `scripts/diamond_chain2.py`

**Interfaces:**
- Consumes: `diamond.*`, `sdp.primal_u1_graph`, `GraphDual.v_dual_cg`.

- [ ] **Paso 1:** guion con:
  1. mapa abeliano `v(fA, fB)` con `primal_u1_graph`, malla 13×13 en [0, π]² (mismos flujos
     en las dos celdas) y los casos `fA = fB`, `fA = −fB`; comprobación: `v ≤ 1/3 + 1e-8`
     siempre y `v = 1/3` a 1e-7 sin holonomía;
  2. simetrías: `v(fA, fB) = v(fB, fA) = v(−fA, −fB)` a 1e-7 en 5 puntos (si alguna no vale,
     la comprobación lo dice y el informe de la tarea lo señala: es información, no un fallo
     del código, y se reporta en vez de borrarse);
  3. orientación relativa: en 4 puntos (θ_A, θ_B, γ) (incluido (π, π, π/2) y el mínimo
     abeliano del paso 1), `min_Q v_dual_cg(relative_pairs(..., Q))` por Nelder-Mead sobre el
     vector de rotación de Q, 5 arranques; comprobación: `Q = I` con datos abelianos reproduce
     el valor de `primal_u1_graph` a 1e-7.
  Escribe `data/diamond_chain2_abel.csv` y `data/diamond_chain2_rel.csv`.
- [ ] **Paso 2:** ejecutar bajo `(ulimit -v 8000000; /usr/bin/time -f "%M KB" .venv/bin/python scripts/diamond_chain2.py)`
  → todas `OK` salvo las del punto 2 que sean información; tiempo y pico en el docstring.
- [ ] **Paso 3:** commit "Diamantes: cadena de 2, mapas abelianos y orientación relativa".

---

### Tarea 5: Extraer la maquinaria periódica de `ladder.py`

**Files:**
- Create: `holonomy/periodic.py`
- Modify: `holonomy/ladder.py` (`momentum_isometries`, `symmetric_blocks`, `LadderDual`)
- Test: `tests/test_core.py` (sin cambios en los existentes; uno nuevo)

**Interfaces:**
- Produces:
  - `momentum_isometries(n_cells: int, basis, perm: list[int])`: como la actual, con la
    traslación dada por la permutación de sitios `perm` (sitio i → perm[i]).
  - `symmetric_blocks(n, edges, kinds, perm, n_cells, z_groups, dense_max=None, half=False, keys=None)`
    → misma salida `[(k, q, {clave: matriz})]`, claves `(nombre, kind)` y `("Z", grupo)`.
  - `class PeriodicDual` con `KEYS` y `kinds` como atributos de instancia, `witness_terms`,
    `norm_grad` genérico (cada arista suma `(2cos φ, 2sin φ, 1)` en las posiciones de
    `("A", kind)`, `("B", kind)`, `("C", kind)` que estén en `KEYS`) y los métodos actuales
    de `LadderDual` (`lam_max`, `top_eig`, `v_dual_sdp`, `v_dual_cg`, `witness_spread`,
    `top_blocks`, `block_gaps`, `v_dual`, `v_dual_grad`).
  - `ladder.LadderDual(PeriodicDual)` con la misma firma pública y el mismo `KEYS` de 8
    claves (los peldaños sin B). `ladder.momentum_isometries(L, basis)` y
    `ladder.symmetric_blocks(L, ...)` siguen existiendo como envoltorios.
- [ ] **Paso 1: test de regresión primero** (Review Focus 5): `test_escalera_regresion_periodic`
  fija `LadderDual(4).v_dual_sdp(abelian_phases(4, 0.5π, periodic=True, psi=0.3π))` y
  `LadderDual(5).lam_max(y)` con un `y` fijo, con los valores calculados **antes** del cambio
  (pegarlos como literales en el test), a 1e-10.
- [ ] **Paso 2:** pytest → PASS con el código actual.
- [ ] **Paso 3:** extraer a `periodic.py` sin cambiar comportamiento.
- [ ] **Paso 4:** `.venv/bin/python -m pytest` completo → PASS, incluidos los cuatro tests de
  escalera existentes sin modificar.
- [ ] **Paso 5:** commit "Escalera: maquinaria de bloques periódicos extraída a periodic.py".

---

### Tarea 6: Cadena cerrada, dual reducido y mapas

**Files:**
- Modify: `holonomy/diamond.py` (añadir `DiamondDual`)
- Create: `scripts/diamond_ring.py`
- Test: `tests/test_diamond.py`

**Interfaces:**
- Consumes: `periodic.PeriodicDual`, `periodic.symmetric_blocks`.
- Produces: `DiamondDual(k, dense_max=None, half=False)`, subclase de `PeriodicDual`, con
  traslación `perm[i] = (i + 3) % 3k`, kinds `KINDS`, grupos Z `("j","u","w")` y `KEYS` las
  15 claves `(A|B|C, kind)`. Reducciones adicionales por reflexión u↔w: decisión del
  implementador, solo si se demuestra que conservan los datos y el test de contraste sigue
  pasando.
- [ ] **Paso 1: tests que fallan:**
  - `test_cadena_cerrada_dual_igual_primal`: k = 2 y k = 3, en
    `(fA, fB, ψ) ∈ {(π/2, π/2, 0), (π, −π, 0), (0, 0, π)}`,
    `DiamondDual(k).v_dual_sdp(abelian_phases(..., periodic=True, psi))` =
    `sdp.primal_u1_graph(n_sites(k,True), chain_edges(k,True), phases)` a 1e-6.
  - `test_cadena_cerrada_cg`: k = 3, `v_dual_cg` con `dense_max=4` y con `half=True` = 
    `v_dual_sdp` a 1e-8, estado `optimal`.
- [ ] **Paso 2:** pytest → FAIL. **Paso 3:** implementar. **Paso 4:** pytest → PASS (medir el
  pico del test k = 3; si pasa de 6 GB, marcarlo `@pytest.mark.slow` y documentarlo).
- [ ] **Paso 5:** guion `scripts/diamond_ring.py k nf npsi`: mapa `v(f, ψ)` para
  `fA = fB = f` y para `fA = −fB = f`, malla `nf × npsi` en [0, π] × [0, π] (ψ → −ψ
  comprobado en 3 puntos, como en `ladder_psi.py`), mínimo en ψ por fila refinado con Brent.
  Escribe `data/diamond_ring_L{k}_{modo}.csv`. Comprobación: `v ≤ 1/3 + 1e-8` y `v = 1/3` sin
  holonomía.
- [ ] **Paso 6:** ejecutar `k = 2, 3, 4` (malla 13×13) uno a uno bajo `ulimit -v 12000000`;
  anotar tiempo y pico en el docstring. Informe de la tarea: el punto (o puntos) de mayor
  déficit `1/3 − v`, candidato para la tarea 7.
- [ ] **Paso 7:** commit "Diamantes: cadena cerrada, dual reducido y mapas en (f, ψ)".

---

### Tarea 7: Límite k → ∞

**Ejecutor:** `general-purpose`; las ejecuciones largas en segundo plano, una cada vez.

**Files:**
- Create: `scripts/diamond_limit.py`

- [ ] **Paso 0 (decisión de Alex):** la sesión principal le muestra los mapas de la tarea 6 y
  el punto de mayor déficit, y Alex elige el punto (o puntos) del límite. No se lanza nada
  antes.
- [ ] **Paso 1:** guion `scripts/diamond_limit.py kmax`, patrón de `ladder_limit.py`: para
  k = 2…kmax, `DiamondDual(k, dense_max=100, half=True).v_dual_cg` en el punto elegido, un
  proceso hijo nuevo por k, reanudable. Comprobaciones: estado `optimal` y diferencia de
  cotas < 1e-8 en cada k; coincidencia con `diamond_ring` en k ≤ 4 a 1e-8.
- [ ] **Paso 2:** ejecutar subiendo k uno a uno; medir el pico de cada k y parar cuando el
  siguiente no quepa con margen bajo `ulimit -v 16000000`. Escribe `data/diamond_limit.csv`.
- [ ] **Paso 3:** commit "Diamantes: límite k → ∞ hasta k = <kmax>".

---

### Tarea 8: Mínimo global y testigo en la cadena de 2

**Files:**
- Create: `scripts/diamond_chain2_min.py`

- [ ] **Paso 1:** guion, patrón de `ladder_open3_min.py`: malla de (θ_A, θ_B, γ) (θ en
  {0, π/4, π/2, 3π/4, π}, γ en {0, π/6, …, π}, θ = 0 con un solo γ), cada punto con el
  mínimo sobre Q de la tarea 4, y Nelder-Mead conjunto en (θ_A, θ_B, γ, Q) desde los tres
  mejores. Reanudable (`data/diamond_chain2_evals.csv`).
- [ ] **Paso 2:** en el mínimo: perturbaciones en los 6 ejes y 12 direcciones al azar
  (mínimo local estricto, como `ladder_perp_min.py`) y `witness_spread` con ε = 1e-4, 1e-5,
  1e-6 sobre los subespacios finales (unicidad del testigo).
- [ ] **Paso 3:** ejecutar bajo `ulimit -v 8000000` (varias horas, en segundo plano).
  Escribe `data/diamond_chain2_min.csv`.
- [ ] **Paso 4:** commit "Diamantes: mínimo global de la cadena de 2 y su testigo".

---

### Tarea 9: README, figuras y `run_all.sh`

**Files:**
- Modify: `README.md`, `scripts/figures.py`, `run_all.sh`

- [ ] **Paso 1:** README: módulos `diamond`, `gdual`, `periodic` en la tabla de paquete y una
  fila por guion `diamond*.py` con qué comprueba, sección del artículo (provisional "diamantes
  §n", se fija en la tarea 10), tiempo y pico medidos.
- [ ] **Paso 2:** `figures.py`: `fig_diamante_mapa.pdf` (mapa `1/3 − v` de la cadena de 2 en
  (fA, fB)) y `fig_diamante_limite.pdf` (límite en k), a partir de `data/`, con el estilo de
  las de escaleras. Copiarlas a `~/fisica`.
- [ ] **Paso 3:** `.venv/bin/python -m pytest` y `./run_all.sh --rapido` → todo correcto.
- [ ] **Paso 4:** commit "Diamantes: README, figuras y run_all".

---

### Tarea 10: El artículo

**Ejecutor:** `general-purpose`, en dos lotes con revisión de Alex entre ellos (ver
`paper-editing-approach`: los números se copian de `data/`, no se reescriben a mano).

**Files:**
- Create: `~/fisica/holonomia_diamantes.tex` (preámbulo copiado de `holonomia_escaleras.tex`)

- [ ] **Paso 1 (lote 1):** introducción, marco (geometría, invariantes citando los Lemas de
  invariantes y de pares de \cite{escaleras}), cotas generales (techo 1/3 por subgrafos +
  triángulo), dual y simetrías, diamante simple. Compilar dos veces con `pdflatex`, sin
  referencias indefinidas. **Parar para revisión de Alex.**
- [ ] **Paso 2 (lote 2):** cadena de 2, cadena cerrada, límite, métodos numéricos, "Estado de
  las afirmaciones y problemas abiertos" (tabla con el estado real de cada afirmación:
  demostrado, numérico, abierto) y "Verificación y reproducibilidad" con la tabla de guiones.
  Cada número del texto se contrasta con su CSV (script de comprobación ad hoc en el
  scratchpad, no en el repo). Compilar dos veces.
- [ ] **Paso 3:** actualizar en el README y los docstrings las secciones provisionales con las
  definitivas; commit "Diamantes: numeración del artículo en README y guiones".

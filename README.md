# Holonomía y pegado de perspectivas de Werner en anillos de qubits

Código que acompaña al artículo del mismo título (A. Gama, 2026). Reproduce
todos los resultados numéricos y simbólicos, las tablas y las figuras.

¿Cuándo pueden varios estados de Werner de dos qubits, cada uno con sus propios
ejes, ser las marginales de arista de un mismo estado global en un anillo? El
valor máximo del parámetro de Werner, `v_max`, depende solo del ángulo de la
holonomía de los ejes a lo largo del anillo. Su dual es la energía fundamental
de un anillo XXZ con flujo.

## Instalación

Python ≥ 3.11. Versiones con las que se obtuvieron los resultados del artículo:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

(NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, Clarabel 0.11.1, SCS 3.3.1,
SymPy 1.14.0; Matplotlib y una instalación de LaTeX para las figuras.)

## Uso

```bash
(ulimit -v 8000000; .venv/bin/pytest -q -m "not slow")   # pruebas rápidas (~1 min; 116 passed, 3 skipped)
(ulimit -v 8000000; .venv/bin/pytest -q -m slow)        # pruebas lentas (más de 4 min: test_cadena2_abeliana sola, 236 s); test_primal_reducido_k3 llega a 5,4 GB de memoria residente y test_cg_igual_sdp_n6 a ~4 GB: no las lance sin ulimit
./run_all.sh --rapido               # comprobaciones de menos de ~2 min (incluye `diamond_single`, `diamond_ring` con k = 2 y `diamond_numeros`)
./run_all.sh                        # todo (~1 h 30 min con la caché de Bethe; ladder.py son ~75 min)
```

Cada guion de `scripts/` imprime una línea `OK`/`FALLO` por comprobación y
termina con código 1 si alguna falla. Los datos se escriben en `data/` y las
figuras en `figures/`.

## Paquete `holonomy`

| Módulo | Contenido |
|---|---|
| `paulis` | Operadores de Pauli en n qubits, rotaciones, estados de Werner rotados |
| `sdp` | SDP primal del pegado, completo y reducido por U(1) (anillo o grafo cualquiera con datos abelianos) |
| `xxz` | Diagonalización exacta del anillo XXZ con flujo, por sectores de magnetización |
| `bethe` | Ecuaciones de Bethe con flujo en los tres regímenes (Δ > 0), con derivada implícita en el flujo |
| `dual` | Objetivo dual (Prop. 4), fórmula de un parámetro, criterio de borde (Prop. 6), anchura de la ventana |
| `ladder` | Escaleras de dos patas: aristas, flujos de plaqueta, dual reducido de la escalera periódica (SDP completo, y por generación de columnas con cotas certificadas para L ≥ 6, con bloques dispersos y Lanczos para L ≥ 8 y, opcionalmente, solo los bloques k ≤ L) |
| `diamond` | Cadenas de diamantes K4−e (abiertas y cerradas): sitios, aristas, flujos de triángulo, gauge simétrico (abeliano) y de árbol (no abeliano), pares de holonomías relativas y dual reducido de la cadena cerrada `DiamondDual` (U(1), traslación de una celda, inversión temporal; 15 coeficientes) |
| `gdual` | Dual general de v_max en un grafo cualquiera con datos en SO(3) (9 coeficientes por arista): SDP completo (n ≤ 6) y generación de columnas `v_dual_cg` con cotas certificadas (el primal completo no abeliano de 7 qubits no cabe en memoria) |
| `periodic` | Dual por bloques de un grafo periódico con datos abelianos (U(1) y traslación); `LadderDual` y `DiamondDual` son subclases suyas |
| `asymptotics` | Coeficiente de 1/n², rigidez finita D_n y anchura asintótica |
| `report` | Salida OK/FALLO y lectura/escritura de datos |

## Guiones y secciones del artículo

| Guion | Qué comprueba | Sección | Tiempo aprox. |
|---|---|---|---|
| `verify_holonomia.py` | SDP primal del anillo de 4: barrido en θ, holonomías de Haar, testigo y estado óptimo | 4 | 10 s |
| `verify_sym.py` | Normalización del testigo, espectro por sectores y bloque T†T (SymPy) | 4, 8 | 10 s |
| `verify_step3.py` | Espacio propio, estados K, valores por arista y las 15 correlaciones (SymPy) | 4, 8 | 5 s |
| `verify_unique.py` | Elementos de matriz de la unicidad (SymPy) | 4 | 1 s |
| `tri.py` | Estado explícito del triángulo | 3 | 1 s |
| `chk.py` | SDP primal independiente para n = 3 a 6, y φ fuera de [0, π/4] | 4, 5 | 2 min |
| `witness_n.py` | Valor dual por sectores frente al primal, n = 3 a 8 | 5.1 | 1 min |
| `primalU1.py` | Primal reducido por U(1) hasta n = 7, también dentro de la ventana | 5.1, 5.5 | 1 min |
| `witness_check.py` | Testigo óptimo explícito de 128 × 128 en n = 7 | 5.1 | 2 s |
| `xxz.py` | λ_max del testigo simétrico = 4rn ε_n (Prop. 4), matrices de 2ⁿ | 5.1 | 1 s |
| `sec.py` | Los sectores de magnetización mínima y el ferromagnético bastan | 5.1 | 15 s |
| `table62.py` | Tabla 2 (v_max por diagonalización exacta, n = 3 a 16) | 5.3 | 1 min |
| `edge.py` | Bordes de la ventana (tabla 1, n ≤ 20), diagonalización y Bethe | 5.3 | 5 min |
| `phistar.py` | Flujo óptimo libre Φ⋆(Θ) y barridos completos en el flujo; datos de la figura 1 | 5.3 | 4 min |
| `rank.py` | Rango del óptimo primal en n = 5, 6 y 7 | 5.5 | 40 s |
| `cusp.py` | Hipótesis (H) del Teorema 8 para n = 6 a 13, pesos del óptimo primal y monotonía de la ventana | 5.4 | 1 min |
| `certify_h.py` | Certificación asistida por ordenador de (H) en todo el tramo de Δ⋆, n = 6 a 18 (bloques de Bloch, cotas de Weyl, bisección); escribe `data/certify_h.csv` | 5.4 | 12 min |
| `ba.py` | Bethe frente a diagonalización exacta (385 combinaciones, n ≤ 14) | 5.2 | 20 s |
| `baedge2.py` | Coeficiente de 1/n², D_n y bordes para n grande (tabla 3) | 6.2, 6.3 | ~10 min en 4 procesos; caché en `data/bethe_cache/` |
| `bafit.py` | Ajustes de escalado con correcciones logarítmicas | 6.2, 6.3 | 1 s |
| `figures.py` | Figuras 1 y 2 a partir de `data/`, y las figuras 1 y 2 del texto de escaleras (`fig_escalera_flujo.pdf`, `fig_escalera_limite.pdf`), y las figuras 1 y 2 del texto de diamantes (`fig_diamante_mapa.pdf`, mapa 1/3 − v de la cadena de 2; `fig_diamante_limite.pdf`, déficit frente a k) | 5.3, 6.2; escaleras 7, 8; diamantes 6, 8 | 10 s |
| `ladder.py` | Exploración: escalera 2×3 abierta (primal completo, holonomías abelianas y no abelianas), cotas de estrella y de plaqueta, primal U(1) en grafos, primal U(1)+traslación y dual reducido de la escalera periódica (2×4, 2×5) frente al primal, barrido del dual hasta 2×6; escribe `data/ladder.csv` | problema abierto 5 | ~75 min |
| `ladder_flux.py` | Exploración: barrido fino del flujo de plaqueta en la escalera periódica 2×4, 2×5 (dual-SDP) y 2×6 (Nelder-Mead), flujo del testigo y mínimo en el flujo; escribe `data/ladder_flux.csv`. No está en `run_all.sh`; conviene limitar la memoria (`ulimit -v 12000000`) | problema abierto 5 | ~2 h |
| `ladder_jump.py` | Exploración: ventana de la escalera 2×5 (f ≈ 0,510π–0,531π) en la que el flujo del testigo se mueve deprisa entre dos regímenes de bloques degenerados; escribe `data/ladder_jump.csv` | problema abierto 5 | ~15 min |
| `ladder_regimes.py` | Exploración: mapa de regímenes (bloques degenerados en λ_max del testigo óptimo) en todo el rango de flujo, bordes por extrapolación lineal; `python scripts/ladder_regimes.py 4` escribe `data/ladder_regimes_L4.csv`; para 2×5: `5 0 1 0.01` (51 min) y `5 0 0.012 0.0005` (14 min) | problema abierto 5 | 7 min (L = 4) |
| `ladder_unique.py` | Exploración: unicidad del testigo óptimo de la 2×4 (rango de cada coeficiente en el conjunto casi óptimo, ε = 10⁻⁴…10⁻⁶) | problema abierto 5 | 11 min |
| `ladder_weights.py` | Exploración: rango y pesos por bloque del estado óptimo simétrico de la 2×4 en f = 0,70π–0,85π; escribe `data/ladder_weights_L4.csv` | problema abierto 5 | 2 min |
| `ladder_psi.py` | Exploración: holonomía de las patas ψ en la escalera periódica; mapa v(f, ψ), cotas de estrella, plaqueta y patas como anillos, simetría ψ → −ψ y contraste con el primal (L = 4); `python scripts/ladder_psi.py 4 21` escribe `data/ladder_psi_L4.csv`; `python scripts/ladder_psi.py 5 11`, `data/ladder_psi_L5.csv`; `python scripts/ladder_psi.py 5 11 41 6` (41 valores de ψ, 6 procesos), `data/ladder_psi_L5_11x41.csv`; `python scripts/ladder_psi.py 6 21 41` (L ≥ 6 por generación de columnas), `data/ladder_psi_L6_21x41.csv`; `python scripts/ladder_psi.py 7 21 41`, `data/ladder_psi_L7_21x41.csv`; cada pasada escribe también el mínimo en ψ de cada fila, refinado por Brent (`data/ladder_psi_min_L<L>…csv`) | problema abierto 5 | 13 min (L = 4; pico de 9,8 GB por el contraste con el primal); 61 min (L = 5, malla 11×11; pico de 2,6 GB); 92 min (L = 5, 11×41 con 6 procesos; ~2,6 GB por proceso); 98 min (L = 6, 21×41; pico de 0,6 GB); 3 h 47 min (L = 7, 21×41; pico de 1,7 GB) |
| `ladder_limit.py` | Exploración: límite L → ∞ en f = π o, con `--F F/π`, con flujo total fijo F (f = F/L); v para L = 4..11 y ψ = 0, π/4, π/2, 3π/4, π por generación de columnas con bloques dispersos (dimensión > 100) y Lanczos, solo los bloques k ≤ L (los k > L tienen el mismo espectro), cada punto en frío en un proceso hijo nuevo que da su pico de memoria; contraste con los mapas de `ladder_psi.py` (L = 4..7) y simetría (f, ψ) → (−f, −ψ); escribe `data/ladder_limit_L<L>.csv` (con `--F`, `data/ladder_limit_F<F/π>_L<L>.csv`) | problema abierto 5 | 80 min (L = 4..10; la 2×10, ~9 min y ~4 GB por punto, antes de reducir a k ≤ L); `python scripts/ladder_limit.py 11`: 2 h 3 min (15–23 min y 5,9–9,3 GB por punto); `--F` 0, 0,5, 1 y 2 con L = 4..10: 29, 15, 14 y 16 min, ~4 GB |
| `ladder_perp.py` | Exploración: testigo óptimo de la escalera abierta 2×3 con holonomías R_z(π), R_x(π) (las tres holonomías de ciclo son rotaciones de π); dual del primal completo (81 coeficientes) frente al primal, testigo XYZ con pesos por enlace sin campos, y unicidad por el rango de cada coeficiente en el conjunto casi óptimo (ε = 10⁻⁴…10⁻⁶; reanudable por ε); escribe `data/ladder_perp_witness.csv` y `data/ladder_perp_spread.csv` | problema abierto 5 | 2 h 19 min (pico de 3,9 GB) |
| `ladder_perp_min.py` | Exploración: ese punto es un mínimo local estricto de v en los invariantes (θ_A, θ_B, ángulo entre ejes): v sube en los tres ejes y en 12 direcciones al azar, sin términos lineales y con forma cuadrática definida positiva; escribe `data/ladder_perp_min.csv` | problema abierto 5 | 19 min (pico de 3,7 GB) |
| `ladder_fmin.py` | Exploración: mínimo de v(f, ψ) en la 2×4 y la 2×5 por minimización alternada (Brent) desde la malla de `ladder_flux.py`; con `perfil`, el perfil de la 2×4 alrededor de (2π/3, 0): asimetría cúbica en f y subida cuártica en ψ; escribe `data/ladder_fmin.csv` y `data/ladder_fmin_perfil_L4.csv` | problema abierto 5 | 36 min; `perfil`, 11 s |
| `ladder_open3_min.py` | Exploración: mínimo global de v en la 2×3 abierta sobre (θ_A, θ_B, γ): malla y Nelder–Mead desde los tres mejores puntos; reanudable (cada evaluación se guarda en `data/ladder_open3_evals.csv`); el menor de los 368 puntos es el de rotaciones perpendiculares (π, π, π/2); escribe `data/ladder_open3_min.csv` | problema abierto 5 | 2,8 h, 3,7 GB |
| `tree_star.py` | Exploración: la estrella K_{1,d} alcanza (d+2)/(3d) con un estado explícito (d = 1..6) y con el primal; árboles no isomorfos de hasta 7 sitios (cota de estrella en todos, por debajo en 10 de 24) y caminos frente a anillos sin holonomía; escribe `data/tree_star.csv` | problema abierto 5 | 84 s |
| `diamond_single.py` | Diamante simple K4−e: v_max = 1/3 para toda holonomía (primal completo en 8 puntos y 6 juegos de rotaciones aleatorias) y estado explícito con tres emparejamientos; escribe `data/diamond_single.csv` | diamantes 5 | 2,4 s, pico 150 MB |
| `diamond_chain2.py` | Exploración: cadena abierta de 2 diamantes (7 qubits); mapa abeliano v(fA, fB) (malla 13×13 y recta fA = −fB) con el primal U(1), simetrías y orientación relativa de las dos celdas con `v_dual_cg` (caché reanudable en `data/diamond_chain2_evals.csv`); escribe `data/diamond_chain2_abel.csv` y `data/diamond_chain2_rel.csv`. No está en `run_all.sh`; `(ulimit -v 8000000; /usr/bin/time -f "%M KB" .venv/bin/python scripts/diamond_chain2.py [--rapido])`; `--rapido` es una prueba de humo con ficheros `_rapido` | diamantes 6 | 21,6 h con la caché vacía (pico 0,94 GB); regenerar desde la caché, 30 min (0,91 GB); `--rapido`, 565 s (968 MB) |
| `diamond_chain2_min.py` | Exploración: mínimo global de v en la cadena de 2 sobre (θ_A, θ_B, γ, Q), cribado en malla y Nelder–Mead desde los tres mejores; el mínimo exacto es x⋆ = (0, π, 0, R_x(π/2)) con v = 0,3255461236, órbita de gauge de dimensión 3, estrictez en las 3 direcciones transversas y anchura del testigo; reanudable (`data/diamond_chain2_min_evals.csv`); escribe `data/diamond_chain2_min*.csv`. No está en `run_all.sh`; `(ulimit -v 8000000; /usr/bin/time -f "%M KB" .venv/bin/python scripts/diamond_chain2_min.py [--rapido])` | diamantes 6.1 | 38 h (136 645 s, pico 1,99 GB) más 2,0 h del paso 2 en x⋆ (pico 0,85 GB); `--rapido`, 2569 s (839 MB) |
| `diamond_ring.py` | Exploración: cadena cerrada de k diamantes, mapa v(f, ψ) en los modos «igual» (fA = fB) y «opuesto» (fA = −fB) con `DiamondDual` por generación de columnas, mínimo en ψ de cada fila (Brent) y contraste con el primal reducido para k ≤ 3; reanudable (`data/diamond_ring_L<k>_evals.csv`); escribe `data/diamond_ring_L<k>_<modo>.csv` y `data/diamond_ring_min_L<k>_<modo>.csv`. `run_all.sh` solo lanza k = 2 (malla 13×13); k = 3 y 4: `(ulimit -v 12000000; /usr/bin/time -f "%M KB %e s" .venv/bin/python scripts/diamond_ring.py 3 13 13)` | diamantes 7 | k = 2: 175 s, pico 150 MB; k = 3: 99 min, pico 5,4 GB; k = 4: 41 h, pico 7,6 GB (regenerar desde la caché, 24 s y 0,22 GB) |
| `diamond_numeros.py` | Los números del artículo de diamantes que salen de los CSV de `data/` y ningún otro guion imprime (recuentos de puntos con v = 1/3, la caché del mínimo, curvaturas, ψ⋆, k(1/3 − v) del punto A, no monotonía del punto B), cada uno con el redondeo del texto; solo lee `data/`, sin SDP | diamantes 6, 6.1, 7, 8 | < 1 s |
| `diamond_limit.py` | Exploración: límite k → ∞ del déficit 1/3 − v de la cadena cerrada en un punto fijo, un proceso hijo por evaluación; A (f = 0, ψ = 0, k = 3, 5, 7) y B (f = 5π/12, ψ minimizado, k = 2..5); reanudable (`data/diamond_limit_<etiqueta>_evals.csv`); escribe `data/diamond_limit_A.csv` y `data/diamond_limit_B.csv`. No está en `run_all.sh`; `(ulimit -v 16000000; /usr/bin/time -f "%M KB %e s" .venv/bin/python scripts/diamond_limit.py A igual 0 0 3 5 7)` y `... B igual 5*pi/12 min 2 3 4 5` | diamantes 8 | A: k = 3, 1,6 s; k = 5, 18 s; k = 7, 12,8 h (pico 2,9 GB); B: k = 2, 5 s; k = 3, 103 s; k = 4, 31 min (pico 1,5 GB); k = 5, 2 h 6 min (pico 5,5 GB) |

`baedge2.py` guarda cada n en `data/bethe_cache/`; el repositorio incluye esa
caché, así que `python scripts/baedge2.py --report` reproduce la tabla 3 sin
recalcular. Para recalcular, borre la caché; se puede repartir entre procesos:
`python scripts/baedge2.py 1024 1025 &  python scripts/baedge2.py 2048 &`.

## Convenciones

El qubit 0 es el más significativo y |0⟩ es el autoestado de Z con valor +1. En
el gauge uniforme cada arista se orienta i → i+1, incluida la de cierre, con
T_e = −R_z(Θ/n). En `holonomy.xxz` el sitio i es el bit `(s >> i) & 1` (orden
inverso); no afecta a las energías.

## Notas numéricas

- Cerca de Δ = −1 y para n ≤ 6, el estado fundamental del anillo XXZ es el
  ferromagnético totalmente polarizado (energía nΔ/4), no el del sector de
  magnetización mínima. `xxz.eps` y `bethe.eps` lo incluyen; sin él, el
  objetivo dual bajaría por debajo de v_max (`sec.py` lo comprueba).
- En el triángulo el ínfimo del dual es el límite Δ → ∞ (testigo de Ising);
  `dual.ising_limit` lo incluye de forma analítica.
- Las ecuaciones de Bethe están implementadas para Δ > 0: con flujo π la
  rapidez extrema escapa a infinito en Δ = 0. Todos los óptimos con n ≥ 6
  tienen Δ⋆ cerca de 1.
- Clarabel termina a menudo en `optimal_inaccurate` con tolerancias de 1e-10;
  cada guion lo indica junto a los errores obtenidos.
- `data/diamond_ring_L4_evals.csv` (la caché del mapa de k = 4) es del formato
  anterior, sin la columna `estado_solver`: se calculó antes de que existiera.
  `diamond_ring.py` la lee igual y deja vacío ese campo en
  `data/diamond_ring_L4_igual.csv` y `_opuesto.csv` ("sin estado del solver,
  de caché antigua"). Los valores siguen certificados por la diferencia de
  cotas; solo falta saber si algún SDP restringido acabó en
  `optimal_inaccurate`.

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
.venv/bin/python -m pytest          # pruebas rápidas (~15 s)
./run_all.sh --rapido               # comprobaciones de menos de ~1 min
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
| `ladder` | Escaleras de dos patas: aristas, flujos de plaqueta, dual reducido de la escalera periódica |
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
| `figures.py` | Figuras 1 y 2 a partir de `data/` | 5.3, 6.2 | 10 s |
| `ladder.py` | Exploración: escalera 2×3 abierta (primal completo, holonomías abelianas y no abelianas), cotas de estrella y de plaqueta, primal U(1) en grafos, primal U(1)+traslación y dual reducido de la escalera periódica (2×4, 2×5) frente al primal, barrido del dual hasta 2×6; escribe `data/ladder.csv` | problema abierto 5 | ~75 min |
| `ladder_flux.py` | Exploración: barrido fino del flujo de plaqueta en la escalera periódica 2×4, 2×5 (dual-SDP) y 2×6 (Nelder-Mead), flujo del testigo y mínimo en el flujo; escribe `data/ladder_flux.csv`. No está en `run_all.sh`; conviene limitar la memoria (`ulimit -v 12000000`) | problema abierto 5 | ~2 h |

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

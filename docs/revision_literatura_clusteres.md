# Revisión de literatura: vía por clústeres para el XXZ retorcido con Δ grande

Fecha: 20 de noviembre de 2026. Objetivo: saber si la demostración autocontenida de las cotas
(P) y (D) del Teorema puente de `robustez_mps.tex` para el XXZ con Δ grande ya existe, y qué
resultados del artículo son nuevos.

## Qué necesitamos

| Clave | Requisito |
|---|---|
| N1 | Volumen finito con condiciones **periódicas** (anillo) |
| N2 | Flujo / salto con fase compleja U(1), uniforme en Φ |
| N3 | Modelo clásico con **varios** fundamentales degenerados (los dos Néel) |
| N4 | Cota bilateral \|E₀(n,Φ) + n e∞\| ≤ C e^{−n/ξ}, uniforme en Φ |
| N5 | Marginales de dos sitios de un estado del anillo exponencialmente cerca de las de la cadena infinita |
| N6 | Hamiltoniano efectivo de una pared de dominio (n impar) |
| N7 | Aplicable al XXZ antiferromagnético 1D con Δ grande |

## Cobertura

✓ cubre · ~ parcial · ✗ no · ? no verificado (sin acceso al texto completo)

| Fuente | N1 | N2 | N3 | N4 | N5 | N6 | N7 | Lectura |
|---|---|---|---|---|---|---|---|---|
| **Datta–Kennedy, CMP 236 (2003), math-ph/0208026** + cond-mat/0104199 | ✓ | ~ | ✓ | ~ | ~ | ✓ | ✓ | texto completo |
| Del Vecchio–Fröhlich–Pizzo–Ranallo, arXiv 2306.02772 (XXZ con campo) | ✗ abierta | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | texto completo |
| Nachtergaele–Sims–Young, AHP 23 (2022), 2010.15337, §8 (ruptura de traslación) | ✓ | ✓ | ✓ | ✗ | ~ | ✗ | ~ | §1, §2, §8 |
| Datta–Fernández–Fröhlich, JSP 84 (1996), mp_arc 95-228 | ✗ | ✓ | ✓ | ✗ | ~ | ✗ | ✗ (d ≥ 2) | texto completo |
| Datta–Fröhlich–Rey-Bellet–Fernández, HPA 69 (1996), mp_arc 96-226 | ✗ | ~ | ✓ | ✗ | ✗ | ~ | ✗ | texto completo |
| Borgs–Kotecký–Ueltschi, CMP 181 (1996), mp_arc 95-233 | ~ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ (d ≥ 2) | texto completo |
| Yarotsky, CMP 261 (2006), math-ph/0412040 | ✓ | ✓ | ✗ (único) | ✗ | ~ | ✗ | ✗ | texto completo |
| Yarotsky, math-ph/0411042 (cuasipartículas) | ~ | ✓ | ✗ | ✗ | ~ | ~ | ✗ | texto completo |
| Fröhlich–Pizzo; Del Vecchio–Fröhlich–Pizzo–Rossi (Lie–Schwinger, 2020–2022) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | texto completo |
| Bravyi–Hastings–Michalakis (2010); Michalakis–Zwolak (2013) | ✓ | ✓ | ✗ (TQO falla con Néel) | ✗ | ✗ | ✗ | ✗ | secciones relevantes |
| Watanabe, PRB 98, 155137 (2018), 1803.06558 | ✓ | ✓ | ✗ (único; degenerado «future work») | ~ (no riguroso) | ~ | ✗ | ✗ | texto completo |
| Albanese, JSP 55 (1989); CMP 134 (1990) | ? | ? | ~ | ? | ? | ~ | ~ | solo resumen |
| Kirkwood–Thomas, CMP 88 (1983) | ? | ✗ (positividad) | ~ | ? | ~ | ✗ | ~ | solo resumen |
| Kennedy–Tasaki, CMP 147 (1992) | ? | ? | ✓ | ? | ? | ? | ✗ (S = 1) | solo resumen |
| Matsui, CMP 253 (2005) | ? | ? | ✓ | ? | ? | ? | ✓ | cita secundaria |

## La fuente clave: Datta–Kennedy

- *Expansions for one quasiparticle states in spin 1/2 systems* (cond-mat/0104199) y *Instability
  of interfaces in the antiferromagnetic XXZ chain at zero temperature* (CMP 236, 2003,
  math-ph/0208026). Desarrollo de punto fijo de tipo Kirkwood–Thomas, ψ = exp(−½ Σ g(X) σ(X)),
  **en el anillo periódico**, para el XZ y el XXZ antiferromagnéticos muy anisótropos.
- n par: los dos niveles de Néel difieren en (c\|ε\|)^{N/2}, y ambos son N e₀ más una corrección
  exponencialmente pequeña (afirmado en la §3; la prueba de la tasa solo está esbozada).
- n impar: las condiciones periódicas fuerzan una pared; Teorema 2: dispersión de la pared
  E(k) = N e₀ + Σ_s ε_s e^{iks}, con \|ε_s\| ≤ (c\|ε\|)^{⌈\|s\|/2⌉}. Es el análogo riguroso de
  nuestra Observación del espinón, sin flujo.
- Trabajan por sectores de momento k, con fases complejas en esa base; el flujo uniforme, que
  conmuta con la traslación, puede caber en el mismo esquema (por comprobar). El argumento de
  Perron–Frobenius (ψ > 0) que identifica el fundamental en k = 0 se pierde con flujo, y hay que
  sustituirlo (p. ej. por continuidad en Φ y la simplicidad del autovalor del punto fijo).

## Novedad de los resultados de `robustez_mps.tex`

| Resultado | Estado en la literatura |
|---|---|
| Problema de marginales con holonomía en el anillo | No encontrado |
| R1: desdoblamiento R_n(1 − cos Φ/2), con su prefactor reticular | La forma de Lüscher n^{-1/2} e^{-n/(2ξ_JKM)} sale de resultados conocidos (Rutkevich 1904.11324, ec. 7: ξ = 1/(2m)); la periodicidad 4π se ha visto numéricamente (2510.19902). La ley en Φ con el prefactor no se ha encontrado; **pendiente**: Babelon–de Vega–Viallet, NPB 220 (1983) 13, y Shastry–Sutherland, PRL 65 (1990) 243 (sin acceso) |
| R2: espinón con carga ½ del flujo en anillos impares, constante c | No encontrado; lo más próximo, sin flujo, es Datta–Kennedy (dispersión de la pared) |
| R3 riguroso: corrección exponencial de la energía y de las marginales, uniforme en el flujo, con el doblete de Néel | No encontrado. Watanabe: fundamental único y no riguroso; NSY: estabilidad del gap, sin asintótica de la energía |

## Consecuencia para el artículo nuevo

No hay que partir de cero: el punto de partida natural es el desarrollo de Datta–Kennedy en el
anillo periódico. Lo que falta, y sería la aportación del artículo nuevo:

1. Añadir el flujo uniforme Φ y obtener cotas uniformes en Φ (N2), sustituyendo el argumento de
   positividad.
2. Escribir como teorema la cota bilateral \|E₀(n,Φ) + n e∞\| ≤ C (c/Δ)^{n/2} (N4).
3. Marginales de dos sitios a partir de la cercanía de g_Λ y g_∞ (N5).
4. Con n impar, la dependencia en Φ de la dispersión de la pared (carga ½) y la ley 1/n³.

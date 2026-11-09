# Rigidez frente a un flujo del testigo óptimo: literatura (Tarea 2)

Fecha: 2026-11-09. Nota de trabajo, no es parte del artículo. Cada afirmación lleva su fuente o
la marca «sin verificar». «Verificado» significa: leído en la página de la revista, en arXiv o en
el PDF de arXiv durante esta tarea. «ED propia» significa: comprobado con diagonalización exacta
hecha aquí (sección 9).

## 0. Resumen de decisiones

1. **Watanabe es autor único.** El artículo «Insensitivity of bulk properties to the twisted boundary
   condition» es de Haruki Watanabe solo (PRB 98, 155137, 2018; arXiv:1803.06558). No son
   «Watanabe, Po, Zaletel, Vishwanath»: Po, Zaletel y Vishwanath aparecen en la lista de referencias
   del artículo, por otro trabajo (PNAS 2015). El enunciado es de nivel físico (argumento basado en
   decaimiento exponencial de correlaciones, no un teorema matemático) y excluye explícitamente el
   caso con degeneración del fundamental.
2. **Fórmula de Sutherland verificada.** Para $H=\sum_i P_{i,i+1}$ (anillo, $P$ = intercambio de
   colores), en el límite termodinámico
   $$e_\infty(N)=1-\frac2N\big[\psi(1)-\psi(1/N)\big]=-1+2\sum_{k\ge2}\frac{(-1)^k\zeta(k)}{N^k},$$
   que es exactamente la serie publicada (Messio–Mila 2012, ec. 4; Dufour–Nataf–Mila 2015, ec. 8).
   $N=2$: $1-2\ln2=-0{,}386294$; $N=3$: $-0{,}703212$; $N=4$: $-0{,}825119$. La candidata del encargo es
   correcta tal cual. Confirmada además con ED propia (sección 9).
3. **Velocidad y carga central.** Con esta normalización $v=2\pi/N$ y $c=N-1$ (SU($N$)$_1$).
   Fuente publicada: Messio–Mila (atribuyen $v$ a Sutherland y $c$ a Affleck); ED propia confirma el
   producto $cv$.
4. **Rigidez del anillo SU($N$) retorcido.** No he encontrado una fórmula explícita publicada para
   $N>2$. Lo que sí hay: para $N=2$, Shastry–Sutherland (exacto) y Laflorencie–Capponi–Sørensen
   (correcciones logarítmicas). Para $N$ general doy una fórmula **derivada aquí** (teoría de campos
   más ED propia), marcada como tal, en la sección 4.
5. **¿Está hecho el límite de qudits?** Parcialmente. Allerstorfer et al. (CMP 2026, sección 5)
   escriben que el valor de $\vmax$ en el anillo de qubits es $\lambda_{\max}$ del hamiltoniano de
   Heisenberg y su límite es $\ln2$. **No** tratan $d>2$, ni calculan valores a $n$ finito, ni hay
   flujo/holonomía. Johnson–Viola solo tratan unión de tres partes y compartibilidad $1\to n$. Que
   para qudits el límite sea $q_\infty(d)=[\psi(1)-\psi(1/d)]/d$ se sigue de combinar la ec. (58) de
   Allerstorfer con Sutherland, pero no está escrito en ninguno de los dos (hasta donde he podido
   ver). La búsqueda no es exhaustiva.
6. **Aviso sobre una cita existente.** `holonomia_werner.tex` (el artículo de anillos) dice que los
   valores $n=3..6$ de $\vmax$ son «resultados conocidos \cite{allerstorfer}». En el PDF de
   Allerstorfer solo aparecen la definición (150) y el límite (151); no he visto una tabla a $n$ finito.
   Conviene matizar a «se obtienen de la definición (150) de \cite{allerstorfer}».

## 1. Referencias en formato `\bibitem`

Las claves son propuestas. `anillos`, `escaleras` y `diamantes` ya existen en los otros trabajos.

```latex
\bibitem{kohn}
W.~Kohn.
\newblock Theory of the insulating state.
\newblock Phys.\ Rev.\ \textbf{133}, A171 (1964).
\newblock doi:10.1103/PhysRev.133.A171.

\bibitem{shastry}
B.~S. Shastry, B.~Sutherland.
\newblock Twisted boundary conditions and effective mass in Heisenberg-Ising and Hubbard
rings.
\newblock Phys.\ Rev.\ Lett.\ \textbf{65}, 243 (1990).
\newblock doi:10.1103/PhysRevLett.65.243.

\bibitem{watanabe}
H.~Watanabe.
\newblock Insensitivity of bulk properties to the twisted boundary condition.
\newblock Phys.\ Rev.\ B \textbf{98}, 155137 (2018). arXiv:1803.06558.
\newblock doi:10.1103/PhysRevB.98.155137.

\bibitem{sutherland75}
B.~Sutherland.
\newblock Model for a multicomponent quantum system.
\newblock Phys.\ Rev.\ B \textbf{12}, 3795 (1975).
\newblock doi:10.1103/PhysRevB.12.3795.

\bibitem{messiomila}
L.~Messio, F.~Mila.
\newblock Entropy dependence of correlations in one-dimensional $\mathrm{SU}(N)$
antiferromagnets.
\newblock Phys.\ Rev.\ Lett.\ \textbf{109}, 205306 (2012). arXiv:1207.1320.
\newblock doi:10.1103/PhysRevLett.109.205306.

\bibitem{dufour}
J.~Dufour, P.~Nataf, F.~Mila.
\newblock Variational Monte Carlo investigation of $\mathrm{SU}(N)$ Heisenberg chains.
\newblock Phys.\ Rev.\ B \textbf{91}, 174427 (2015). arXiv:1502.01895.
\newblock doi:10.1103/PhysRevB.91.174427.

\bibitem{affleck88}
I.~Affleck.
\newblock Nucl.\ Phys.\ B \textbf{305}, 582 (1988).
% Título: «Critical behaviour of SU(n) quantum chains and topological non-linear
% sigma-models»; el título NO está verificado (volumen y página sí, citados en
% Messio-Mila y Dufour et al.).

\bibitem{affleck86}
I.~Affleck.
\newblock Nucl.\ Phys.\ B \textbf{265}, 409 (1986).
% Volumen y página verificados (citados en Majumdar-Mukherjee y Dufour et al.); título sin verificar.

\bibitem{majumdar}
K.~Majumdar, M.~Mukherjee.
\newblock Logarithmic corrections to finite size spectrum of $\mathrm{SU}(N)$ symmetric quantum
chains.
\newblock arXiv:cond-mat/0206359 (2002).
% Si salió publicado en revista: sin verificar.

\bibitem{laflorencie}
N.~Laflorencie, S.~Capponi, E.~S. S{\o}rensen.
\newblock Finite size scaling of the spin stiffness of the antiferromagnetic $S=\frac12$
$XXZ$ chain.
\newblock Eur.\ Phys.\ J.\ B \textbf{24}, 77 (2001). arXiv:cond-mat/0105462.

\bibitem{johnson}
P.~D. Johnson, L.~Viola.
\newblock Compatible quantum correlations: Extension problems for Werner and isotropic states.
\newblock Phys.\ Rev.\ A \textbf{88}, 032323 (2013). arXiv:1305.1342.
\newblock doi:10.1103/PhysRevA.88.032323.

\bibitem{allerstorfer}
R.~Allerstorfer, M.~Christandl, D.~Grinko, I.~Nechita, M.~Ozols, D.~Rochette,
P.~Verduyn Lunel.
\newblock Monogamy of highly symmetric states.
\newblock Commun.\ Math.\ Phys.\ \textbf{407}, 21 (2026). arXiv:2309.16655.
\newblock doi:10.1007/s00220-025-05483-3.

\bibitem{jakab}
D.~Jakab, A.~Solymos, Z.~Zimborás.
\newblock Extendibility of Werner states.
\newblock arXiv:2208.13743 (2022).
% Publicación en revista: sin verificar.
```

Notas de verificación de las citas:

- Kohn: título, volumen, página, año y doi leídos en la página de APS (journals.aps.org). Recibido el 30
  de agosto de 1963, publicado el 6 de enero de 1964.
- Shastry–Sutherland: autores, título, PRL 65(2), 243 y doi leídos en APS; publicado el 9 de julio de 1990.
  El rango de páginas 243–246 viene de un registro bibliográfico secundario (no de APS). Existe un
  comentario de Kawakami y Yang, PRL 65, 3063 (1990), mencionado en la página de APS.
- Watanabe: autor único, PRB 98, 155137 y arXiv leídos en la página de arXiv (v3, 23 feb 2019). El doi
  `10.1103/PhysRevB.98.155137` es el que figura como «related DOI» en esa página; no lo he resuelto en
  doi.org.
- Sutherland 1975: título, PRB 12(9), 3795–3805, noviembre 1975, y doi vistos en el registro de APS vía
  buscador; **no he leído el texto del artículo**. El resultado de energía lo cito a través de
  Messio–Mila y Dufour et al., que sí lo reproducen.
- Allerstorfer et al.: CMP 407(1), artículo 21 (pp. 21:1–21:50), publicado 2026-01-02, doi
  `10.1007/s00220-025-05483-3` (registro de CWI y de arXiv). El artículo existente cita solo arXiv; hay que
  actualizar.
- Johnson–Viola: PRA 88, 032323 (2013), doi y arXiv:1305.1342 (v3, 11 sep 2013) leídos en arXiv.
  **El título publicado lleva «Extension problems»** (la versión arXiv dice «on extension problems»).

## 2. Kohn (1964): criterio aislante/conductor

**Qué afirma** (lo verificado: resumen de APS y descripción de Resta, cond-mat/0003014). Kohn propone
una caracterización del estado aislante que incluye aislantes de bandas y también aislantes tipo Mott
que la teoría de bandas clasificaría como metales. El criterio es de localización de la función de
onda: la función de muchos cuerpos se descompone en una suma $\Psi=\sum_J\Psi_J$ de componentes
localizadas en regiones esencialmente desconectadas del espacio de configuraciones, con solapamientos
exponencialmente pequeños; bajo esa hipótesis demuestra que la conductividad dc se anula. Un apéndice
resuelve un modelo con transición aislante–conductor.

**Qué se cita habitualmente de Kohn en términos de giro** (no verificado en el texto de Kohn): la
sensibilidad de la energía fundamental a un flujo (o a un cambio de condiciones de contorno de un
anillo) es exponencialmente pequeña en un aislante y $O(1/L)$ en un conductor, y la segunda derivada
$\partial_\theta^2E_\theta$ da el peso de Drude («fórmula de Kohn»). Esto último sí lo he visto citado así
en Watanabe (sección VII, cita a Kohn, Scalapino–White–Zhang y Oshikawa): «$D\propto L\,\partial_\theta^2E_\theta$».
No he podido leer la prueba original de Kohn; la formulación en términos de energía frente a flujo es
la lectura moderna (Shastry–Sutherland la usan explícitamente).

**Qué no dice.** No da una fórmula para sistemas concretos. El criterio de localización es sobre la
función de onda, no un enunciado cuantitativo sobre cuánto vale la dependencia en el flujo. En sistemas
críticos (no aislantes ni conductores clásicos) el criterio no decide: la rigidez es $\propto1/n$, no
cero ni $O(1)$.

**Frase para el artículo.** «Siguiendo a Kohn~\cite{kohn}, la respuesta de la energía fundamental a un
flujo en el anillo distingue fases: exponencialmente pequeña si hay gap, y de orden $1/n$ con
coeficiente la rigidez $D$ si el sistema es crítico. La sensibilidad de $\vmax$ a la holonomía mide,
por tanto, esa rigidez del testigo óptimo.»

## 3. Insensibilidad exponencial (Watanabe 2018)

**Enunciado y autoría (verificado, texto completo de arXiv v3).** H. Watanabe, PRB 98, 155137.
Para gaps de un hamiltoniano con simetría U(1) e interacciones de alcance finito, las propiedades de
volumen (valores esperados de operadores locales, energía fundamental, gap de excitación, respuestas
estáticas y de baja frecuencia) no dependen del ángulo del giro en el límite de sistema grande, con
supresión exponencial en la dimensión lineal. Para valores esperados de operadores sumas de locales
(incluido $\hat H$, es decir, la planitud de $E_\theta$), la supresión es un factor
$L^{3/2}e^{-L/(2\xi)}$ (ec. 20 y texto siguiente), con $\xi=\xi_0+2v/\Delta$, donde $\xi_0$ y $v$ son
las constantes de la cota de Lieb–Robinson y $\Delta$ el gap.

**Hipótesis exactas (del texto):**

- Localidad: $\hat H=\sum_{\vec x}\hat H_{\vec x}$ con alcance finito $r$ independiente del tamaño.
- Simetría U(1) cuyo parámetro de fase es el que se tuerce (conservación de una carga con
  densidad $\hat n_x$). El giro es $\theta_i$ en cada dirección $i=1..d$.
- Gap no nulo y **fundamental único para un valor de $\theta$** (p. ej. $\theta=0$), en el límite
  $L\to\infty$. No se supone el gap uniforme a lo largo del giro: se **deduce** que el gap
  depende de $\theta$ como mucho en una cantidad exponencialmente pequeña (sección VI, por reducción al
  absurdo variacional).
- Dimensión: arbitraria ($d$ general); para $d=1$ es el anillo con un flujo.
- Sistema finito con condiciones periódicas de tamaño lineal $L_i$.

**Qué no dice, y avisos importantes:**

- Es un argumento físico con inputs rigurosos (decaimiento exponencial de correlaciones conectadas a
  tiempos iguales por Lieb–Robinson), no un teorema con todas las constantes controladas.
- **Excluye la degeneración del fundamental**: «In the derivation we assumed the uniqueness of the
  ground state», y para el caso cuasi-degenerado dice que lo esperable es análogo, pero «we will
  leave the concrete analysis to future work». El anillo XXZ con $\Delta>1$ y $n$ par tiene
  fundamental cuasi-degenerado (orden de Néel), así que la hipótesis estricta no se cumple allí; solo el
  comentario heurístico (supresión $e^{-V/\xi_d}$ con rotura de simetría discreta).
- No es uniforme cuando el gap se cierra: $\xi\to\infty$ para $\Delta\to0$. En el problema del anillo, esto
  significa que la constante de la supresión explota al acercarse al punto isótropo desde
  $\Delta>1$.
- No dice nada de sistemas críticos, donde la dependencia es $D\tilde\Phi^2/(2n)$.
- La simetría que se tuerce debe ser U(1). Para un giro en una dirección de Cartan de SU($N$) la
  hipótesis se cumple (el hamiltoniano conmuta con ella); no se afirma nada sobre giros no abelianos.

**Frase para el artículo.** «Si el testigo óptimo tiene gap, la energía fundamental del anillo depende del
flujo solo de forma exponencialmente pequeña en $n$~\cite{watanabe} (para un fundamental único; en
fases con rotura de simetría discreta es el comportamiento esperado, no demostrado); la dependencia
de $\vmax$ en la holonomía procede entonces de la región crítica $\Delta\to1$.»

## 4. Shastry–Sutherland (1990) y rigidez del anillo SU($N$)

**Shastry–Sutherland (verificado en APS y, para la fórmula, en Laflorencie et al., ec. 3).** La
energía de condiciones de contorno retorcidas es la inversa de la masa efectiva (rigidez) de portadores.
Calculan la rigidez exacta (ansatz de Bethe) del anillo Heisenberg–Ising 1D y del Hubbard 1D; para
Hubbard también una rigidez de espín (ansatz de Bethe–Yang anidado). Para el anillo XXZ con
$\Delta=\cos\mu$ (convención de Laflorencie et al.: $H=J\sum[S^xS^x+S^yS^y+\Delta S^zS^z]$, $J=1$,
$E_0(\phi)=E_0(0)+\rho\phi^2/(2L)$):
$$\frac{\rho}{J}=\frac{\pi\sin\mu}{4\mu(\pi-\mu)},\qquad \rho(\Delta=1)=\tfrac14 .$$
Es la fórmula que ya usa el artículo de anillos ($D(\gamma)$). Interpretan la rigidez como sonda de la
transición metal–aislante (Mott en $\Delta=1$).

**Qué no dice.** No trata SU($N$) con $N>2$ (el SU($N$) con flujo requiere el ansatz de Bethe anidado de
Sutherland; en el resumen solo aparece anidado para Hubbard). No da el tamaño finito ni las
correcciones logarítmicas.

**Correcciones logarítmicas SU(2) (verificado, Laflorencie–Capponi–Sørensen, ec. 22).**
$\rho(L)/J\simeq\frac14+\frac{\rho(L_0)/J-1/4}{1+8(\rho(L_0)/J-1/4)\ln(L/L_0)}$, que para $L\to\infty$ da
$\rho(L)\simeq\frac14+\frac1{8\ln L}$ independientemente de $\rho(L_0)$. Es la amplitud universal $a=1/2$
del artículo de anillos ($D_n=D_\infty[1+a/\ln n]$).

**Fórmula para SU($N$): DERIVADA AQUÍ, no publicada (hasta donde he encontrado).** Con $H=\sum P$ y el
giro en el enlace de cierre $e^{i(\varphi_a-\varphi_b)}$ para el intercambio de colores $a,b$ (equivale a
dar a cada color $a$ la fase $e^{i\varphi_a}$ al cruzar el enlace; solo importa la parte sin traza
$t=\varphi-\bar\varphi\,(1,\dots,1)$), SU($N$)$_1$ es $N-1$ bosones libres sobre la red de raíces, y
$$\Delta E(\varphi)=E_0(\varphi)-E_0(0)=\min_{\lambda\in\Lambda_w}\frac{v\,|t-2\pi\lambda|^2}{2\pi n}
=\min_{\lambda\in\Lambda_w}\frac{|t-2\pi\lambda|^2}{N\,n}\;(1+O(1/\ln n)),\quad v=\frac{2\pi}N,$$
con $\Lambda_w$ la red de pesos de su(N) (proyección de $\mathbb Z^N$ sobre el hiperplano sin traza) y
$n$ múltiplo de $N$. La fórmula es cuadrática (con la misma rigidez en todas las direcciones de
Cartan, por simetría de Weyl) hasta la frontera de la celda de Voronoi de $\Lambda_w$, donde hay un cruce
de niveles (cúspide).

Comprobaciones: (i) $N=2$, $t=(\Theta/2,-\Theta/2)$: $\Delta E=\Theta^2/(4n)$, igual que el resultado
publicado ($H=\sum P=2\sum\vec S\cdot\vec S+\text{cte}$ y $\rho=1/4$ da $2\cdot\rho\,\Theta^2/(2n)$); cúspide en $\Theta=\pi$.
(ii) ED propia (sección 9): rigidez isótropa en las direcciones de Cartan probadas para $N=2,3,4$, cúspide
en $t=2\pi/3$ para $N=3$ con $t=(t,-t,0)$ ($n=9$, observada en $t\approx2{,}09$), y la forma cuadrática con un
exceso $\approx10$–$15\,\%$ a $n\le14$ que decrece despacio, compatible con $1/\ln n$.
**Para $n$ no múltiplo de $N$** el mínimo se desplaza (generalización del efecto de paridad): visto en ED
($N=3$, $n=10$: la energía baja al aumentar $t$); no lo he derivado ni comprobado cuantitativamente.

**Correcciones logarítmicas para $N>2$: amplitud NO encontrada.** Majumdar–Mukherjee (preprint
cond-mat/0206359; SU($N$)$_1$ perturbado por el operador marginal $\varphi\propto-\sum_AJ_L^AJ_R^A$,
coeficiente de RG $b=2N/\sqrt{N^2-1}$, $g(n)=1/[\pi b\ln n]$ con corrección $\ln\ln n$) dan correcciones de
niveles excitados primarios y de la energía fundamental (esta, a orden $g^3$), pero no la del estado
con giro. Estimación mía, sin verificar: como el acoplamiento efectivo escala como $1/(N\ln n)$ y la
estructura del valor esperado de $J_L\!\cdot\!J_R$ en el vacío retorcido es universal, la corrección
relativa sería $\approx a_N/\ln n$ con $a_N=1/N$ ($a_2=1/2$ es el valor publicado). Los exactos de ED
($n\le14$ para $N=2$, $n\le12$ para $N=3$, $n=8$ para $N=4$) son compatibles en orden de magnitud con
esto, pero son pocos tamaños para fijar la amplitud.

**Frase para el artículo.** «La rigidez del anillo SU($N$) con flujo en una dirección de Cartan es
$\Delta E=|t|^2/(Nn)$ a orden dominante (SU($N$)$_1$ con $v=2\pi/N$; para $N=2$ se reduce a
$D=1/4$ de~\cite{shastry}), con correcciones $O(1/\ln n)$ del operador marginal; Resultado derivado de
teoría de campos, no teorema.»

## 5. Energía por sitio del modelo SU($N$) de Uimin–Lai–Sutherland

**Fórmula verificada.** Para $H=\sum_{i=1}^nP_{i,i+1}$ ($P$ = permutación de colores, SU($N$) con la
fundamental en cada sitio), en el límite $n\to\infty$:
$$e_\infty(N)=-1+2\sum_{k=2}^\infty\frac{(-1)^k\zeta(k)}{N^k}
=1-\frac2N\Big[\psi(1)-\psi\!\big(\tfrac1N\big)\Big]=1+\frac2N\Big[\gamma+\psi\!\big(\tfrac1N\big)\Big].$$

- La serie es la ec. (4) de Messio–Mila (PRL 109, 205306), con $H=\sum P_{ij}$ (su ec. 2), y la ec. (8) de
  Dufour–Nataf–Mila (PRB 91, 174427), donde $H=\sum_{\langle ij\rangle}\sum_{\alpha\beta}
  a^\dagger_{i\alpha}a_{i\beta}a^\dagger_{j\beta}a_{j\alpha}$, que en ocupación 1 es $P_{ij}$. Ambos la atribuyen
  al ansatz de Bethe de Sutherland (PRB 12, 3795). **No he leído el original de Sutherland.**
- Equivalencia con la forma digamma: $\psi(1+x)-\psi(1)=\sum_{k\ge2}(-1)^k\zeta(k)x^{k-1}$ y
  $\psi(1/N)=\psi(1+1/N)-N$, de donde $\psi(1)-\psi(1/N)=N-\sum_{k\ge2}(-1)^k\zeta(k)N^{1-k}$ y
  $1-\tfrac2N[\cdot]=-1+2\sum(-1)^k\zeta(k)N^{-k}$. Comprobado también numéricamente a 6 cifras para
  $N=2..10$ (scipy, $k\le400$).
- **$N=2$:** $\psi(1)-\psi(1/2)=2\ln2$, luego $e_\infty(2)=1-2\ln2=-0{,}386294$. Coincide con
  Hulthén: $P=(1+\vec\sigma\cdot\vec\sigma)/2$ y $\sum\vec\sigma\cdot\vec\sigma/n\to1-4\ln2$ dan
  $\tfrac12[1+(1-4\ln2)]=1-2\ln2$.
- **Valores:** $N=3$: $1-\frac{\pi}{3\sqrt3}-\ln3=-0{,}7032120767$; $N=4$: $1-\frac\pi4-\frac32\ln2=-0{,}8251189342$;
  $N=5$: $-0{,}8847296927$; $N=6$: $-0{,}9183039468$; $N\to\infty$: $-1$.
- **Conversión de normalización.** Con $H'=\sum(P-1)$: $e'_\infty=e_\infty-1=-\frac2N[\psi(1)-\psi(1/N)]$
  ($N=2$: $-2\ln2$). Con $H''=\sum(1-P)/2$ (el hamiltoniano de Allerstorfer): $\lambda_{\max}(H''/n)=
  (1-e_\infty)/2=[\psi(1)-\psi(1/N)]/N$ ($N=2$: $\ln2$). Con $H=-\sum P$ (ferromagnético) el fundamental
  es el simétrico con $e=-1$ exacto, que no es el modelo crítico.
- **ED propia (sección 9)**, $n^2(e_n-e_\infty)$ frente a la predicción CFT $-\pi^2(N-1)/(3N)$: $N=2$ ($n=14$):
  $-1{,}6657$ vs $-1{,}6449$; $N=3$ ($n=12$): $-2{,}2318$ vs $-2{,}1932$; $N=4$ ($n=8$): $-2{,}5605$ vs $-2{,}4674$.
  Converge a la predicción con $n$ (con correcciones logarítmicas).

**Qué no dice.** La fórmula es el límite $n\to\infty$; no da $e_n$ a $n$ finito ni el efecto de
$n\not\equiv0\pmod N$. Uimin (1970) y Lai (1974) son los orígenes del modelo ULS (citas de memoria, sin verificar:
G. V. Uimin, JETP Lett. 12, 225 (1970); C. K. Lai, J. Math. Phys. 15, 1675 (1974)).

**Frase para el artículo.** «La energía por sitio del anillo SU($d$) $\sum P_{i,i+1}$ en el límite
termodinámico es $e_\infty(d)=1-\frac2d[\psi(1)-\psi(1/d)]$~\cite{sutherland75,messiomila}; para
$d=2$ es $1-2\ln2$ y para $d=3,4$ vale $-0{,}7032$ y $-0{,}8251$.»

## 6. Velocidad y carga central de SU($N$)$_1$

- **Carga central $c=N-1$:** Affleck (NPB 305, 582) la propone; cita en Messio–Mila (verificado como cita).
  Corroborada por entropía de entrelazamiento en DMRG/QMC (Nataf–Mila PRB 97, 134420, 2018, y
  D'Emidio–Block–Kaul, arXiv:1502.04739; solo leí sus resúmenes).
- **Velocidad $v=2\pi/N$** con $H=\sum P$ (J = 1): Messio–Mila (PRL 109, 205306): «$N-1$ ramas de
  excitaciones elementales con la misma velocidad $v=2\pi/N$ a $k$ pequeño», resultado atribuido a Sutherland.
  Chequeo: $N=2$ da $v=\pi$, que es $\pi J/2$ del Heisenberg con $J=2$ (porque $\sum P=2\sum\vec S\cdot\vec S+$cte).
  La entropía $S(T)=k_B\pi cT/(3v)=k_BN(N-1)T/6$ (Messio–Mila ec. 5, citando Lee 1994).
- **Corrección de tamaño finito:** $E_0(n)/n=e_\infty-\pi cv/(6n^2)=e_\infty-\pi^2(N-1)/(3Nn^2)$. Es el
  esquema estándar $E_0=n\,e_\infty-\pi cv/(6n)$ (Nataf–Mila 2016 lo usan con $2\pi cv/(12N_s^2)$).
  ED propia (sección 5) confirma el producto $cv$.

**Qué no dice.** No hay fórmula de la velocidad con flujo; la renormalización logarítmica de $v$ no
la he visto para SU($N$).

**Frase.** «Con la normalización $\sum P$ las excitaciones de baja energía tienen velocidad $2\pi/N$ y
carga central $N-1$~\cite{messiomila,affleck88}.»

## 7. Espectro de tamaño finito con giro: estado de la literatura

- **SU(2), XXZ retorcido:** Alcaraz–Barber–Batchelor (PRL 58, 771, 1987; ya en el artículo de anillos) y
  Klümper–Batchelor–Pearce (J. Phys. A 24, 3111, 1991; ya citado). No re-verificados aquí.
- **SU($N$), giro general:** una referencia candidata es Ribeiro–Martins–Galleas, «Integrable SU(N) vertex
  models with general toroidal boundary conditions» (arXiv:nlin/0308011), que resuelve el Bethe anidado con
  giros generales; la búsqueda solo devolvió su título. No sé si da el espectro de tamaño finito.
  Alcaraz–Martins (J. Phys. A 23, 1990, el dato de página no está verificado) son la fuente de las
  dimensiones conformes del $\mathrm{SU}_q(N)$ de Perk–Schultz. **Sin verificar** que alguna dé explícitamente
  la rigidez con giro del ULS.
- **Resumen:** no he encontrado una fórmula explícita de $E_0(\varphi)$ a orden $1/n$ para el anillo SU($N$)
  retorcido, ni correcciones logarítmicas con flujo. La fórmula de la sección 4 sale de aplicar
  SU($N$)$_1$ como $N-1$ bosones libres en la red de pesos. Es un resultado de teoría de campos que se puede
  presentar como «Resultado derivado», consistente con ED.

## 8. Extensibilidad y «joinability» de Werner e isotrópicos

**Johnson–Viola (PRA 88, 032323, 2013; verificado en arXiv).** Problema de marginales cuánticas para estados
de Werner e isotrópicos en dimensión $d$. Dos escenarios: *joinability* de tres partes (Alicia–Bob y
Alicia–Carlos, cada par en un Werner o isotrópico) y compartibilidad $m$-$n$ (un Werner compartido entre $N$
subsistemas, repartidos en grupos de $m$ y $n$). Dan condiciones necesarias y suficientes para
*joinability* de tres partes y compartibilidad $1\to n$ en cualquier $d$; relación con la clonación cuántica.
Los estados de Werner e isotrópicos y la fórmula (149) de Allerstorfer para $K_{n,1}$, $q_W(K_{n,1},d)
=\min\{(n+d-1)/(2n),1\}$, se atribuyen a este trabajo.
**No tratan** anillos, ciclos ni grafos generales ni flujo. **No** calculan $\vmax$ en anillos de qudits.

**Allerstorfer et al. («Monogamy of highly symmetric states», CMP 407, 21, 2026; arXiv:2309.16655; leído el PDF
de arXiv v2).** Definen $G$-extendibilidad para grafos arbitrarios y la estudian en detalle para el grafo
completo $K_n$: valor máximo de la proyección antisimétrica $q_W(n,d)$ y de la maximalmente entrelazada
$p_B,p_I$, con SDP y teoría de representaciones. Para grafos transitivos en aristas, la ec. (58) da
$p_\Pi^w(G,d)=\lambda_{\max}\big(\tfrac1{|E|}\sum_e\Pi_e\big)$.

- **Sección 5 (ciclo $C_n$, solo $d=2$):** $p_W(C_n,2)=\lambda_{\max}(H_{C_n})$, con
  $H_{C_n}=\frac1n\sum_{e}\frac{I_e-F_e}2$ (ec. 150), lo identifican con el Heisenberg XXX$_{1/2}$ antiferromagnético
  (el texto escribe $H_{XXX}=-H_{C_n}$, cierto salvo una constante: con $\vec S\cdot\vec S=(P-\tfrac12)/2$ se tiene
  $H_{C_n}=\tfrac14-\frac1n\sum\vec S\cdot\vec S$), y usan el resultado de Bethe $\to\ln2$ para $n\to\infty$ (ec. 151,
  citan Faddeev 1996 y ABBBQ87 ec. 2.50). Lo aplican a discriminación de estados de Bell en comunicación
  simultánea (cota $\ln2\approx0{,}69$) y verificación de posición cuántica.
- **Convención de Werner:** $q$ = peso del proyector antisimétrico normalizado,
  $\rho=q\,\Pi_A/\mathrm{Tr}\,\Pi_A+(1-q)\Pi_S/\mathrm{Tr}\,\Pi_S$. Relación con el $v$ de la Tarea:
  $\mathrm{tr}(\rho P)=1-2q=1/d-v(1+1/d)$, luego $q=\tfrac12\big[1-\tfrac1d+v(1+\tfrac1d)\big]$ y para $d=2$
  $v=(4q-1)/3$ (el artículo de anillos). Comprobado: $n=6$, $d=2$: $v=0{,}62284$, como en el artículo.
- **No tratan** $d>2$ en el ciclo, ni valores a $n$ finito, ni flujo/holonomía entre aristas.

**Otros:** Jakab–Solymos–Zimborás (arXiv:2208.13743) resuelven la extendibilidad de Werner en el
grafo bipartito completo $K_{n,m}$ (extendibilidad simétrica a dos lados) como problema de estado
fundamental de un hamiltoniano con simetría; Solymos–Jakab–Zimborás (arXiv:2411.04597) hacen
lo propio con estados de Brauer. Ninguno trata ciclos (por lo leído en resúmenes y en la lista de
contenidos de Allerstorfer).

**¿Está hecho el límite de qudits?** No explícitamente en lo que he podido ver. Sí es inmediato:
- Por la ec. (58) de Allerstorfer (válida para cualquier $d$ y grafo transitivo en aristas),
  $\vmax(n,\Theta=0)$ en el anillo de qudits corresponde a $q_{\max}(C_n,d)=\tfrac12(1-e_n(d))$ con
  $e_n(d)=E_0(\sum P)/n$.
- Con Sutherland, $q_\infty(d)=[\psi(1)-\psi(1/d)]/d$ y, con la conjetura de trabajo de la Tarea,
  $v_\infty(d)=\tfrac d{d+1}(\tfrac1d-e_\infty(d))$: $d=2$: $0{,}590863$ $[=(4\ln2-1)/3]$; $d=3$: $0{,}777409$;
  $d=4$: $0{,}860095$; $d=5$: $0{,}903941$. Valores a $n$ finito (ED propia, $d=3$): $n=6$: $0{,}82569$,
  $n=9$: $0{,}79829$, $n=12$: $0{,}78903$; ($d=4$, $n=8$): $0{,}89210$.
- Por tanto el límite de qudits es nuevo **como enunciado explícito**, pero técnicamente es un corolario de
  dos resultados publicados. El efecto del flujo (holonomía $U_e\in SU(d)$, giro en dirección de Cartan) no está
  en ninguno. La búsqueda no ha sido exhaustiva; conviene redactar con «hasta donde sabemos».

**Frase para el artículo.** «Para holonomía trivial, el valor de $\vmax$ en el anillo de qudits es el de la
energía fundamental del anillo SU($d$) $\sum P$: esto sigue de la ec.~(58) de~\cite{allerstorfer} (que
solo explicita $d=2$, con límite $\ln2$) y de la solución de Sutherland~\cite{sutherland75}; lo que aporta este
trabajo es la dependencia en la holonomía.»

## 9. Verificaciones numéricas propias (ED)

ED de $H=\sum_{i}P_{i,i+1}$ en el anillo completo de $N^n$ estados con giro en el enlace de cierre
(matriz hermítica con fase $e^{i(\varphi_a-\varphi_b)}$ en el intercambio $a\leftrightarrow b$ de colores),
`scipy.sparse.linalg.eigsh`, $N^n\le4^8$ o $3^{12}$ o $2^{14}$ estados, 681 MB de pico, 14 s (sin giro), 91 s
(con giro). Scripts temporales, no en el repositorio (en la carpeta de borradores de la sesión).
Resultados de esta nota:

| $N$ | $n$ | $e_n$ | $n^2(e_n-e_\infty)$ | predicción $-\pi^2(N-1)/(3N)$ |
|---|---|---|---|---|
| 2 | 14 | $-0{,}394793$ | $-1{,}6657$ | $-1{,}6449$ |
| 3 | 12 | $-0{,}718711$ | $-2{,}2318$ | $-2{,}1932$ |
| 4 | 8 | $-0{,}865126$ | $-2{,}5605$ | $-2{,}4674$ |

Rigidez: cociente $R(n)=\Delta E(\varphi)\,N\,n/|t|^2$ a $\varphi$ pequeño (predicción dominante 1), igual
para todas las direcciones de Cartan probadas dentro de $\sim10^{-3}$:

| $N$ | $n$ | $R(n)$ |
|---|---|---|
| 2 | 4, 8, 12, 14 | 1,333; 1,185; 1,149; 1,139 |
| 3 | 6, 9, 12 | 1,209; 1,150; 1,125 |
| 4 | 4, 8 | 1,333; 1,153 |

Cúspide: $N=3$, $n=9$, $t=(t,-t,0)$: $\Delta E$ crece como parábola hasta $t=2\pi/3$ ($\Delta E=0{,}3596$,
predicción cuadrática $0{,}3249$, cociente 1,107) y luego baja: el cruce está en $2\pi/3$, no en $\pi$ (frontera
de Voronoi de la red de pesos). $N=2$, $n=12$: periodicidad $2\pi$ y máximo en $\pi$ ($0{,}2311$ frente a $0{,}2056$).

Estas cifras son comprobaciones de coherencia de la sección 4. Si se quieren en el repositorio, deberían
pasar por el patrón de `scripts/` de la tarea correspondiente; aquí no se ha creado ningún guion.

## 10. Lista de lo no verificado

- Texto original de Sutherland PRB 12, 3795 (la fórmula se verifica vía dos artículos que la citan y por ED);
  título de Affleck NPB 305, 582 y NPB 265, 409; Uimin y Lai (citas de memoria); estado de publicación
  de Majumdar–Mukherjee y de Jakab–Solymos–Zimborás.
- Que Kohn (1964) contenga literalmente la afirmación «exponencialmente pequeña en aislantes, $O(1/L)$ en
  conductores» y la fórmula de Drude (lo leído: resumen de APS, descripción de Resta, cita en Watanabe).
- Rango de páginas 243–246 de Shastry–Sutherland (registro secundario).
- Amplitud $a_N=1/N$ de la corrección logarítmica de la rigidez SU($N$), y el desplazamiento del mínimo
  para $n\not\equiv0\pmod N$: conjetura propia, solo consistente con ED a tamaños pequeños.
- Alcaraz–Martins, Ribeiro–Martins–Galleas, de Vega, Fowler: solo títulos o menciones; no sé si dan la
  rigidez retorcida del ULS.
- Que ningún otro trabajo calcule $\vmax$ en anillos de qudits: la búsqueda fue de unas pocas consultas.

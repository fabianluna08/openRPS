# OpenRPS: Radiowave Propagation Simulator & Channel Estimator
**Simulador de propagación de ondas de radio de código abierto, para uso docente**

`OpenRPS` es un simulador modular de propagación de radio y estimador de canal multitrayecto, escrito en Python y publicado bajo licencia **GPL-2.0** (la misma licencia que el kernel de Linux). Incluye modelos empíricos de pérdida de trayecto y un motor de ray-tracing 3D, y está pensado como herramienta de apoyo para estudiantes.

Desde la versión 1.4.0 incluye una **interfaz gráfica** al estilo de las herramientas de planificación como RPS (`open-rps-gui`). Tiene un editor 2D del escenario con edificios 2.5D, muros, materiales y antenas, una vista 3D con los rayos, mapas de cobertura y la respuesta al impulso en cualquier punto. Se abre en el navegador y no necesita nada más que numpy (sección 4.E).

La versión 1.5.0 añade la **difracción en aristas con la Teoría Uniforme de la Difracción (UTD)**. Se aplica en las esquinas y los bordes de techo de los edificios y en los bordes de los muros, y con ella la sombra detrás de un obstáculo ya no queda vacía. También añade las **reflexiones dobles** muro–muro y muro–suelo (sección 1.F).

La versión 1.6.0 añade la **difracción doble**: el rayo puede pasar por dos aristas seguidas, como sobre un edificio alto y ancho o rodeando las esquinas de dos edificios. También añade los **rayos que combinan una reflexión con una difracción** (sección 1.G).

La versión 1.7.0 conecta la simulación con la realidad:

* **Plano de fondo:** se carga la imagen de un plano, se escala con dos puntos de distancia conocida y se dibujan los muros encima.
* **Mediciones:** se importan desde un CSV o se anotan en el mapa con lo que marca el teléfono.
* **Calibración:** se ve cuánto se aleja la predicción de lo medido, y OpenRPS ajusta las pérdidas de los materiales y la potencia de las antenas para que coincidan (sección 1.H).

---

## 1. Arquitectura y Modelos Físicos Implementados

El simulador proporciona tanto modelos empíricos macrocelulares como un motor determinista de **Ray-Tracing 3D** para la caracterización de canales multitrayecto (Multipath):

### A. Modelos de Pérdida de Propagación (Path Loss)
1. **Free-Space Path Loss (FSPL)**:
   $$PL_{\text{FSPL}}(d) = 20\log_{10}(d) + 20\log_{10}(f) + 20\log_{10}\left(\frac{4\pi}{c}\right)$$
2. **Two-Ray Ground Reflection**:
   Suma coherente del rayo directo y el rayo reflejado en tierra con coeficientes de reflexión de Fresnel complejos para polarización vertical y horizontal:
   $$\Gamma_{\parallel} = \frac{\epsilon_c \sin\psi - \sqrt{\epsilon_c - \cos^2\psi}}{\epsilon_c \sin\psi + \sqrt{\epsilon_c - \cos^2\psi}}, \quad \text{donde } \epsilon_c = \epsilon_r - j\frac{\sigma}{\omega \epsilon_0}$$
   Lejos del punto de quiebre ($d \gg 4 h_t h_r/\lambda$) tiende a $PL = 40\log_{10}(d) - 20\log_{10}(h_t h_r)$.
3. **Log-Distance Path Loss**:
   $$PL(d) = PL(d_0) + 10\, n \log_{10}\left(\frac{d}{d_0}\right)$$
4. **Okumura-Hata** (`OkumuraHata`):
   Modelo empírico estándar para celdas urbanas, suburbanas y rurales abiertas. Válido para 150–1500 MHz, antena base de 30–200 m, móvil de 1–10 m y distancias de 1–20 km.
5. **COST-231 Hata** (`Cost231Hata`):
   Extensión de Hata para 1500–2000 MHz (mismos rangos de alturas y distancia).

**Convenciones comunes a todos los modelos**

* `basic_path_loss(d, f, ...)` devuelve la pérdida entre antenas isotrópicas. `path_loss(d, f, tx_gain_dbi=0, rx_gain_dbi=0, ...)` devuelve la pérdida de enlace $PL - G_t - G_r$, de modo que $P_r = P_t - \texttt{path\_loss}$ en **todos** los modelos (en la versión 1.0 solo FSPL aplicaba las ganancias).
* `distance_m` es la distancia 3D entre antenas en FSPL y Log-Distance, y la distancia horizontal en Two-Ray, Hata y COST-231 (las alturas se pasan con `ht_m` / `hr_m`). El atributo de clase `uses_ground_distance` lo indica.
* Todos aceptan un `float` o un arreglo de numpy como distancia.
* Entradas inválidas (frecuencia ≤ 0, distancia negativa, alturas ≤ 0 en Hata, entorno desconocido) lanzan `ValueError`.
* Hata y COST-231 emiten un `ModelRangeWarning` cuando se evalúan fuera de su rango de validez; el resultado en ese caso es una extrapolación. Con `strict=True` (p. ej. `OkumuraHata("urban_medium", strict=True)`) lanzan `ValueError` en lugar de extrapolar, con un mensaje que enumera qué parámetro está fuera de rango.

---

### B. Motor de Multitrayecto y Respuesta al Impulso del Canal (CIR)
Cada rayo $k$ simulado en el espacio contiene:
* Amplitud compleja $a_k$ (voltaje/campo eléctrico)
* Retardo temporal de propagación $\tau_k = d_k / c$ (en nanosegundos)
* Ángulos de Salida (AoD) en la Estación Base: $\phi_{\text{BS}}$ (azimut) y $\theta_{\text{BS}}$ (elevación)
* Ángulos de Llegada (AoA) en la Estación Móvil: $\phi_{\text{MS}}$ y $\theta_{\text{MS}}$

A partir de cada rayo, se calculan las siguientes métricas:

1. **Potencia Total Recibida ($P_r$)**. Por defecto la suma es **incoherente** (potencia media local, sin desvanecimiento rápido); con `coherent=True` se suman las amplitudes complejas de los rayos de cada transmisor (interferencia constructiva/destructiva):
   $$P_r^{\text{incoh}} = \sum_{k=1}^K |a_k|^2, \qquad P_r^{\text{coh}} = \Big|\sum_{k=1}^K a_k\Big|^2 \quad [\text{mW}], \qquad P_{r,\text{dBm}} = 10\log_{10}(P_r)$$

2. **Pérdida de Trayecto (Path Loss)**:
   $$PL = P_{t,\text{dBm}} - P_{r,\text{dBm}} \quad [\text{dB}]$$

3. **Retardo Medio ($\overline{\tau}$)**, con $P_k = |a_k|^2$:
   $$\overline{\tau} = \frac{\sum_{k} P_k \tau_k}{\sum_{k} P_k} \quad [\text{ns}]$$
   `cir.mean_excess_delay()` devuelve este retardo medio **absoluto** ($\tau_k = d_k/c$); `cir.mean_excess_delay(relative_to_first=True)` le resta el retardo del primer rayo que llega ($\overline{\tau} - \min_k \tau_k$), que es el retardo medio *en exceso* en sentido estricto.

4. **Dispersión de Retardo RMS (RMS Delay Spread, $\sigma_\tau$)**:
   $$\sigma_\tau = \sqrt{\frac{\sum_{k} P_k \tau_k^2}{\sum_{k} P_k} - (\overline{\tau})^2} \quad [\text{ns}]$$

5. **Dispersión Angular RMS (Angular Spread, $\sigma_\phi$)**:
   Calculado mediante estadísticas circulares vectoriales en $[-180^\circ, 180^\circ]$ respecto al ángulo medio ponderado por potencia.

Las métricas 3–5 ponderan siempre cada rayo con su propia potencia $|a_k|^2$, también cuando la potencia total se calcula con `coherent=True`: describen cómo se reparte la energía entre los rayos, no su suma de banda estrecha.

---

### C. Opciones de la simulación (nuevas en 1.2.0)

Todas son opcionales y, con su valor por defecto, los resultados son los de la versión 1.1.0 (única excepción: los píxeles a menos de 1 m de una antena, ver *Campo lejano*).

**Suma coherente** — `run_raytracing_simulation(..., coherent=True)`, `cir.total_received_power_dbm(coherent=True)`, `RayField.total_power_linear(coherent=True)`. Suma las amplitudes complejas (la fase $-k\,d_k$ más la del coeficiente de reflexión) de los rayos de cada transmisor, lo que produce desvanecimiento rápido. Sin muros (rayo directo + suelo) el resultado coincide con `TwoRayGroundReflection` con el mismo suelo ($\epsilon_r$, $\sigma$), polarización vertical, alturas y ganancia: la diferencia máxima medida es de $2\cdot10^{-10}$ dB (redondeo; grillas de hasta 250 000 píxeles a 0.9, 2.4 y 3.5 GHz) y los tests exigen menos de $10^{-8}$ dB. Ambos usan la misma geometría (distancia horizontal más alturas en Two-Ray, coordenadas 3D en el trazador) y el mismo ángulo rasante. Diferencias de convención que quedan: (a) en un nulo perfecto el trazador devuelve $-\infty$ dBm y Two-Ray limita la pérdida a 200 dB; (b) con antenas coincidentes ($d = 0$) Two-Ray devuelve 0 dB y el trazador no tiene rayo directo (y el píxel queda en `NaN` por `min_distance_m`); (c) con antenas sectoriales el trazador aplica el diagrama a cada rayo según su propio ángulo de salida, mientras que el modo empírico Two-Ray usa para ambos rayos la dirección directa Tx→Rx. El patrón de desvanecimiento tiene un periodo espacial del orden de la longitud de onda: una grilla más gruesa solo lo *muestrea*.

```python
res = sim.run_raytracing_simulation((0, 1000), (0, 1000), resolution_m=5.0, coherent=True)
cir = sim.ray_tracer.trace_paths(sim.transmitters[0], (650.0, 450.0, 1.5))
print(cir.total_received_power_dbm(), cir.total_received_power_dbm(coherent=True))
```

**Antenas sectoriales** — `Transmitter(..., pattern="sector", azimuth_deg=0, beamwidth_az_deg=65, beamwidth_el_deg=65, downtilt_deg=0)`. Diagrama parabólico de 3GPP (TR 36.814 / TR 38.901), con `antenna_gain_dbi` como ganancia máxima (en el eje):
$$A_H(\varphi) = -\min\left[12\left(\frac{\varphi}{\varphi_{3\text{dB}}}\right)^2, 30\right], \quad A_V(\theta) = -\min\left[12\left(\frac{\theta - \theta_{\text{tilt}}}{\theta_{3\text{dB}}}\right)^2, 30\right], \quad A(\varphi,\theta) = -\min\left[-(A_H + A_V),\ 30\right] \ \text{dB}$$
$\varphi$ es el azimut medido desde `azimuth_deg` (0° = eje +x, 90° = eje +y) y $\theta - \theta_{\text{tilt}}$ la elevación medida desde el eje del haz; `downtilt_deg` positivo inclina el haz hacia el suelo. La atenuación es 0 dB en el eje, 3 dB a media anchura de haz y satura en 30 dB (lóbulo trasero). En ray tracing se aplica a **cada rayo** en su dirección de salida (AoD), en los dos modos (vectorizado y punto a punto); en los modos empíricos, en la dirección Tx→Rx. `tx.pattern_gain_db(phi, theta)` devuelve $A$ y `tx.gain_dbi(phi, theta)` la ganancia total. Por defecto `pattern="isotropic"`.

```python
sim.add_transmitter(Transmitter(tx_id=0, x=500, y=500, z=30, antenna_gain_dbi=17,
                                pattern="sector", azimuth_deg=90, downtilt_deg=6, beamwidth_el_deg=10))
```

**Campo lejano** — `min_distance_m` (por defecto 1 m) en `run_raytracing_simulation` y `run_empirical_simulation`. Los píxeles cuya distancia 3D a la antena de un transmisor es menor que el umbral se marcan como `NaN` para ese transmisor (potencia y pérdida no fiables: Friis no vale junto a la antena, y a $d \to 0$ daba valores absurdos como $P_r = P_t + G$). Ese transmisor no puede ser el mejor servidor del píxel: se elige entre los demás, y si no hay ninguno el píxel queda sin servidor (`best_server_id = -1`, potencia `NaN`). Los mapas, el CSV y las estadísticas del JSON tratan esos píxeles como "sin dato", sin avisos de numpy. `min_distance_m=None` (o `0`) lo desactiva. Se eligió 1 m porque es la distancia de referencia $d_0$ habitual de los modelos de pérdida, equivale a 8 λ a 2.4 GHz (3 λ a 900 MHz) y no altera ningún escenario con antenas en mástil (el píxel más cercano queda a $|h_t - h_r|$); por debajo de ~300 MHz (λ > 1 m) conviene subirlo a unas pocas longitudes de onda.

**Desvanecimiento por sombra** — `shadow_fading_sigma_db` (por defecto 0) y `seed` en las dos simulaciones. Con $\sigma > 0$ se suma a la pérdida de cada (transmisor, píxel) una muestra independiente $X \sim \mathcal{N}(0, \sigma^2)$ dB (log-normal en unidades lineales), generada con `numpy.random.default_rng(seed)`: misma semilla y misma grilla ⇒ mismo mapa, en cualquier modo (vectorizado, punto a punto o empírico).

```python
res = sim.run_empirical_simulation(OkumuraHata("urban_medium"), (0, 5000), (0, 5000), resolution_m=50.0,
                                   shadow_fading_sigma_db=8.0, seed=1)
```

**SINR** — `result.compute_sinr(noise_figure_db=7.0, bandwidth_hz=20e6, temperature_k=290.0)` calcula por píxel el SINR del mejor servidor en el enlace descendente:
$$\text{SINR} = \frac{P_{\text{mejor}}}{\sum_{i \ne \text{mejor},\ f_i = f_{\text{mejor}}} P_i + N}, \qquad N = k\,T\,B\,F \quad (-93.97\ \text{dBm con 20 MHz y } F = 7\ \text{dB})$$
Solo interfieren los transmisores de la **misma frecuencia** que el servidor (co-canal); los de otra frecuencia se ignoran por completo. El método devuelve el mapa y lo guarda en `result.sinr_db`, junto con `result.interference_plus_noise_dbm` ($I + N$ en dBm) y `result.sinr_settings`. A partir de ese momento `export_csv` añade al final las columnas `sinr_db` e `interference_plus_noise_dbm` y `export_json` un bloque `"sinr"`; si no se llama, los archivos son idénticos a los de 1.1.0. `result.sinr_map(...)` calcula lo mismo sin guardar nada y `thermal_noise_dbm(bandwidth_hz, noise_figure_db)` da $N$. Un píxel sin servidor, o con un interferente co-canal de potencia desconocida (a menos de `min_distance_m` de él), tiene SINR `NaN`.

```python
sinr = result.compute_sinr(noise_figure_db=7.0, bandwidth_hz=20e6)
MapVisualizer.export_svg(result, "sinr.svg", metric="sinr")      # o metric="interference" para I + N
result.export_csv("con_sinr.csv")                                 # incluye las dos columnas nuevas
```

---

### D. Muros que obstruyen (nuevo en 1.3.0)

Hasta la versión 1.2.0 un muro solo *añadía* un rayo reflejado: el rayo directo, el de suelo y los de otros muros lo atravesaban sin pérdida. Desde 1.3.0 cada muro puede, además, **atenuar o bloquear** los rayos que lo cruzan. Es opcional y se activa muro a muro:

```python
PlanarReflector(p0=[500, 0, 0], normal=[1, 0, 0], reflectivity=0.5, transmission_loss_db=12.0)
```

| `transmission_loss_db` | Efecto sobre un rayo que cruza el muro |
|---|---|
| `None` (por defecto) | Ninguno: el muro no obstruye (comportamiento de 1.2.0, resultados idénticos bit a bit). |
| $L > 0$ (finito) | Pérdida de penetración: la potencia del rayo baja $L$ dB por cada cruce (amplitud $\times 10^{-L/20}$). |
| `float("inf")` | Muro opaco: el rayo llega con potencia 0. |

`0` equivale a `None`; un valor negativo o `NaN` lanza `ValueError`.

**Qué se comprueba.** Cada tramo recto de cada rayo contra cada muro con `transmission_loss_db`: el rayo directo (Tx→Rx), el de suelo (Tx→punto de rebote y punto de rebote→Rx) y cada rayo reflejado en un muro (Tx→punto de reflexión y punto de reflexión→Rx). Se respeta `bounds` (parche finito; sin `bounds`, el plano entero). Las pérdidas de todos los muros cruzados **se suman en dB** (dos muros de 7 y 5 dB quitan 12 dB; un rayo reflejado que atraviesa un muro a la ida y a la vuelta lo paga dos veces). Un muro **nunca obstruye su propia reflexión**, pero sí es atravesado por los rayos directo, de suelo y reflejados en otros muros.

**Convenciones en los casos límite** (las mismas en las dos implementaciones):

* El tramo es un segmento *abierto*: cruza el muro solo si sus dos extremos quedan estrictamente a lados opuestos del plano, a más de 1 µm de él (`OBSTRUCTION_TOL_M`). Una antena o un píxel situados exactamente sobre el plano de un muro no quedan tapados por ese muro (ven ambos lados; sí los tapan los demás). Un rayo paralelo al muro, o contenido en su plano, no lo cruza.
* El parche es *cerrado*: un cruce justo por el borde de `bounds` cuenta (mismo margen de 1 nm que el punto de reflexión). Un rayo que pasa exactamente por la arista común de dos muros cruza los dos.
* Si un rayo rebota exactamente sobre el plano de *otro* muro (al pie de un muro, o en la esquina de un edificio), ese muro cuenta como cruzado una vez cuando queda *entre* los dos tramos del rayo (Tx y Rx a lados opuestos de su plano, y el muro del lado hacia el que se abre el ángulo del rayo): el rayo no se "cuela" por la arista hacia el interior de un edificio ni por debajo de un muro apoyado en el suelo, pero sí puede rozar por fuera el extremo de un muro.

**Qué pasa con un rayo bloqueado.** La obstrucción nunca cambia *qué* rayos existen, solo su amplitud: un rayo tapado por un muro opaco sigue en la lista (`cir.rays`, `RayField.valid`) con su retardo y sus ángulos, pero con amplitud 0. Cada rayo informa de lo que perdió en `MultipathRay.obstruction_loss_db` / `RayField.obstruction_loss_db` (0 = camino despejado, `inf` = bloqueado). Los rayos de potencia 0 no cuentan para la potencia total ni para la dispersión de retardo o angular. Si **todos** los rayos de un transmisor están bloqueados en un píxel, su potencia es `-inf` dBm y su pérdida `NaN`, y no puede ser el mejor servidor; si ningún transmisor llega, el píxel queda sin servidor (`best_server_id = -1`, potencia `-inf`, dispersión y SINR `NaN`) y los mapas lo dibujan como "sin datos". Un interferente bloqueado aporta 0 mW al SINR.

**Suma coherente.** Con `coherent=True` la obstrucción solo escala la amplitud de cada rayo; **no añade fase** (el muro no tiene espesor ni retardo propio).

**Un edificio** son cuatro muros finitos (más un quinto horizontal si se quiere techo):

```python
from open_rps import PropagationSimulator, Transmitter, PlanarReflector

def edificio(x0, x1, y0, y1, altura, perdida_db=float("inf"), reflectividad=0.5):
    """Las cuatro fachadas de un edificio rectangular, cada una un muro finito."""
    caras = [([x0, y0, 0], [1, 0, 0], (x0, x0, y0, y1, 0, altura)),    # oeste
             ([x1, y0, 0], [1, 0, 0], (x1, x1, y0, y1, 0, altura)),    # este
             ([x0, y0, 0], [0, 1, 0], (x0, x1, y0, y0, 0, altura)),    # sur
             ([x0, y1, 0], [0, 1, 0], (x0, x1, y1, y1, 0, altura))]    # norte
    return [PlanarReflector(p0=p0, normal=n, bounds=b, reflectivity=reflectividad,
                            transmission_loss_db=perdida_db) for p0, n, b in caras]

sim = PropagationSimulator()
sim.add_transmitter(Transmitter(tx_id=0, x=200, y=500, z=25, frequency_hz=2.4e9))
for muro in edificio(450, 550, 400, 600, altura=40):        # opaco: sombra detrás del edificio
    sim.add_reflector(muro)
res = sim.run_raytracing_simulation((0, 1000), (0, 1000), resolution_m=5.0)

cir = sim.ray_tracer.trace_paths(sim.transmitters[0], (800.0, 500.0, 1.5))    # receptor detrás del edificio
print([r.obstruction_loss_db for r in cir.rays], cir.total_received_power_dbm())   # [inf, inf, inf, inf] -inf
print(sim.ray_tracer.obstruction_loss_db([200, 500, 25], [800, 500, 1.5]))         # inf: no hay línea de vista
```

`PlanarReflector.crosses(p_inicio, p_fin)` dice si un segmento cruza un muro (geometría pura) y `RayTracer.obstruction_loss_db(p_inicio, p_fin)` suma las pérdidas de todos los muros que cruza un segmento (0 = hay línea de vista). Si el edificio es más bajo que la antena, la sombra tiene longitud finita (el rayo pasa por encima) y, como no hay techo, parte del interior queda iluminada desde arriba: añade un muro horizontal (`normal=[0, 0, 1]`, `bounds=(x0, x1, y0, y1, altura, altura)`) para cerrarlo.

**Lo que no hace** (ver *Limitaciones físicas*): por sí sola la obstrucción no difracta, así que el borde de la sombra es un salto brusco y, tras un obstáculo opaco, puede no llegar nada (desde 1.5.0 las aristas `DiffractionEdge` rellenan la sombra, sección 1.F); la pérdida es un número fijo por muro (no depende del ángulo, la frecuencia ni el espesor) e independiente de `reflectivity`/`eps_r`; y solo actúa en el modo de ray tracing: los modelos empíricos (FSPL, Two-Ray, Hata…) ignoran los muros.

**Coste.** Sin ningún muro con `transmission_loss_db` no se ejecuta nada nuevo. Con ellos, la ruta vectorizada sigue siendo numpy puro (sin bucles por píxel); en una grilla de 201 × 201 con 2 transmisores y 6 muros, activar la obstrucción en los 6 alarga la simulación en torno a un 30–40 % (medido con Python 3.14 y numpy 2.4).

---

### E. Edificios 2.5D, techos poligonales y geometría de los rayos (nuevo en 1.4.0)

**Techos poligonales.** `PlanarReflector(..., footprint=[(x0, y0), (x1, y1), ...])` limita un muro *horizontal* (normal según z) a un polígono simple, por ejemplo el techo de un edificio en L o con fachadas oblicuas. Hasta 1.3.0 `bounds` solo admitía una caja alineada con los ejes, así que el techo de un edificio que no fuera un rectángulo alineado no se podía representar. Un punto pertenece al techo si cae dentro del polígono, bordes incluidos, con el mismo criterio de parche cerrado y el mismo margen de 1 nm que `bounds`. Actúa en el punto de reflexión y en la obstrucción, en las dos implementaciones del trazador, que siguen dando los mismos resultados. Sin `bounds`, la caja se toma del polígono a la altura del plano.

```python
from open_rps import PlanarReflector
en_L = [(0, 0), (40, 0), (40, 15), (15, 15), (15, 40), (0, 40)]
techo = PlanarReflector(p0=[0, 0, 12], normal=[0, 0, 1], footprint=en_L, transmission_loss_db=float("inf"))
```

Las fachadas oblicuas no necesitan nada nuevo: un muro vertical sobre el segmento (x0, y0)→(x1, y1) con `bounds` igual a la caja del segmento es exacto, porque su plano corta esa caja justo a lo largo del segmento. `open_rps.project.vertical_wall(start, end, z_min, z_max, nombre)` lo construye.

**Geometría de cada rayo.** `RayTracer.trace_paths` rellena tres campos nuevos de `MultipathRay`, que sirven para dibujar los rayos. No forman parte de `==` ni del `repr`, así que los rayos se comparan y se imprimen como en 1.3.0:

* `kind`: `"direct"`, `"ground"` o `"wall"`.
* `path`: los vértices del rayo, Tx → [punto de rebote] → Rx, como tuplas (x, y, z).
* `reflector_index`: la posición del muro en `RayTracer.reflectors`, o −1.

```python
cir = sim.ray_tracer.trace_paths(sim.transmitters[0], (300.0, 190.0, 1.5))
for ray in cir.rays:
    print(ray.kind, ray.path, ray.obstruction_loss_db)
```

**Materiales.** `open_rps.project.Material` describe de qué está hecho un muro:

* **Reflexión:** coeficiente constante, o Fresnel con εr y σ = c·f_GHz^d de la Recomendación ITU-R P.2040 (Tabla 3).
* **Penetración:** puede ser nula, fija, lineal en la frecuencia (L = a + b·f_GHz, la forma de la Tabla 7.4.3-1 de 3GPP TR 38.901) u opaca.

La biblioteca por defecto (`default_materials()`) trae concreto, ladrillo, vidrio, vidrio IRR, madera, tablaroca, metal y un "muro ideal" que solo refleja. Como un muro lo comparten todos los transmisores, los valores que dependen de la frecuencia se evalúan a una sola **frecuencia de referencia**: la de `settings.reference_frequency_hz` o, si no se indica, la del primer transmisor activo.

### F. Difracción UTD y reflexiones dobles (nuevo en 1.5.0)

**Difracción en aristas.** Un `DiffractionEdge` es el borde recto de una cuña: la esquina de un edificio, el borde de su techo o el borde de un muro. Añade un rayo Tx → Q → Rx, con el punto Q sobre la arista elegido por la ley de Keller (ángulos iguales con la arista). Su amplitud es la del campo incidente en Q por el coeficiente de difracción $D$ de Kouyoumjian y Pathak (UTD) y por el factor de dispersión de una onda esférica:

$$E_d = E_i(Q)\, D\, \sqrt{\frac{s'}{s\,(s'+s)}}\, e^{-jks}, \qquad D = \frac{-e^{-j\pi/4}}{2n\sqrt{2\pi k}\,\sin\beta_0}\Big[T_1 + T_2 + R_0\,T_3 + R_n\,T_4\Big]$$

* **Términos y caras.** Los cuatro términos $T_i = \cot(\cdot)\, F(kLa^\pm)$ usan la función de transición $F$, que va calculada con integrales de Fresnel propias (sin scipy), y la cuña tiene un ángulo exterior $n\pi$. Las caras entran con los coeficientes heurísticos de Luebbers ($R_0$, $R_n$), que son de Fresnel con los $\varepsilon_r$ y $\sigma$ del material, o $\mp\rho$ con una reflectividad constante: $\rho = 1$ es el conductor perfecto.
* **Polarización.** En una arista vertical se usa el caso *soft* (campo E paralelo a la arista) y en una horizontal el caso *hard*. Es coherente con la polarización vertical del resto del motor.
* **Validación** (`tests/test_v1_5.py`):
  * Una pantalla delgada coincide con el filo de cuchillo exacto de Fresnel–Kirchhoff de ITU-R P.526 con menos de 0.35 dB de diferencia, en las dos polarizaciones, para ν de −1.5 a 3.2. Las funciones especiales coinciden con mpmath a 10⁻¹³.
  * En el borde de sombra de la esquina de un edificio opaco, el campo total (directo + difractado) es continuo: varía menos de 0.3 dB entre receptores separados 10 cm y vale −6 dB respecto al espacio libre justo en el borde.
  * El rayo difractado de una cuña conductora es recíproco.
* **Cuñas que dejan pasar parte del campo.** Con `shadow_loss_db` finito, los términos del borde de sombra incidente se escalan por $1 - 10^{-L/20}$, de modo que el campo transmitido más el difractado también es continuo.

```python
from open_rps import RayTracer, DiffractionEdge
tracer = RayTracer()
# Esquina de un edificio en (40, 40), de 30 m de alto: la cara 0 va hacia -x y su exterior hacia +y
tracer.add_edge(DiffractionEdge(start=[40, 40, 0], end=[40, 40, 30], face_direction=[-1, 0, 0],
                                face_normal=[0, 1, 0], wedge_n=1.5, polarization="soft", eps_r=5.24, sigma=0.09))
```

En un proyecto (`Settings.diffraction`, activado por defecto) las aristas salen solas de la geometría:

* **Edificios:** una arista vertical por cada esquina convexa de la planta (la esquina interior de una L no difracta hacia fuera) y, si el edificio tiene techo, una horizontal por cada lado del techo ($n = 1.5$).
* **Muros:** el borde superior, el inferior si el muro no llega al suelo, y los dos extremos verticales, que son semiplanos ($n = 2$). Un extremo que toca otro muro o un edificio no se considera arista libre.

Los muros y fachadas del escenario obstruyen los dos tramos de cada rayo difractado como a cualquier otro rayo.

**Reflexiones dobles.** `RayTracer(max_reflections=2)` (en un proyecto, `Settings.max_reflections = 2`) añade, por el método de las imágenes, un rayo Tx → P1 → P2 → Rx por cada par ordenado de planos distintos: muro–muro, suelo–muro y muro–suelo. Usa los coeficientes de reflexión de cada plano en su ángulo rasante, los mismos parches (`bounds` y `footprint`) y la misma obstrucción tramo a tramo. Con N muros son N·(N+1) rayos candidatos más, así que es bastante más lento: unos 9 s en lugar de 3.5 s en el ejemplo urbano a 4 m. Los pares en los que el parche del segundo muro queda entero detrás del primero se descartan sin calcular.

`MultipathRay.kind` vale `"diffraction"` o `"double"` en estos rayos, `MultipathRay.interactions` dice qué tocaron, por ejemplo `(("wall", 3), ("ground", -1))` o `(("edge", 7),)`, y `RayTracer.candidate_rays()` da el orden del eje K de `trace_points`. Sin aristas y con `max_reflections=1` (los valores por defecto de la librería), los resultados son idénticos bit a bit a los de 1.4.0.

### G. Difracción doble y rayos mixtos (nuevo en 1.6.0)

Detrás de un edificio alto y ancho puede no haber ninguna arista que vea a la vez al transmisor y al receptor. Con una sola difracción por rayo esa sombra se queda vacía, porque el rayo tendría que pasar por dos aristas seguidas: subir al techo por un borde y bajar por el otro, o rodear dos esquinas.

**Difracción doble.** `RayTracer(max_diffractions=2)` (en un proyecto, `Settings.max_diffractions = 2`) añade un rayo Tx → Q1 → Q2 → Rx por cada par ordenado de aristas distintas:

* **Puntos de difracción.** Q1 y Q2 hacen mínima la longitud del camino, que es la ley de Keller en las dos aristas. En aristas paralelas hay solución cerrada: el camino desplegado es una recta. En aristas que se cruzan sin cortarse se usa el método de Newton sobre las dos posiciones a lo largo de las aristas, que es un problema convexo. Cada receptor itera hasta el redondeo, así que el resultado no depende del resto de la grilla.
* **Amplitud.** Es la UTD en cascada: la onda difractada por la primera arista llega a la segunda como un rayo con su propia divergencia.

$$E = E_0\, D_1\, D_2\, \frac{e^{-jk(s_1+s_2+s_3)}}{\sqrt{s_1\, s_2\, s_3\, (s_1+s_2+s_3)}}, \qquad L_1 = \frac{s_1 s_2 \sin^2\beta_1}{s_1+s_2}, \quad L_2 = \frac{s_2 s_3 \sin^2\beta_2}{s_2+s_3}$$

* **Tramo rasante.** Cuando el tramo entre las dos aristas va rasante a una cara, como entre dos bordes del mismo techo, $D_2$ se divide entre dos (Kouyoumjian y Pathak). Las caras que toca ese tramo usan además el coeficiente del conductor perfecto (±1). El de Luebbers vale −1 en incidencia rasante para cualquier cara con pérdidas y anularía el campo sobre el techo.
* **Dos esquinas de la misma fachada.** Las aristas verticales son *soft*, y el campo tangente a una cara opaca es cero. En el tramo que roza la fachada el coeficiente UTD se anula exactamente, así que ese rayo no se genera, en lugar de quedar como un residuo de redondeo de −700 dBm. Detrás de una torre opaca la señal entra por los bordes del techo; rodear dos esquinas sí aporta cuando son de edificios distintos.

**Rayos mixtos.** `RayTracer(mixed_interactions=True)` (en un proyecto, `Settings.mixed_interactions = True`) añade dos familias de rayos, en muros y en el suelo:

* **Reflexión → difracción:** Tx → P → Q → Rx. La arista ve la imagen del transmisor en el muro.
* **Difracción → reflexión:** Tx → Q → P → Rx. La arista se observa desde la imagen del receptor.

Cada uno lleva el coeficiente de reflexión del plano en su ángulo rasante (Fresnel o constante) y el coeficiente UTD con las distancias del camino desplegado. El parche del muro (`bounds`, `footprint`) y la obstrucción tramo a tramo funcionan como en los demás rayos.

**Solo donde hacen falta.** Un rayo de segundo orden con una difracción solo se calcula donde el receptor queda en la sombra de la arista o en su zona de transición. Esa zona es un margen de $\sqrt{20/(kL)}$ rad alrededor de los bordes de sombra y de reflexión, donde la función de transición se aparta de 1. En la difracción doble se pide en las dos aristas. Fuera de la zona, el rayo que toca una arista menos ya llega con el campo completo y el de segundo orden sería una corrección pequeña contada dos veces. El criterio es simétrico en los dos ángulos, así que los rayos siguen siendo recíprocos.

**Validación** (`tests/test_v1_6.py`):

* **Dos pantallas** a 200 m una de otra. Ahí solo llega el rayo doblemente difractado, y se compara con un cálculo numérico independiente de Fresnel–Kirchhoff: el campo del primer filo se integra sobre la abertura del segundo. La UTD en cascada queda a menos de 0.85 dB en cada polarización. Soft y hard caen a los dos lados del resultado de Kirchhoff, que no depende de la polarización, y su promedio queda a menos de 0.18 dB.
* **Geometría:** la ley de Keller se cumple en las dos aristas con Newton, y la solución cerrada de aristas paralelas coincide con Newton.
* **Reciprocidad:** al intercambiar Tx y Rx salen los mismos rayos en orden inverso, con la misma amplitud a 10⁻¹². Se comprueba para reflexión ↔ difracción y para la difracción doble, con cuñas conductoras perfectas.
* **Torre metálica** de 40 m de alto y 20 × 40 m de planta. Con una difracción, detrás de ella quedan 165 píxeles sin datos. Con dos, todos reciben señal, a −87 dBm de mediana, y el interior sigue vacío.
* **Motor:** el trazado vectorizado y el punto a punto dan lo mismo, y apagar los filtros de la sección siguiente no cambia ningún número.

**Rendimiento.** Con E aristas y N planos (muros y suelo) hay E·(E−1) rayos de difracción doble y 2·N·E mixtos candidatos. Antes de calcular nada se descartan los que no pueden existir:

* **Pruebas exactas:** el transmisor dentro de una cuña, aristas que se tocan o que quedan dentro de la cuña de la otra, y la imagen fuera del parche del muro.
* **Visibilidad muestreada** a través de los muros opacos.
* **Condiciones necesarias por par y por receptor:** que el punto de Keller pueda caer sobre la arista, y que algún ángulo posible caiga en la zona de transición, con el menor $L$ posible.

Un test comprueba que estos filtros no cambian ningún rayo. En el ejemplo urbano a 4 m (151 × 101 píxeles, 2 transmisores, 62 aristas) la simulación tarda:

| Opciones | Tiempo |
|---|---|
| una difracción por rayo (1.5.0) | 2.1 s |
| difracción doble | 7.6 s |
| rayos mixtos | 15 s |
| las dos | 25 s |

En `MultipathRay.kind` estos rayos valen `"double-diffraction"`, `"reflection-diffraction"` y `"diffraction-reflection"`. `MultipathRay.interactions` da el orden, por ejemplo `(("wall", 3), ("edge", 7))`. `trace_points(..., compact=True)` devuelve solo las filas de los rayos que pueden existir para ese transmisor, con sus etiquetas en `RayField.labels`. La simulación de grilla usa ese modo.

### H. Plano de fondo, mediciones y calibración (nuevo en 1.7.0)

Un modelo de propagación solo es tan bueno como sus parámetros. La pérdida real de una pared de tablaroca o la potencia que de verdad emite un punto de acceso casi nunca se conocen de antemano. Esta versión permite medir, comparar y ajustar.

**Plano de fondo.** En la interfaz, *Archivo → Plano de fondo* carga una imagen en PNG, JPEG, WebP o GIF:

* **Escala y posición.** Se marcan dos puntos de distancia conocida (una pared, una cota) y se escribe esa distancia. En el mismo diálogo se pueden dar las coordenadas del primer punto (por ejemplo, 0 y 0 en la esquina que se usa de origen), así que escalar y colocar es un solo paso.
* **Dibujo.** El plano se dibuja en modo *multiply*: lo blanco deja ver el mapa de cobertura y las líneas quedan oscuras. Con el plano seleccionado se puede arrastrar, y en 3D aparece sobre el suelo.
* **Archivo.** La imagen se guarda dentro del `.orps.json` como *data URL* (`Project.floor_plan`, clase `FloorPlan`), así que el archivo es autosuficiente. Las imágenes de más de 4096 px de lado se reducen al cargarlas.
* **Simulación.** El plano solo se dibuja: la simulación no lo usa y moverlo no deja los resultados desactualizados.

**Mediciones.** Cada medición (`open_rps.measurements.Measurement`) es un punto (x, y y, opcionalmente, z), la potencia medida en dBm y, opcionalmente, el transmisor al que corresponde. Sin transmisor se compara con el más fuerte. Hay dos formas de cargarlas:

* **Desde un CSV**, con *Archivo → Importar mediciones* o `read_measurements_csv`:
  * **Columnas** (en la primera fila): `x`, `y` y la potencia, que se reconoce como `rssi`, `dbm`, `potencia`, `señal` o `signal`.
  * **Columnas opcionales:** `z` o `altura`, `tx` o `ap` (id o nombre del transmisor) y una etiqueta.
  * **Formato.** Los nombres se reconocen sin acentos ni mayúsculas, y una unidad entre paréntesis se ignora ("Señal (dBm)"). Se aceptan coma, punto y coma o tabulador como separador, coma decimal con los dos últimos, y líneas de comentario con `#`.
* **Con la herramienta Medición (N):** clic donde mediste y escribe lo que marcó el teléfono o el analizador.

**Comparación.** La predicción se evalúa exactamente en cada punto, con la misma cuenta que los mapas, sin interpolar la grilla ni sumar sombra aleatoria:

* **El error** es predicho − medido; si es positivo, el modelo es optimista.
* **Estadísticas:** sesgo, desviación, RMSE, error máximo y correlación.
* **Puntos que quedan fuera:** sin señal predicha, a menos de la distancia mínima de una antena, o con un transmisor desactivado. Se cuentan aparte.
* **En la interfaz** se compara después de cada ejecución. Cada punto muestra su error en el mapa y la pestaña *Mediciones* tiene la gráfica de predicho contra medido y los puntos con más error.

**Calibración** (`calibrate`, el botón *Calibrar* o el comando `open-rps-calibrate`). Ajusta por mínimos cuadrados sobre el error en dB, con el método de Levenberg–Marquardt y límites:

* **Pérdida por cruce de cada material** (ray tracing). Es el valor a la frecuencia de referencia, entre 0 y 60 dB. Un material con ley lineal en la frecuencia conserva su pendiente.
* **Potencia de los transmisores.** Puede ser un ajuste por transmisor, si las mediciones dicen de cuál son, o uno común: este absorbe, por ejemplo, la ganancia desconocida del equipo de medición.
* **El exponente de pérdida**, en el modo log-distancia.

Cada paso vuelve a trazar los rayos en los puntos medidos: las derivadas son numéricas y salen del mismo motor, sin aproximaciones. No se ajusta un parámetro del que dependan menos de 3 puntos medidos, porque esos puntos no lo distinguen del error de medición. Pasa, por ejemplo, con una pared que ningún rayo medido cruza; el resultado lo dice. Cada valor trae su incertidumbre (1σ), calculada con la dispersión que queda y la curvatura del ajuste. El proyecto original no cambia: `CalibrationResult.project` es una copia calibrada, y en la interfaz *Aplicar al proyecto* se puede deshacer con Ctrl+Z.

**Validación** (`tests/test_v1_7.py`):

* **Sin ruido** se recuperan los valores con los que se generaron las mediciones:
  * tablaroca a menos de 0.05 dB, concreto a menos de 0.3 dB y potencia a menos de 0.05 dB;
  * el exponente de log-distancia (3.4) a menos de 0.01.
* **Con 3 dB de ruido**, en el escenario urbano con difracción, los cuatro valores ajustados quedan a menos de 1.4 σ de los verdaderos (el test acepta hasta 3.5 σ).
* **Equivalencia con el mapa:** la predicción en los puntos coincide con el mapa en todos los modos.

`examples/mediciones_oficina.csv` trae 56 mediciones **sintéticas** para practicar, con `examples/plano_oficina.png` (un plano generado). Las produjo OpenRPS con una oficina "real" que tiene la tablaroca a 5.5 dB, el concreto a 18 dB y el AP a 17 dBm, en vez de los 3 dB, 25.8 dB y 20 dBm del ejemplo, más 3 dB de ruido. La calibración las recupera en 1.1 s:

| Parámetro | Ejemplo (antes) | Calibrado | Verdad |
|---|---|---|---|
| Tablaroca | 3.00 dB | 5.85 ± 0.56 dB | 5.5 dB |
| Concreto | 25.8 dB | 17.9 ± 1.3 dB | 18 dB |
| Potencia del AP | 20 dBm | 17.6 ± 0.7 dBm | 17 dBm |
| RMSE | 6.3 dB | 3.0 dB | (ruido: 3 dB) |

El vidrio queda igual: ningún punto medido depende de él.

**Lo que conviene saber al medir:**

* **Promedia.** El RSSI de un teléfono oscila varios dB de una lectura a otra (desvanecimiento rápido, el cuerpo de quien mide). La predicción, en cambio, es la potencia media local: promedia varias lecturas en cada punto.
* **Los valores calibrados son los que mejor explican tus mediciones con este modelo, no la verdad física.** Si el escenario no tiene muebles, gente o pisos, sus pérdidas acaban dentro de las paredes. Si faltan muros en el dibujo, otros los compensan.
* **Con pocos puntos el ajuste se acomoda al ruido.** Mide en zonas donde cada material importe: detrás de las paredes, no solo en la sala del AP. Revisa las incertidumbres y comprueba el resultado con mediciones que no usaste para calibrar.

---

## 2. Estructura del Proyecto

```
open_rps/
├── pyproject.toml        # Metadatos y empaquetado (comando `open-rps`)
├── LICENSE               # GNU GPL v2.0
├── requirements.txt      # Única dependencia: numpy
├── src/open_rps/
│   ├── __init__.py       # Exportaciones del paquete y __version__
│   ├── models.py         # Modelos analíticos (FSPL, Two-Ray, Log-Distance, Hata, COST-231 Hata)
│   ├── multipath.py      # MultipathRay y ChannelImpulseResponse (CIR, Delay/Angular Spread)
│   ├── raytracer.py      # Motor 3D de trazado de rayos (directo, tierra, muros, aristas UTD, dobles y mixtos)
│   ├── diffraction.py    # Integrales de Fresnel, función de transición y coeficiente UTD (nuevo en 1.5.0)
│   ├── simulation.py     # Orquestador de grillas 2D (Coverage Maps, Best Server, CSV/JSON)
│   ├── visualizer.py     # Mapas ASCII para terminal y SVG vectoriales
│   ├── cli.py            # Interfaz de línea de comandos
│   ├── project.py        # Proyectos .orps.json: escenario, edificios 2.5D, materiales (nuevo en 1.4.0)
│   ├── measurements.py   # Mediciones (CSV), comparación con la predicción y calibración (nuevo en 1.7.0)
│   ├── calibrate_cli.py  # Comando open-rps-calibrate (nuevo en 1.7.0)
│   └── gui/              # Interfaz gráfica (nuevo en 1.4.0)
│       ├── server.py     # Servidor local (http.server) y API JSON
│       └── static/       # Página: HTML, CSS, JavaScript y three.js (vendor/)
├── examples/
│   ├── example_sim.py           # Demostración con 2 transmisores y reflector
│   ├── practicas_openrps.ipynb  # Cuaderno con 5 prácticas para alumnos
│   ├── proyecto_urbano.orps.json   # Microcelda urbana (ábrelo en la interfaz gráfica)
│   ├── proyecto_oficina.orps.json  # Planta de oficinas con Wi-Fi a 5.2 GHz
│   ├── plano_oficina.png           # Plano de esa oficina, para practicar el plano de fondo (1.7.0)
│   └── mediciones_oficina.csv      # 56 mediciones sintéticas de esa oficina, para practicar la calibración
├── tests/                # Tests unitarios (unittest de la librería estándar)
└── output/               # Archivos de ejemplo generados (CSV, JSON, mapas SVG)
```

---

## 3. Instalación

Requiere Python 3.10 o superior y `numpy` (se instala solo). Desde la carpeta del proyecto:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install .            # o, para desarrollo: pip install -e .
```

Para instalarlo directamente desde GitHub, sin clonar el repositorio:

```bash
pip install git+https://github.com/fabianluna08/openRPS.git
# con las dependencias del cuaderno de prácticas (matplotlib y Jupyter):
pip install "open-rps[notebook] @ git+https://github.com/fabianluna08/openRPS.git"
```

Cualquiera de las opciones instala la librería (`import open_rps`) y los comandos `open-rps`, `open-rps-gui` y `open-rps-calibrate`. Probado con Python 3.14 y numpy 2.4; no se ha verificado con versiones anteriores.

Para generar un paquete distribuible (wheel) que se pueda compartir con los alumnos:

```bash
pip wheel . --no-deps -w dist/      # crea dist/open_rps-1.7.0-py3-none-any.whl
pip install dist/open_rps-1.7.0-py3-none-any.whl
```

---

## 4. Guía de Uso

### A. Ejecutar el script de demostración:
```bash
python3 examples/example_sim.py
```
Este script:
1. Instancia dos transmisores (Tx0 en 200m,250m y Tx1 en 800m,750m) en 2.4 GHz.
2. Agrega una fachada reflectante en el plano x = 500 m.
3. Computa la respuesta al impulso en una grilla de 1000m x 1000m.
4. Muestra mapas de calor en la consola mediante arte ASCII.
5. Genera en `./output/` (relativo al directorio desde el que lo ejecutas):
   - `simulation_results.csv`: Tabla completa con coordenadas X, Y, Tx ganador, potencia, pérdida y dispersión.
   - `simulation_summary.json`: Estadísticas clave.
   - `coverage_rx_power.svg`: Mapa de cobertura vectorial a color.
   - `coverage_delay_spread.svg`: Mapa de retardo multitrayecto.
   - `coverage_best_server.svg`: Mapa de cobertura por celda/servidor.

### B. Uso mediante CLI:
```bash
# Simulación con Ray-Tracing (multitrayecto), con y sin el muro reflector:
open-rps --mode raytracing --size 1200 --res 20 --freq 2.4e9
open-rps --mode raytracing --reflector off

# Okumura-Hata urbano (900 MHz, antenas a 30 m: dentro del rango del modelo salvo la distancia < 1 km):
open-rps --mode hata --size 2000 --res 50 --freq 900e6 --tx-height 30

# Two-Ray (tierra reflectante) y espacio libre:
open-rps --mode tworay --size 1000 --res 25
open-rps --mode fspl

# Novedades 1.2.0: suma coherente, antenas sectoriales, sombra log-normal reproducible y mapa de SINR
open-rps --mode raytracing --res 5 --coherent --outdir out_coherente
open-rps --mode raytracing --sector --downtilt 4 --sinr --outdir out_sector
open-rps --mode hata --freq 900e6 --tx-height 30 --size 5000 --res 100 --shadow-sigma 8 --seed 1 --sinr
open-rps --mode hata --strict        # termina con error (código 1) e indica qué está fuera del rango de COST-231 Hata

# Novedades 1.3.0: muros que obstruyen (pérdida de penetración en dB, o "inf" para un muro opaco)
open-rps --mode raytracing --wall-loss 15 --outdir out_muro_15dB
open-rps --mode raytracing --building --wall-loss inf --res 10 --outdir out_edificio   # edificio opaco: sombra

# Ayuda:
open-rps --help          # o: python3 -m open_rps.cli --help
```

| Opción | Valores | Por defecto | Descripción |
|---|---|---|---|
| `--mode` | `raytracing`, `tworay`, `hata`, `fspl` | `raytracing` | Modelo de propagación. `hata` usa Okumura-Hata hasta 1500 MHz y COST-231 Hata por encima. |
| `--size` | metros | `1000` | Lado del área cuadrada. |
| `--res` | metros | `25` | Paso de la grilla. |
| `--freq` | Hz | `2.4e9` | Frecuencia portadora. |
| `--power` | dBm | `43` | Potencia de Tx0 (Tx1 emite 3 dB menos). |
| `--tx-height` | metros | `25` | Altura de las dos estaciones base. |
| `--reflector` | `on`, `off` | `on` | Muro reflector en x = size/2, o el edificio de `--building` (solo afecta a `raytracing`). |
| `--wall-loss` | dB, o `inf` | sin obstrucción | Pérdida de penetración de los muros de demostración: cada rayo que cruza un muro pierde esos dB; `inf` = muros opacos. Si se omite, los muros solo reflejan y no tapan nada (como en 1.2.0). Solo afecta a `raytracing`. |
| `--building` | (sin valor) | desactivado | Sustituye el muro infinito por un edificio rectangular de 4 muros finitos (x de 0.45 a 0.60 × size, y de 0.35 a 0.50 × size, el doble de alto que las estaciones). Con `--wall-loss inf` proyecta sombra; sin `--wall-loss` solo refleja. |
| `--coherent` | (sin valor) | desactivado | Suma coherente de los rayos: desvanecimiento rápido (solo afecta a `raytracing`). |
| `--sector` | (sin valor) | desactivado | Antenas sectoriales 3GPP de 65° apuntando al centro del área (Tx0 a 45°, Tx1 a −135°) en lugar de isotrópicas. |
| `--downtilt` | grados | `0` | Inclinación hacia abajo de las antenas sectoriales (solo con `--sector`). |
| `--shadow-sigma` | dB | `0` | Desviación típica del desvanecimiento por sombra log-normal (0 = sin sombra). |
| `--seed` | entero | aleatoria | Semilla de la sombra, para repetir exactamente un mapa. |
| `--min-distance` | metros | `1` | Los píxeles más cercanos que esto a una antena quedan sin dato; `0` lo desactiva. |
| `--strict` | (sin valor) | desactivado | Con `--mode hata`: termina con error (código de salida 1) en lugar de extrapolar fuera del rango de validez. En el escenario de demostración las estaciones están dentro del área, así que siempre hay píxeles a menos de 1 km y `--strict` falla siempre: sirve para ver qué parámetros están fuera de rango. |
| `--sinr` | (sin valor) | desactivado | Calcula el SINR del mejor servidor: columnas extra en el CSV, bloque `"sinr"` en el JSON y `coverage_sinr.svg`. |
| `--noise-figure` | dB | `7` | Figura de ruido del receptor (para `--sinr`). |
| `--bandwidth` | Hz | `20e6` | Ancho de banda de ruido (para `--sinr`). |
| `--theme` | `light`, `dark` | `light` | Tema de color de los SVG. |
| `--outdir` | ruta | `./output` | Carpeta de salida, **relativa al directorio actual**. |

El CLI escribe en `--outdir` los mismos nombres de archivo que el script de demostración: `simulation_results.csv`, `simulation_summary.json`, `coverage_rx_power.svg`, `coverage_best_server.svg`, solo en modo `raytracing` `coverage_delay_spread.svg` y, solo con `--sinr`, `coverage_sinr.svg`. Cada ejecución sobrescribe la anterior: usa un `--outdir` distinto por escenario si quieres conservarlas. En los modos empíricos las columnas de dispersión de retardo y angular del CSV valen `nan` (y `null` en el JSON), porque esos modelos no describen el multitrayecto.

Con los valores por defecto (2.4 GHz, antenas a 25 m, área de 1 km) el modo `hata` queda fuera del rango de validez de ambos modelos de Hata; el CLI lo indica con una línea `[!] Warning: ...`.

Con `--wall-loss` o `--building` el CLI añade una línea `[*] Walls: ...` que describe el escenario. Con `--building --wall-loss inf` los píxeles del interior del edificio no reciben ningún rayo: aparecen como "sin datos" en los SVG, con `best_tx_id = -1` y potencia `-inf` en el CSV (en el mapa ASCII son un hueco en blanco). Con el muro infinito y `--wall-loss inf` cada estación solo cubre su lado del muro.

### C. Uso como librería de Python:
```python
from open_rps import PropagationSimulator, Transmitter, PlanarReflector, MapVisualizer

sim = PropagationSimulator()
sim.add_transmitter(Transmitter(tx_id=0, x=500, y=500, z=30, tx_power_dbm=43.0, frequency_hz=1.8e9))

# Muro en el plano x = 700 m, limitado a un tramo de 200 m de largo y 20 m de alto:
sim.add_reflector(PlanarReflector(p0=[700, 0, 0], normal=[1, 0, 0], reflectivity=0.6,
                                  bounds=(699, 701, 400, 600, 0, 20)))

# Ejecutar sobre una grilla de 1 km x 1 km con resolución de 20m:
result = sim.run_raytracing_simulation(x_range=(0, 1000), y_range=(0, 1000), resolution_m=20.0)

print(f"Potencia máxima: {result.max_rx_power_dbm.max():.2f} dBm")
result.export_csv("mi_simulacion.csv")
result.export_json("mi_simulacion.json")
MapVisualizer.export_svg(result, "mi_mapa.svg", metric="rx_power")   # o "delay_spread", "best_server", "sinr", "interference"

# Respuesta al impulso (CIR) en un punto concreto:
cir = sim.ray_tracer.trace_paths(sim.transmitters[0], (650.0, 450.0, 1.5))
print(len(cir.rays), cir.rms_delay_spread(), cir.angular_spread(plane="phi", station="bs"))
print(cir.mean_excess_delay(), cir.mean_excess_delay(relative_to_first=True))   # absoluto / en exceso
```

Las opciones nuevas de la versión 1.2.0 (`coherent`, antenas `pattern="sector"`, `min_distance_m`, `shadow_fading_sigma_db`/`seed`, `compute_sinr`) se describen, con ejemplos, en la sección 1.C, y los muros que obstruyen (`transmission_loss_db`, nuevo en 1.3.0), en la sección 1.D.

Notas sobre `SimulationResult`:

* El primer eje de `rx_power_dbm` y `path_loss_db` sigue el **orden en que se añadieron** los transmisores, no su `tx_id`. `tx_ids` da el identificador de cada posición y `best_server_index` la posición del mejor servidor por píxel (`best_server_id` contiene los `tx_id`; `-1` = sin servidor).
* `path_loss_db` es $P_t - P_r$, es decir, incluye la ganancia de antena del transmisor (y su diagrama, si es sectorial) y la sombra.
* Valores sin dato: `-inf` dBm significa que no llega potencia (p. ej. todos los rayos de ese transmisor están bloqueados por muros opacos); `NaN`, que el valor no es fiable o no existe (píxel a menos de `min_distance_m` de la antena, métrica que el modelo no proporciona, píxel sin servidor).
* `tx_frequencies_hz` guarda la frecuencia de cada transmisor (la usa el SINR); `sinr_db`, `interference_plus_noise_dbm` y `sinr_settings` valen `None` hasta que se llama a `compute_sinr()`.
* `PlanarReflector` acepta `bounds=(xmin, xmax, ymin, ymax, zmin, zmax)` para limitar el parche reflectante (sin `bounds` el plano es infinito), opcionalmente `eps_r`/`sigma` para usar el coeficiente de Fresnel dependiente del ángulo en lugar de `reflectivity`, y `transmission_loss_db` para que el muro atenúe o bloquee los rayos que lo cruzan (sección 1.D; por defecto no obstruye).
* `run_raytracing_simulation` evalúa toda la grilla con numpy. `vectorized=False` ejecuta la implementación de referencia punto a punto (mismos resultados, decenas de veces más lenta).

### D. Cuaderno de prácticas (Jupyter)

`examples/practicas_openrps.ipynb` contiene cinco prácticas guiadas, con preguntas y ejercicios "Tu turno": (1) pérdida de trayecto vs. distancia entre modelos, (2) cobertura multi-celda y efecto de un muro reflector, (3) respuesta al impulso del canal, dispersión de retardo y angular, (4) suma coherente (desvanecimiento rápido y comparación con Two-Ray) y mapa de SINR con antenas sectoriales frente a isotrópicas, (5) obstrucción: sombra de un edificio opaco y pérdida de penetración. Necesita matplotlib y Jupyter, que se instalan con el extra `notebook`:

```bash
pip install -e ".[notebook]"
jupyter lab examples/practicas_openrps.ipynb
```

### E. Interfaz gráfica (nuevo en 1.4.0)

```bash
open-rps-gui                                   # abre el navegador con el ejemplo urbano
open-rps-gui examples/proyecto_oficina.orps.json
python3 -m open_rps.gui --port 0 --no-browser  # puerto libre, sin abrir el navegador
```

El comando levanta un servidor local en `http://127.0.0.1:8765/`, que solo es accesible desde tu equipo, y abre la interfaz en el navegador. Para cerrarla, pulsa Ctrl+C en la consola. La página la sirve el propio paquete, three.js va incluido y funciona sin internet. Necesita un navegador moderno con WebGL para la vista 3D. Su distribución sigue la de una herramienta de planificación:

* **Árbol del proyecto** (izquierda): área de predicción, configuración, transmisores, edificios, muros, plano de fondo, mediciones y materiales.
* **Propiedades** (derecha): el formulario del elemento seleccionado. Incluye la posición, potencia, frecuencia y diagrama de cada antena, la altura y el material de cada edificio, los vértices de su planta, y los parámetros ITU-R/3GPP de cada material.
* **Vista 2D:** editor a escala con cuadrícula en metros. Puedes colocar transmisores (T), dibujar edificios rectangulares (R) o poligonales (P), muros (W), mover objetos y vértices (V), medir (M) y desplazar la vista con la rueda o el botón derecho. Se puede deshacer y rehacer (Ctrl+Z / Ctrl+Y).
* **Vista 3D** (2, o 3 para verla junto a la 2D): edificios extruidos, muros, antenas con el cono del sector y el mapa de cobertura en el plano del receptor, opcionalmente en relieve. Se gira con el ratón.
* **Mapas:** potencia recibida, pérdida de trayecto, mejor servidor, SINR, interferencia + ruido, dispersión de retardo y dispersión angular. La potencia y la pérdida se pueden ver por transmisor. La escala es automática o manual, con la misma rampa de color y la misma paleta Okabe-Ito que los SVG. Al pasar el cursor se ve el valor de cada transmisor.
* **Receptor de prueba** (S): haz clic o arrastra en 2D o 3D. Se dibujan todos los rayos, incluidos los bloqueados, en discontinua. El panel inferior muestra la potencia y las dispersiones por transmisor, el perfil potencia–retardo (respuesta al impulso) y la tabla de rayos con sus ángulos y pérdidas por muros. Se recalcula al mover cualquier objeto.
* **Estadísticas:** porcentaje del área cubierta sobre un umbral de potencia, área con SINR ≥ 0 dB, percentiles e histograma de la métrica, y reparto del área entre servidores.
* **Plano de fondo y mediciones** (1.7.0, sección 1.H): *Archivo → Plano de fondo* y *Archivo → Importar mediciones*. La herramienta Medición (N) sirve para anotar puntos. La pestaña *Mediciones* compara la predicción con lo medido y calibra el modelo.
* **Archivo:** abrir y guardar proyectos `.orps.json`; el último proyecto se guarda además en el navegador. Hay ejemplos incluidos y se puede exportar el resultado como CSV, JSON o SVG (los mismos archivos que el CLI) o la vista como PNG.

Cada edificio se convierte en el motor en un muro vertical por arista y un techo poligonal (sección 1.E). Con el material por defecto (concreto) las fachadas atenúan 14.6 dB a 2.4 GHz. Con "Metal" son opacas y con "Muro ideal" solo reflejan. Los modelos empíricos ignoran edificios y muros, como en la librería.

### F. Proyectos desde Python (nuevo en 1.4.0)

Un proyecto `.orps.json` es el escenario completo, y la interfaz y los scripts usan el mismo archivo:

```python
from open_rps.project import Project
project = Project.load("examples/proyecto_urbano.orps.json")
project.settings.mode = "raytracing"
project.area.resolution_m = 2.0
result = project.run()                      # SimulationResult, como run_raytracing_simulation
result.export_csv("urbano.csv")
probe = project.probe((300.0, 190.0))       # rayos y métricas en un punto (sin sombra log-normal)
sim = project.build_simulator()             # PropagationSimulator con todos los muros
project.save("copia.orps.json")
```

### G. Mediciones y calibración (nuevo en 1.7.0)

```bash
open-rps-calibrate examples/proyecto_oficina.orps.json examples/mediciones_oficina.csv -o oficina_calibrada.orps.json
open-rps-calibrate oficina_calibrada.orps.json --compare-only --errors errores.csv   # solo comparar
open-rps-calibrate proyecto.orps.json medidas.csv --materials concreto,tablaroca --tx-power common
```

```python
from open_rps.project import Project
from open_rps.measurements import read_csv, compare, calibrate

project = Project.load("examples/proyecto_oficina.orps.json")
project.measurements = read_csv("examples/mediciones_oficina.csv", project.transmitters)
print(compare(project).stats)                 # sesgo, desviación, RMSE, error máximo, r
result = calibrate(project)                   # o materials=["tablaroca"], tx_power="common"
for p in result.parameters:
    print(p.name, p.before, "->", p.after, "±", p.uncertainty)
result.project.save("oficina_calibrada.orps.json")
```

Un proyecto con plano de fondo o con mediciones se guarda como versión 2 del formato. Sin ellos sigue siendo la versión 1, y lo abren OpenRPS 1.4 a 1.6.

`Project.from_dict` valida todo el documento. Rechaza campos desconocidos, polígonos que se cruzan, materiales inexistentes, `tx_id` repetidos y grillas de más de 4 millones de píxeles, y en cada caso lanza un `ProjectError` (subclase de `ValueError`) que dice qué elemento y qué campo fallan.

---

## 5. Tests

Los tests usan `unittest` de la librería estándar (no hace falta pytest). Con el paquete instalado (o con `PYTHONPATH=src`), desde la raíz del proyecto:

```bash
python3 -m unittest discover -s tests
```

Cubren valores de referencia de FSPL, Hata y COST-231, la asíntota $d^4$ del modelo Two-Ray, la dispersión de retardo y angular (incluido el cruce por ±180°), la geometría de los rayos de suelo y de muro, la equivalencia entre el trazado vectorizado y el de referencia punto a punto, simulaciones completas, las exportaciones CSV/JSON/SVG (con `tx_id` no consecutivos y mapas sin datos) y el CLI. `tests/test_v1_2.py` cubre lo añadido en 1.2.0: regresión numérica frente a 1.1.0 con los valores por defecto, `relative_to_first`, el umbral de campo lejano, `strict`, la suma coherente (incluida su igualdad con Two-Ray), la sombra log-normal (reproducibilidad y estadística), el diagrama sectorial, el SINR con valores calculados a mano, la paleta Okabe-Ito y la fusión de celdas de los SVG. `tests/test_v1_3.py` cubre la obstrucción de 1.3.0: la geometría del cruce segmento–muro (con y sin `bounds`, bordes y tolerancias), muro opaco, pérdida finita exacta, suma de pérdidas en dB, que un muro no se obstruye a sí mismo, casos límite (Tx/Rx sobre el plano, rayos paralelos, rebote justo al pie de un muro o en la esquina de un edificio), la equivalencia vectorizado/punto a punto con muros oblicuos y finitos, la sombra de un edificio de cuatro muros frente a un cálculo geométrico independiente, los mapas y exportaciones con píxeles sin datos, el CLI y la regresión frente a 1.2.0 (valores de referencia calculados con 1.2.0 e igualdad bit a bit cuando no hay obstrucción).

`tests/test_v1_4.py` cubre lo añadido en 1.4.0:

* La geometría de los rayos, incluida la ley de la reflexión y el hecho de que no cambia `==` ni `repr`.
* El punto en polígono vectorizado frente al escalar, los techos poligonales (reflexión y obstrucción contra una referencia geométrica independiente) y la equivalencia entre el trazado vectorizado y el punto a punto con edificios en L y oblicuos.
* La corrección del receptor situado sobre el plano de un muro, los materiales (valores de ITU-R P.2040 a 9 GHz y modelos de penetración) y los proyectos: ida y vuelta por JSON, validación, que el receptor de prueba coincida con la grilla y todos los modos de simulación.

`tests/test_v1_5.py` cubre lo añadido en 1.5.0:

* **Funciones especiales:** integrales de Fresnel y función de transición contra valores calculados con mpmath, y filo de cuchillo de ITU-R P.526.
* **UTD:** comparación con el filo de cuchillo en las dos polarizaciones, continuidad del campo en el borde de sombra con cuñas opacas y semitransparentes, reciprocidad, y casos en los que no hay rayo (receptor dentro de la cuña, punto de Keller fuera de la arista).
* **Reflexiones dobles:** geometría en un pasillo (ley de la reflexión, longitud de la imagen, amplitud), parches finitos, obstrucción y pares descartados.
* **Motor y proyectos:** el orden de `candidate_rays`, la igualdad entre el trazado vectorizado y el punto a punto con aristas y reflexiones dobles, la evaluación por bloques, las aristas que genera un proyecto (esquinas convexas, techos y extremos libres de muros) y la sombra rellenada detrás de un edificio bajo.

`tests/test_v1_6.py` cubre lo añadido en 1.6.0:

* **Difracción doble:** comparación con el cálculo numérico de Fresnel–Kirchhoff de dos filos, la ley de Keller en las dos aristas (Newton y solución cerrada) y la sombra detrás de una torre que ninguna arista sola rodea, con techo conductor y con pérdidas.
* **Reciprocidad** de los rayos reflexión → difracción, difracción → reflexión y de la difracción doble.
* **Motor:** la zona de transición (y su simetría), el rango angular de un segmento o de un parche visto desde una arista, que los filtros no cambien ningún rayo, el modo compacto de `trace_points`, la igualdad entre el trazado vectorizado y el punto a punto con todas las opciones, la configuración y el informe de los proyectos, y las etiquetas del receptor de prueba.

`tests/test_v1_7.py` cubre lo añadido en 1.7.0:

* **CSV:** coma, punto y coma y tabulador, coma decimal, encabezados con acentos y unidades, transmisor por nombre o id, BOM, y errores que dicen la fila y la columna.
* **Predicción en los puntos:** igual al mapa en ray tracing y en los cuatro modos empíricos, con antena sectorial.
* **Comparación:** signo del error, estadísticas, mejor servidor, y puntos sin señal, demasiado cerca o con un transmisor ausente.
* **Calibración:**
  * recupera las pérdidas, la potencia y el exponente sin ruido, y con ruido los encuentra a pocas σ de los verdaderos;
  * la ley lineal conserva su pendiente y no cambia el proyecto original;
  * un parámetro con pocos puntos se deja igual.
* **Formato, comando y API:** el formato de archivo (versión 1 sin plano ni mediciones, validación del plano y de las mediciones), el comando `open-rps-calibrate` y la API de la interfaz (leer CSV, comparar y calibrar).

`tests/test_gui_server.py` arranca el servidor de la interfaz en un puerto libre y prueba la API: archivos estáticos, protección frente a rutas `../`, simulación y decodificación de las grillas, exportaciones, receptor de prueba, errores y rechazo de peticiones de otros orígenes.

Para repetir la comparación bit a bit contra una copia real de la versión 1.2.0 (el test se omite si la variable no está definida):

```bash
git worktree add /tmp/openrps-1.2.0 396030c        # commit de la versión 1.2.0 (búscalo con `git log --oneline`)
OPEN_RPS_REFERENCE_SRC=/tmp/openrps-1.2.0/src python3 -m unittest tests.test_v1_3.RegressionAgainst120Tests
```

---

## 6. Limitaciones físicas

OpenRPS es un simulador didáctico de primer orden. Antes de usar sus resultados conviene tener presente lo que **no** modela:

* **Sin validación experimental propia.** Los modelos están comprobados contra sus fórmulas, casos calculados a mano y entre sí (p. ej. ray tracing coherente frente a Two-Ray), no contra campañas de medición ni contra el RPS original u otra herramienta de referencia. Desde 1.7.0 puedes comparar con tus propias mediciones y calibrar el escenario (sección 1.H). Ten en cuenta dos cosas:
  * La calibración ajusta los parámetros del modelo, no corrige lo que el modelo no tiene. Las pérdidas calibradas absorben los muebles, la gente o los muros que falten en el dibujo, así que solo valen para ese escenario.
  * Las mediciones de ejemplo son sintéticas.
* **Suma incoherente por defecto; la coherente es idealizada.** Por defecto el trazador suma $|a_k|^2$ rayo a rayo: da la potencia media local, sin interferencia ni desvanecimiento rápido, y por eso difiere de `TwoRayGroundReflection`. Con `coherent=True` aparecen los desvanecimientos (y la igualdad con Two-Ray), pero la posición exacta de cada nulo depende de fases calculadas sobre una geometría perfecta (suelo y muros lisos, posiciones exactas al milímetro, una sola frecuencia): sirve para ver el fenómeno y su estadística, no para predecir dónde habrá un nulo en un entorno real. Además el patrón varía en distancias del orden de λ, así que una grilla de metros solo lo muestrea (los mapas coherentes parecen "ruidosos" por ese motivo).
* **A lo más dos interacciones por rayo.** Rayo directo, un rebote en el suelo, un rebote por muro, una difracción por arista y, opcionalmente, dos rebotes, dos difracciones o un rebote y una difracción (secciones 1.F y 1.G). Las limitaciones de la difracción son estas:
  * No hay rayos de tres o más interacciones. Detrás de dos edificios altos seguidos, o donde hay que rodear dos esquinas y además pasar sobre un techo, la sombra sigue siendo pesimista. Tampoco hay dispersión difusa.
  * Con polarización vertical, el rayo que rodea dos esquinas de una misma fachada vale cero en la UTD de primer orden y no se genera. Haría falta la difracción de pendiente (*slope diffraction*).
  * Los rayos de segundo orden con difracción solo se calculan en la zona de transición de sus aristas (sección 1.G). En el límite de esa zona el rayo aparece o desaparece de golpe, aunque con poca potencia frente al rayo de primer orden.
  * Los coeficientes de Luebbers son heurísticos para cuñas con pérdidas y no exactamente recíprocos fuera del conductor perfecto. Entre dos aristas, la cara que roza el tramo intermedio se trata como conductor perfecto.
  * En incidencia rasante sobre una cara se usa la misma polarización (soft/hard) en todo el rayo.
  * Un receptor exactamente sobre un borde de sombra se evalúa a 10⁻⁷ rad del borde, del lado que el trazador decide con sus parches cerrados.
* **Los muros solo obstruyen si se pide.** Por defecto un reflector añade un rayo reflejado pero no bloquea ni atenúa ningún otro: siempre hay línea de vista (comportamiento de 1.2.0). Con `transmission_loss_db` (sección 1.D) los rayos que cruzan el muro se atenúan o se bloquean y aparecen zonas de sombra geométrica, pero con estas simplificaciones:
  * *Sin aristas no hay difracción.* La obstrucción por sí sola deja el borde de la sombra como un salto brusco de un píxel al siguiente, y detrás de un obstáculo opaco solo llegan los rayos reflejados que lo rodeen: a menudo ninguno (`-inf` dBm, píxel "sin datos"). Con las aristas de la sección 1.F la sombra se rellena con una transición suave, de unos −6 dB en el borde geométrico. Con `max_diffractions=2` el rayo puede pasar además por dos aristas seguidas (sección 1.G). Detrás de varios obstáculos seguidos haría falta más, así que ahí la simulación sigue siendo **pesimista**. En los proyectos y en la interfaz las aristas se generan solas (`Settings.diffraction`). Si construyes los muros a mano con la librería, tienes que añadirlas con `add_edge`.
  * *La transmisión es un número fijo por muro.* La pérdida no depende del ángulo de incidencia, la frecuencia, la polarización ni el espesor (el muro es un plano sin grosor), no añade fase ni retardo, y es independiente de `reflectivity`/`eps_r`: no se impone balance de energía entre lo reflejado y lo transmitido. No hay reflexiones múltiples dentro del muro ni entre muros: un rayo que atraviesa un muro no genera reflexiones nuevas al otro lado más allá de las de un rebote ya descritas.
  * *Los edificios son muros sueltos.* Un edificio se construye con cuatro muros finitos (y un quinto horizontal como techo, si hace falta); no hay volúmenes. Los rayos que pasan sobre el tejado lo hacen por difracción en el borde del techo, si se añaden sus aristas (1.F). Los muros reflejan por las dos caras, así que también existen reflexiones "desde dentro".
  * *Casos límite por convenio.* Una antena o un píxel exactamente sobre el plano de un muro no quedan tapados por él, y un rayo por la arista común de dos muros cruza ambos (sección 1.D).
  * *Solo en ray tracing.* Los modelos empíricos (`run_empirical_simulation`: FSPL, Two-Ray, Log-Distance, Hata, COST-231) ignoran por completo los muros: ni reflejan ni obstruyen.
* **Muros idealizados.** Planos lisos que reflejan por ambas caras; infinitos salvo que se indique `bounds`. Con `reflectivity` el coeficiente es constante (independiente del ángulo, la frecuencia y la polarización) con un desfase fijo de 180°. Con `eps_r` se usa Fresnel para campo eléctrico paralelo al muro, una aproximación válida para polarización vertical, muros verticales y propagación casi horizontal.
* **Terreno plano.** El suelo es el plano z = 0, liso y homogéneo (por defecto $\epsilon_r = 15$, $\sigma = 0.005$ S/m), con polarización vertical. No hay relieve, curvatura terrestre, vegetación ni efectos atmosféricos.
* **Antenas: isotrópica o sector parabólico.** El transmisor es isotrópico (por defecto) o usa el diagrama parabólico de 3GPP (`pattern="sector"`), que es un modelo analítico de un lóbulo principal con un suelo de −30 dB: no hay lóbulos laterales reales, diagramas medidos, agrupaciones, MIMO ni conformación de haz. El receptor es siempre isotrópico (0 dBi) y no hay pérdidas por desajuste de polarización. En los modos empíricos el diagrama se aplica en la dirección geométrica Tx→Rx; combinarlo con Hata (que ya promedia el entorno) es una aproximación habitual pero burda.
* **Campo lejano con un umbral fijo.** Se aplica Friis a cualquier distancia por encima de `min_distance_m`; por debajo el píxel se marca `NaN`. El valor por defecto (1 m) es un umbral sencillo, no la distancia de Fraunhofer $2D^2/\lambda$ de una antena concreta (su tamaño $D$ no se modela): con antenas grandes o frecuencias bajas hay que subirlo a mano. Los píxeles a pocos metros de la antena siguen siendo poco representativos de un despliegue real.
* **Modelos empíricos.** Hata y COST-231 son ajustes estadísticos para macroceldas (1–20 km, antena base de 30–200 m). Usarlos a 2.4 GHz, con antenas más bajas o a menos de 1 km es extrapolar; por defecto el código avisa pero no lo impide (`strict=True` sí lo impide). Estos modos no producen dispersión de retardo ni angular.
* **Sombra sin correlación espacial.** `shadow_fading_sigma_db` añade una muestra log-normal independiente en cada píxel y para cada transmisor. La sombra real está correlacionada a lo largo de decenas de metros y entre emplazamientos; aquí el mapa resultante tiene aspecto de "sal y pimienta". Es adecuada para estudiar estadísticas (márgenes, probabilidad de cobertura), no para obtener mapas de aspecto realista. No depende del entorno ni de la distancia.
* **SINR simplificado.** Enlace descendente, todos los transmisores emitiendo siempre (carga completa), receptor ideal salvo por su figura de ruido. Solo interfieren los transmisores con exactamente la misma frecuencia portadora; no hay interferencia de canal adyacente, ni reparto de recursos, ni control de potencia, ni traspasos. El mejor servidor sigue siendo el de mayor potencia recibida (no el de mayor SINR). La dispersión de retardo y angular de cada píxel se refiere solo a los rayos del mejor servidor.
* **Banda estrecha.** Una única frecuencia por transmisor; los coeficientes no dependen del ancho de banda (`bandwidth_hz` solo interviene en la potencia de ruido).
* **Materiales a una sola frecuencia.** Un muro lo comparten todos los transmisores, así que la conductividad y la penetración de un material se evalúan a la frecuencia de referencia del proyecto (sección 1.E) aunque haya transmisores en varias bandas. La pérdida por penetración sigue siendo un número fijo por cruce: no depende del ángulo ni del espesor.
* **Reflexiones desde el interior de los edificios.** Como cada fachada es un muro suelto que refleja por las dos caras, puede aparecer un rayo que entra por una fachada, se refleja en la opuesta por dentro y vuelve a salir. Paga la pérdida de penetración de cada cruce, así que con materiales opacos o muy atenuantes no aporta nada.
* **Mapas SVG.** Hasta 40 000 celdas hay un rectángulo por celda con su valor al pasar el ratón. Por encima se omiten esos valores y las celdas vecinas del mismo color se fusionan: una grilla de 400 × 400 pasa de 11.5 MB a unos 2 MB en los mapas continuos y a 15 kB en el de mejor servidor (misma imagen), pero para grillas mucho mayores sigue siendo preferible el CSV. El mapa de mejor servidor usa la paleta Okabe-Ito de 8 colores, distinguibles con las formas habituales de daltonismo; a partir del noveno transmisor comparten un gris ("Other"), por lo que cada transmisor se rotula además sobre el mapa.

---

## 7. Historial de cambios

### Versión 1.7.0

* **Novedades** (sección 1.H):
  * **Plano de fondo** en la interfaz: imagen en PNG, JPEG, WebP o GIF que se escala y se coloca con dos puntos, se dibuja en modo *multiply* en 2D y sobre el suelo en 3D, y se guarda dentro del proyecto (`Project.floor_plan`, `FloorPlan`).
  * **Mediciones:** se leen de CSV (`read_measurements_csv`), con coma, punto y coma o tabulador, coma decimal y transmisor por nombre o id, o se anotan con la herramienta Medición (N). Se guardan en el proyecto (`Project.measurements`).
  * **Comparación** con la predicción exacta en cada punto (`compare_measurements`, `predict_points`): error por punto, estadísticas y, en la interfaz, el error sobre el mapa, la gráfica de predicho contra medido y los puntos con más error.
  * **Calibración** (`calibrate`): pérdida de los materiales, potencia de los transmisores (por transmisor o común) y exponente de log-distancia, por Levenberg–Marquardt con incertidumbre 1σ. Está en la interfaz (*Calibrar*, *Aplicar al proyecto*, que se deshace con Ctrl+Z) y en el comando nuevo `open-rps-calibrate`.
  * **Ejemplos:** `examples/plano_oficina.png` y `examples/mediciones_oficina.csv`, sintéticas.
  * **API de la interfaz:** `/api/measurements/parse`, `/api/compare` y `/api/calibrate`.
* **Cambios que conviene conocer:**
  * El formato de proyecto pasa a la versión 2 solo cuando hay plano o mediciones. Sin ellos se guarda como versión 1 y lo siguen abriendo las versiones anteriores.
  * En la interfaz, cambiar el plano, las mediciones, el nombre o la descripción ya no deja los resultados desactualizados.
  * El servidor acepta peticiones de hasta 64 MB (antes 32), por la imagen del plano.
* **Simulación:** sin cambios. Con el mismo proyecto da exactamente lo mismo que 1.6.0.

### Versión 1.6.0

* **Novedades** (sección 1.G):
  * **Difracción doble:** `RayTracer(max_diffractions=2)`. Los puntos de Keller salen en forma cerrada para aristas paralelas y con Newton para aristas cruzadas, y la amplitud con la UTD en cascada.
  * **Rayos mixtos:** `RayTracer(mixed_interactions=True)` añade reflexión → difracción y difracción → reflexión, en muros y en el suelo.
  * **Filtros exactos** que descartan los pares y los receptores donde un rayo de segundo orden no puede existir o no pasa la prueba de la zona de transición.
  * **Modo compacto:** `trace_points(..., compact=True)` y el campo `RayField.labels`.
  * **Proyectos:** `Settings.max_diffractions` y `Settings.mixed_interactions`, desactivadas por defecto. El informe incluye `max_diffractions`, `mixed_interactions` y `active_rays_per_tx`.
  * **Interfaz:**
    * La configuración tiene las opciones "Difracciones por rayo" y "Reflexión + difracción", visibles con la difracción activada.
    * En 2D y 3D, los puntos de cada rayo se marcan según lo que tocaron: cuadrado o cubo en una arista, círculo o esfera en una reflexión.
    * El registro dice cuántos rayos quedan tras los filtros.
* **Corrección en la interfaz.** La vista 3D no actualizaba el mapa del suelo al cambiar de métrica, de transmisor o de escala, ni al repetir la simulación sin tocar nada: seguía mostrando el mapa anterior. Venía de 1.4.0.
* **Cambios que conviene conocer:**
  * Con los valores por defecto (una difracción, sin rayos mixtos) los resultados coinciden con 1.5.0 hasta 10⁻¹³ dB en potencia y 10⁻¹² ns en dispersión de retardo: cambió el orden de algunas sumas. Sin aristas siguen siendo idénticos bit a bit a 1.4.0.
  * Hay una excepción en un receptor exactamente sobre la cara de una cuña *soft*. Ahí el coeficiente UTD es cero y sale de una cancelación que deja un residuo de redondeo, por debajo de −300 dBm, que puede cambiar.
  * `run_raytracing_simulation` recorre ahora los transmisores por fuera y elige el tamaño de bloque según los rayos que quedan tras los filtros.
  * La fracción continua de la función de transición deja de iterar en cada elemento en cuanto este converge.

### Versión 1.5.0

* **Novedades** (sección 1.F):
  * **Difracción UTD:**
    * Las clases y funciones nuevas son `DiffractionEdge`, `RayTracer.add_edge` y `PropagationSimulator.add_edge`, junto con el módulo `open_rps.diffraction`: integrales de Fresnel, función de transición, coeficiente de Kouyoumjian–Pathak con Luebbers y filo de cuchillo de ITU-R P.526.
    * Funciona en `trace_paths` y en `trace_points` (mismos rayos), con suma incoherente o coherente, antenas sectoriales, obstrucción, sombra log-normal y SINR.
  * **Reflexiones dobles:** `RayTracer(max_reflections=2)` añade los rayos muro–muro, suelo–muro y muro–suelo. `RayTracer.candidate_rays()` da el orden de todos los rayos candidatos.
  * **Proyectos:** `Settings.diffraction` (activada por defecto) y `Settings.max_reflections`, y las aristas se generan solas a partir de edificios y muros (`Building.edges`, `Wall.edges`). El informe de la simulación incluye `edges`, `max_reflections` y `candidate_rays`.
  * **Interfaz:**
    * La configuración tiene las opciones "Difracción (UTD)" y "Reflexiones".
    * Los rayos difractados se marcan con un cuadrado en el punto de la arista, y las reflexiones dobles con sus dos puntos de rebote.
    * Las etiquetas del receptor dicen qué esquina, borde o par de muros tocó cada rayo.
* **Cambios que conviene conocer:**
  * Con los valores por defecto de la librería (sin aristas, un rebote) los resultados son idénticos bit a bit a 1.4.0. Los **proyectos** sí cambian, porque la difracción viene activada: la sombra detrás de los obstáculos deja de estar vacía. Para recuperar los resultados de 1.4.0 pon `"diffraction": false` en `settings`.
  * `MultipathRay` tiene un campo más, `interactions`, que tampoco forma parte de `==` ni del `repr`.
  * `run_raytracing_simulation` reparte la grilla en bloques según el número de rayos candidatos (`simulation._MAX_RAY_POINTS`), sin cambiar ningún número.

### Versión 1.4.0

* **Novedades:**
  * Interfaz gráfica `open-rps-gui` (`python3 -m open_rps.gui`): editor 2D, vista 3D, mapas, receptor de prueba con respuesta al impulso, estadísticas, proyectos y exportaciones (sección 4.E). Solo usa la biblioteca estándar y numpy en Python; incluye three.js r187 (MIT) para la vista 3D.
  * `open_rps.project`: proyectos `.orps.json` con área, configuración, transmisores, edificios 2.5D poligonales, muros y biblioteca de materiales (ITU-R P.2040 y 3GPP TR 38.901); `Project.run()`, `probe()`, `build_simulator()`, `load()` y `save()`. Dos ejemplos en `examples/`.
  * `PlanarReflector(footprint=...)`: muros horizontales (techos) limitados por un polígono.
  * `MultipathRay.kind`, `path` y `reflector_index`: geometría de cada rayo, rellenada por `trace_paths`.
* **Corrección.** Un receptor o un transmisor situado sobre el plano de un muro (a menos de 1 µm, `OBSTRUCTION_TOL_M`) ya no recibe un rayo reflejado en ese muro. Es el mismo criterio que la obstrucción ya usaba: un punto sobre el plano ve las dos caras.
  * Antes, con muros oblicuos, que el rayo existiera dependía del signo con que se redondeaba la distancia al plano: ±5·10⁻¹⁷.
  * Cuando existía, duplicaba el rayo directo, con unos +1.2 dB en el escenario de referencia de los tests.
  * Además, el trazado vectorizado y el punto a punto podían discrepar en esos píxeles.
  * Solo cambian los píxeles que caen exactamente sobre el plano de un muro. Todo lo demás es idéntico bit a bit a 1.2.0 y 1.3.0: el test de regresión contra 1.2.0 excluye solo esos píxeles, y se comprobó también contra 1.3.0 y con los archivos de `output/`.

### Versión 1.3.0

La obstrucción es opcional y se activa muro a muro: sin `transmission_loss_db` todos los resultados numéricos, el CSV, el JSON y los SVG son idénticos bit a bit a los de 1.2.0 (comprobado contra una copia de 1.2.0; los archivos de `output/` no cambian).

* **Novedades** (detalles, convenciones y ejemplos en la sección 1.D):
  * `PlanarReflector(..., transmission_loss_db=None)`: pérdida de penetración en dB aplicada a todo rayo que cruce el muro (`float("inf")` = muro opaco). Afecta al rayo directo, al de suelo (sus dos tramos) y a los reflejados en otros muros (sus dos tramos); las pérdidas de varios muros se suman en dB. Funciona en `trace_paths` y `trace_points`, en la simulación vectorizada y en la de referencia punto a punto (mismos resultados), con suma incoherente o coherente, y junto con antenas sectoriales, `min_distance_m`, sombra log-normal y SINR.
  * `PlanarReflector.crosses()`, `signed_distance()` y la propiedad `obstructs`; `RayTracer.obstruction_loss_db(p_inicio, p_fin)` (pérdida de un segmento: 0 = línea de vista).
  * `MultipathRay.obstruction_loss_db` y `RayField.obstruction_loss_db`: pérdida por muros de cada rayo (0 si no cruza ninguno, `inf` si lo bloquea un muro opaco).
  * CLI: `--wall-loss DB` (acepta `inf`) y `--building` (edificio de 4 muros finitos en lugar del muro infinito).
  * Cuaderno: nueva Práctica 5 (sombra de un edificio y pérdida de penetración).
* **Cambios que conviene conocer** (ninguno altera resultados existentes):
  * `MultipathRay` tiene un campo más al final, `obstruction_loss_db = 0.0`: su `repr` lo muestra, y dos rayos solo son iguales (`==`) si también coincide ese campo. `RayField` gana el campo opcional homónimo.
  * Con muros opacos pueden aparecer píxeles con `-inf` dBm y sin servidor en simulaciones de ray tracing (antes solo ocurría en casos degenerados). El resto del código ya trataba esos valores como "sin dato".
  * En un `ChannelImpulseResponse` puede haber rayos con amplitud 0 (bloqueados): `10 * log10(ray.power_linear)` da `-inf` con un aviso de numpy; filtra con `ray.power_linear > 0` o mira `ray.obstruction_loss_db`.

### Versión 1.2.0

Todo lo nuevo es opcional: con los argumentos por defecto, los resultados numéricos, el CSV y el JSON son idénticos a los de 1.1.0, salvo en los dos puntos indicados como *cambios de comportamiento*.

* **Novedades** (detalles y ejemplos en la sección 1.C):
  * `coherent=True` en `run_raytracing_simulation`, `ChannelImpulseResponse.total_received_power_linear/dbm` y `RayField.total_power_linear`: suma coherente de los rayos (desvanecimiento rápido), equivalente a Two-Ray cuando solo hay rayo directo y de suelo.
  * Antenas sectoriales 3GPP: `Transmitter(pattern="sector", azimuth_deg, beamwidth_az_deg, beamwidth_el_deg, downtilt_deg)`, con `Transmitter.pattern_gain_db()` y `gain_dbi()`.
  * `min_distance_m` (campo lejano), `shadow_fading_sigma_db` y `seed` (sombra log-normal) en las dos simulaciones.
  * SINR: `SimulationResult.compute_sinr()`, `sinr_map()`, los campos `sinr_db`, `interference_plus_noise_dbm`, `sinr_settings` y `tx_frequencies_hz`, la función `thermal_noise_dbm()` y los mapas `metric="sinr"` / `"interference"`; columnas y bloque JSON extra solo si se calculó.
  * `mean_excess_delay(relative_to_first=True)`: retardo medio en exceso respecto al primer rayo.
  * `OkumuraHata(..., strict=True)` y `Cost231Hata(..., strict=True)`: `ValueError` fuera del rango de validez.
  * CLI: `--coherent`, `--sector`, `--downtilt`, `--shadow-sigma`, `--seed`, `--min-distance`, `--strict`, `--sinr`, `--noise-figure`, `--bandwidth`.
  * Cuaderno: nueva Práctica 4 (suma coherente y SINR con antenas sectoriales).
* **Cambios de comportamiento**:
  * Los píxeles a menos de 1 m (distancia 3D) de la antena de un transmisor valen ahora `NaN` para ese transmisor en lugar de un valor no físico, y ese transmisor no puede ser su mejor servidor. Solo afecta a escenarios con la antena a la altura del receptor y sobre un nodo de la grilla; `min_distance_m=None` recupera el comportamiento anterior. En un píxel sin servidor, `max_rx_power_dbm` vale `NaN` (en vez de `-inf`) si la potencia de algún transmisor es desconocida.
  * El mapa SVG de mejor servidor usa la paleta Okabe-Ito (cambian los colores, no la geometría); `output/coverage_best_server.svg` se ha regenerado. En el tema oscuro el negro de la paleta se sustituye por blanco.
* **SVG más pequeños**: en grillas de más de 40 000 celdas (las que ya no llevaban valores al pasar el ratón) las celdas contiguas del mismo color se fusionan por filas y columnas y se escriben como un `<path>` por color. Misma imagen (comprobado píxel a píxel con librsvg), entre 5 y 760 veces menos bytes en una grilla de 400 × 400. Los SVG de grillas pequeñas no cambian.

### Versión 1.1.0

* **Correcciones**: ángulos de elevación del rayo de suelo (el punto de rebote se calculaba mal); `export_csv` con `tx_id` que no fueran 0..n-1 (fallaba o tomaba la pérdida de otro transmisor); el JSON ya no contiene `NaN`/`Infinity`; los mapas sin datos ya no generan avisos de numpy.
* **Cambios de comportamiento**: todos los modelos empíricos aplican la ganancia de antena (Two-Ray y Hata la ignoraban: sus mapas suben `antenna_gain_dbi` dB); FSPL ya no recorta a 0 dB la pérdida con ganancias; Two-Ray y Hata reciben la distancia horizontal en la simulación de grilla (antes la 3D); en los modos empíricos la dispersión de retardo/angular es `NaN` en lugar de 0; `tx_id` duplicados o negativos, normales nulas y reflectividades fuera de [0, 1] lanzan `ValueError`.
* **Novedades**: `Cost231Hata`, `ModelRangeWarning`, `PlanarReflector(bounds=..., eps_r=..., sigma=...)`, `RayTracer.trace_points` (trazado vectorizado), simulación de grilla vectorizada, SVG a escala con ejes en metros y leyendas, CLI con `--reflector`, `--tx-height`, `--theme` y exportación JSON, y la batería de tests.

---

## Licencia

Este proyecto se distribuye bajo la **GNU General Public License v2.0** (ver el archivo `LICENSE`). Eres libre de usarlo, estudiarlo, modificarlo y redistribuirlo bajo los términos de esa licencia.

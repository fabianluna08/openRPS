# OpenRPS: Radiowave Propagation Simulator & Channel Estimator
**Simulador de propagación de ondas de radio de código abierto, para uso docente**

`OpenRPS` es un simulador modular de propagación de radio y estimador de canal multitrayecto, escrito en Python y publicado bajo licencia **GPL-2.0** (la misma licencia que el kernel de Linux). Incluye modelos empíricos de pérdida de trayecto y un motor de ray-tracing 3D, y está pensado como herramienta de apoyo para estudiantes.

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
* Hata y COST-231 emiten un `ModelRangeWarning` cuando se evalúan fuera de su rango de validez; el resultado en ese caso es una extrapolación.

---

### B. Motor de Multitrayecto y Respuesta al Impulso del Canal (CIR)
Cada rayo $k$ simulado en el espacio contiene:
* Amplitud compleja $a_k$ (voltaje/campo eléctrico)
* Retardo temporal de propagación $\tau_k = d_k / c$ (en nanosegundos)
* Ángulos de Salida (AoD) en la Estación Base: $\phi_{\text{BS}}$ (azimut) y $\theta_{\text{BS}}$ (elevación)
* Ángulos de Llegada (AoA) en la Estación Móvil: $\phi_{\text{MS}}$ y $\theta_{\text{MS}}$

A partir de cada rayo, se calculan las siguientes métricas (la potencia se suma de forma **incoherente**: la fase de cada rayo se calcula pero no interviene en ninguna métrica):

1. **Potencia Total Recibida ($P_r$)**:
   $$P_r = \sum_{k=1}^K |a_k|^2 \quad [\text{mW}], \qquad P_{r,\text{dBm}} = 10\log_{10}(P_r)$$

2. **Pérdida de Trayecto (Path Loss)**:
   $$PL = P_{t,\text{dBm}} - P_{r,\text{dBm}} \quad [\text{dB}]$$

3. **Retardo Medio en Exceso ($\overline{\tau}$)**:
   $$\overline{\tau} = \frac{\sum_{k} P_k \tau_k}{\sum_{k} P_k} \quad [\text{ns}]$$

4. **Dispersión de Retardo RMS (RMS Delay Spread, $\sigma_\tau$)**:
   $$\sigma_\tau = \sqrt{\frac{\sum_{k} P_k \tau_k^2}{\sum_{k} P_k} - (\overline{\tau})^2} \quad [\text{ns}]$$

5. **Dispersión Angular RMS (Angular Spread, $\sigma_\phi$)**:
   Calculado mediante estadísticas circulares vectoriales en $[-180^\circ, 180^\circ]$ respecto al ángulo medio ponderado por potencia.

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
│   ├── raytracer.py      # Motor 3D de trazado de rayos (directo, tierra y muros)
│   ├── simulation.py     # Orquestador de grillas 2D (Coverage Maps, Best Server, CSV/JSON)
│   ├── visualizer.py     # Mapas ASCII para terminal y SVG vectoriales
│   └── cli.py            # Interfaz de línea de comandos
├── examples/
│   ├── example_sim.py           # Demostración con 2 transmisores y reflector
│   └── practicas_openrps.ipynb  # Cuaderno con 3 prácticas para alumnos
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

Cualquiera de las opciones instala la librería (`import open_rps`) y el comando `open-rps`. Probado con Python 3.14 y numpy 2.4; no se ha verificado con versiones anteriores.

Para generar un paquete distribuible (wheel) que se pueda compartir con los alumnos:

```bash
pip wheel . --no-deps -w dist/      # crea dist/open_rps-1.1.0-py3-none-any.whl
pip install dist/open_rps-1.1.0-py3-none-any.whl
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
| `--reflector` | `on`, `off` | `on` | Muro reflector en x = size/2 (solo afecta a `raytracing`). |
| `--theme` | `light`, `dark` | `light` | Tema de color de los SVG. |
| `--outdir` | ruta | `./output` | Carpeta de salida, **relativa al directorio actual**. |

El CLI escribe en `--outdir` los mismos nombres de archivo que el script de demostración: `simulation_results.csv`, `simulation_summary.json`, `coverage_rx_power.svg`, `coverage_best_server.svg` y, solo en modo `raytracing`, `coverage_delay_spread.svg`. Cada ejecución sobrescribe la anterior: usa un `--outdir` distinto por escenario si quieres conservarlas. En los modos empíricos las columnas de dispersión de retardo y angular del CSV valen `nan` (y `null` en el JSON), porque esos modelos no describen el multitrayecto.

Con los valores por defecto (2.4 GHz, antenas a 25 m, área de 1 km) el modo `hata` queda fuera del rango de validez de ambos modelos de Hata; el CLI lo indica con una línea `[!] Warning: ...`.

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
MapVisualizer.export_svg(result, "mi_mapa.svg", metric="rx_power")   # o "delay_spread", "best_server"

# Respuesta al impulso (CIR) en un punto concreto:
cir = sim.ray_tracer.trace_paths(sim.transmitters[0], (650.0, 450.0, 1.5))
print(len(cir.rays), cir.rms_delay_spread(), cir.angular_spread(plane="phi", station="bs"))
```

Notas sobre `SimulationResult`:

* El primer eje de `rx_power_dbm` y `path_loss_db` sigue el **orden en que se añadieron** los transmisores, no su `tx_id`. `tx_ids` da el identificador de cada posición y `best_server_index` la posición del mejor servidor por píxel (`best_server_id` contiene los `tx_id`; `-1` = sin servidor).
* `path_loss_db` es $P_t - P_r$, es decir, incluye la ganancia de antena del transmisor.
* `PlanarReflector` acepta `bounds=(xmin, xmax, ymin, ymax, zmin, zmax)` para limitar el parche reflectante (sin `bounds` el plano es infinito) y, opcionalmente, `eps_r`/`sigma` para usar el coeficiente de Fresnel dependiente del ángulo en lugar de `reflectivity`.
* `run_raytracing_simulation` evalúa toda la grilla con numpy. `vectorized=False` ejecuta la implementación de referencia punto a punto (mismos resultados, decenas de veces más lenta).

### D. Cuaderno de prácticas (Jupyter)

`examples/practicas_openrps.ipynb` contiene tres prácticas guiadas, con preguntas y ejercicios "Tu turno": (1) pérdida de trayecto vs. distancia entre modelos, (2) cobertura multi-celda y efecto de un muro reflector, (3) respuesta al impulso del canal, dispersión de retardo y angular. Necesita matplotlib y Jupyter, que se instalan con el extra `notebook`:

```bash
pip install -e ".[notebook]"
jupyter lab examples/practicas_openrps.ipynb
```

---

## 5. Tests

Los tests usan `unittest` de la librería estándar (no hace falta pytest). Con el paquete instalado, desde la raíz del proyecto:

```bash
python3 -m unittest discover -s tests
```

Cubren valores de referencia de FSPL, Hata y COST-231, la asíntota $d^4$ del modelo Two-Ray, la dispersión de retardo y angular (incluido el cruce por ±180°), la geometría de los rayos de suelo y de muro, la equivalencia entre el trazado vectorizado y el de referencia punto a punto, simulaciones completas, las exportaciones CSV/JSON/SVG (con `tx_id` no consecutivos y mapas sin datos) y el CLI.

---

## 6. Limitaciones físicas

OpenRPS es un simulador didáctico de primer orden. Antes de usar sus resultados conviene tener presente lo que **no** modela:

* **Sin validación experimental.** Los modelos están comprobados contra sus fórmulas y casos calculados a mano, no contra mediciones ni contra el RPS original u otra herramienta de referencia.
* **Suma incoherente de potencias.** El trazador suma $|a_k|^2$ rayo a rayo: no hay interferencia constructiva/destructiva, ni desvanecimiento rápido, ni la caída $d^4$ tras el punto de quiebre. Por eso el modo `raytracing` (directo + suelo) y el modelo `TwoRayGroundReflection` (suma coherente de los mismos dos rayos) dan resultados distintos.
* **Solo reflexiones especulares de un rebote.** Rayo directo, un rebote en el suelo y un rebote por muro. No hay rebotes múltiples (ni muro–suelo), difracción en aristas, dispersión difusa ni transmisión a través de obstáculos.
* **Los muros no obstruyen.** Un reflector añade un rayo reflejado pero no bloquea ni atenúa el rayo directo, el de suelo ni los de otros muros: siempre hay línea de vista. No existen edificios como volúmenes ni zonas de sombra.
* **Muros idealizados.** Planos lisos que reflejan por ambas caras; infinitos salvo que se indique `bounds`. Con `reflectivity` el coeficiente es constante (independiente del ángulo, la frecuencia y la polarización) con un desfase fijo de 180°. Con `eps_r` se usa Fresnel para campo eléctrico paralelo al muro, una aproximación válida para polarización vertical, muros verticales y propagación casi horizontal.
* **Terreno plano.** El suelo es el plano z = 0, liso y homogéneo (por defecto $\epsilon_r = 15$, $\sigma = 0.005$ S/m), con polarización vertical. No hay relieve, curvatura terrestre, vegetación ni efectos atmosféricos.
* **Antenas isotrópicas.** `antenna_gain_dbi` se aplica por igual en todas las direcciones (sin diagrama de radiación, sectores ni inclinación); el receptor tiene 0 dBi; no hay pérdidas por desajuste de polarización.
* **Campo lejano siempre.** Se aplica Friis a cualquier distancia; los píxeles a pocos metros (o pocas longitudes de onda) de una antena no son fiables.
* **Modelos empíricos.** Hata y COST-231 son ajustes estadísticos para macroceldas (1–20 km, antena base de 30–200 m). Usarlos a 2.4 GHz, con antenas más bajas o a menos de 1 km es extrapolar; el código avisa pero no lo impide. No se añade desvanecimiento por sombra (log-normal) y estos modos no producen dispersión de retardo ni angular.
* **Mejor servidor = mayor potencia recibida.** No se calcula interferencia, ruido ni SINR. La dispersión de retardo y angular de cada píxel se refiere solo a los rayos del mejor servidor.
* **`mean_excess_delay`** devuelve el retardo medio absoluto (ponderado por potencia), no el exceso respecto al primer rayo.
* **Banda estrecha.** Una única frecuencia por transmisor; los coeficientes no dependen del ancho de banda.
* **Mapas SVG.** Un rectángulo por celda: para grillas de cientos de miles de celdas los archivos son muy grandes (use el CSV). El mapa de mejor servidor usa una paleta fija de 8 colores; a partir del noveno transmisor comparten un gris ("Other"), y con más de 3 transmisores algunos pares de colores son difíciles de distinguir (sobre todo con daltonismo), por lo que cada transmisor se rotula además sobre el mapa.

---

## 7. Cambios en la versión 1.1.0

* **Correcciones**: ángulos de elevación del rayo de suelo (el punto de rebote se calculaba mal); `export_csv` con `tx_id` que no fueran 0..n-1 (fallaba o tomaba la pérdida de otro transmisor); el JSON ya no contiene `NaN`/`Infinity`; los mapas sin datos ya no generan avisos de numpy.
* **Cambios de comportamiento**: todos los modelos empíricos aplican la ganancia de antena (Two-Ray y Hata la ignoraban: sus mapas suben `antenna_gain_dbi` dB); FSPL ya no recorta a 0 dB la pérdida con ganancias; Two-Ray y Hata reciben la distancia horizontal en la simulación de grilla (antes la 3D); en los modos empíricos la dispersión de retardo/angular es `NaN` en lugar de 0; `tx_id` duplicados o negativos, normales nulas y reflectividades fuera de [0, 1] lanzan `ValueError`.
* **Novedades**: `Cost231Hata`, `ModelRangeWarning`, `PlanarReflector(bounds=..., eps_r=..., sigma=...)`, `RayTracer.trace_points` (trazado vectorizado), simulación de grilla vectorizada, SVG a escala con ejes en metros y leyendas, CLI con `--reflector`, `--tx-height`, `--theme` y exportación JSON, y la batería de tests.

---

## Licencia

Este proyecto se distribuye bajo la **GNU General Public License v2.0** (ver el archivo `LICENSE`). Eres libre de usarlo, estudiarlo, modificarlo y redistribuirlo bajo los términos de esa licencia.

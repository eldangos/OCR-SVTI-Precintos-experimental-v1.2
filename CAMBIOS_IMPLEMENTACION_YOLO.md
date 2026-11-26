# Implementación YOLO local en SeaAIScan SVTI

## Entrega V5 · 26 de noviembre de 2026

### Consolidación de la instalación

`Step_by_step.md` reúne la instalación, configuración, ejecución y solución de errores. El README dirige a esa guía y `requirements.txt` usa marcadores para los entornos Python 3.10 y 3.11. Se retiraron las tres guías/manifiestos redundantes después de revisar la información incorporada, sus respaldos temporales y las referencias del código. Los scripts de instalación y arranque se conservan como atajos documentados en la guía única.

Se instalaron las dependencias en dos entornos temporales nuevos y ambos pasaron `pip check`. El entorno 3.10 importó la aplicación y ejecutó PaddleOCR con la caché existente sobre una imagen en blanco. Cuatro pruebas focalizadas de interfaz, captura, OCR y persistencia con mocks pasaron. El entorno 3.11 ejecutó el worker V5 a 1280 con Torch 2.11.0+cpu y Torchvision 0.26.0+cpu, obtuvo una detección y cerró con código 0. La inferencia CPU medida fue de 2722 ms para esa imagen. La instalación desde cero de CUDA, la primera descarga de los modelos OCR y la prueba en un segundo PC siguen pendientes.

La consolidación actualizó documentación, dependencias, el instalador y el enlace de ayuda del error de pesos ausentes. El hash SHA-256 de la base original se mantuvo en `49526a620a7d3b8ca5912715cb0c5c2a5b1e229cfbdb0e9832be71e50bf39bfc`. La base está excluida del índice de entrega.

El worker de la aplicación verifica el SHA-256 del checkpoint antes de cargarlo con Ultralytics. La ruta, el tamaño de inferencia y la confianza se resuelven en `modulos/yolo_runtime.py`. El proceso se comunica con PySide6 por JSON de una línea sobre stdin/stdout. Al iniciar, devuelve el archivo efectivamente abierto, tamaño configurado, tarea, etiquetas y dispositivo. La interfaz registra la identidad del modelo y los errores en el diagnóstico local.

| Característica | V3 histórico | V5 verificado en esta entrega |
| --- | --- | --- |
| Checkpoint | `SeaAIScan_V3_2042.pt` | `modelos/seaaiscan-svti-precintos-5-yolo26m-t1.pt` |
| Arquitectura | YOLO26 Small según el registro anterior | `yolo26m.yaml`, escala `m`, 21 774 430 parámetros |
| Resolución | Inferencia configurada a 640 | Entrenamiento e inferencia inicial a 1280 |
| Salida | Precinto | Tarea `detect`, clase única `0: Precinto` |

V5 ocupa 44 230 041 bytes (42,18 MiB). SHA-256: `d3681fa0467a7dd733bdaf645dbfe26d05efe5c3077711d463c196a005597c93`. El checkpoint registra Ultralytics 8.4.55. La copia recibida ya estaba en `modelos/`. El respaldo V3 no se encontró en la copia de trabajo ni en los objetos de las referencias Git locales. Si el equipo conserva V3 en otro medio puede guardarlo por separado en `modelos/`. El worker rechaza cualquier archivo que no coincida con la huella V5.

**Evaluación del entrenamiento en Roboflow.** mAP@50 90,4 %, precisión 92,2 %, recall 90,8 % y F1 91,5 %. Para estimar esas métricas en Hikvision haría falta material independiente con anotaciones de referencia.

### Verificación local

| Comprobación | Resultado observado |
| --- | --- |
| Worker real de PySide6, canal IPC y GPU | Inicio en 37,56 s en la corrida medida. Respuesta de disponibilidad con la ruta V5 y clase correcta. Cierre `NormalExit`, código 0, cero reinicios. Se repitió la carga y el cierre con la verificación SHA-256 final y el intérprete configurado en `.env`. |
| Foto local `img1.png` | Un precinto con confianza 0,983 e ID 1, inferencia 113 ms. |
| Seguimiento híbrido | Sobre una pequeña traslación de la foto, el frame intermedio devolvió `auto-flow` con ID 1. YOLO reconfirmó el ID con confianza 0,959. |
| Video local | Cinco fotogramas iniciales consecutivos del original, cero cajas. Inferencia de 75 a 119 ms por frame, redondeada. No se estimó exactitud ni FPS sostenidos. |
| Interfaz, captura y OCR | Cuatro pruebas `UiSafetyTests` pasaron con servicios simulados y almacenamiento temporal. Cubren indicador/cadencia YOLO, entrega de la foto completa, PNG y hash de entrada, y persistencia de la lectura. |
| Dependencias, sintaxis y configuración | `pip check` sin conflictos en ambos entornos. Compilación de 62 fuentes y 26 importaciones de la aplicación principal. Importación de worker, demo y diagnóstico en YOLO. Rechazo comprobado de archivo ausente y de hash incorrecto. Sintaxis PowerShell de ambos scripts válida. |

La RTX 2050 de 4 GB completó estas pruebas a 1280 sin falta de memoria. Se conserva ese valor inicial. Sigue pendiente medir rendimiento sostenido, CPU y seguimiento en condiciones reales. Se respetó la separación entre parada pan/tilt/zoom y órdenes de foco. Las pruebas usaron archivos locales o servicios simulados. Quedan pendientes la calibración física, las pruebas en terreno y una instalación limpia en un segundo PC.

### Entornos de ejecución

| Entorno | Versiones de la máquina verificada |
| --- | --- |
| Aplicación Windows x64 | Python 3.10.11, PaddlePaddle 2.6.2, PaddleOCR 2.7.3, NumPy 1.26.4, opencv-python 4.6.0.66, PySide6 6.12.0, python-dotenv 1.2.4, openpyxl 3.1.5, requests 2.34.2 y psutil 7.2.2. |
| Detector Windows x64 | Python 3.11.9, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128, Ultralytics 8.4.55, NumPy 2.4.6, opencv-python 5.1.0.95 y lap 0.5.14. |

`requirements.txt` fija las dependencias de ambos procesos mediante marcadores de versión de Python. Pip instala dependencias transitivas como Pillow para OCR y SciPy para YOLO. El instalador selecciona el índice de Torch/Torchvision según el dispositivo y usa el mismo archivo para fijar sus versiones. El proceso YOLO desactiva la comprobación en línea de Ultralytics y la instalación automática de paquetes.

### Archivos de la intervención

- `config.py` y `modulos/settings_store.py` cargan los ajustes desde la raíz antes de evaluar opciones de la aplicación. La interfaz conserva QSettings para la cámara.
- `modulos/yolo_runtime.py`, `modulos/yolo_worker.py` y `ocr_tools/yolo_worker_process.py` conectan V5 con el supervisor existente. `ocr_tools/yolo_local_demo.py` y `ocr_tools/diagnose_yolo_frame.py` utilizan la misma configuración inicial. La demo conserva la selección explícita de otro archivo para comparaciones locales.
- `requirements.txt`, `.env.example`, `.gitignore`, `instalar_windows.ps1` e `iniciar_seaaiscan.ps1` preparan los dos entornos y el arranque. En el PC de desarrollo, el `.env` privado referencia el entorno YOLO existente mediante una ruta relativa.
- El README enlaza la guía de instalación y arranque para Windows. [Instalación paso a paso](Step_by_step.md)

### Limpieza y revisión antes de publicar

| Material de desarrollo | Tratamiento para la entrega y motivo |
| --- | --- |
| `__pycache__` de la raíz, `ui/`, `modulos/`, `ocr_tools/` y `tests/` | Se retiraron 87 archivos de bytecode regenerable tras verificar sus copias en una carpeta temporal. La entrega solo contiene las fuentes. |
| `reports/` (61 archivos rastreados en la revisión de origen) | Incluye trazas OCR, rutas personales, informes, controles sintéticos y copias de evaluadores. Se conserva el archivo histórico en desarrollo. La aplicación crea sus salidas cuando las necesita. Los benchmarks históricos que requieren esos datos se ejecutan en la copia de investigación. |
| `videos_prueba/`, `capturas_precintos/`, ZIP, SQLite y exportaciones | Originales y evidencia privados. Se conservan en disco en desarrollo. |
| `reports/yolo_debug/`, salidas de las demos y directorio de resultados del entorno anterior | Diagnósticos, fotos y logs generados. Fuera del repositorio académico. |
| Código, demos y pruebas | Se conservaron los módulos actuales, las pruebas y los scripts reutilizables. Algunas pruebas de comparación histórica necesitan la revisión antigua `e35236e`, ausente en las referencias disponibles. |

La revisión Git alcanzó las referencias locales, incluidos los blobs de los dos commits históricos. No se detectaron el valor de la contraseña o IP configuradas, claves privadas ni tokens con los patrones revisados. Sí se identificaron un video original, resultados OCR y rutas personales ya confirmados. Ignorar sus nombres protege los archivos nuevos, pero los commits históricos aún conservan los anteriores. Para el primer push se preparó una copia académica con un Git independiente, sin esos commits ni los datos privados. La revisión de origen `fa305a811adceb9fedff246e33b5f67f33d397b3` se conserva en la carpeta de desarrollo.

La carpeta `entrega/SeaAIScan-SVTI/` del equipo de desarrollo contiene la copia académica en un índice independiente, sin remoto. En la revisión V5 se cotejó su contenido con las fuentes y se repitieron las cuatro pruebas con mocks y las comprobaciones de importación y sintaxis usando el intérprete existente. Para revisar la entrega desde la raíz de desarrollo, ejecuta `git -C entrega/SeaAIScan-SVTI diff --cached --stat`.

**Distribución de pesos.** El checkpoint cabe en Git ordinario por línea de comandos. GitHub advierte desde 50 MiB y bloquea los archivos mayores de 100 MiB. Se entrega la huella y el procedimiento de obtención autorizada. La copia para GitHub excluye archivos `.pt` mientras se revisan los permisos. [Límites de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

**Licencias.** El checkpoint declara AGPL-3.0. Ultralytics y Roboflow explican las condiciones del uso abierto de estos modelos y las opciones comerciales. Antes de redistribuir, corresponde confirmar derechos del equipo y del dataset, licencia aplicable al conjunto, avisos y disponibilidad del código fuente correspondiente. La separación en dos procesos es una decisión técnica. [Ultralytics](https://www.ultralytics.com/license) · [Roboflow](https://docs.roboflow.com/models/model-licenses)

Los metadatos instalados declaran Apache-2.0 para PaddleOCR, PaddlePaddle y requests, BSD para Torch/Torchvision, NumPy, psutil y python-dotenv, y MIT para openpyxl. Qt for Python ofrece alternativas LGPL/GPL o comerciales según componentes. Las dependencias transitivas también cuentan, por ejemplo PyMuPDF a través de la cadena PDF de PaddleOCR. Los wheels de OpenCV contienen avisos de terceros. Los paquetes se instalan desde sus índices y conservan sus licencias propias. [Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html) · [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

---

## Historial anterior a V5 · automatización al 25 de noviembre de 2026

### Adquisición real hacia PTZ simulado

El centrado calcula cx=(x1+x2)/2 y cy=(y1+y2)/2 en el frame 102.
Los errores normalizados son ex=2*cx/ancho-1 y ey=2*cy/alto-1.
Pan usa pan_sign*sign(ex) y tilt usa tilt_sign*sign(ey), fuera de
la zona muerta de ±0,12 (equivale a ±6 % del ancho/alto completo).
Las señales +1/-1 actuales describen la convención simulada. La convención
física aún requiere observar desplazamiento y calibrar ambos ejes.
AutoPtzController rechaza habilitar un transporte marcado physical=True
sin directions_verified=True. La ventana sigue conectada al adaptador virtual.
El pulso real aislado conserva su autorización separada, pan +5 y 80 ms.

Tras cada STOP confirmado se exige un frame con timestamp estrictamente
posterior antes de otro pulso, zoom o captura. Una secuencia nueva que contenga
un frame anterior al STOP no satisface esa condición. Después del zoom se
comprueba de nuevo el centrado y se corrige antes del siguiente acercamiento.
test_ptz_coordinate_control.py verifica los cuatro cuadrantes, dos convenciones
de signos, independencia de resolución, zona muerta, STOP e imagen posterior,
pérdida del objetivo y rechazo del transporte físico sin calibración.

La revisión del registro del pulso inmóvil conserva los cinco resultados
positivos del servicio. No contiene trazas detalladas nuevas que demuestren
desplazamiento, rechazo ni la duración efectiva de movimiento. El temporizador
previo a la respuesta HTTP puede reducir la ventana posterior al acuse, pero
la causa física de la inmovilidad continúa pendiente. Se mantienen duración,
velocidad y límites de seguridad sin modificaciones.

Diagnóstico del primer pulso físico: detector.log contiene cinco pares
started=True/stopped=True entre 17:28:25 y 17:29:20 del 25/11. El botón llama a
PtzWorker.enviar_pulso y este usa CameraService.mover_camara, igual que el mando
manual. Esos registros prueban retorno positivo del servicio, pero no desplazamiento
óptico. Se confirmó que el temporizador anterior comenzaba antes de invocar el
servicio y podía consumir los 80 ms mientras esperaba su respuesta. A petición
del operador se corrigió: ahora los 80 ms nominales comienzan después del
retorno positivo de mover_camara. Pan +5 y duración nominal siguen iguales.
Un watchdog threading.Timer separado comienza antes de la llamada, con plazo
de 500 ms tomado del timeout de transporte existente, y solicita STOP si la
operación sigue vigente. El watchdog permanece activo mientras corre el pulso
nominal y puede truncarlo. Rechazo, excepción y timeout solicitan STOP sin
iniciar temporización nominal. Una respuesta posterior al vencimiento no
reactiva el pulso. Las órdenes manuales posteriores invalidan esos temporizadores.
La petición STOP se encola de forma independiente de Qt, pero el envío HTTP
se serializa después de la operación de red en curso. No se garantiza una
parada mecánica a los 80 o 500 ms.

La traza pulse_diagnostic registra token, reloj monotónico, comando, duración,
llamada al servicio, resultado booleano, tiempo de respuesta, error, solicitud
de STOP y descarte/cancelación en cola. service_call significa invocación local
del servicio; service_response registra su confirmación ISAPI o fallo, no
telemetría de posición ni un HTTP status crudo en respuestas exitosas. Se
conserva el detalle del servicio para rechazos. Las excepciones inesperadas se
convierten en fallo y se solicita STOP, evitando terminar el worker sin parada.
La prueba test_ptz_pulse_diagnostics.py cubrió con HTTP simulado aceptación,
403, excepción, ReadTimeout, respuesta de 120 ms, respuesta posterior al plazo
de seguridad y descarte por antigüedad en cola. Los eventos safety_timer_started,
pulse_timer_started y stop_requested permiten distinguir ambos relojes.

Corrección del presupuesto de zoom virtual: la ganancia anterior de 1,25 por
pulso dejaba un objetivo inicial de tamaño relativo 0,03 en 0,179 tras ocho
pulsos, inferior al objetivo 0,30. La escena virtual utiliza ahora ganancia 1,40
por pulso nominal de 0,12 s a velocidad 15, ajustada por duración y velocidad.
La traslación virtual se limita además al 35 % del lado de la caja para conservar
solapamiento. Estos parámetros describen exclusivamente la escena simulada.
Los límites de 8 pasos, 1,2 s de zoom y velocidades físicas se conservan.
Una caja virtual demasiado pequeña todavía termina con ERROR y STOP.
El historial muestra ganancia, tamaño, pasos requeridos y disponibles, zoom
acumulado, límites, centrado, nitidez y razón de transición. Los eventos del
worker real quedan en el historial sin reemplazar el resultado virtual.

Validación focalizada: `test_ptz_zoom_simulation.py` ejecuta un único caso con dos
escenas y reloj virtual, utilizando el simulador real de la aplicación. La
primera recorre centrado, zoom, estabilidad, captura y OCR virtuales hasta
FINALIZADO. La segunda agota ocho pulsos, solicita STOP y no captura. Ambas
mantienen exactamente los mismos presupuestos del controlador.

Una confirmación YOLO válida entra en DETECTADO y siembra una escena virtual con
la caja, ID y dimensiones originales. Desde ese instante la imagen se rotula
DEMO VIRTUAL y el controlador muestra PTZ SIMULADO. Los pulsos actualizan las
coordenadas virtuales según el signo de pan/tilt y su escala según el zoom.
Se registran comando, duración, token, STOP y caja resultante. La cámara física
permanece quieta. Detener una misión antes de solicitar su primera fotografía
libera la adquisición para confirmarla de nuevo. Tras procesar una fotografía,
el mismo objetivo permanece bloqueado hasta cooldown y ausencia observada.

El control PTZ REAL permite preparar únicamente pan +5 durante 80 ms nominales,
con confirmación explícita por diálogo cuyo valor inicial es No. Usa
enviar_pulso y su STOP temporizado. No habilita zoom ni automatización física.
El resultado de movimiento y STOP se registra y un timeout solicita parada.
Su sentido óptico y latencia aún requieren autorización y prueba con Hikvision.

La vigilancia consume únicamente las muestras existentes del 102. En la ventana,
**Simular seguimiento PTZ** conecta el controlador exclusivamente a
`modulos/ptz_simulation.py`. Este adaptador registra los pulsos y simula sus
confirmaciones, AUTO y, si se marca **Simular captura 101 y OCR**, la finalización
del procesamiento. Ambos controles comienzan desactivados. **DETENER PTZ** cancela
el ciclo simulado. Fuera de simulación usa el mando de parada existente. El operador conserva la
captura manual 101 y sus funciones normales de OCR y registro. Los estados del
simulador describen acciones virtuales. Un pulso simulado no modifica el video,
por lo que una escena fija y descentrada puede alcanzar el límite de intentos.

La adquisición exige tres observaciones estables, confianza de 0,65 y al menos
0,8 s por defecto. El detector puede mostrar cajas por debajo de ese umbral.
Durante el seguimiento, el controlador prioriza muestras recientes (intervalo
máximo solicitado de 0,30 s), con un frame en vuelo. Una respuesta solo libera
la inferencia pendiente si coinciden generación, época y secuencia. Cero cajas
es una respuesta normal. Tres errores de inferencia consecutivos utilizan el
mismo presupuesto limitado de reinicios que una respuesta ausente. Una pérdida
anterior al primer disparo permite una adquisición nueva con confirmación completa.
Después del disparo siguen vigentes el tiempo de espera y la ausencia observada.

El marco central punteado es la guía fija de encuadre de VideoWidget. YOLO,
KLT, texto candidato OCR y ROI manual tienen etiquetas distintas. KLT comprueba
ida y vuelta, escala, distribución de puntos y consenso geométrico. Su ventana
de un segundo se mide desde el timestamp del frame con detección. El archivo
`reports/yolo_debug/detector.log` recibe diagnóstico y acciones simuladas.

Los parámetros experimentales de confianza, tamaño relativo, nitidez, estabilidad,
caducidad, pulsos y límites de zoom están en `modulos/yolo_runtime.py`. Los
valores deben calibrarse con escenas de trabajo. En tres muestras guardadas del
102 (154401, 162941, 165010) el modelo V3 devolvió cero cajas a 640 y confianza
0,25 en inferencia directa. Esta comprobación no mide precisión ni falsos positivos.

Antes de habilitar una captura automática física queda pendiente registrar pares
102/101 con el mismo precinto, centro y bordes a varios niveles de zoom. Comparar
los cuatro bordes de la caja en coordenadas normalizadas para detectar recorte,
desplazamiento o distinta relación de aspecto. La simple normalización de las
coordenadas no verifica que ambos canales compartan campo de visión. También
faltan las mediciones físicas de sentido de ejes, latencia de STOP y calidad OCR.

Cambios de esta revisión: adaptador de simulación, conexión y estado de la ventana,
control de adquisición y caducidad (`auto_ptz`, `yolo_vigilance`, `yolo_runtime`),
correlación IPC y reloj del seguimiento (`yolo_worker`, `yolo_worker_process`),
y pruebas específicas de UI y worker. Las secciones siguientes describen la
integración original y sus mediciones históricas.

## Objetivo y límites

Se integró el checkpoint local `modelos/SeaAIScan_V3_2042.pt` para detectar y seguir precintos sobre los frames que la aplicación ya recibe del canal RTSP 102. No se utiliza inferencia Roboflow/cloud ni se abre otra conexión RTSP.

Se conservaron el canal 101 para fotografía/OCR, el canal 102 para video, `CameraService`, OCR, almacenamiento, reportes, autoenfoque, zoom, iris y PTZ. El entorno original de SeaAIScan no recibió Torch, Torchvision, Ultralytics ni OpenCV 5.

## Archivos

### Nuevos

- `modulos/yolo_runtime.py`: ruta central de pesos/intérprete, tamaño de inferencia, confianza y antigüedad máxima de resultados. Permite sustituir el checkpoint mediante `SEAI_YOLO_WEIGHTS` y el Python YOLO mediante `SEAI_YOLO_PYTHON`.
- `modulos/yolo_worker.py`: puente Qt con `QProcess` al entorno YOLO independiente. Serializa frames JPEG por NDJSON, procesa como máximo un frame simultáneo y conserva solo el más reciente pendiente.
- `ocr_tools/yolo_worker_process.py`: proceso hijo que carga `YOLO(weights)`, usa `model.track(..., persist=True, tracker="bytetrack.yaml")`, devuelve cajas/confianza/IDs y mantiene una recuperación manual KLT por flujo óptico cuando la detección se pierde.
- `ocr_tools/yolo_local_demo.py`: herramienta independiente para inferencia en fotos, video local, video sintético de paneo y cálculo aproximado de FPS. No usa RTSP.
- `tests/test_yolo_worker.py`: pruebas simuladas de coalescencia de frames y selección ROI.

### Modificados

- `ui/main_window.py`: agrega botones Activar/Desactivar YOLO y Seleccionar/recuperar objetivo; envía al worker únicamente frames nuevos recibidos desde `CameraService.copiar_muestra()` en `_tick_video()`; valida generación, epoch y antigüedad; detiene el proceso al desactivar o cerrar.
- `ui/video_widget.py`: dibuja cajas YOLO, confianza e ID sobre la imagen existente y convierte el ROI arrastrado por el usuario a coordenadas del frame original. El overlay OCR existente se conserva.
- `tests/test_ui_safety.py`: verifica que YOLO consume los frames del servicio existente sin abrir otro lector RTSP.
- Las dependencias del entorno aislado se encuentran ahora en `requirements.txt`, en las líneas para Python 3.11.
- Los comandos de instalación vigentes están en `Step_by_step.md`.

## Flujo

1. El lector persistente existente obtiene video de canal 102; el timer de la UI toma frames nuevos, sin pedir otra conexión.
2. Al activar YOLO, `YoloWorker` inicia el Python indicado por `SEAI_YOLO_PYTHON` y pasa `SEAI_YOLO_WEIGHTS` al script hijo. El modelo se carga una sola vez.
3. Los frames se codifican a JPEG y se envían por una línea JSON. Hay un frame en inferencia como máximo; las muestras nuevas reemplazan la pendiente, evitando acumulación y latencia creciente.
4. Ultralytics mantiene predictor/tracker con `persist=True` y ByteTrack. Los IDs persisten en la sesión y se reinician al cambiar generación o stream epoch.
5. El worker devuelve cajas `xyxy`, clase, confianza, ID y secuencia. La UI descarta resultados de conexiones anteriores o con más de 0,8 s.
6. El widget dibuja clase, confianza e ID. Un ROI arrastrado envía coordenadas al proceso; KLT conserva el objetivo temporalmente si desaparece la detección. La caja manual usa ID `M<n>` mientras no haya asociación YOLO.
7. Desactivar envía `shutdown`; la app espera el cierre asíncrono y fuerza el proceso solo tras un timeout.

## Pruebas y resultados

- Checkpoint: contenedor PyTorch válido; metadatos indican `ultralytics 8.4.55`, `yolo26s.yaml`, `DetectionModel`, `task=detect` y clase `Precinto`.
- Imagen real del proyecto: detección en `capturas_precintos/img1.png` a confianza 0,6523 y 0,4455. La segunda caja casi se superpone con la primera; revisar NMS/confianza con más material.
- Video sintético local: 45/45 frames con detección y un ID; 15,12 FPS de pared a 15 FPS de entrada.
- Puente Qt/QProcess simulado con video: respondió al primer y último frame enviado; los intermedios se coalescieron; proceso hijo terminó con `NormalExit`.
- UI offscreen: botón visible, detección/caja en frame local, stop/restart y ROI simulado probados. El smoke test final usó un servicio falso incompleto al cerrar, sin `detener_lectura()`; la excepción fue del mock de cierre, no del servicio real. El proceso YOLO ya había sido detenido.
- Suite UI/worker: 28 pruebas pasaron. Suite completa: 119 pasaron y dos pruebas históricas no pudieron cargar baseline porque el commit `e35236e` no existe en el repositorio.
- Video real de teléfono `Videos (3).mp4`: a confianza 0,20, 270° detectó en 23/65 muestras frente a 6/65 en 0° y 6/65 en 90°; se confirmó visualmente el precinto en el frame anotado. No se aplica esa rotación al stream Hikvision.
- Video `Videos (9).mp4`: 1.731/4.493 frames con detección, 110 IDs totales, 42,42 FPS de pared (29,96 FPS de fuente). Muestra que los IDs pueden fragmentarse tras pérdidas/oclusión.

## Limitaciones pendientes

- No se ha hecho una prueba física con el canal 102; resolución, foco, movimiento, oclusiones y rendimiento real deben verificarse con el compañero en una prueba controlada.
- Los ejemplos locales incluyen videos de teléfono con orientación distinta; no se rota automáticamente el video Hikvision. El demo local aún no tiene un flag de rotación integrado.
- Algunas imágenes producen cajas duplicadas y otras no detectan el precinto pequeño/ocluido; ajustar confianza, NMS, `imgsz` o revisar dataset requiere evaluación, no se cambió el modelo.
- ByteTrack generó 110 IDs en uno de los videos largos por pérdidas/reapariciones; confirmar continuidad con video real del puerto.
- La recuperación manual KLT es temporal y basada en apariencia/movimiento; no acredita identidad semántica y puede desviarse con cambios de cámara u oclusión larga.
- El primer arranque del proceso tarda mientras carga pesos. CPU está soportada como fallback, pero no se midió FPS CPU en este equipo.
- La instalación de desarrollo utilizó un entorno YOLO externo. La entrega usa `.venv-yolo/Scripts/python.exe` por defecto y admite `SEAI_YOLO_PYTHON` para otras ubicaciones. Ver `Step_by_step.md`.

## Entorno comprobado

- Entorno SeaAIScan sin cambios: Python 3.10.11, PaddlePaddle 2.6.2, PaddleOCR 2.7.3, NumPy 1.26.4 y OpenCV 4.6.0.66.
- YOLO aislado: Python 3.11.9, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128, Ultralytics 8.4.55, OpenCV 5.1.0.95, NumPy 2.4.6, lap 0.5.14.
- GPU de desarrollo: GeForce RTX 2050; `torch.cuda.is_available()` verdadero; runtime CUDA PyTorch 12.8.
- `pip check` del entorno YOLO: sin dependencias rotas.

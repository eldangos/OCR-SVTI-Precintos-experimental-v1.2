# Implementación YOLO local en SeaAIScan SVTI

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
- `requirements-yolo.txt`: fija dependencias del entorno aislado; Torch/Torchvision se instalan aparte según CPU/CUDA.
- `INSTALACION_YOLO.md`: procedimientos PowerShell/CMD para preparar equipos nuevos.

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
- El venv YOLO de desarrollo está fuera del repositorio en el Escritorio; en equipos nuevos debe configurarse `SEAI_YOLO_PYTHON` según `INSTALACION_YOLO.md`.

## Entorno comprobado

- Entorno SeaAIScan sin cambios: Python 3.10.11, PaddlePaddle 2.6.2, PaddleOCR 2.7.3, NumPy 1.26.4 y OpenCV 4.6.0.66.
- YOLO aislado: Python 3.11.9, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128, Ultralytics 8.4.55, OpenCV 5.1.0.95, NumPy 2.4.6, lap 0.5.14.
- GPU de desarrollo: GeForce RTX 2050; `torch.cuda.is_available()` verdadero; runtime CUDA PyTorch 12.8.
- `pip check` del entorno YOLO: sin dependencias rotas.

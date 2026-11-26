# SeaAIScan SVTI

SeaAIScan ayuda a un operador a localizar precintos en video, fotografiarlos a mayor resolución y revisar su lectura OCR. La aplicación de escritorio guarda la foto y el resultado de la lectura en el computador. El checkpoint local corresponde a la **versión 5 del dataset** y la arquitectura es **Ultralytics YOLO26 Medium**.

## Instalación en Windows

Para preparar los dos entornos, obtener los pesos V5 e iniciar SeaAIScan, sigue la [guía paso a paso](Step_by_step.md).

## Del video a la lectura

| Componente | Función |
| --- | --- |
| Interfaz PySide6 y OpenCV | Muestra video RTSP 102, superpone cajas y ofrece control manual PTZ, zoom, enfoque y parada de emergencia. |
| Detector en Python 3.11 | La interfaz envía fotogramas por la entrada estándar de un proceso independiente. PyTorch y Ultralytics ejecutan YOLO localmente a `imgsz=1280`. |
| Seguimiento y vigilancia | ByteTrack asocia detecciones. KLT sigue regiones entre inferencias. El controlador exige reconfirmación y gestiona pérdida, recuperación y antigüedad de los resultados. |
| Captura y OCR en Python 3.10 | La captura manual obtiene una foto completa del canal 101. PaddleOCR procesa la imagen y se conserva evidencia en `capturas_precintos/`. La interfaz permite revisar y registrar lecturas en SQLite y exportarlas a Excel. |
| Automatización PTZ | El centrado y el zoom automáticos usan el adaptador simulado. La parada de movimiento está separada del foco. El ensayo físico aislado requiere una acción explícita del operador. |

## Modelo y alcance de las pruebas

V5 detecta una clase, `0 = Precinto`. Roboflow reportó mAP@50 de 90,4 %, precisión de 92,2 %, recall de 90,8 % y F1 de 91,5 % a una resolución de entrenamiento de 1280 × 1280. Son métricas de la evaluación de Roboflow. La precisión en condiciones de puerto queda pendiente de una evaluación independiente.

Con una fotografía local, la aplicación obtuvo una detección con confianza 0,983. ByteTrack y KLT conservaron la identidad en una secuencia corta derivada de esa fotografía. En cinco fotogramas consecutivos de un video local no se obtuvieron cajas. La automatización física sigue pendiente de calibración. Las pruebas de interfaz, captura y OCR utilizaron servicios simulados. [Registro de implementación y pruebas](CAMBIOS_IMPLEMENTACION_YOLO.md)

## Datos y distribución

Las credenciales, capturas, manifiestos, videos, lecturas y entornos son locales. La instalación usa Internet para descargar dependencias y PaddleOCR puede descargar sus modelos la primera vez. La inferencia YOLO usa el checkpoint verificado en disco.

El checkpoint declara AGPL-3.0. Antes de distribuir código o pesos, el responsable debe revisar esa licencia, los derechos del dataset y las licencias de las dependencias. La entrega académica aún requiere esa decisión. [Licencias de modelos de Roboflow](https://docs.roboflow.com/models/model-licenses) · [Licencia de Ultralytics](https://www.ultralytics.com/license)

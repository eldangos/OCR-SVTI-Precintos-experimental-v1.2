# Instalación de SeaAIScan y YOLO local en Windows

Esta guía conserva dos entornos virtuales independientes: el entorno original de SeaAIScan y el entorno de inferencia YOLO. El primero sigue usando sus dependencias actuales; Torch, Torchvision, Ultralytics y OpenCV 5 viven solo en el entorno YOLO.

## 1. Requisitos

- Windows 10/11 de 64 bits.
- Proyecto SeaAIScan completo y pesos `modelos\SeaAIScan_V3_2042.pt` copiados localmente. Los archivos `.pt` están ignorados por Git; distribuirlos por el medio autorizado del equipo.
- Python 3.10 de 64 bits para el entorno original, conforme al procedimiento actual del proyecto.
- Python 3.11 de 64 bits para YOLO. Fue el intérprete validado en la integración.
- Mínimo 8 GB de RAM recomendado. Para CUDA, GPU NVIDIA y driver compatible. CPU funciona, pero puede ser demasiado lento para video en vivo.
- Espacio adicional para PyTorch, que ocupa varios GB.

Abrir **PowerShell** o **CMD** y comprobar el Python Launcher:

```powershell
py -0p
py -3.10 --version
py -3.11 --version
```

Instalar Python desde python.org si falta alguna versión, habilitando Python Launcher. Cerrar y volver a abrir la terminal después de instalar.

## 2. Entorno original de SeaAIScan

Desde la carpeta del proyecto:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

El `requirements.txt` del proyecto no necesita paquetes YOLO y no debe recibir Torch, Torchvision ni OpenCV 5. Para iniciar SeaAIScan:

```powershell
.\venv\Scripts\python.exe .\main.py
```

Equivalente en **CMD**:

```bat
venv\Scripts\python.exe main.py
```

Si el entorno `venv` ya existe y funciona, no lo recrees ni reinstales sus dependencias.

## 3. Crear el entorno YOLO aislado

En PowerShell, desde la raíz del proyecto:

```powershell
py -3.11 -m venv .venv-yolo
$YOLOPY = (Resolve-Path .\.venv-yolo\Scripts\python.exe).Path
& $YOLOPY -m pip install --upgrade pip
```

El entorno se crea dentro del proyecto, pero `.venv-yolo` es privado de esa copia. No reutilizar ni reemplazar el `venv` original.

## 4. Instalar PyTorch y Torchvision

**No instales estos dos paquetes desde `requirements-yolo.txt`.** El wheel depende del dispositivo. PyTorch incluye el runtime CUDA en sus wheels; normalmente no hace falta instalar CUDA Toolkit, pero sí un driver NVIDIA compatible.

### Equipo con NVIDIA

1. Ejecuta `nvidia-smi` para identificar GPU y driver.
2. Abre el selector oficial: <https://pytorch.org/get-started/locally/>.
3. Selecciona Windows, Pip, Python y la variante CUDA que el selector ofrezca para ese driver.
4. Ejecuta el comando mostrado, usando el Python YOLO explícito. Ejemplo validado en la máquina de desarrollo con RTX 2050:

```powershell
& $YOLOPY -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Ese ejemplo CUDA 12.8 no es universal: para otra GPU/driver, usar la selección oficial actual, no copiar `cu128` a ciegas.

### Equipo solo CPU

```powershell
& $YOLOPY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 5. Instalar Ultralytics y el tracker

Instala Torch/Torchvision primero. Después:

```powershell
& $YOLOPY -m pip install -r .\requirements-yolo.txt
& $YOLOPY -m pip check
```

`requirements-yolo.txt` fija las versiones probadas de Ultralytics y sus dependencias directas, incluido `lap` requerido por ByteTrack. No incluye Torch/Torchvision para evitar escoger un wheel CPU/CUDA incorrecto. Si se ejecuta el requirements antes de instalar PyTorch, pip podría resolver una distribución propia de Torch: respeta el orden anterior.

## 6. Configurar las rutas del runtime

Por defecto, `modulos/yolo_runtime.py` busca los pesos en:

```text
<raíz del proyecto>\modelos\SeaAIScan_V3_2042.pt
```

El intérprete predeterminado es `%USERPROFILE%\Desktop\SeaAIScan-YOLO-311\Scripts\python.exe`, que coincide con la instalación de desarrollo. Para usar `.venv-yolo` o una ruta distinta, establece `SEAI_YOLO_PYTHON` antes de iniciar la app.

PowerShell:

```powershell
$env:SEAI_YOLO_PYTHON = (Resolve-Path .\.venv-yolo\Scripts\python.exe).Path
.\venv\Scripts\python.exe .\main.py
```

CMD:

```bat
set "SEAI_YOLO_PYTHON=%CD%\.venv-yolo\Scripts\python.exe"
venv\Scripts\python.exe main.py
```

Para cambiar el checkpoint, establece `SEAI_YOLO_WEIGHTS` a su ruta completa. No mover ni renombrar el checkpoint por defecto si se usan los valores predeterminados.

## 7. Verificar librerías, GPU y carga del modelo

PowerShell:

```powershell
& $YOLOPY -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $YOLOPY .\ocr_tools\yolo_local_demo.py --image .\capturas_precintos\img1.png --device auto
```

`--device auto` elige CUDA si `torch.cuda.is_available()` es verdadero; de lo contrario usa CPU. En la prueba de integración, `YOLO('SeaAIScan_V3_2042.pt')` cargó el checkpoint como tarea `detect`, clase `Precinto`.

## 8. Activar YOLO en la aplicación

Inicia SeaAIScan con el entorno original, conecta normalmente la cámara y pulsa **Activar detector YOLO**. La app inicia el intérprete configurado como proceso separado, carga localmente el checkpoint y alimenta los frames ya recibidos por el canal 102. No abre un stream RTSP adicional.

Las cajas muestran clase, confianza e ID. Para reubicar un objetivo perdido, pulsa **Seleccionar / recuperar objetivo** y arrastra un ROI alrededor del precinto. Desactivar el botón cierra el proceso; se puede volver a activar sin reiniciar la app.

El canal 101 continúa reservado para fotos/OCR. No se usa el canal 103.

## 9. Prueba local de video

Con un video grabado en `videos_prueba`:

```powershell
& $YOLOPY .\ocr_tools\yolo_local_demo.py --video ".\videos_prueba\mi_video.mp4" --device auto --output-dir "$env:USERPROFILE\Desktop\SeaAIScan-YOLO-Results"
```

El comando crea `mi_video_tracked.avi` con cajas, confianza e IDs. Para comparar orientación de un video de teléfono sin alterar la cámara, usa las imágenes de evidencia ya generadas o una copia rotada del archivo; la app **no rota el canal Hikvision 102**.

## 10. Errores frecuentes

- **No se encuentra el intérprete YOLO:** establece `SEAI_YOLO_PYTHON` a `Scripts\python.exe` del entorno YOLO.
- **No se encuentran los pesos:** confirma que `modelos\SeaAIScan_V3_2042.pt` exista o define `SEAI_YOLO_WEIGHTS`.
- **CUDA no disponible:** confirma `nvidia-smi`, driver, wheel CUDA y `torch.cuda.is_available()`. La aplicación debe poder usar CPU con `--device auto`, pero será más lenta.
- **DLL/CUDA o error de carga de Torch:** reinstala Torch/Torchvision usando el selector oficial para ese equipo; no cambies el entorno SeaAIScan.
- **`lap` no está instalado:** instala `requirements-yolo.txt`; `lap` se requiere para ByteTrack.
- **Sin detecciones:** prueba con imagen/video donde el precinto sea grande y visible; la prueba con video de teléfono indicó que la orientación puede afectar mucho. No gires el stream Hikvision sin una prueba específica.
- **Ventana termina antes de cargar:** espera al estado “YOLO listo”; el primer arranque carga el modelo y puede tardar.

## Versiones verificadas en la máquina de desarrollo

| Paquete | Versión |
|---|---|
| Python YOLO | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| Ultralytics | 8.4.55 |
| OpenCV del entorno YOLO | 5.1.0.95 |
| NumPy del entorno YOLO | 2.4.6 |
| lap | 0.5.14 |
| GPU | NVIDIA GeForce RTX 2050, compute capability 8.6 |
| Runtime reportado por PyTorch | CUDA 12.8 |

Estas versiones describen la máquina verificada, no un requisito para copiar sin revisar en cualquier equipo. Elegir el wheel PyTorch según GPU/driver.

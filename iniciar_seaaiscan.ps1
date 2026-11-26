$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $app = ".\venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $app -PathType Leaf)) {
        throw "Falta Python de SeaAIScan. Ejecuta instalar_windows.ps1."
    }
    if (!(Test-Path -LiteralPath ".env")) {
        Write-Warning "Falta .env. Puedes copiar .env.example a .env o configurar la camara desde la interfaz."
    }
    $check = @'
import config
from modulos.yolo_runtime import YOLO_PYTHON_PATH, YOLO_WEIGHTS_PATH, verify_v5
if not YOLO_PYTHON_PATH.is_file():
    print(f'Falta el Python de YOLO: {YOLO_PYTHON_PATH}. Revisa SEAI_YOLO_PYTHON.')
try:
    verify_v5(YOLO_WEIGHTS_PATH)
except (OSError, ValueError) as error:
    print(error)
'@
    & $app -c $check
    if ($LASTEXITCODE -ne 0) { throw "Fallo al leer la configuracion. Revisa el entorno principal y .env." }
    & $app ".\main.py"
    if ($LASTEXITCODE -ne 0) { throw "SeaAIScan finalizo con codigo $LASTEXITCODE." }
}
finally { Pop-Location }

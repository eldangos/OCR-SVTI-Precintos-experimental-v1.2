param([ValidateSet("cpu", "cu128")][string]$Dispositivo = "cpu")
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    function Invoke-Checked([string]$Program, [string[]]$Arguments) {
        & $Program @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Fallo al ejecutar $Program $Arguments (codigo $LASTEXITCODE)." }
    }

    function Initialize-Environment([string]$Folder, [string]$Version) {
        $exe = Join-Path $Folder 'Scripts\python.exe'
        if (!(Test-Path -LiteralPath $exe -PathType Leaf)) {
            if (!(Get-Command py -ErrorAction SilentlyContinue)) {
                throw "Falta el lanzador py. Instala Python $Version x64 con Python Launcher."
            }
            Invoke-Checked py @("-$Version", "--version")
            Invoke-Checked py @("-$Version", "-m", "venv", $Folder)
        }
        $check = "import struct, sys; sys.exit(0 if sys.version_info[:2] == tuple(map(int, '$Version'.split('.'))) and struct.calcsize('P') == 8 else 'Se requiere Python $Version x64 en $Folder')"
        Invoke-Checked $exe @("-c", $check)
    }

    Initialize-Environment ".\venv" "3.10"
    Initialize-Environment ".\.venv-yolo" "3.11"
    $app = ".\venv\Scripts\python.exe"
    $yolo = ".\.venv-yolo\Scripts\python.exe"
    Invoke-Checked $app @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $app @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked $yolo @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $yolo @("-m", "pip", "install", "--force-reinstall", "--no-deps", "--index-url", "https://download.pytorch.org/whl/$Dispositivo", "-c", "requirements.txt", "torch", "torchvision")
    Invoke-Checked $yolo @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked $app @("-m", "pip", "check")
    Invoke-Checked $yolo @("-m", "pip", "check")
    if ($Dispositivo -eq "cu128") {
        Invoke-Checked $yolo @("-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA no disponible. Revisa el driver o instala con -Dispositivo cpu.')")
    }
    if (!(Test-Path -LiteralPath ".env")) {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
        Write-Host "Creado .env con los campos de camara vacios."
    }
    Invoke-Checked $app @("-c", "import config; from modulos.yolo_runtime import verify_v5, YOLO_WEIGHTS_PATH; verify_v5(YOLO_WEIGHTS_PATH)")
    Write-Host "Instalacion completa. Ejecuta .\iniciar_seaaiscan.ps1"
}
finally { Pop-Location }

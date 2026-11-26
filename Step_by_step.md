1. Crear los entornos virtuales (solo la primera vez)
py -3.10 -m venv venv
py -3.11 -m venv .venv-yolo

.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-yolo\Scripts\python.exe -m pip install -r requirements.txt


.\venv\Scripts\python.exe main.py
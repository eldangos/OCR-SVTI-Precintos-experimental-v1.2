"""Paths and tunables for the optional, isolated local YOLO runtime."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS_PATH = Path(
    os.environ.get(
        "SEAI_YOLO_WEIGHTS",
        PROJECT_ROOT / "modelos" / "SeaAIScan_V3_2042.pt",
    )
)
YOLO_PYTHON_PATH = Path(
    os.environ.get(
        "SEAI_YOLO_PYTHON",
        Path.home() / "Desktop" / "SeaAIScan-YOLO-311" / "Scripts" / "python.exe",
    )
)
YOLO_PROCESS_PATH = PROJECT_ROOT / "ocr_tools" / "yolo_worker_process.py"
YOLO_IMAGE_SIZE = int(os.environ.get("SEAI_YOLO_IMGSZ", "640"))
YOLO_CONFIDENCE = float(os.environ.get("SEAI_YOLO_CONFIDENCE", "0.25"))
YOLO_RESULT_MAX_AGE_S = 0.8
YOLO_JPEG_QUALITY = 82
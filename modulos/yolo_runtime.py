"""Paths and tunables for the optional, isolated local YOLO runtime."""

import hashlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(variable, default):
    path = Path(os.environ.get(variable) or default).expanduser()
    return (PROJECT_ROOT / path).resolve()


YOLO_WEIGHTS_PATH = project_path(
    "SEAI_YOLO_WEIGHTS", "modelos/seaaiscan-svti-precintos-5-yolo26m-t1.pt"
)
V5_SHA256 = "d3681fa0467a7dd733bdaf645dbfe26d05efe5c3077711d463c196a005597c93"


def verify_v5(path):
    if not path.is_file():
        raise FileNotFoundError(f"Faltan los pesos V5: {path}. Ver Step_by_step.md")
    with path.open("rb") as checkpoint:
        digest = hashlib.sha256()
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != V5_SHA256:
        raise ValueError(f"El SHA-256 de {path} no corresponde al checkpoint V5 de la entrega")


YOLO_PYTHON_PATH = project_path("SEAI_YOLO_PYTHON", ".venv-yolo/Scripts/python.exe")
YOLO_PROCESS_PATH = PROJECT_ROOT / "ocr_tools" / "yolo_worker_process.py"
YOLO_DEBUG_FRAME_DIR = project_path("SEAI_YOLO_DEBUG_DIR", "reports/yolo_debug")
YOLO_IMAGE_SIZE = int(os.environ.get("SEAI_YOLO_IMGSZ", "1280"))
YOLO_CONFIDENCE = float(os.environ.get("SEAI_YOLO_CONFIDENCE", "0.25"))
YOLO_RESULT_MAX_AGE_S = 2.0
YOLO_JPEG_QUALITY = 82

# Surveillance cadence and capture gate (seconds, confidence in [0, 1]).
YOLO_SEARCH_INTERVAL_S = float(os.environ.get('SEAI_YOLO_SEARCH_INTERVAL_S', '.5'))
YOLO_TRACK_INTERVAL_S = float(os.environ.get('SEAI_YOLO_TRACK_INTERVAL_S', '.15'))
YOLO_CORRECT_INTERVAL_S = float(os.environ.get('SEAI_YOLO_CORRECT_INTERVAL_S', '.65'))
YOLO_FLOW_INTERVAL_S = float(os.environ.get('SEAI_YOLO_FLOW_INTERVAL_S', '.075'))
if not .05 <= YOLO_FLOW_INTERVAL_S <= .5:
    raise ValueError('SEAI_YOLO_FLOW_INTERVAL_S debe estar entre 0,05 y 0,5 s')
YOLO_HYBRID_OPTIONS = {
    'recheck_s': float(os.environ.get('SEAI_YOLO_RECHECK_S', '1.2')),
    'flow_max_s': float(os.environ.get('SEAI_YOLO_FLOW_MAX_S', '5')),
    'retain_s': float(os.environ.get('SEAI_YOLO_RETAIN_S', '.55')),
    'keep_confidence': float(os.environ.get('SEAI_YOLO_KEEP_CONFIDENCE', '.35')),
    'appearance_min': float(os.environ.get('SEAI_YOLO_APPEARANCE_MIN', '.70')),
}
YOLO_CONFIRM_HITS = int(os.environ.get('SEAI_YOLO_CONFIRM_HITS', '3'))
YOLO_CONFIRM_S = float(os.environ.get('SEAI_YOLO_CONFIRM_S', '.8'))
YOLO_CAPTURE_CONFIDENCE = float(os.environ.get('SEAI_YOLO_CAPTURE_CONFIDENCE', '.65'))
YOLO_CAPTURE_COOLDOWN_S = float(os.environ.get('SEAI_YOLO_CAPTURE_COOLDOWN_S', '15'))
YOLO_REARM_ABSENCE_S = float(os.environ.get('SEAI_YOLO_REARM_ABSENCE_S', '3'))
YOLO_CONFIRM_MAX_GAP_S = float(os.environ.get(
    'SEAI_YOLO_CONFIRM_MAX_GAP_S', str(max(1.5, YOLO_SEARCH_INTERVAL_S * 2.5)),
))

# Relative zoom budget (no absolute optical-position telemetry is available).
AUTO_PTZ_OPTIONS = {
    'settle_s': float(os.environ.get('SEAI_AUTO_SETTLE_S', '.8')),
    'target_age_s': float(os.environ.get('SEAI_AUTO_TARGET_AGE_S', '1.2')),
    'speed': int(os.environ.get('SEAI_AUTO_PTZ_SPEED', '12')),
    'zoom_speed': int(os.environ.get('SEAI_AUTO_ZOOM_SPEED', '15')),
    'pulse_s': float(os.environ.get('SEAI_AUTO_PULSE_S', '.12')),
    'deadband': float(os.environ.get('SEAI_AUTO_CENTER_TOLERANCE', '.12')),
    'target_size': float(os.environ.get('SEAI_AUTO_TARGET_SIZE', '.30')),
    'max_zoom_s': float(os.environ.get('SEAI_AUTO_MAX_ZOOM_S', '1.2')),
    'max_zoom_steps': int(os.environ.get('SEAI_AUTO_MAX_ZOOM_STEPS', '8')),
    'max_moves': int(os.environ.get('SEAI_AUTO_MAX_MOVES', '16')),
    'max_session_s': float(os.environ.get('SEAI_AUTO_MAX_SESSION_S', '35')),
    'sharpness': float(os.environ.get('SEAI_AUTO_SHARPNESS', '45')),
    'photo_sharpness': float(os.environ.get('SEAI_AUTO_PHOTO_SHARPNESS', '35')),
    'retries': int(os.environ.get('SEAI_AUTO_OCR_RETRIES', '1')),
    'pan_sign': int(os.environ.get('SEAI_AUTO_PAN_SIGN', '1')),
    'tilt_sign': int(os.environ.get('SEAI_AUTO_TILT_SIGN', '-1')),
}

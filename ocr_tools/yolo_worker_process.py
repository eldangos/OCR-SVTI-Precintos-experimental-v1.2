"""NDJSON frame worker for an isolated local Ultralytics installation."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modulos.yolo_quality import image_quality
from modulos.yolo_runtime import YOLO_CONFIDENCE, YOLO_IMAGE_SIZE, verify_v5
from modulos.yolo_vigilance import covers_frame

os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "False"
os.environ["YOLO_OFFLINE"] = "True"


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def overlap(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def initialize_manual_tracker(frame, box, cv2):
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.rectangle(
        np.zeros((height, width), dtype=np.uint8),
        (x1, y1),
        (x2, y2),
        255,
        thickness=-1,
    )
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=3,
        mask=mask,
    )
    if points is None or len(points) < 3:
        return None
    return {
        "box": [float(x1), float(y1), float(x2), float(y2)],
        "gray": gray,
        "points": points,
        "confidence": None,
        "track_id": None,
        "selected_at": time.monotonic(),
    }


def advance_manual_tracker(state, frame, cv2):
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = state["points"]
    if points is None or len(points) < 3:
        return None

    forward, status, _ = cv2.calcOpticalFlowPyrLK(state["gray"], gray, points, None)
    if forward is None or status is None:
        return None
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, state["gray"], forward, None)
    if backward is None or back_status is None:
        return None

    valid = (
        (status.ravel() == 1)
        & (back_status.ravel() == 1)
        & (abs(backward - points).reshape(-1, 2).max(axis=1) < 2.0)
    )
    if int(valid.sum()) < 3:
        return None

    good_old = points[valid]
    good_new = forward[valid]
    transform, inliers = cv2.estimateAffinePartial2D(
        good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return None
    inliers = inliers.ravel().astype(bool)
    if int(inliers.sum()) < max(4, int(np.ceil(len(points) * .55))):
        return None
    scale = np.linalg.norm(transform[:, 0])
    if not .75 <= scale <= 1.4:
        return None
    x1, y1, x2, y2 = state["box"]
    spread = np.ptp(good_old[inliers].reshape(-1, 2), axis=0)
    if spread[0] < .2 * (x2-x1) or spread[1] < .2 * (y2-y1):
        return None
    center = transform @ np.array([(x1+x2)/2, (y1+y2)/2, 1.])
    half = scale * np.array([x2-x1, y2-y1]) / 2
    next_box = [*(center-half), *(center+half)]
    height, width = gray.shape
    if next_box[0] < 0 or next_box[1] < 0 or next_box[2] > width or next_box[3] > height:
        return None

    state.update(
        box=[float(value) for value in next_box],
        gray=gray,
        points=good_new[inliers].reshape(-1, 1, 2),
    )
    return state


def filter_detections(detections, shape, *, discarded=None):
    """Apply the worker's box filters. Optionally record each rejection."""
    import math

    height, width = shape[:2]
    clean = []
    for item in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        box = item['box']
        reason = None
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            reason = 'Coordenadas inválidas'
        else:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                reason = 'Caja degenerada o fuera del fotograma'
            elif covers_frame(box, width, height):
                reason = 'Cubre al menos el 95 % de ambas dimensiones'
            elif any(overlap(box, other['box']) > 0.6 for other in clean):
                reason = 'Duplicada: IoU > 0,60 con una caja de mayor o igual confianza'
        if reason is None:
            clean.append(item)
        elif discarded is not None:
            discarded.append(dict(item, reason=reason))
    return clean


def follow_target(state, frame, detections, cv2, *, now=None, track_visual=True):
    """Seed KLT automatically from YOLO; bridge only short, reliable gaps."""
    now = time.monotonic() if now is None else now
    if not track_visual:
        state = None
    clean = filter_detections(detections, frame.shape)

    had_target = state is not None
    predicted = None
    if state is not None and now - state.get('last_detection', now) <= 1.0:
        predicted = advance_manual_tracker(state, frame, cv2)
    elif state is not None:
        state = None
    reference = predicted or state
    best = None
    if reference is not None:
        associated = [d for d in clean if reference.get('track_id') is None
                      or d['track_id'] == reference['track_id'] or reference.get('source') == 'manual-flow']
        best = max(associated, key=lambda d: overlap(d['box'], reference['box']), default=None)
        if best is not None and overlap(best['box'], reference['box']) < 0.25:
            best = None
    if best is None and state is None:
        best = clean[0] if clean else None
    if best is not None:
        refreshed = initialize_manual_tracker(frame, best['box'], cv2) if track_visual else None
        state = refreshed
        if state is not None:
            state.update(confidence=best['confidence'], track_id=best['track_id'],
                         last_detection=now, source='auto-flow')
        best['selected'] = True
        return state, clean, 'Siguiendo'
    if predicted is not None:
        predicted.setdefault('last_detection', now)
        clean.append(dict(box=predicted['box'], confidence=predicted['confidence'],
                          track_id=predicted['track_id'], class_id=0,
                          class_name='Precinto', source=predicted.get('source', 'manual-flow'),
                          selected=True))
        return predicted, clean, 'Siguiendo'
    # Retain only the association anchor until expiry. Never publish its old box.
    return state, clean, 'Objetivo perdido' if had_target else 'Buscando precinto'


def flow_frame(frame, shape, cv2):
    height, width = shape[:2]
    scale = min(1., 480 / max(height, width))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    if frame.shape[:2] == (size[1], size[0]):
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def scale_box(box, x_scale, y_scale):
    return [float(value * (x_scale if i % 2 == 0 else y_scale))
            for i, value in enumerate(box)]


def appearance_patch(gray, box, cv2):
    x1, y1, x2, y2 = map(round, box)
    roi = gray[y1:y2, x1:x2]
    if min(roi.shape) < 4:
        return None
    return cv2.resize(roi, (24, 24), interpolation=cv2.INTER_AREA)


def advance_locked_flow(state, frame, shape, lock, stamp, cv2):
    if state is None:
        return None, [], 'Sin puntos KLT fiables'
    if not 0 < stamp - state['last_detection'] <= lock['max_age_s']:
        return None, [], 'Venció el plazo de reconfirmación YOLO'
    reduced = flow_frame(frame, shape, cv2)
    state = advance_manual_tracker(state, reduced, cv2)
    if state is None:
        return None, [], 'KLT sin consenso geométrico'
    patch = appearance_patch(state['gray'], state['box'], cv2)
    anchor = state['appearance']
    if (patch is None or anchor is None or patch.std() < 4 or anchor.std() < 4
            or not float(cv2.matchTemplate(patch, anchor, cv2.TM_CCOEFF_NORMED)[0, 0])
            >= lock['appearance_min']):
        return None, [], 'KLT perdió similitud de apariencia'
    detection = dict(
        box=scale_box(state['box'], shape[1]/reduced.shape[1], shape[0]/reduced.shape[0]),
        track_id=lock['track_id'], confidence=lock['confidence'],
        class_name='Precinto', source='auto-flow')
    return state, [detection], None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=YOLO_CONFIDENCE)
    args = parser.parse_args()

    try:
        weights = Path(args.weights).resolve()
        verify_v5(weights)
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        device = "0" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(sys.stderr):
            model = YOLO(str(weights))
            if model.task != "detect" or not model.names:
                raise ValueError(f"Checkpoint incompatible: task={model.task}, clases={model.names}")
        emit(
            {
                "type": "ready",
                "weights": str(weights),
                "imgsz": args.imgsz,
                "task": model.task,
                "names": model.names,
                "device": torch.cuda.get_device_name(0) if device == "0" else "cpu",
            }
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    stream_epoch = None
    generation = None
    manual_tracker = None
    seed_frame = None
    active_seed = None
    manual_track_counter = 0
    try:
        for raw_line in sys.stdin.buffer:
            request = {}
            try:
                request = json.loads(raw_line)
                request_type = request.get("type")
                if request_type == "shutdown":
                    break
                if request_type != "frame":
                    continue

                if (
                    request.get("reset")
                    or request.get("stream_epoch") != stream_epoch
                    or request.get("generation") != generation
                ):
                    model.predictor = None
                    manual_tracker = None
                    seed_frame = active_seed = None
                    stream_epoch = request.get("stream_epoch")
                    generation = request.get("generation")

                encoded = base64.b64decode(request["jpeg"], validate=True)
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("El JPEG del frame no se pudo decodificar")

                infer = request.get('infer', True)
                lock = request.get('lock')
                shape = request.get('shape', list(frame.shape))
                if lock is None and active_seed is not None:
                    manual_tracker = active_seed = None
                if lock is not None and lock['sequence'] != active_seed:
                    active_seed = lock['sequence']
                    manual_tracker = None
                    if seed_frame is not None and seed_frame[0] == active_seed:
                        seed = seed_frame[1]
                        box = scale_box(lock['box'], seed.shape[1]/shape[1], seed.shape[0]/shape[0])
                        manual_tracker = initialize_manual_tracker(seed, box, cv2)
                        if manual_tracker is not None:
                            manual_tracker['last_detection'] = seed_frame[2]
                            manual_tracker['appearance'] = appearance_patch(
                                manual_tracker['gray'], manual_tracker['box'], cv2)
                tracked, flow_reason = [], None
                if lock is not None:
                    manual_tracker, tracked, flow_reason = advance_locked_flow(
                        manual_tracker, frame, shape, lock, request['submitted_at'], cv2)
                if not infer:
                    emit(dict(type='result', sequence=request['sequence'], generation=generation,
                              stream_epoch=stream_epoch, submitted_at=request['submitted_at'],
                              shape=shape, detections=tracked, flow_only=True, flow_reason=flow_reason,
                              tracking_status='Siguiendo' if tracked else 'Sin posición KLT fiable'))
                    continue

                requested_roi = request.get("manual_roi")
                if requested_roi is not None:
                    manual_tracker = initialize_manual_tracker(frame, requested_roi, cv2)
                    if manual_tracker is not None:
                        manual_track_counter += 1
                        manual_tracker["track_id"] = f"M{manual_track_counter}"
                        manual_tracker["last_detection"] = request['submitted_at']
                        manual_tracker["source"] = "manual-flow"

                started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    result = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        imgsz=args.imgsz,
                        conf=args.confidence,
                        device=device,
                        verbose=False,
                    )[0]
                inference_ms = float(result.speed.get("inference", 0.0))
                detections = []
                for index, box in enumerate(result.boxes):
                    class_id = int(box.cls.item())
                    track_id = None
                    if result.boxes.id is not None:
                        track_id = int(result.boxes.id[index].item())
                    detections.append(
                        {
                            "box": [float(value) for value in box.xyxy[0].tolist()],
                            "confidence": float(box.conf.item()),
                            "class_id": class_id,
                            "class_name": str(model.names[class_id]),
                            "track_id": track_id,
                            "source": "yolo",
                            "selected": False,
                        }
                    )

                # With a locked target only the UI-accepted YOLO box may reseed KLT.
                tracker, detections, tracking_status = follow_target(
                    manual_tracker if lock is None else None, frame, detections, cv2,
                    now=request['submitted_at'],
                    track_visual=lock is None and bool(request.get('track_visual', True)),
                )
                if lock is None:
                    manual_tracker = tracker
                seed_frame = (request['sequence'], flow_frame(frame, shape, cv2), request['submitted_at'])
                height, width = frame.shape[:2]
                for detection in detections:
                    x1, y1, x2, y2 = detection['box']
                    quality = image_quality(frame, (x1 / width, y1 / height, x2 / width, y2 / height))
                    detection['sharpness'] = quality['sharpness']
                detections.extend(tracked)

                emit(
                    {
                        "type": "result",
                        "sequence": request["sequence"],
                        "generation": request["generation"],
                        "stream_epoch": stream_epoch,
                        "submitted_at": request["submitted_at"],
                        "shape": list(frame.shape),
                        "detections": detections,
                        "flow_reason": flow_reason,
                        "tracking_status": tracking_status,
                        "inference_ms": inference_ms,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                    }
                )
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit(
                    {
                        "type": "error",
                        "sequence": request.get("sequence"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        model.predictor = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

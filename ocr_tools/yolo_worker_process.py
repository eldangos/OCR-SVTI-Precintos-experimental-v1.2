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

os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "False"


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
    delta = (good_new - good_old).reshape(-1, 2)
    motion = delta.mean(axis=0)
    x1, y1, x2, y2 = state["box"]
    next_box = [x1 + motion[0], y1 + motion[1], x2 + motion[0], y2 + motion[1]]
    height, width = gray.shape
    if next_box[0] < 0 or next_box[1] < 0 or next_box[2] > width or next_box[3] > height:
        return None

    mask = cv2.rectangle(
        np.zeros((height, width), dtype=np.uint8),
        (int(next_box[0]), int(next_box[1])),
        (int(next_box[2]), int(next_box[3])),
        255,
        thickness=-1,
    )
    refreshed = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=3,
        mask=mask,
    )
    state.update(
        box=[float(value) for value in next_box],
        gray=gray,
        points=refreshed if refreshed is not None and len(refreshed) >= 3 else good_new.reshape(-1, 1, 2),
    )
    return state


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        device = "0" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(sys.stderr):
            model = YOLO(args.weights)
        emit(
            {
                "type": "ready",
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
                    stream_epoch = request.get("stream_epoch")
                    generation = request.get("generation")

                encoded = base64.b64decode(request["jpeg"], validate=True)
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("El JPEG del frame no se pudo decodificar")

                requested_roi = request.get("manual_roi")
                if requested_roi is not None:
                    manual_tracker = initialize_manual_tracker(frame, requested_roi, cv2)
                    if manual_tracker is not None:
                        manual_track_counter += 1
                        manual_tracker["track_id"] = f"M{manual_track_counter}"

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

                if manual_tracker is not None:
                    best = max(
                        detections,
                        key=lambda item: overlap(item["box"], manual_tracker["box"]),
                        default=None,
                    )
                    if best is not None and overlap(best["box"], manual_tracker["box"]) >= 0.08:
                        refreshed_tracker = initialize_manual_tracker(
                            frame,
                            best["box"],
                            cv2,
                        )
                        if refreshed_tracker is not None:
                            manual_tracker = refreshed_tracker
                        manual_tracker.update(
                            box=best["box"],
                            confidence=best["confidence"],
                            track_id=best["track_id"],
                        )
                        best["selected"] = True
                    else:
                        manual_tracker = advance_manual_tracker(manual_tracker, frame, cv2)

                    if manual_tracker is not None and not any(item["selected"] for item in detections):
                        manual_tracker["selected_at"] = time.monotonic()
                        detections.append(
                            {
                                "box": manual_tracker["box"],
                                "confidence": manual_tracker["confidence"],
                                "class_id": 0,
                                "class_name": str(model.names[0]),
                                "track_id": manual_tracker["track_id"],
                                "source": "manual-flow",
                                "selected": True,
                            }
                        )

                emit(
                    {
                        "type": "result",
                        "sequence": request["sequence"],
                        "generation": request["generation"],
                        "stream_epoch": stream_epoch,
                        "submitted_at": request["submitted_at"],
                        "shape": list(frame.shape),
                        "detections": detections,
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
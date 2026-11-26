"""Compare predict and fresh ByteTrack on one saved channel 102 PNG, offline."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modulos.yolo_runtime import YOLO_CONFIDENCE, YOLO_IMAGE_SIZE, YOLO_WEIGHTS_PATH, verify_v5
from ocr_tools.yolo_worker_process import filter_detections


def compare(method, frame, device, yolo):
    model = yolo(str(YOLO_WEIGHTS_PATH))
    params = dict(imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, device=device, verbose=False)
    tensor_shape = None

    def record_resolution(predictor):
        nonlocal tensor_shape
        preprocess = predictor.preprocess

        def preprocess_with_shape(images):
            nonlocal tensor_shape
            tensor = preprocess(images)
            tensor_shape = list(tensor.shape[-2:])
            return tensor

        predictor.preprocess = preprocess_with_shape

    model.add_callback('on_predict_start', record_resolution)
    started = time.perf_counter()
    if method == 'track':
        result = model.track(source=frame.copy(), persist=True, tracker='bytetrack.yaml', **params)[0]
    else:
        result = model.predict(source=frame.copy(), **params)[0]
    total_ms = (time.perf_counter() - started) * 1000
    boxes = []
    if result.boxes is not None:
        ids = result.boxes.id
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            boxes.append(dict(box=box.xyxy[0].tolist(), confidence=float(box.conf.item()),
                              class_id=class_id, class_name=str(model.names[class_id]),
                              track_id=int(ids[i].item()) if ids is not None else None))
    discarded = []
    kept = filter_detections(boxes, frame.shape, discarded=discarded)
    return dict(returned_count=len(boxes), returned_boxes=boxes, kept_count=len(kept),
                discarded_count=len(discarded), discarded=discarded,
                inference_ms=result.speed.get('inference'), total_call_ms=total_ms,
                model_input_height_width=tensor_shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame', type=Path, help='Un PNG real guardado del canal 102')
    args = parser.parse_args()
    if not args.frame.is_file() or args.frame.suffix.lower() != '.png':
        parser.error('Selecciona un PNG local del canal 102')
    # Require local assets so Ultralytics cannot resolve missing weights by name.
    try:
        verify_v5(YOLO_WEIGHTS_PATH)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import torch
        from ultralytics import YOLO

        frame = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error('No se pudo decodificar el PNG')
        device = '0' if torch.cuda.is_available() else 'cpu'
        results = {method: compare(method, frame, device, YOLO)
                   for method in ('predict', 'track')}
    print(json.dumps(dict(
        frame=str(args.frame), source_height_width=list(frame.shape[:2]),
        weights=str(YOLO_WEIGHTS_PATH), imgsz=YOLO_IMAGE_SIZE,
        confidence_threshold=YOLO_CONFIDENCE, device=device,
        note='Instancias nuevas, sin historial de ByteTrack. Filtros del worker; sin adquisición temporal ni KLT.',
        **results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

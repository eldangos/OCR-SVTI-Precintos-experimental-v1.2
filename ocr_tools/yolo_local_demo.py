"""Local YOLO26 image/video probe; never connects to RTSP or cloud inference.

Run with the isolated environment, for example:
    python ocr_tools/yolo_local_demo.py --image capturas_precintos/img1.png
    python ocr_tools/yolo_local_demo.py --make-video capturas_precintos/img1.png
    python ocr_tools/yolo_local_demo.py --video C:/path/to/recording.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "modelos" / "SeaAIScan_V3_2042.pt"
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "SeaAIScan-YOLO-Results"
VIDEO_SIZE = (704, 576)


def selected_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def make_synthetic_video(
    image_path: Path,
    output_path: Path,
    *,
    frame_count: int = 45,
    fps: float = 15.0,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    if frame_count < 2 or fps <= 0:
        raise ValueError("El video necesita al menos 2 frames y FPS positivos")

    width, height = VIDEO_SIZE
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    vertical_pad = (height - resized.shape[0]) // 2
    horizontal_pad = (width - resized.shape[1]) // 2
    base = cv2.copyMakeBorder(
        resized,
        vertical_pad,
        height - resized.shape[0] - vertical_pad,
        horizontal_pad,
        width - resized.shape[1] - horizontal_pad,
        cv2.BORDER_REPLICATE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        VIDEO_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV no pudo crear el video: {output_path}")

    try:
        for index in range(frame_count):
            phase = 2 * math.pi * index / frame_count
            offset_x = round(12 * math.sin(phase))
            offset_y = round(7 * math.cos(phase))
            transform = np.float32(
                [[1, 0, offset_x], [0, 1, offset_y]]
            )
            frame = cv2.warpAffine(
                base,
                transform,
                VIDEO_SIZE,
                borderMode=cv2.BORDER_REFLECT,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def infer_image(
    model: YOLO,
    image_path: Path,
    output_path: Path,
    *,
    device: str,
    image_size: int,
    confidence: float,
) -> int:
    results = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
    )
    result = results[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result.plot()):
        raise OSError(f"No se pudo guardar la anotación: {output_path}")

    print(
        f"IMAGE source={image_path} output={output_path} "
        f"shape={result.orig_shape} count={len(result.boxes)} speed_ms={result.speed}"
    )
    for box in result.boxes:
        class_id = int(box.cls.item())
        print(
            f"  class={model.names[class_id]} "
            f"confidence={float(box.conf.item()):.4f} "
            f"xyxy={[round(value, 1) for value in box.xyxy[0].tolist()]}"
        )
    return len(result.boxes)


def track_video(
    model: YOLO,
    video_path: Path,
    output_path: Path,
    *,
    device: str,
    image_size: int,
    confidence: float,
) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"OpenCV no pudo abrir el video: {video_path}")

    fps_in = capture.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in if math.isfinite(fps_in) and fps_in > 0 else 15.0
    capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    frame_count = 0
    detected_frames = 0
    tracked_ids: set[int] = set()
    inference_seconds = 0.0
    run_started = perf_counter()
    results = model.track(
        source=str(video_path),
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=image_size,
        conf=confidence,
        device=device,
        stream=True,
        verbose=False,
    )
    try:
        for result in results:
            frame_count += 1
            inference_seconds += result.speed.get("inference", 0.0) / 1000
            height, width = result.orig_img.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps_out,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"OpenCV no pudo crear el video anotado: {output_path}"
                    )
            if len(result.boxes):
                detected_frames += 1
                ids = result.boxes.id
                if ids is not None:
                    tracked_ids.update(int(value) for value in ids.tolist())
            writer.write(result.plot())
    finally:
        results.close()
        if writer is not None:
            writer.release()

    elapsed = perf_counter() - run_started
    summary: dict[str, float | int] = {
        "frames": frame_count,
        "detected_frames": detected_frames,
        "unique_track_ids": len(tracked_ids),
        "elapsed_s": elapsed,
        "wall_fps": frame_count / elapsed if elapsed else 0.0,
        "inference_fps": frame_count / inference_seconds if inference_seconds else 0.0,
        "source_fps": fps_in,
    }
    print(f"VIDEO source={video_path} output={output_path} summary={summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--make-video", type=Path, metavar="IMAGE")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu o índice CUDA como 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"No se encuentra el modelo: {args.weights}")

    device = selected_device(args.device)
    print(
        f"RUNTIME torch={torch.__version__} ultralytics="
        f"{__import__('ultralytics').__version__} device={device}"
    )

    if args.make_video:
        output = args.output_dir / "synthetic_seal_pan.avi"
        make_synthetic_video(
            args.make_video,
            output,
            frame_count=args.frames,
            fps=args.fps,
        )
        print(f"SYNTHETIC_VIDEO output={output} frames={args.frames} fps={args.fps}")

    if args.image is None and args.video is None:
        return 0

    load_started = perf_counter()
    model = YOLO(str(args.weights))
    print(
        f"MODEL loaded_s={perf_counter() - load_started:.3f} "
        f"task={model.task} names={model.names}"
    )

    detections = None
    if args.image:
        image_output = args.output_dir / f"{args.image.stem}_detected.jpg"
        detections = infer_image(
            model,
            args.image,
            image_output,
            device=device,
            image_size=args.imgsz,
            confidence=args.confidence,
        )
    if args.video:
        video_output = args.output_dir / f"{args.video.stem}_tracked.avi"
        track_video(
            model,
            args.video,
            video_output,
            device=device,
            image_size=args.imgsz,
            confidence=args.confidence,
        )
    return 0 if detections is None or detections > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
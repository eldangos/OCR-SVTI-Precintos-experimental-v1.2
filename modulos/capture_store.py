"""Nombres cortos para capturas, conservando fecha y evidencia."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class CapturePaths:
    image: Path
    report: Path
    metadata: Path
    captured_at: str


def reservar_captura(root: Path, captured_at: datetime) -> CapturePaths:
    """Reserva un número único dentro de la carpeta del día."""
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("La fecha de captura debe incluir zona horaria")

    day = Path(root) / captured_at.date().isoformat()
    day.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^P(\d+)(?:_|\.)")
    numbers = [
        int(match.group(1))
        for path in day.iterdir()
        if (match := pattern.match(path.name))
    ]

    number = max(numbers, default=0) + 1
    timestamp = captured_at.isoformat(timespec="microseconds")

    while True:
        stem = f"P{number:04d}"
        image = day / f"{stem}.png"
        report = day / f"{stem}_ocr.json"
        metadata = day / f"{stem}_meta.json"

        if image.exists() or report.exists():
            number += 1
            continue

        try:
            stream = metadata.open("x", encoding="utf-8")
        except FileExistsError:
            number += 1
            continue

        with stream:
            json.dump(
                {
                    "capture_id": stem,
                    "captured_at": timestamp,
                    "image": image.name,
                    "report": report.name,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )

        return CapturePaths(image, report, metadata, timestamp)


def guardar_png_exclusivo(path, frame):
    """Guarda el PNG completo sin sobrescribir una imagen existente."""
    import cv2

    encoded, data = cv2.imencode(".png", frame)

    if not encoded:
        raise OSError("No se pudo codificar la evidencia visual")

    with open(path, "xb") as stream:
        stream.write(data.tobytes())
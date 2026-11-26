"""Asynchronous bridge to the isolated local YOLO Python environment."""

from __future__ import annotations

import base64
import json
import time

import cv2
from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from modulos.yolo_runtime import (
    PROJECT_ROOT,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_PROCESS_PATH,
    YOLO_PYTHON_PATH,
    YOLO_WEIGHTS_PATH,
)


class YoloWorker(QObject):
    status_changed = Signal(str)
    result_ready = Signal(object)
    failed = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent=None, *, process=None):
        super().__init__(parent)
        self._process = process or QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.started.connect(lambda: self.status_changed.emit("YOLO iniciando"))
        self._process.errorOccurred.connect(self._process_error)
        self._process.finished.connect(self._process_finished)
        self._stdout_buffer = bytearray()
        self._pending = None
        self._pending_roi = None
        self._busy = False
        self._ready = False
        self._stopping = False
        self._stream_epoch = None
        self._latest_sequence = -1
        self._enabled = False
        self._recovering = False
        self._restarts = []
        self._deadline = None
        self._inflight = None
        self._frame_errors = 0
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_health)
        self._watchdog.start()

    def _check_health(self):
        if self._enabled and self._deadline is not None and time.monotonic() > self._deadline:
            phase = 'respuesta de frame' if self._busy else 'inicio del modelo'
            self._recover(f"Tiempo de respuesta YOLO agotado ({phase})")

    def _recover(self, reason):
        if not self._enabled or self._recovering:
            return
        now = time.monotonic()
        self._restarts = [stamp for stamp in self._restarts if now - stamp < 120]
        self._deadline = None
        self._ready = self._busy = False
        self._pending = self._pending_roi = None
        self.log_message.emit(reason)
        if len(self._restarts) >= 3:
            self._enabled = False
            self._stopping = True
            self.failed.emit("YOLO detenido: tres recuperaciones en 120 s; revisar registro")
            if self.is_running():
                self._process.kill()
            return
        self._restarts.append(now)
        self._recovering = True
        self.status_changed.emit("YOLO iniciando · recuperando proceso sin respuesta")
        if self.is_running():
            self._process.kill()
        else:
            self._process_finished(-1, None)

    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def is_ready(self) -> bool:
        return self._ready and self.is_running()

    def accepts_frame(self) -> bool:
        return self.is_ready() and not (self._busy or self._stopping or self._recovering)

    def activity(self):
        if self._recovering:
            return 'Recuperando proceso'
        if self._busy and self._ready:
            return 'Procesando fotograma'
        if self.is_ready():
            return 'Detector listo'
        return 'YOLO iniciando' if self.is_running() else 'Worker detenido'

    def start(self) -> None:
        if self.is_running():
            if self._stopping:
                self._enabled = self._recovering = True
            return
        if not YOLO_PYTHON_PATH.is_file():
            self.failed.emit(f"No se encuentra el Python YOLO: {YOLO_PYTHON_PATH}")
            return
        if not YOLO_PROCESS_PATH.is_file():
            self.failed.emit(f"No se encuentra el worker YOLO: {YOLO_PROCESS_PATH}")
            return
        if not YOLO_WEIGHTS_PATH.is_file():
            self.failed.emit(f"No se encuentran los pesos YOLO: {YOLO_WEIGHTS_PATH}")
            return

        if not self._enabled:
            self._restarts.clear()
        self._enabled = True
        self._recovering = False
        self._stopping = False
        self._ready = False
        self._busy = False
        self._pending = None
        self._sent_epoch = None
        self._inflight = None
        self._frame_errors = 0
        self._deadline = time.monotonic() + 60
        self._stdout_buffer.clear()
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.start(
            str(YOLO_PYTHON_PATH),
            [
                "-u",
                str(YOLO_PROCESS_PATH),
                "--weights",
                str(YOLO_WEIGHTS_PATH),
                "--imgsz",
                str(YOLO_IMAGE_SIZE),
                "--confidence",
                str(YOLO_CONFIDENCE),
            ],
        )
        self.status_changed.emit("YOLO iniciando")

    def submit_frame(self, frame, *, sequence, submitted_at, generation, stream_epoch,
                     track_visual=True, infer=True, lock=None) -> None:
        if not self.is_running() or self._stopping:
            return
        if stream_epoch != self._stream_epoch:
            self._stream_epoch = stream_epoch
        self._latest_sequence = sequence
        self._pending = {
            "frame": frame,
            "sequence": int(sequence),
            "submitted_at": float(submitted_at),
            "generation": int(generation),
            "stream_epoch": int(stream_epoch),
            "track_visual": bool(track_visual),
            "infer": infer,
            "lock": lock,
        }
        self._send_pending()

    def select_roi(self, box) -> None:
        if not self._stopping:
            self._pending_roi = [float(value) for value in box]

    def stop(self) -> None:
        self._enabled = False
        self._recovering = False
        self._deadline = None
        if not self.is_running():
            self._ready = False
            self._pending = None
            return
        self._stopping = True
        self._ready = False
        self._pending = None
        self._pending_roi = None
        self._process.write(b'{"type":"shutdown"}\n')
        self._process.closeWriteChannel()
        QTimer.singleShot(2500, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._stopping and self.is_running():
            self._process.kill()

    def _send_pending(self) -> None:
        if not self._ready or self._busy or self._pending is None or not self.is_running():
            return
        pending = self._pending
        self._pending = None
        frame = pending["frame"]
        if not pending["infer"]:
            h, w = frame.shape[:2]
            scale = min(1., 480 / max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))),
                                   interpolation=cv2.INTER_AREA)
        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not encoded:
            self.failed.emit("No se pudo codificar el fotograma para YOLO")
            return

        message = {
            "type": "frame",
            "sequence": pending["sequence"],
            "submitted_at": pending["submitted_at"],
            "generation": pending["generation"],
            "stream_epoch": pending["stream_epoch"],
            "track_visual": pending["track_visual"],
            "infer": pending["infer"],
            "lock": pending["lock"],
            "shape": list(pending["frame"].shape),
            "jpeg": base64.b64encode(jpeg).decode("ascii"),
        }
        if self._pending_roi is not None and pending["infer"] and pending["lock"] is None:
            message["manual_roi"] = self._pending_roi
            self._pending_roi = None
        if pending["stream_epoch"] != getattr(self, "_sent_epoch", None):
            message["reset"] = True
            self._sent_epoch = pending["stream_epoch"]

        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        if self._process.write(payload) != len(payload):
            self._recover("No se pudo enviar el fotograma completo al proceso YOLO")
            return
        self._busy = True
        self._inflight = (message['generation'], message['stream_epoch'], message['sequence'])
        self._deadline = time.monotonic() + 10

    def _read_stdout(self) -> None:
        self._stdout_buffer.extend(bytes(self._process.readAllStandardOutput()))
        if len(self._stdout_buffer) > 1_000_000:
            self._stdout_buffer.clear()
            self._recover('Salida YOLO sin protocolo válido o demasiado grande')
            return
        while b"\n" in self._stdout_buffer:
            line, _, rest = self._stdout_buffer.partition(b"\n")
            self._stdout_buffer = bytearray(rest)
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.log_message.emit(line.decode("utf-8", errors="replace")[:300])
                continue

            if not isinstance(message, dict):
                self.log_message.emit('Salida YOLO ignorada: se esperaba un objeto JSON')
                continue
            kind = message.get("type")
            if self._stopping or self._recovering:
                continue
            if kind == "ready":
                self._ready = True
                self._deadline = None
                self.log_message.emit(
                    f"Modelo local: {message.get('weights')} · imgsz={message.get('imgsz')} "
                    f"· {message.get('task')} · {message.get('names')}"
                )
                self.status_changed.emit(
                    f"YOLO listo · {message.get('device', 'device')} · {message.get('names', {})}"
                )
                self._send_pending()
            elif kind == "result":
                if not self._busy or self._inflight != tuple(message.get(k) for k in ('generation', 'stream_epoch', 'sequence')):
                    continue
                self._inflight = None
                self._frame_errors = 0
                self._busy = False
                self._deadline = None
                self.result_ready.emit(message)
                self._send_pending()
            elif kind in ("error", "fatal"):
                self._busy = False
                self._deadline = None
                error = str(message.get("error", "Error de inferencia YOLO"))
                self.failed.emit(error)
                if kind == "fatal":
                    self._recover(error)
                else:
                    self._frame_errors += 1
                    if self._frame_errors >= 3:
                        self._recover('Tres errores consecutivos de inferencia: ' + error)
                    else:
                        self._send_pending()

    def _read_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.log_message.emit(line[:400])

    def _process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._ready = False
            self._busy = False
            self._recover("No se pudo iniciar el entorno local de YOLO")

    def _process_finished(self, exit_code, exit_status) -> None:
        self._ready = False
        self._busy = False
        self._pending = None
        self._deadline = None
        if self._recovering and self._enabled:
            QTimer.singleShot(1000, self._restart)
            return
        if self._enabled and not self._stopping:
            self._recover(f"Proceso YOLO terminó: {exit_code}")
            return
        self.status_changed.emit("YOLO detenido")

    def _restart(self):
        if self._enabled and self._recovering:
            self.start()

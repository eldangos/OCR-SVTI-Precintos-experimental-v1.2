"""Simulated YOLO IPC and manual video ROI tests; no camera or model required."""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QProcess, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from modulos.yolo_worker import YoloWorker
from ui.video_widget import VideoWidget

_PROCESS_PATH = Path(__file__).resolve().parents[1] / "ocr_tools" / "yolo_worker_process.py"
_PROCESS_SPEC = importlib.util.spec_from_file_location("yolo_worker_process", _PROCESS_PATH)
_PROCESS_MODULE = importlib.util.module_from_spec(_PROCESS_SPEC)
_PROCESS_SPEC.loader.exec_module(_PROCESS_MODULE)


class FakeProcess(QObject):
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    started = Signal()
    errorOccurred = Signal(object)
    finished = Signal(int, object)

    def __init__(self):
        super().__init__()
        self.writes = []
        self.stdout = bytearray()
        self.stderr = bytearray()
        self._state = QProcess.ProcessState.Running

    def state(self):
        return self._state

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def readAllStandardOutput(self):
        data = bytes(self.stdout)
        self.stdout.clear()
        return data

    def readAllStandardError(self):
        data = bytes(self.stderr)
        self.stderr.clear()
        return data

    def feed_json(self, message):
        self.stdout.extend(json.dumps(message).encode("utf-8") + b"\n")
        self.readyReadStandardOutput.emit()


class YoloWorkerTests(unittest.TestCase):
    def test_inflight_frame_is_not_queued_and_only_latest_pending_frame_is_sent(self):
        process = FakeProcess()
        worker = YoloWorker(process=process)
        worker._ready = True
        frame = np.zeros((72, 96, 3), dtype=np.uint8)

        worker.submit_frame(
            frame, sequence=10, submitted_at=1.0, generation=4, stream_epoch=8
        )
        worker.submit_frame(
            frame, sequence=11, submitted_at=1.1, generation=4, stream_epoch=8
        )
        worker.submit_frame(
            frame, sequence=12, submitted_at=1.2, generation=4, stream_epoch=8
        )

        self.assertEqual(len(process.writes), 1)
        self.assertEqual(worker._pending["sequence"], 12)
        sent_first = json.loads(process.writes[0])
        self.assertEqual(sent_first["sequence"], 10)
        self.assertEqual(sent_first["generation"], 4)
        self.assertTrue(sent_first["reset"])

        results = []
        worker.result_ready.connect(results.append)
        process.feed_json(
            {
                "type": "result",
                "sequence": 10,
                "generation": 4,
                "stream_epoch": 8,
                "submitted_at": 1.0,
                "detections": [],
            }
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(process.writes), 2)
        sent_latest = json.loads(process.writes[1])
        self.assertEqual(sent_latest["sequence"], 12)
        self.assertEqual(sent_latest["generation"], 4)
        self.assertNotIn("reset", sent_latest)


class VideoRoiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_manual_roi_drag_maps_viewport_to_original_frame_coordinates(self):
        widget = VideoWidget()
        widget.resize(720, 405)
        widget.mostrar_frame(np.zeros((200, 360, 3), dtype=np.uint8))
        widget.habilitar_seleccion_roi(True)
        selected = []
        widget.roi_selected.connect(selected.append)
        widget.show()
        self.app.processEvents()

        QTest.mousePress(
            widget,
            Qt.MouseButton.LeftButton,
            pos=QPoint(100, 50),
        )
        QTest.mouseRelease(
            widget,
            Qt.MouseButton.LeftButton,
            pos=QPoint(300, 200),
        )

        self.assertEqual(len(selected), 1)
        x1, y1, x2, y2 = selected[0]
        self.assertAlmostEqual(x1, 50, delta=2)
        self.assertAlmostEqual(y1, 24, delta=2)
        self.assertAlmostEqual(x2, 150, delta=2)
        self.assertAlmostEqual(y2, 99, delta=2)
        widget.close()

    def test_selected_roi_follows_synthetic_translation_with_optical_flow(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for y in range(32, 90, 12):
            for x in range(42, 105, 12):
                cv2.rectangle(frame, (x, y), (x + 5, y + 5), (255, 255, 255), -1)

        tracker = _PROCESS_MODULE.initialize_manual_tracker(
            frame,
            [35, 25, 115, 100],
            cv2,
        )
        self.assertIsNotNone(tracker)

        translated = cv2.warpAffine(
            frame,
            np.float32([[1, 0, 4], [0, 1, 3]]),
            (160, 120),
            borderMode=cv2.BORDER_REFLECT,
        )
        tracked = _PROCESS_MODULE.advance_manual_tracker(tracker, translated, cv2)

        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(tracked['box'][0], 39, delta=2)
        self.assertAlmostEqual(tracked['box'][1], 28, delta=2)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
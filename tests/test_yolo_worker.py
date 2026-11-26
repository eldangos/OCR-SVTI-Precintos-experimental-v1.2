"""Simulated YOLO IPC and manual video ROI tests; no camera or model required."""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import time

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

    def kill(self):
        self._state = QProcess.ProcessState.NotRunning
        self.finished.emit(-1, None)

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
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_timeout_kills_worker_and_schedules_bounded_recovery(self):
        process = FakeProcess()
        worker = YoloWorker(process=process)
        worker._enabled = worker._busy = True
        worker._deadline = time.monotonic() - 1
        with patch('modulos.yolo_worker.QTimer.singleShot') as restart:
            worker._check_health()
            self.assertFalse(worker.is_running())
            self.assertTrue(worker._recovering)
            restart.assert_called_once_with(1000, worker._restart)
        worker._restarts = [time.monotonic()] * 3
        worker._recovering = False
        errors = []
        worker.failed.connect(errors.append)
        worker._recover('fixture')
        self.assertFalse(worker._enabled)
        self.assertEqual(len(errors), 1)
        worker._restart()
        self.assertFalse(worker.is_running())

    def test_empty_result_is_a_healthy_reply(self):
        process = FakeProcess()
        worker = YoloWorker(process=process)
        worker._enabled = worker._busy = True
        worker._deadline = time.monotonic() + 1
        results = []
        worker.result_ready.connect(results.append)
        worker._inflight = (1, 2, 3)
        process.feed_json({'type': 'result', 'generation': 1, 'stream_epoch': 2, 'sequence': 2, 'detections': []})
        self.assertTrue(worker._busy)
        process.feed_json({'type': 'result', 'generation': 1, 'stream_epoch': 2, 'sequence': 3, 'detections': []})
        self.assertFalse(worker._busy)
        self.assertIsNone(worker._deadline)
        self.assertEqual(results[0]['detections'], [])
        self.assertEqual(worker._restarts, [])

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

    def test_automatic_flow_bridges_gap_then_expires_and_reacquires(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for y in range(32, 90, 12):
            for x in range(42, 105, 12):
                cv2.rectangle(frame, (x, y), (x + 5, y + 5), (255, 255, 255), -1)
        detection = dict(box=[35, 25, 115, 100], confidence=.8, track_id=7,
                         class_id=0, class_name='Precinto', source='yolo')
        follow = _PROCESS_MODULE.follow_target
        state, boxes, status = follow(None, frame, [detection], cv2, now=1)
        self.assertIsNotNone(state)
        self.assertEqual(status, 'Siguiendo')
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 3]]), (160, 120))
        state, boxes, status = follow(state, moved, [], cv2, now=1.1)
        self.assertEqual(boxes[0]['source'], 'auto-flow')
        self.assertEqual(boxes[0]['track_id'], 7)
        self.assertAlmostEqual(boxes[0]['box'][0], 39, delta=2)
        state, boxes, status = follow(state, moved, [], cv2, now=2.2)
        self.assertIsNone(state)
        self.assertEqual(status, 'Objetivo perdido')
        state, boxes, status = follow(state, frame, [detection], cv2, now=2.3)
        self.assertIsNotNone(state)
        self.assertEqual(status, 'Siguiendo')

    def test_oversized_duplicate_and_invalid_boxes_are_filtered(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        def detection(box, confidence=.8):
            return dict(box=box, confidence=confidence, track_id=1)
        _, boxes, _ = _PROCESS_MODULE.follow_target(None, frame, [
            detection([0, 0, 160, 120]), detection([10, 10, 40, 40]),
            detection([11, 11, 41, 41], .7), detection([float('nan'), 0, 1, 1]),
        ], cv2)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['box'], [10, 10, 40, 40])


if __name__ == "__main__":
    unittest.main()

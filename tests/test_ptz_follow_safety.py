"""Only local synthetic images and a fake PTZ transport are used."""
import threading
import unittest

import cv2
import numpy as np
from modulos.ptz_worker import PtzWorker
from ocr_tools.yolo_worker_process import follow_target
from modulos.yolo_quality import image_quality


class TrackingSafetyTests(unittest.TestCase):
    def test_zoom_and_translation_between_model_observations(self):
        rng = np.random.default_rng(21)
        frame = np.zeros((160, 200, 3), np.uint8)
        frame[50:110, 70:130] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
        d = dict(box=[70, 50, 130, 110], confidence=.64, track_id=6, source='yolo')
        state, _, _ = follow_target(None, frame, [d], cv2, now=1)
        transform = cv2.getRotationMatrix2D((100, 80), 0, 1.15)
        transform[:, 2] += [3, 2]
        moved = cv2.warpAffine(frame, transform, (200, 160))
        state, boxes, _ = follow_target(state, moved, [], cv2, now=1.4)
        self.assertEqual((boxes[0]['source'], boxes[0]['track_id']), ('auto-flow', 6))
        x1, _, x2, _ = boxes[0]['box']
        self.assertAlmostEqual(x2-x1, 69, delta=3)
        self.assertAlmostEqual((x1+x2)/2, 103, delta=2)
        self.assertGreater(image_quality(moved, (.38, .34, .62, .66))['sharpness'], 45)
        _, boxes, status = follow_target(state, moved, [], cv2, now=2.2)
        self.assertFalse(boxes)
        self.assertEqual(status, 'Objetivo perdido')

    def test_worker_sends_one_zoom_then_stop_without_ui_timer(self):
        calls, stopped = [], threading.Event()
        class SimulatedCamera:
            def mover_camara(self, address, **command):
                calls.append(command)
                return True

            def detener_ptz(self, address):
                calls.append(dict(pan=0, tilt=0, zoom=0))
                stopped.set()
                return True

            def cerrar(self):
                pass
        worker = PtzWorker()
        worker.configurar(SimulatedCamera(), 'simulado')
        worker.enviar_pulso(zoom=8, duration_s=.07)
        worker.start()
        try:
            self.assertTrue(stopped.wait(2))
            self.assertEqual([c['zoom'] for c in calls], [8, 0])
        finally:
            worker.detener_hilo()
            self.assertTrue(worker.wait(2000))

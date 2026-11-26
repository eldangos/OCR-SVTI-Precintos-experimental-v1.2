"""Focused offline tests using synthetic images and simulated controller callbacks."""
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from modulos.auto_ptz import AutoPtzController
from modulos.yolo_vigilance import VigilanceController
from ocr_tools.yolo_worker_process import advance_locked_flow, appearance_patch, initialize_manual_tracker


class HybridTrackingTest(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.seq = 0
        self.shape = [160, 200, 3]
        self.gate = VigilanceController(clock=lambda: self.now, track_interval_s=.15, correct_interval_s=.65)
        self.gate.set_context(1, 1)
        self.physical = Mock(side_effect=AssertionError('Unexpected movement or capture'))
        self.controller = AutoPtzController(self.gate, pulse=self.physical, stop=self.physical,
            focus=self.physical, capture=self.physical, focus_pending=lambda: False,
            available=lambda: True, observe_only=True, clock=lambda: self.now)
        self.controller.enable()
        self.box = [70, 50, 130, 110]
        self.image = np.zeros(self.shape, np.uint8)
        texture = np.random.default_rng(21).integers(0, 255, (60, 60), dtype=np.uint8)
        self.image[50:110, 70:130] = texture[:, :, None]
        self.detection = dict(box=self.box, confidence=.90, track_id=6, class_name='Precinto', source='yolo')
        for _ in range(3):
            self.packet([self.detection], dt=.5)
        self.assertTrue(self.controller.tracking)
        self.seed = self.controller.flow_seed
        self.flow = initialize_manual_tracker(self.image, self.box, cv2)
        self.assertIsNotNone(self.flow)
        self.flow['last_detection'] = self.now
        self.flow['appearance'] = appearance_patch(self.flow['gray'], self.flow['box'], cv2)

    def packet(self, detections, *, dt=.1, flow_only=False, flow_reason=None):
        self.now += dt
        self.seq += 1
        self.controller.observe(dict(generation=1, stream_epoch=1, sequence=self.seq,
            submitted_at=self.now, shape=self.shape, detections=detections,
            flow_only=flow_only, flow_reason=flow_reason))
        self.controller.tick()

    def flow_packet(self, frame, *, infer=False):
        self.flow, detections, reason = advance_locked_flow(
            self.flow, frame, self.shape, self.seed, self.now + .1, cv2)
        self.packet(detections, flow_only=not infer, flow_reason=reason)
        return detections

    def test_stationary_with_detection_misses_and_lower_confidence_reconfirmation(self):
        original_stamp = self.controller._last_yolo
        for i in range(18):
            self.assertTrue(self.flow_packet(self.image, infer=i % 4 == 0))
        self.assertGreater(self.now - original_stamp, 1.)
        self.assertEqual(self.controller.visual_target()[0]['source'], 'auto-flow')
        self.assertIsNone(self.controller.decision)
        self.assertTrue(self.controller.recovering)
        self.packet([dict(self.detection, confidence=.40)])
        self.assertEqual(self.controller.target['source'], 'yolo')
        self.assertGreater(self.controller._last_yolo, original_stamp)
        self.assertFalse(self.controller.recovering)
        self.physical.assert_not_called()

    def test_moving_object_updates_box_but_keeps_identity(self):
        moved = cv2.warpAffine(self.image, np.float32([[1, 0, 4], [0, 1, 3]]), (200, 160))
        self.assertTrue(self.flow_packet(moved))
        target = self.controller.visual_target()[0]
        self.assertEqual(target['track_id'], 6)
        self.assertAlmostEqual(target['box'][0], 74, delta=2)
        self.assertAlmostEqual(target['box'][1], 53, delta=2)
        self.physical.assert_not_called()

    def test_occlusion_retention_expiry_and_automatic_recovery(self):
        self.assertFalse(self.flow_packet(np.zeros_like(self.image)))
        self.assertEqual(self.controller.visual_target()[0]['source'], 'retained')
        self.assertIsNone(self.controller.decision)
        self.packet([], dt=.5, flow_only=True)
        self.assertFalse(self.controller.visual_target())
        self.packet([], dt=.7, flow_only=True)
        self.assertEqual(self.controller.state, 'PERDIDO')
        self.packet([], dt=1.1)
        self.assertEqual(self.controller.state, 'VIGILANCIA')
        for _ in range(3):
            self.packet([self.detection], dt=.5)
        self.assertTrue(self.controller.tracking)
        self.assertEqual(self.controller.target['track_id'], 6)
        self.physical.assert_not_called()

    def test_cadence_and_bounded_optical_lock(self):
        self.gate.submitted()
        self.now += .16
        self.assertFalse(self.gate.due(tracking=True))
        self.assertTrue(self.gate.due(tracking=True, recovering=True))
        self.now += .50
        self.assertTrue(self.gate.due(tracking=True))
        stamp = self.flow['last_detection'] + self.seed['max_age_s'] + .01
        state, boxes, reason = advance_locked_flow(self.flow, self.image, self.shape, self.seed, stamp, cv2)
        self.assertIsNone(state)
        self.assertEqual(boxes, [])
        self.assertIn('reconfirmación', reason)

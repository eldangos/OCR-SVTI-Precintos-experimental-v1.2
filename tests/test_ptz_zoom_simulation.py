"""One targeted virtual-time test. No camera, model, capture or OCR is loaded."""
import unittest
from unittest.mock import patch
from modulos.ptz_simulation import PtzSimulation
from modulos.auto_ptz import AutoPtzController
from modulos.yolo_vigilance import VigilanceController


class ZoomSimulationTests(unittest.TestCase):
    def test_success_and_unreachable_target_keep_same_safety_limits(self):
        for box, expected in (([405, 291, 435, 309], 'FINALIZADO'),
                              ([495, 297, 505, 303], 'ERROR')):
            with self.subTest(expected=expected):
                now = [100.]
                events, callbacks, states = [], [], set()
                gate = VigilanceController(clock=lambda: now[0])
                gate.set_context(1, 1)
                sim = PtzSimulation(None, events.append)
                controller = AutoPtzController(gate, pulse=sim.pulse, stop=sim.stop,
                    focus=sim.focus, focus_pending=lambda: False, capture=sim.capture,
                    available=lambda: True, clock=lambda: now[0])
                sim.controller = controller
                sim.demo_box = list(box)
                controller.enable()
                limits = (controller.max_zoom_steps, controller.max_zoom_s, controller.pulse_s)
                self.assertEqual(limits, (8, 1.2, .12))
                def schedule(delay, callback):
                    callbacks.append((now[0] + delay/1000, callback))
                with patch('modulos.ptz_simulation.QTimer.singleShot', side_effect=schedule):
                    for seq in range(200):
                        end = now[0] + .3
                        while callbacks and min(t for t, _ in callbacks) <= end:
                            pending = min(callbacks, key=lambda item: item[0])
                            callbacks.remove(pending)
                            now[0] = pending[0]
                            pending[1]()
                            states.add(controller.state)
                        now[0] = end
                        controller.observe(dict(generation=1, stream_epoch=1, sequence=seq,
                            submitted_at=now[0], shape=sim.shape, detections=[
                                dict(box=list(sim.demo_box), source='yolo', track_id=1,
                                     confidence=.9, class_name='Precinto', sharpness=100)]))
                        states.add(controller.state)
                        controller.tick()
                        states.add(controller.state)
                        if controller.state in ('FINALIZADO', 'ERROR'):
                            break
                self.assertEqual(controller.state, expected, controller.reason)
                self.assertEqual((controller.max_zoom_steps, controller.max_zoom_s, controller.pulse_s), limits)
                self.assertLessEqual(controller._zooms, 8)
                self.assertLessEqual(controller._zoom_time, 1.2)
                self.assertTrue(any('zoom acumulado' in e for e in events))
                if expected == 'FINALIZADO':
                    self.assertTrue({'DETECTADO', 'CENTRANDO', 'ACERCANDO', 'ESTABILIZANDO',
                        'LISTO PARA CAPTURAR', 'CAPTURANDO', 'OCR', 'FINALIZADO'} <= states, states)
                    self.assertGreaterEqual(controller.geometry()[2], controller.target_size)
                else:
                    self.assertIn('Límite de zoom relativo', controller.reason)
                    self.assertEqual(controller._zooms, 8)
                    self.assertEqual(events[-1], 'SIM · STOP')
                    self.assertFalse(any('captura 101' in e for e in events))

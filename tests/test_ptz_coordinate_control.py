"""Coordinate feedback tests with fake PTZ callbacks. No device/model imports."""
import unittest
from test_auto_ptz import Simulator


class CoordinateControlTests(unittest.TestCase):
    def test_quadrants_and_configured_axis_directions(self):
        for x, y in ((140, 150), (860, 150), (140, 450), (860, 450)):
            for pan_sign, tilt_sign in ((1, -1), (-1, 1)):
                with self.subTest(center=(x, y), signs=(pan_sign, tilt_sign)):
                    s = Simulator(pan_sign=pan_sign, tilt_sign=tilt_sign)
                    s.confirm([x-40, y-50, x+40, y+50])
                    c = s.controller
                    ex, ey, size = c.geometry()
                    self.assertAlmostEqual(ex, (x-500)/500)
                    self.assertAlmostEqual(ey, (y-300)/300)
                    c.target['box'] = [v*2 for v in c.target['box']]
                    c.shape = [1200, 2000, 3]
                    self.assertEqual(c.geometry(), (ex, ey, size))
                    c.tick()
                    command = s.commands[-1][1]
                    self.assertEqual(command['pan'], pan_sign * (12 if x > 500 else -12))
                    self.assertEqual(command['tilt'], tilt_sign * (12 if y > 300 else -12))
                    self.assertEqual(command['duration_s'], .12)

    def test_stop_and_post_stop_frame_are_both_required(self):
        s = Simulator()
        s.confirm([100, 240, 180, 360])
        c = s.controller
        c.tick()
        count = len(s.commands)
        s.frame([100, 240, 180, 360])
        c.tick()
        self.assertEqual(len(s.commands), count)
        s.stopped()
        stop_stamp = s.now
        count = len(s.commands)
        s.now += .5
        s.seq += 1
        c.observe(dict(sequence=s.seq, generation=1, stream_epoch=1,
                       submitted_at=stop_stamp, shape=[600, 1000, 3], detections=[]))
        c.tick()
        self.assertEqual(len(s.commands), count)
        self.assertTrue(c._waiting_after_stop)
        s.frame([110, 240, 190, 360])
        c.tick()
        self.assertEqual(len(s.commands), count+1)
        s.frame()
        self.assertEqual(c.state, 'PERDIDO')
        self.assertEqual(s.commands[-1], ('stop',))

    def test_zoom_rechecks_center_and_deadband_prevents_oscillation(self):
        s = Simulator()
        s.confirm([410, 240, 490, 360])
        c = s.controller
        c.tick()
        self.assertEqual(c.state, 'ACERCANDO')
        s.stopped()
        s.frame([364, 216, 476, 384])
        c.tick()
        self.assertEqual(c.state, 'CENTRANDO')
        self.assertLess(s.commands[-1][1]['pan'], 0)
        self.assertNotIn('zoom', s.commands[-1][1])
        s = Simulator()
        s.confirm([420, 200, 560, 400])
        s.controller.tick()
        self.assertEqual(s.controller.state, 'ESTABILIZANDO')
        self.assertFalse(any(item[0] == 'pulse' for item in s.commands))

    def test_uncalibrated_physical_transport_cannot_enable(self):
        s = Simulator(physical=True)
        self.assertFalse(s.controller.enable())
        self.assertFalse(s.controller.enabled)
        self.assertIn('sin calibrar', s.controller.reason)
        self.assertEqual(s.commands, [])

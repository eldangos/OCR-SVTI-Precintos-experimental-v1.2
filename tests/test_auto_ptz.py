"""Deterministic PTZ/capture/OCR simulator. No Qt, network, camera or model.

Run: python -B tests/test_auto_ptz.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modulos.auto_ptz import AutoPtzController
from modulos.yolo_vigilance import VigilanceController


class Simulator:
    def __init__(self, **options):
        self.now, self.seq = 100., 0
        self.commands, self.photos, self.saved, self.states = [], [], [], []
        self.gate = VigilanceController(clock=lambda: self.now)
        self.gate.set_context(1, 1)
        self.controller = AutoPtzController(self.gate, pulse=self.pulse, stop=self.stop,
            focus=lambda: self.commands.append(('focus',)), focus_pending=lambda: False,
            capture=self.capture, available=lambda: True, clock=lambda: self.now, **options)

    def pulse(self, **command):
        self.commands.append(('pulse', command))
        return (1, len(self.commands))

    def stop(self):
        self.commands.append(('stop',))

    def capture(self):
        self.photos.append(len(self.photos) + 1)
        return self.photos[-1]

    def frame(self, box=None, sharpness=100, source='yolo'):
        self.now += .5
        self.seq += 1
        d = [] if box is None else [dict(box=box, confidence=.9, class_name='Precinto',
                                    track_id=7, source=source, sharpness=sharpness)]
        self.controller.observe(dict(generation=1, stream_epoch=1, sequence=self.seq,
            submitted_at=self.now, shape=[600, 1000, 3], detections=d))
        self.states.append(self.controller.state)

    def stopped(self):
        c = self.controller
        self.commands.append(('pulse-stop', c._token))
        c.motion_result(dict(token=c._token, phase='stopped', ok=True))

    def confirm(self, box):
        self.controller.enable()
        for _ in range(3): self.frame(box)
        assert self.controller.state == 'DETECTADO'

    def approach(self):
        self.confirm([100, 240, 180, 360])
        x = 100
        while x < 420:
            self.controller.tick()
            assert self.controller.state == 'CENTRANDO'
            assert self.commands[-1][1]['pan'] < 0
            self.stopped()
            x += 40
            self.frame([x, 240, x + 80, 360])
        self.controller.tick()
        assert self.controller.state == 'ACERCANDO'
        self.stopped()
        self.frame([420, 200, 560, 400])
        self.controller.tick()
        assert self.controller.state == 'ESTABILIZANDO'
        self.controller.focus_result('Modo AUTO verificado por lectura ISAPI')
        for i in range(5): self.frame([420, 200, 560, 400], source='auto-flow' if i % 2 == 0 else 'yolo')
        assert self.controller.state == 'LISTO PARA CAPTURAR'
        self.controller.tick()
        assert self.controller.state == 'CAPTURANDO'


def test_complete_cycle_with_simulated_ptz_capture_ocr_and_save():
    s = Simulator()
    assert not s.controller.enabled
    s.approach()
    assert s.photos == [1]
    assert s.controller.photo_checked(1, True)
    assert s.controller.state == 'OCR'
    s.saved.append(dict(image='simulated.png', code='SVTI123', metadata={'capture_id': 1}))
    s.controller.ocr_result(1, success=True)
    assert s.controller.state == 'FINALIZADO' and len(s.saved) == 1
    s.now += 1.1
    s.controller.tick()
    assert s.controller.state == 'VIGILANCIA'
    for _ in range(50): s.frame([420, 200, 560, 400])
    assert s.photos == [1]  # No duplicate while same seal stays in scene.


def test_loss_stops_active_motion_immediately():
    s = Simulator()
    s.confirm([100, 240, 180, 360])
    s.controller.tick()
    s.frame()
    assert s.controller.state == 'PERDIDO' and s.commands[-1] == ('stop',)
    assert not s.photos


def test_stop_and_late_ack_cannot_restart_motion():
    s = Simulator()
    s.confirm([100, 240, 180, 360])
    s.controller.tick()
    token = s.controller._token
    s.controller.disable('STOP operador')
    count = len(s.commands)
    s.controller.motion_result(dict(token=token, phase='stopped', ok=True))
    for _ in range(10): s.controller.tick()
    assert len(s.commands) == count and not s.controller.enabled


def test_blurred_video_never_triggers_capture():
    s = Simulator()
    s.confirm([420, 200, 560, 400])
    s.controller.tick()
    for _ in range(10): s.frame([420, 200, 560, 400], sharpness=1)
    assert not s.photos


def test_bad_photo_and_ocr_retries_are_bounded():
    s = Simulator(retries=1)
    s.approach()
    assert not s.controller.photo_checked(1, False)
    s.controller.focus_result('Modo AUTO verificado por lectura ISAPI')
    for _ in range(5): s.frame([420, 200, 560, 400])
    s.controller.tick()
    assert s.photos == [1, 2]
    s.controller.photo_checked(2, True)
    s.controller.ocr_result(2, success=False)
    assert s.controller.state == 'ERROR' and not s.controller.enabled


def test_zoom_and_attempt_limits_do_not_capture_distant_target():
    s = Simulator(max_zoom_steps=1)
    s.confirm([460, 240, 540, 360])
    s.controller.tick()
    s.stopped()
    s.frame([460, 240, 540, 360])
    s.controller.tick()
    assert s.controller.state == 'ERROR' and not s.photos


def test_no_frame_or_unconfirmed_stop_aborts():
    s = Simulator()
    s.confirm([100, 240, 180, 360])
    s.controller.tick()
    s.now += 2
    s.controller.tick()
    assert s.controller.state == 'PERDIDO' and s.commands[-1] == ('stop',)


def test_single_detection_or_manual_roi_does_not_move():
    s = Simulator()
    s.controller.enable()
    s.frame([100, 240, 180, 360])
    for _ in range(10):
        s.frame([100, 240, 180, 360], source='manual-flow')
        s.controller.tick()
    assert not s.commands and not s.photos


def test_ocr_box_or_different_identity_cannot_continue_zoom():
    for source, target_id in [('ocr', 7), ('manual-flow', 7), ('yolo', 8)]:
        s = Simulator()
        box = [460, 240, 540, 360]
        s.confirm(box)
        s.controller.tick()
        assert s.controller.state == 'ACERCANDO'
        s.controller.observe(dict(generation=1, stream_epoch=1, sequence=10,
            submitted_at=s.now, shape=[600, 1000, 3], detections=[
                dict(box=box, source=source, class_name='Precinto', track_id=target_id, confidence=.9)]))
        assert s.controller.state == 'PERDIDO' and s.commands[-1] == ('stop',)


def test_stale_result_cancels_unfinished_zoom():
    s = Simulator()
    s.confirm([460, 240, 540, 360])
    s.controller.tick()
    s.controller.observe(dict(sequence=15, generation=1, stream_epoch=1,
                              submitted_at=s.now-3, detections=[]))
    assert s.controller.state == 'PERDIDO' and s.commands[-1] == ('stop',)


def test_preview_never_captures_and_ready_is_revoked_on_motion():
    s = Simulator(capture_allowed=lambda: False)
    s.confirm([420, 200, 560, 400])
    s.controller.tick()
    s.controller.focus_result('Modo AUTO verificado')
    for _ in range(6):
        s.frame([420, 200, 560, 400])
        s.controller.tick()
    assert s.controller.state == 'LISTO PARA CAPTURAR' and not s.photos
    s.frame([480, 200, 620, 400], sharpness=0)
    s.controller.capture_allowed = lambda: True
    s.controller.tick()
    assert not s.photos and s.controller.state == 'ESTABILIZANDO'


if __name__ == '__main__':
    tests = [v for n, v in list(globals().items()) if n.startswith('test_')]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} simulated automatic PTZ checks passed; no physical camera')

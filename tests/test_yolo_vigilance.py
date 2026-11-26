"""Targeted, dependency-free capture simulation. Never opens a camera or model.

Run: python -B tests/test_yolo_vigilance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modulos.yolo_vigilance import VigilanceController


class Simulator:
    def __init__(self, **settings):
        self.now = 100.
        self.calls = 0
        self.sequence = 0
        self.gate = VigilanceController(clock=lambda: self.now, **settings)
        self.gate.set_context(1, 2)

    def capture(self):
        assert self.gate.processing and not self.gate.armed
        self.calls += 1
        return True

    def frame(self, *, boxes=True, source='yolo', confidence=.9, box=None,
              available=True, age=0, generation=1, track_id=7):
        self.sequence += 1
        self.now += .5
        packet = dict(generation=generation, stream_epoch=2, sequence=self.sequence,
                      submitted_at=self.now - age, shape=[120, 160, 3], detections=[])
        if boxes:
            packet['detections'] = [dict(box=box or [40, 30, 65, 80], source=source,
                                         confidence=confidence, class_name='Precinto', track_id=track_id)]
        return self.gate.observe(packet, self.capture, available=available)


def test_stable_detection_captures_once_and_pauses_inference():
    s = Simulator()
    assert not s.frame() and not s.frame() and s.frame()
    assert s.calls == 1 and not s.gate.due()
    for _ in range(100):
        s.frame()
    assert s.calls == 1


def test_single_false_positive_and_unstable_boxes_do_not_capture():
    s = Simulator()
    s.frame()
    s.frame(boxes=False)
    for i in range(20):
        s.frame(box=[10, 10, 30, 40] if i % 2 else [80, 40, 100, 70])
    assert s.calls == 0


def test_flow_manual_and_low_confidence_cannot_confirm():
    s = Simulator()
    s.frame()
    for _ in range(10):
        s.frame(source='auto-flow')
        s.frame(source='manual-flow')
        s.frame(confidence=.3)
    assert s.calls == 0


def test_stationary_seal_and_id_change_never_retrigger_after_cooldown():
    s = Simulator(cooldown_s=2)
    for _ in range(3): s.frame()
    s.gate.finished()
    for i in range(80): s.frame(track_id=i)
    assert s.calls == 1 and not s.gate.armed and s.gate.due()


def test_absence_and_cooldown_rearm_then_full_confirmation_is_required():
    s = Simulator(cooldown_s=4, absence_s=2)
    for _ in range(3): s.frame()
    s.gate.finished()
    for _ in range(5): s.frame(boxes=False)
    assert not s.gate.armed
    for _ in range(4): s.frame(boxes=False)
    assert s.gate.armed
    assert not s.frame() and not s.frame() and s.frame()
    assert s.calls == 2


def test_no_result_is_not_absence_and_restart_does_not_rearm():
    s = Simulator(cooldown_s=1, absence_s=1)
    for _ in range(3): s.frame()
    s.gate.finished()
    s.frame(boxes=False)
    s.now += 30
    s.gate.invalidate()
    s.gate.set_context(2, 2)
    for _ in range(10): s.frame(generation=2)
    assert s.calls == 1 and not s.gate.armed


def test_missing_frames_break_absence_evidence():
    s = Simulator(cooldown_s=1, absence_s=2)
    for _ in range(3): s.frame()
    s.gate.finished()
    s.frame(boxes=False)
    s.now += 30
    s.frame(boxes=False)
    assert not s.gate.armed


def test_stale_wrong_context_and_invalid_boxes_never_capture():
    s = Simulator()
    for _ in range(10):
        s.frame(age=10)
        s.frame(generation=0)
        s.frame(box=[0, 0, 160, 120])
        s.frame(box=[float('nan'), 0, 20, 30])
    assert s.calls == 0


def test_unavailable_capture_requires_fresh_confirmation_when_ready():
    s = Simulator()
    for _ in range(20): s.frame(available=False)
    assert s.calls == 0
    assert not s.frame() and not s.frame() and s.frame()


def test_gap_between_confirmations_resets_hits():
    s = Simulator()
    s.frame()
    s.frame()
    s.now += 10
    assert not s.frame() and s.calls == 0


def test_scheduler_limits_frequency():
    s = Simulator(interval_s=1)
    assert s.gate.due()
    s.gate.submitted()
    for _ in range(29):
        s.now += .03
        assert not s.gate.due()
    s.now += .2
    assert s.gate.due()


def test_rejected_capture_is_latched_without_busy_state():
    s = Simulator()
    s.capture = lambda: False
    for _ in range(10): s.frame()
    assert not s.gate.processing and not s.gate.armed


def test_duplicate_sequence_does_not_count_as_confirmation():
    s = Simulator()
    for _ in range(10):
        s.sequence = 0
        s.frame()
    assert s.calls == 0


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} capture simulator checks passed; no camera/Qt/model loaded')

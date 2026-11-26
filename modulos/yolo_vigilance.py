"""Low-rate surveillance and conservative capture gate, independent of Qt/camera."""

import math
import time


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def covers_frame(box, width, height):
    """Reject frame-sized boxes, while allowing large close-up seals."""
    x1, y1, x2, y2 = box
    return x2 - x1 >= .95 * width and y2 - y1 >= .95 * height


class VigilanceController:
    def __init__(self, *, interval_s=.5, confirm_hits=3, confirm_s=.8,
                 confidence=.65, cooldown_s=15, absence_s=3,
                 max_gap_s=1.5, max_age_s=2, stable_iou=.5, clock=time.monotonic,
                 track_interval_s=.30, correct_interval_s=None):
        correct_interval_s = track_interval_s if correct_interval_s is None else correct_interval_s
        values = (interval_s, confirm_s, confidence, cooldown_s, absence_s,
                  max_gap_s, max_age_s, stable_iou, track_interval_s, correct_interval_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Los parámetros de vigilancia deben ser finitos')
        if (interval_s <= 0 or track_interval_s <= 0 or correct_interval_s < track_interval_s
                or confirm_hits < 2 or confirm_s <= 0
                or cooldown_s <= 0 or absence_s <= 0 or max_gap_s <= 0
                or max_age_s <= 0 or not 0 < confidence <= 1 or not 0 < stable_iou <= 1):
            raise ValueError('Parámetros de vigilancia fuera de rango')
        self.interval_s, self.confirm_hits, self.confirm_s = interval_s, confirm_hits, confirm_s
        self.track_interval_s = track_interval_s
        self.correct_interval_s = correct_interval_s
        self.confidence, self.cooldown_s, self.absence_s = confidence, cooldown_s, absence_s
        self.max_gap_s, self.max_age_s, self.stable_iou = max_gap_s, max_age_s, stable_iou
        self.clock = clock
        self.context = None
        self.processing = False
        self.armed = True
        self.cooldown_until = 0
        self._last_submit = float('-inf')
        self.invalidate()

    def invalidate(self):
        """Lost video/restart is NOT evidence that the seal left the scene."""
        self._candidate = None
        self._absent_since = self._last_empty = None
        self._sequence = -1
        self.confirmed_target = None
        self.reason = 'Sin resultados recientes'

    def set_context(self, generation, epoch):
        if self.context != (generation, epoch):
            self.context = (generation, epoch)
            self.invalidate()

    def due(self, *, tracking=False, recovering=False):
        if recovering or self.confirming:
            interval = min(self.interval_s, self.track_interval_s)
        else:
            interval = self.correct_interval_s if tracking else self.interval_s
        return (tracking or not self.processing) and self.clock() - self._last_submit >= interval

    def submitted(self):
        self._last_submit = self.clock()

    def finished(self, *, success=False):
        if self.processing or success:
            self.processing = False
            if success:
                self.armed = False
                self.cooldown_until = self.clock() + self.cooldown_s
            self.invalidate()

    def started(self):
        self.processing = True
        self.invalidate()

    @property
    def confirming(self):
        return self.armed and not self.processing and self._candidate is not None

    @property
    def status(self):
        if self.processing:
            return 'Procesando captura / OCR'
        if not self.armed:
            wait = max(0., self.cooldown_until - self.clock())
            return f'Objetivo ya procesado · espera {wait:.1f} s · requiere {self.absence_s:.1f} s de ausencia observada'
        if self._candidate:
            return f'Confirmando precinto ({self._candidate["hits"]}/{self.confirm_hits})'
        return 'Buscando precinto · ' + self.reason

    def observe(self, packet, capture, *, available=True):
        """capture() returns True only if it actually accepted the request."""
        now = self.clock()
        if self.processing or (packet.get('generation'), packet.get('stream_epoch')) != self.context:
            return False
        stamp, sequence = packet.get('submitted_at'), packet.get('sequence')
        if (not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                or not 0 <= now - stamp <= self.max_age_s):
            self.invalidate()
            return False
        if not isinstance(sequence, int) or sequence <= self._sequence:
            return False
        self._sequence = sequence
        shape = packet.get('shape', [])
        raw = packet.get('detections', [])
        if (not isinstance(shape, (list, tuple)) or len(shape) < 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in shape[:2])
                or not isinstance(raw, list)):
            self._candidate = None
            self._absent_since = None
            return False
        height, width = shape[:2]
        visible, real = [], []
        for item in raw:
            if not isinstance(item, dict):
                continue
            box, conf = item.get('box', []), item.get('confidence')
            if (str(item.get('class_name', '')).strip().lower() != 'precinto'
                    or not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            if covers_frame(box, width, height):
                continue
            visible.append(item)
            # KLT/manual boxes may bridge visualization, never count as detector confirmation.
            if (item.get('source') == 'yolo' and isinstance(conf, (int, float))
                    and math.isfinite(conf) and self.confidence <= conf <= 1):
                real.append(item)
        if not visible:
            self.reason = 'Sin detecciones YOLO' if not raw else 'Cajas rechazadas por geometría/clase'
            self._candidate = None
            if raw:  # Malformed/filtered output must not rearm the trigger.
                self._absent_since = self._last_empty = None
                return False
            if self._last_empty is None or now - self._last_empty > self.max_gap_s:
                self._absent_since = now
            self._last_empty = now
            if (not self.armed and now >= self.cooldown_until
                    and now - self._absent_since >= self.absence_s):
                self.armed = True
            return False
        self._absent_since = self._last_empty = None
        if not self.armed or not available:
            self._candidate = None
            return False
        candidate = self._candidate
        if candidate and now - candidate['last'] > self.max_gap_s:
            candidate = self._candidate = None
        if not real:
            scores = [d.get('confidence') for d in visible if d.get('source') == 'yolo']
            scores = [v for v in scores if isinstance(v, (int, float)) and math.isfinite(v)]
            self.reason = (f'Confianza insuficiente: {max(scores):.2f} < {self.confidence:.2f}'
                           if scores and max(scores) < self.confidence else 'Solo seguimiento; esperando YOLO')
            return False
        matching = [d for d in real if candidate and
                    (candidate['track_id'] is None or d.get('track_id') == candidate['track_id']) and
                    overlap(d['box'], candidate['anchor']) >= self.stable_iou and
                    overlap(d['box'], candidate['box']) >= self.stable_iou]
        best = max(matching or real, key=lambda d: d['confidence'])
        if not matching:
            self.reason = ('Confirmación reiniciada por cambio de ID o geometría'
                           if candidate else 'Primer candidato válido')
            self._candidate = dict(anchor=best['box'], box=best['box'], hits=1, first=now,
                                   last=now, track_id=best.get('track_id'))
            return False
        candidate.update(box=best['box'], hits=candidate['hits'] + 1, last=now)
        if candidate['hits'] < self.confirm_hits or now - candidate['first'] < self.confirm_s:
            return False
        # Latch BEFORE invoking the callback (including synchronous errors/reentrancy).
        self.started()
        self.confirmed_target = dict(best)
        try:
            accepted = bool(capture())
        except Exception:
            self.finished()
            raise
        if not accepted:
            self.finished()
        return accepted
